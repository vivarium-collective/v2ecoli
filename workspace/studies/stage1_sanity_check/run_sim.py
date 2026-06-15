"""Stage 1 sanity-check multi-generation simulation runner.

Mirrors ``reports/multigeneration_report.run_multigeneration`` but wraps
each generation's composite construction in its own ``sqlite_emitter()``
context. This:
  - Routes the rich emit schema (bulk + listeners.mass +
    listeners.monomer_counts + listeners.rna_counts +
    listeners.replication_data + listeners.rnap_data +
    listeners.rna_synth_prob + full_chromosome + active_replisome +
    active_RNAP + chromosome_domain) into a queryable SQLite DB.
  - Gives each generation its own ``simulation_id`` (so the history
    table's ``(simulation_id, step)`` primary key doesn't collide).
  - Writes all generations to the same DB file (``--db-path``) for easy
    downstream stitching.

Usage:
    python studies/stage1_sanity_check/run_sim.py \\
        --cache-dir   out/cache_stage1 \\
        --generations 5 \\
        --seed        0 \\
        --max-duration 12000 \\
        --db-path     out/stage1_sanity_runs.db
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reports.multigeneration_report import (
    GenerationResult,
    MAX_GENERATION_DURATION,
    SNAPSHOT_INTERVAL,
    _extract_cell_data,
    _get_emitter_instance,
)
from v2ecoli.composites._helpers import sqlite_emitter, parquet_emitter


def _run_generation(composite, gen_idx, max_duration):
    """Like ``reports.multigeneration_report._run_generation`` but tolerates
    emitters that don't expose a ``.history`` attribute (SQLiteEmitter
    persists to disk; RAMEmitter is in-memory). When history is unavailable
    the final dry_mass falls back to the composite's current state."""
    cell = composite.state["agents"]["0"]
    initial_dry = float(cell["listeners"]["mass"].get("dry_mass", 0))

    # Keep an emitter handle for the RAMEmitter case; ignored otherwise.
    emitter_instance = _get_emitter_instance(composite)

    t_wall0 = time.time()
    total_run = 0.0
    divided = False
    last_cell_data = None

    while total_run < max_duration:
        chunk = min(SNAPSHOT_INTERVAL, max_duration - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            err_str = str(e)
            if (
                "divide" in err_str.lower()
                or "_add" in err_str
                or "_remove" in err_str
            ):
                total_run += chunk
                divided = True
                break
            if composite.state.get("agents", {}).get("0") is None:
                divided = True
                break
            raise
        total_run += chunk
        cur_cell = composite.state.get("agents", {}).get("0")
        if cur_cell is None:
            divided = True
            break
        last_cell_data = _extract_cell_data(cur_cell)

    wall_time = time.time() - t_wall0

    # Parquet / async emitters buffer batches; call close() so any
    # unflushed rows hit disk before we tear down the composite. No-ops
    # for emitters that don't define close().
    if emitter_instance is not None and hasattr(emitter_instance, "close"):
        try:
            emitter_instance.close(success=True)
        except TypeError:
            # Older emitters don't accept the success kwarg.
            emitter_instance.close()

    # Final dry_mass: prefer in-memory history if present, otherwise read
    # from the cell's current state.
    history = getattr(emitter_instance, "history", None) or []
    final_dry = 0.0
    for snap in reversed(history):
        m = snap.get("listeners", {}).get("mass", {}) if isinstance(snap.get("listeners"), dict) else {}
        d = float(m.get("dry_mass", 0))
        if d > 0:
            final_dry = d
            break
    if final_dry == 0.0:
        final_dry = float(
            composite.state.get("agents", {})
            .get("0", {})
            .get("listeners", {})
            .get("mass", {})
            .get("dry_mass", 0)
        )

    return GenerationResult(
        index=gen_idx,
        duration=total_run,
        wall_time=wall_time,
        divided=divided,
        initial_dry_mass=initial_dry,
        final_dry_mass=final_dry,
        snapshots=[],  # SQLiteEmitter writes to disk; readback happens in plotting phase
        cell_data_after=last_cell_data,
    )


def run_multigeneration_sqlite(
    n_generations: int,
    max_duration_per_gen: float,
    cache_dir: str,
    db_dir: str,
    db_file: str,
    db_name_prefix: str = "stage1-sanity",
    seed: int = 0,
    emitter: str = "sqlite",
    parquet_out_dir: str | None = None,
    experiment_id: str = "default",
) -> list[GenerationResult]:
    """Run ``n_generations`` cells; each gen wrapped in its own emitter
    context (unique simulation_id) writing to the shared DB / parquet store.

    ``emitter`` selects between ``"sqlite"`` (default, writes to
    ``<db_dir>/<db_file>``) and ``"parquet"`` (writes hive-partitioned
    parquet files under ``parquet_out_dir`` using the vEcoli-compatible
    preset; ~30-50× smaller than SQLite at full tick resolution)."""
    from v2ecoli import build_composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline, seed_mass_listener
    from v2ecoli.library.division import divide_cell
    from process_bigraph import Composite

    def _emitter_ctx(gen_idx: int):
        if emitter == "parquet":
            assert parquet_out_dir is not None, "parquet_out_dir is required when emitter='parquet'"
            return parquet_emitter(
                out_dir=parquet_out_dir,
                experiment_id=experiment_id,
                lineage_seed=seed,
                generation=gen_idx,
                agent_id=str(gen_idx),
            )
        return sqlite_emitter(
            file_path=db_dir,
            db_file=db_file,
            name=f"{db_name_prefix}-seed{seed}-gen{gen_idx}",
        )

    results: list[GenerationResult] = []

    print(f"  Gen 1: building from cache {cache_dir} (emitter={emitter})")
    with _emitter_ctx(1):
        composite = build_composite("baseline", cache_dir=cache_dir)
        cell0 = composite.state["agents"]["0"]
        print(
            f"    initial dry_mass={cell0['listeners']['mass'].get('dry_mass', 0):.1f} fg"
        )
        gen = _run_generation(composite, 1, max_duration_per_gen)
        print(
            f"    gen 1: {gen.wall_time:.0f}s wall, sim {gen.duration:.0f}s, "
            f"dry_mass {gen.initial_dry_mass:.0f}→{gen.final_dry_mass:.0f}, "
            f"divided={gen.divided}"
        )
    results.append(gen)
    prev_cell_data = gen.cell_data_after

    for gen_idx in range(2, n_generations + 1):
        if prev_cell_data is None or "bulk" not in prev_cell_data:
            print(f"    gen {gen_idx}: no prior cell state — stopping")
            break
        print(f"  Gen {gen_idx}: dividing previous cell, keeping daughter 1")
        d1_state, _d2_state = divide_cell(prev_cell_data)

        with _emitter_ctx(gen_idx):
            t_build0 = time.time()
            core = build_core()
            doc = baseline(core=core, seed=gen_idx, cache_dir=cache_dir)
            agent = doc["state"]["agents"]["0"]
            for key in ("bulk", "unique", "environment", "boundary"):
                if key in d1_state:
                    agent[key] = d1_state[key]
            agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
            seed_mass_listener(agent, core)
            composite = Composite(doc, core=core)
            build_time = time.time() - t_build0
            print(f"    built daughter composite in {build_time:.1f}s")

            gen = _run_generation(composite, gen_idx, max_duration_per_gen)
            print(
                f"    gen {gen_idx}: {gen.wall_time:.0f}s wall, "
                f"sim {gen.duration:.0f}s, "
                f"dry_mass {gen.initial_dry_mass:.0f}→{gen.final_dry_mass:.0f}, "
                f"divided={gen.divided}"
            )
        results.append(gen)
        prev_cell_data = gen.cell_data_after

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="out/cache_stage1")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=12000)
    parser.add_argument("--db-path", default="out/stage1_sanity_runs.db",
                        help="SQLite DB path (used only when --emitter=sqlite).")
    parser.add_argument("--db-name-prefix", default="stage1-sanity")
    parser.add_argument("--emitter", choices=["sqlite", "parquet"],
                        default="sqlite",
                        help="Choose the v2ecoli emitter. Parquet is ~30-50× "
                             "smaller on disk at full tick resolution.")
    parser.add_argument("--parquet-out-dir", default=None,
                        help="Hive-partitioned parquet output dir (required when "
                             "--emitter=parquet).")
    parser.add_argument("--experiment-id", default="default",
                        help="experiment_id metadata field for parquet partitioning.")
    args = parser.parse_args()

    db_dir = os.path.dirname(os.path.abspath(args.db_path))
    db_file = os.path.basename(args.db_path)

    if args.emitter == "parquet":
        if not args.parquet_out_dir:
            parser.error("--parquet-out-dir is required when --emitter=parquet")
        os.makedirs(args.parquet_out_dir, exist_ok=True)
        sink_desc = f"parquet @ {args.parquet_out_dir}"
    else:
        os.makedirs(db_dir, exist_ok=True)
        sink_desc = f"sqlite @ {args.db_path}"

    print("=" * 60)
    print(f"Stage 1 sanity-check sim — {args.generations} gens, seed={args.seed}")
    print(f"  cache: {args.cache_dir}")
    print(f"  sink:  {sink_desc}")
    print(f"  max_duration_per_gen: {args.max_duration:.0f}s")
    print("=" * 60)

    t0 = time.time()
    results = run_multigeneration_sqlite(
        n_generations=args.generations,
        max_duration_per_gen=args.max_duration,
        cache_dir=args.cache_dir,
        db_dir=db_dir,
        db_file=db_file,
        db_name_prefix=args.db_name_prefix,
        seed=args.seed,
        emitter=args.emitter,
        parquet_out_dir=args.parquet_out_dir,
        experiment_id=args.experiment_id,
    )
    wall = time.time() - t0

    print()
    print("=" * 60)
    print(f"Wall time: {wall:.0f}s ({wall/60:.1f} min)")
    print(f"Sink:      {sink_desc}")
    print(f"Per-gen results:")
    for r in results:
        print(
            f"  gen {r.index}: {r.duration:6.0f}s sim, "
            f"divided={r.divided!s:5s}, "
            f"dry_mass {r.initial_dry_mass:5.0f}→{r.final_dry_mass:5.0f} fg"
        )


if __name__ == "__main__":
    main()
