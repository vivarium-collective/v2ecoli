"""Multi-generation baseline parquet sim for one condition.

Adapts run_acetate_multigen.py's daughter-bundle pattern + the per-gen
ctx mgr pattern in studies/stage1_sanity_check/run_sim.py for parquet.
Each generation wraps in its own ``parquet_emitter()`` context so the
metadata (generation, agent_id) is fresh per gen.

Agent IDs follow vEcoli's lineage convention: gen N uses agent_id "0"*N
("0", "00", "000", ...). This way ``generation = len(agent_id)`` matches
vEcoli's parquet partitioning semantics.

Transient daughter emitters spawned inside the composite at division time
write to ``gen=N+1/agent_id=00 or 01/`` partitions (see Division step's
metadata override). When the next runner-driven gen opens its ctx mgr,
its ``_write_configuration`` wipes whichever of those partitions matches —
so the runner-driven sim's history is the source of truth per gen.

Usage:
    python scripts/run_condition_multigen_parquet.py \\
        --cache-dir out/cache_glycerol_stage1_all_knobs \\
        --out-dir out/glycerol_stage1_no_crp_5gen_parquet \\
        --experiment-id glycerol_stage1_no_crp_5gen \\
        --generations 5 \\
        --max-min 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import dill

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from process_bigraph import Composite

from v2ecoli import build_composite
from v2ecoli.composites._helpers import parquet_emitter
from v2ecoli.composites.baseline import baseline as baseline_doc
from v2ecoli.core import build_core
from v2ecoli.library.division import divide_cell
from v2ecoli.library.parquet_emitter import ParquetEmitter


SNAPSHOT_INTERVAL = 60


def _last_cell_state(comp: Composite) -> dict | None:
    """Snapshot the parent agent's biological state for divide_cell()."""
    cell = comp.state.get("agents", {}).get("0")
    if cell is None:
        return None
    keys = {"bulk", "unique", "listeners", "environment", "boundary",
            "global_time", "timestep", "divide", "division_threshold",
            "process_state", "allocator_rng"}
    return {k: v for k, v in cell.items()
            if k in keys or k.startswith("request_") or k.startswith("allocate_")}


def _run_gen(comp: Composite, max_duration: int, gen_idx: int) -> tuple[float, bool, dict | None]:
    total = 0.0
    divided = False
    last_state = _last_cell_state(comp)
    cell0 = comp.state["agents"]["0"]
    dry0 = float(cell0["listeners"]["mass"].get("dry_mass", 0))
    print(f"    gen {gen_idx}: initial dry_mass={dry0:.1f} fg")

    while total < max_duration:
        chunk = min(SNAPSHOT_INTERVAL, max_duration - total)
        try:
            comp.run(chunk)
        except Exception as e:
            err = str(e)
            if "divide" in err.lower() or "_add" in err or "_remove" in err:
                total += chunk
                divided = True
                break
            if comp.state.get("agents", {}).get("0") is None:
                divided = True
                break
            raise
        total += chunk
        cur = comp.state.get("agents", {}).get("0")
        if cur is None:
            divided = True
            break
        last_state = _last_cell_state(comp)
        if int(total) % 600 == 0 or total >= max_duration - 60:
            dry = float(cur["listeners"]["mass"].get("dry_mass", 0))
            print(f"    gen {gen_idx}: t={total/60:5.1f} min  dry_mass={dry:.1f} fg")

    # Pre-divide finalize hook (in division.py) already closed the parent
    # emitter for us if divided=True, but call flush_all_in_composite as a
    # belt-and-suspenders no-op for any in-composite daughter emitters or
    # the non-divided edge case.
    n_closed = ParquetEmitter.flush_all_in_composite(comp, success=divided)
    if n_closed:
        print(f"    gen {gen_idx}: flushed {n_closed} ParquetEmitter instance(s)")

    return total, divided, last_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="Parquet out-dir root (under <root>/<experiment-id>/...)")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--max-min", type=float, default=200.0,
                    help="Max duration per generation, minutes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dill-dir", default=None,
                    help="Optional dir to dump per-gen daughter dills "
                         "(default: out/<experiment-id>/gen_dills)")
    args = ap.parse_args()

    max_duration = int(args.max_min * 60)
    dill_dir = args.dill_dir or f"out/{args.experiment_id}/gen_dills"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(dill_dir, exist_ok=True)

    print("=" * 60)
    print(f"Multi-gen lineage — {args.generations} gens, seed={args.seed}")
    print(f"  cache:        {args.cache_dir}")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  experiment:   {args.experiment_id}")
    print(f"  max/gen:      {args.max_min:.0f} min")
    print("=" * 60)

    # Load configs/unique_names/dry_mass_inc once — same cache feeds every gen.
    with open(os.path.join(args.cache_dir, "sim_data_cache.dill"), "rb") as f:
        cache = dill.load(f)
    try:
        from v2ecoli.library.unit_bridge import rebind_cache_quantities
        rebind_cache_quantities(cache)
    except ImportError:
        pass
    configs = cache.get("configs", {})
    unique_names = cache.get("unique_names", [])
    dry_mass_inc = cache.get("dry_mass_inc_dict", {})

    core = build_core()
    t_pipeline = time.time()
    summary = []
    prev_cell_data = None

    for gen_idx in range(1, args.generations + 1):
        agent_id = "0" * gen_idx
        gen_label = f"gen {gen_idx} (agent_id={agent_id})"
        print(f"\n[{time.strftime('%H:%M:%S')}] {gen_label}")
        t0 = time.time()

        with parquet_emitter(
            out_dir=args.out_dir,
            experiment_id=args.experiment_id,
            lineage_seed=args.seed,
            agent_id=agent_id,
            generation=gen_idx,
        ):
            t_build = time.time()
            if gen_idx == 1:
                comp = build_composite("baseline", cache_dir=args.cache_dir)
            else:
                d1_state, _d2_state = divide_cell(prev_cell_data)
                bundle = {
                    "initial_state": d1_state,
                    "configs": configs,
                    "unique_names": unique_names,
                    "dry_mass_inc_dict": dry_mass_inc,
                }
                doc = baseline_doc(core=core, seed=args.seed + gen_idx,
                                   cache_dir=args.cache_dir, bundle=bundle)
                comp = Composite(doc, core=core)
            print(f"    composite built in {time.time()-t_build:.1f}s")

            duration, divided, last_state = _run_gen(comp, max_duration, gen_idx)

        dry_f = float((last_state or {}).get("listeners", {}).get("mass", {}).get("dry_mass", 0))
        wall = time.time() - t0
        result = {
            "gen": gen_idx,
            "agent_id": agent_id,
            "duration_min": duration / 60.0,
            "divided": divided,
            "final_dry_mass_fg": dry_f,
            "wall_seconds": wall,
        }
        summary.append(result)
        print(f"    gen {gen_idx} summary: tau={duration/60:.1f} min  "
              f"final_dry={dry_f:.1f} fg  divided={divided}  wall={wall:.0f}s")

        # Persist daughter state for the next gen / for restart.
        prev_cell_data = last_state
        if last_state is not None:
            dill_path = os.path.join(dill_dir, f"gen{gen_idx}.dill")
            with open(dill_path, "wb") as f:
                dill.dump(last_state, f)

        if not divided:
            print(f"    gen {gen_idx} did not divide — stopping lineage.")
            break

    pipeline_wall = time.time() - t_pipeline
    out_summary = {
        "experiment_id": args.experiment_id,
        "cache_dir": args.cache_dir,
        "generations_requested": args.generations,
        "generations_completed": len(summary),
        "pipeline_wall_seconds": pipeline_wall,
        "gens": summary,
    }
    summary_path = os.path.join(args.out_dir, f"{args.experiment_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE — {len(summary)} gens, {pipeline_wall/60:.1f} min wall")
    for s in summary:
        print(f"  gen {s['gen']}: tau={s['duration_min']:5.1f} min  "
              f"final_dry={s['final_dry_mass_fg']:7.1f} fg  "
              f"divided={s['divided']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
