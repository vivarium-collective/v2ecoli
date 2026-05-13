"""v2ecoli multigeneration report.

Runs a single cell to division, keeps exactly one daughter, runs that to
division, keeps one of its daughters, etc. — for a configurable number
of generations. Dispatches the concatenated trajectory (rows tagged by
``generation``) to
v2ecoli.visualizations.multigeneration.MultigenerationVisualization.

    python reports/multigeneration_report.py --generations 3

Output: out/multigeneration/multigeneration_report.html
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


OUTPUT_DIR = "out/multigeneration"
SNAPSHOT_INTERVAL = 50  # seconds between mass captures
MAX_GENERATION_DURATION = 3600  # safety cap per generation

# Keys copied when carrying a divided daughter state forward.
_CELL_DATA_KEYS = {
    "bulk",
    "unique",
    "listeners",
    "environment",
    "boundary",
    "global_time",
    "timestep",
    "divide",
    "division_threshold",
    "process_state",
    "allocator_rng",
}


@dataclass
class GenerationResult:
    index: int
    duration: float
    wall_time: float
    divided: bool
    initial_dry_mass: float
    final_dry_mass: float
    snapshots: list[dict] = field(default_factory=list)
    cell_data_after: dict[str, Any] | None = None


def _get_emitter_instance(composite):
    """Return the emitter instance for agent 0 (while the agent still exists)."""
    cell = composite.state["agents"].get("0")
    if not cell:
        return None
    emitter_edge = cell.get("emitter", {})
    if isinstance(emitter_edge, dict):
        return emitter_edge.get("instance")
    return None


def _snapshots_from_history(history) -> list[dict]:
    """Turn an emitter history list into mass snapshots."""
    snaps = []
    for snap in history:
        t = snap.get("global_time", 0)
        if int(t) % SNAPSHOT_INTERVAL != 0 and t != 1:
            continue
        mass = (
            snap.get("listeners", {}).get("mass", {})
            if isinstance(snap.get("listeners"), dict)
            else {}
        )
        snaps.append(
            {
                "time": float(t),
                "dry_mass": float(mass.get("dry_mass", 0)),
                "cell_mass": float(mass.get("cell_mass", 0)),
                "protein_mass": float(mass.get("protein_mass", 0)),
                "dna_mass": float(mass.get("dna_mass", 0)),
                "rRna_mass": float(mass.get("rRna_mass", 0)),
                "tRna_mass": float(mass.get("tRna_mass", 0)),
                "mRna_mass": float(mass.get("mRna_mass", 0)),
                "smallMolecule_mass": float(mass.get("smallMolecule_mass", 0)),
            }
        )
    return snaps


def _extract_cell_data(cell: dict) -> dict[str, Any]:
    """Copy the state keys needed to seed the next generation."""
    return {
        k: v
        for k, v in cell.items()
        if k in _CELL_DATA_KEYS
        or k.startswith("request_")
        or k.startswith("allocate_")
    }


def _run_generation(
    composite,
    gen_idx: int,
    max_duration: float,
) -> GenerationResult:
    """Run the composite forward in SNAPSHOT_INTERVAL chunks until division
    or the per-generation duration cap. Returns a GenerationResult plus
    the last observed cell_data snapshot (for carrying into the next
    generation)."""
    cell = composite.state["agents"]["0"]
    initial_dry = float(cell["listeners"]["mass"].get("dry_mass", 0))

    # Grab the emitter instance NOW — the agent node (and our edge handle)
    # gets detached from composite.state once the Division step fires.
    emitter_instance = _get_emitter_instance(composite)

    t_wall0 = time.time()
    total_run = 0.0
    divided = False
    last_cell_data: dict[str, Any] | None = None

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
            raise
        total_run += chunk

        cur_cell = composite.state.get("agents", {}).get("0")
        if cur_cell is None:
            divided = True
            break
        last_cell_data = _extract_cell_data(cur_cell)

    wall_time = time.time() - t_wall0
    history = emitter_instance.history if emitter_instance is not None else []
    snaps = _snapshots_from_history(history)
    # Drop post-division snapshots where the listener has been reset
    # (agent removed) — we want the mass at the moment of division.
    snaps = [s for s in snaps if s.get("dry_mass", 0) > 0]
    final_dry = (
        snaps[-1]["dry_mass"]
        if snaps
        else float(
            composite.state.get("agents", {})
            .get("0", {})
            .get("listeners", {})
            .get("mass", {})
            .get("dry_mass", 0)
        )
    )

    return GenerationResult(
        index=gen_idx,
        duration=total_run,
        wall_time=wall_time,
        divided=divided,
        initial_dry_mass=initial_dry,
        final_dry_mass=final_dry,
        snapshots=snaps,
        cell_data_after=last_cell_data,
    )


def run_multigeneration(
    n_generations: int,
    max_duration_per_gen: float,
    cache_dir: str = "out/cache",
    seed: int = 0,
) -> list[GenerationResult]:
    """Run n_generations of single-lineage cells, carrying one daughter
    forward across each division."""
    from v2ecoli import build_composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline, seed_mass_listener
    from v2ecoli.library.division import divide_cell
    from process_bigraph import Composite

    results: list[GenerationResult] = []

    # Generation 1 — start from the fresh initial state the workflow/canary use.
    print(f"  Gen 1: building from cache {cache_dir}")
    composite = build_composite("baseline", cache_dir=cache_dir)
    cell0 = composite.state["agents"]["0"]
    print(
        f"    initial dry_mass={cell0['listeners']['mass'].get('dry_mass', 0):.1f} fg"
    )

    gen = _run_generation(composite, 1, max_duration_per_gen)
    print(
        f"    gen 1: {gen.wall_time:.0f}s wall, "
        f"sim {gen.duration:.0f}s, "
        f"dry_mass {gen.initial_dry_mass:.0f}→{gen.final_dry_mass:.0f}, "
        f"divided={gen.divided}"
    )
    results.append(gen)

    prev_cell_data = gen.cell_data_after

    # Generations 2..N — divide, keep daughter 1, build a fresh composite.
    for gen_idx in range(2, n_generations + 1):
        if prev_cell_data is None or "bulk" not in prev_cell_data:
            print(f"    gen {gen_idx}: no prior cell state — stopping")
            break
        print(f"  Gen {gen_idx}: dividing previous cell, keeping daughter 1")
        d1_state, _d2_state = divide_cell(prev_cell_data)

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


def _results_to_history(results: list[GenerationResult]) -> list[dict]:
    """Convert GenerationResult snapshots to a flat history list with
    ``generation`` tags, as expected by MultigenerationVisualization."""
    history: list[dict] = []
    for gen in results:
        for snap in gen.snapshots:
            row = dict(snap)
            row["generation"] = gen.index
            history.append(row)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations to run (default: 3).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_GENERATION_DURATION,
        help="Safety cap (seconds) per generation (default: 3600).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(OUTPUT_DIR, "multigeneration_report.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        default="out/cache" if os.path.isdir("out/cache") else "out/workflow/cache",
        help="Directory holding ParCa cache.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"v2ecoli multigeneration report — {args.generations} generation(s)")
    print("=" * 60)

    t_pipeline = time.time()
    results = run_multigeneration(
        args.generations,
        args.max_duration,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    pipeline_wall = time.time() - t_pipeline

    print("  Dispatching to MultigenerationVisualization…")
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization

    history = _results_to_history(results)

    viz = MultigenerationVisualization(
        config={"title": f"v2ecoli {args.generations}-generation lineage"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": history,
        "metadata": {
            "n_generations": args.generations,
            "seed": args.seed,
            "cache_dir": args.cache_dir,
            "pipeline_wall_time": pipeline_wall,
        },
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(result["html"])

    print("=" * 60)
    print(f"Pipeline wall time: {pipeline_wall:.0f} s")
    print(f"Report: {args.out}")

    try:
        import subprocess
        subprocess.run(["open", args.out], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
