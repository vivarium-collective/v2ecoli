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
            # Post-add realize errors (e.g. growth_limits Array) fire after
            # the structural divide has already happened. If the mother agent
            # is gone, treat as divided — don't keep running the broken tree.
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


def _load_study(study_path: str) -> dict:
    """Load + lightly validate a v3-shape study.yaml driving this report.

    Expected shape::

        baseline:
        - name: <any>                     # informational
          composite: v2ecoli.composites.baseline.baseline
          params: {seed: int, cache_dir: str}
        lineage:
          generations: int                # how many divisions to chain
          max_duration: float             # per-gen safety cap (seconds)
          seed_strategy: daughter_carry_forward
        visualizations:
        - name: <any>
          address: local:MultigenerationVisualization
          config: {title: str, ...}

    Only the ``baseline`` (single entry; must be the partitioned ``baseline``
    composite — multigeneration_report hardcodes that architecture) and
    ``lineage`` blocks are required. Other fields are passed through but
    ignored.
    """
    import yaml as _yaml
    with open(study_path) as fh:
        spec = _yaml.safe_load(fh) or {}

    baseline_entries = spec.get("baseline") or []
    if len(baseline_entries) != 1:
        raise ValueError(
            f"study {study_path!r}: `baseline:` must have exactly one entry "
            f"(multigeneration_report chains a single architecture); got "
            f"{len(baseline_entries)}"
        )
    composite_ref = baseline_entries[0].get("composite") or ""
    if not composite_ref.endswith(".baseline.baseline"):
        raise ValueError(
            f"study {study_path!r}: baseline composite must be the "
            f"partitioned `baseline` architecture (e.g. "
            f"v2ecoli.composites.baseline.baseline); got {composite_ref!r}"
        )

    lineage = spec.get("lineage") or {}
    if "generations" not in lineage:
        raise ValueError(
            f"study {study_path!r}: missing required `lineage.generations`"
        )
    strategy = lineage.get("seed_strategy", "daughter_carry_forward")
    if strategy != "daughter_carry_forward":
        raise ValueError(
            f"study {study_path!r}: lineage.seed_strategy must be "
            f"'daughter_carry_forward' (the only one implemented); got "
            f"{strategy!r}"
        )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", default=None,
        help="Path to a v3-shape study.yaml driving the lineage. When set, "
             "the baseline entry's params and the lineage block (generations, "
             "max_duration) become defaults; CLI flags below still override.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations to run (default: 3, or study.lineage.generations).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Safety cap (seconds) per generation "
             "(default: 3600, or study.lineage.max_duration).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(OUTPUT_DIR, "multigeneration_report.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory holding ParCa cache "
             "(default: out/cache or out/workflow/cache, or study param; "
             "use e.g. out/cache_plasmid for a plasmid-enabled run).",
    )
    args = parser.parse_args()

    study_spec = _load_study(args.study) if args.study else None
    study_params = ((study_spec or {}).get("baseline") or [{}])[0].get("params") or {}
    study_lineage = (study_spec or {}).get("lineage") or {}

    n_generations = args.generations if args.generations is not None \
        else int(study_lineage.get("generations", 3))
    max_duration = args.max_duration if args.max_duration is not None \
        else float(study_lineage.get("max_duration", MAX_GENERATION_DURATION))
    seed = args.seed if args.seed is not None \
        else int(study_params.get("seed", 0))
    cache_dir = args.cache_dir or study_params.get("cache_dir") \
        or ("out/cache" if os.path.isdir("out/cache") else "out/workflow/cache")

    viz_config = {"title": f"v2ecoli {n_generations}-generation lineage"}
    if study_spec is not None:
        for v in (study_spec.get("visualizations") or []):
            if isinstance(v, dict) and "MultigenerationVisualization" in (v.get("address") or ""):
                viz_config = dict(v.get("config") or viz_config)
                break

    print("=" * 60)
    print(f"v2ecoli multigeneration report — {n_generations} generation(s)")
    if study_spec is not None:
        print(f"Study: {study_spec.get('name', '(unnamed)')}")
    print("=" * 60)

    t_pipeline = time.time()
    results = run_multigeneration(
        n_generations,
        max_duration,
        cache_dir=cache_dir,
        seed=seed,
    )
    pipeline_wall = time.time() - t_pipeline

    print("  Dispatching to MultigenerationVisualization…")
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization

    history = _results_to_history(results)

    viz = MultigenerationVisualization(
        config=viz_config,
        core=allocate_core(),
    )
    result = viz.update({
        "history": history,
        "metadata": {
            "n_generations": n_generations,
            "seed": seed,
            "cache_dir": cache_dir,
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
