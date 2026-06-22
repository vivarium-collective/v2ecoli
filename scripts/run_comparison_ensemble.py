"""Multi-generation, multi-seed ensemble driver for the v2ecoli↔vEcoli comparison.

Runs EITHER engine through the IDENTICAL process-bigraph + XArray path so the
comparison is apples-to-apples (same framework, same compact emitter, same
compute), with NO Nextflow:

  --composite v2ecoli  → v2ecoli's ported ``baseline()`` composite
  --composite vecoli   → the original vEcoli model as a bigraph composite via
                         ``build_composite_native`` (the vEcoli ``composite``
                         branch, auto-wrapped through ``wrap_vivarium_process``)

Each seed runs ``max_generations`` past divisions (``run_multigen_xarray``,
daughter-following) and emits ONLY the compact comparison ``view`` (8 scalar
observables — no bulk/unique arrays) to a per-seed zarr store, which the Ray
backend ships to S3. Seeds run in parallel via ``run_seeds_parallel`` (Ray).

    python scripts/run_comparison_ensemble.py --composite v2ecoli \
        --condition basal --n-seeds 16 --max-generations 16 \
        --out-root s3://.../vecoli-output/<exp> [--chunk 60]

The compact view is the storage-minimizing piece the comparison needs: just the
report-card axes + a few diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Compact comparison observables (confirmed set): report-card axes + diagnostics.
# Dotted paths under the followed agent; mapped to an XArray ``view`` below.
COMPARISON_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.rna_mass",
    "listeners.mass.instantaneous_growth_rate",
    "listeners.unique_molecule_counts.active_RNAP",
    "listeners.unique_molecule_counts.active_ribosome",
    "listeners.rna_synth_prob.total_rna_init",
]


def _build_v2ecoli(seed: int, cache_dir: str):
    """v2ecoli ported composite (baseline) for a condition cache."""
    from v2ecoli import build_composite
    return build_composite("baseline", cache_dir=cache_dir, seed=seed)


def _build_vecoli(seed: int, condition: str):
    """Original vEcoli model as a process-bigraph composite (no Nextflow).

    Mirrors scripts/run_vecoli_composite.py: build_composite_native from the
    vEcoli ``composite`` branch. REQUIRES the vEcoli checkout to be on a branch
    that has both ``ecoli.composites.ecoli_composite.build_composite_native``
    AND the ppGpp soft-floor (i.e. the soft-floor applied onto the composite
    branch). See the module docstring.
    """
    vecoli_dir = str(REPO_ROOT.parent / "vEcoli")
    sys.path.insert(0, vecoli_dir)
    # The installed vEcoli release may already be cached in sys.modules (pulled
    # in transitively when v2ecoli imports loaded), and that release lacks
    # ``ecoli.composites.ecoli_composite``. A bare ``sys.path.insert`` does NOT
    # re-resolve an already-imported package, so drop every ``ecoli*`` module
    # to force a fresh import from ``vecoli_dir`` (the composite-softfloor
    # branch that carries ``build_composite_native``).
    for _m in [m for m in sys.modules if m == "ecoli" or m.startswith("ecoli.")]:
        del sys.modules[_m]
    prev = os.getcwd()
    os.chdir(vecoli_dir)
    try:
        from ecoli.experiments.ecoli_master_sim import EcoliSim
        from ecoli.composites.ecoli_composite import build_composite_native
        from ecoli.library.bigraph_types import ECOLI_TYPES
        from process_bigraph import Composite
        from bigraph_schema import allocate_core
        import v2ecoli.types  # noqa: F401  (register v2ecoli types/dispatches)

        core = allocate_core()
        for name, schema in ECOLI_TYPES.items():
            core.register(name, schema)
        sim = EcoliSim.from_cli()
        # condition → media is set via the vEcoli sim config; lineage_seed=seed.
        sim.config["condition"] = condition
        sim.config["seed"] = seed
        state = build_composite_native(core, sim.config)
        return Composite({"state": state}, core=core)
    finally:
        os.chdir(prev)


def make_run_one(*, composite_kind: str, condition: str, cache_dir: str,
                 max_generations: int, max_steps: int, chunk: int,
                 out_root: str):
    """Return a ``run_one(seed)`` closure for ``run_seeds_parallel``."""
    from v2ecoli.library.xarray_run import run_multigen_xarray, view_from_emit_paths

    def run_one(seed: int) -> dict:
        t0 = time.time()
        store_path = f"{out_root.rstrip('/')}/{composite_kind}_seed{seed:02d}.zarr"
        # local stores: clear stale
        if "://" not in str(store_path) and Path(store_path).exists():
            shutil.rmtree(store_path)
        if composite_kind == "v2ecoli":
            composite = _build_v2ecoli(seed, cache_dir)
        elif composite_kind == "vecoli":
            composite = _build_vecoli(seed, condition)
        else:
            raise ValueError(f"unknown composite_kind {composite_kind!r}")
        view = view_from_emit_paths(COMPARISON_PATHS, include_vectors=False)
        metadata_base = {
            "experiment_id": f"cmp-{composite_kind}-{condition}-seed{seed:02d}",
            "engine": composite_kind,
            "condition": condition,
            "variant": 0,
            "lineage_seed": seed,
            "time_step": 1.0,
            "max_duration": float(max_steps),
            "agent_id": "0",
        }
        result = run_multigen_xarray(
            composite,
            store_path=store_path,
            view=view,
            metadata_base=metadata_base,
            max_steps=max_steps,
            max_generations=max_generations,
            chunk=chunk,
        )
        return {"seed": seed, "wall_seconds": round(time.time() - t0, 1),
                "store": str(store_path), **{k: result.get(k) for k in
                ("steps", "generations")}}

    return run_one


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--composite", required=True, choices=["v2ecoli", "vecoli"])
    p.add_argument("--condition", default="basal")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "out" / "cache"),
                   help="v2ecoli condition cache (v2ecoli engine only).")
    p.add_argument("--n-seeds", type=int, default=16)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-generations", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=60000,
                   help="hard tick cap across all generations (safety stop).")
    p.add_argument("--chunk", type=int, default=60)
    p.add_argument("--out-root", required=True,
                   help="dir or s3:// prefix for per-seed zarr stores.")
    p.add_argument("--mode", default="ray", help="run_seeds_parallel mode (ray/serial).")
    args = p.parse_args(argv)

    from v2ecoli.library.parallel_seeds import run_seeds_parallel
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    run_one = make_run_one(
        composite_kind=args.composite, condition=args.condition,
        cache_dir=args.cache_dir, max_generations=args.max_generations,
        max_steps=args.max_steps, chunk=args.chunk, out_root=args.out_root)
    parallel = run_seeds_parallel(seeds, run_one, mode=args.mode)
    summaries = getattr(parallel, "results", parallel)
    ensemble = {"composite": args.composite, "condition": args.condition,
                "n_seeds": len(seeds), "max_generations": args.max_generations,
                "wall_s": getattr(parallel, "wall_s", None),
                "seeds": list(summaries)}
    print(json.dumps(ensemble, indent=2))


if __name__ == "__main__":
    main()
