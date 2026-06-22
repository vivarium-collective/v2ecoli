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


# Cache for the once-per-process vEcoli import + global registrations. Re-importing
# ``ecoli.*`` per seed would re-trigger module-level emitter registration (e.g.
# ``parquet``) into the process-global registry → "registry already contains an
# entry for parquet" on the 2nd seed. So do the path-fix + purge + imports +
# resolve-dispatch registration EXACTLY ONCE, then reuse the cached symbols.
_VECOLI: dict = {}


def _ensure_vecoli() -> dict:
    """Import the vEcoli composite stack once per process; cache its symbols.

    The v2ecoli image bakes the vEcoli composite-softfloor checkout at
    ``/app/vEcoli``, but an installed vEcoli release (lacking
    ``ecoli.composites.ecoli_composite``) may already be cached in sys.modules
    transitively. ``sys.path.insert`` does NOT re-resolve an already-imported
    package, so we drop every ``ecoli*`` module ONCE to force a fresh import
    from the composite-softfloor branch. Mirrors scripts/run_vecoli_composite.py
    (incl. the SharedProcess resolve dispatches the composite branch needs).
    """
    if _VECOLI:
        return _VECOLI
    vecoli_dir = str(REPO_ROOT.parent / "vEcoli")
    sys.path.insert(0, vecoli_dir)

    # ecoli/__init__.py registers parquet/updaters/dividers/serializers into
    # vivarium's PROCESS-GLOBAL registries as an import side effect. A shadow
    # ecoli release may already have registered those keys, and vivarium's
    # Registry.register RAISES on a duplicate key whose item object differs —
    # which re-importing ecoli guarantees (new class/function identities). Make
    # register idempotent (skip keys already present) so re-importing ecoli from
    # the composite checkout below doesn't blow up on "registry already contains
    # an entry for parquet". The freshly imported ecoli.composites.* and the
    # soft-floor transcription still come from vecoli_dir; only the shared
    # registry helpers fall back to whatever registered first (functionally
    # equivalent — the soft-floor lives in transcription, not in these helpers).
    import vivarium.core.registry as _vreg
    if not getattr(_vreg.Registry, "_idempotent_patched", False):
        _orig_register = _vreg.Registry.register

        def _idempotent_register(self, key, item, alternate_keys=tuple()):
            if key in self.registry:
                return None
            return _orig_register(self, key, item, alternate_keys)

        _vreg.Registry.register = _idempotent_register
        _vreg.Registry._idempotent_patched = True

    # Drop any cached (possibly shadow) ecoli modules so the imports below
    # resolve from vecoli_dir (the composite-softfloor checkout).
    for _m in [m for m in sys.modules if m == "ecoli" or m.startswith("ecoli.")]:
        del sys.modules[_m]

    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.composites.ecoli_composite import build_composite_native
    from ecoli.library.bigraph_types import ECOLI_TYPES, SharedProcess
    from process_bigraph import Composite
    from bigraph_schema import allocate_core
    import v2ecoli.types  # noqa: F401  (register v2ecoli types/dispatches)

    # SharedProcess (new in the composite branch) needs resolve dispatches.
    from bigraph_schema.methods.resolve import resolve as _resolve
    from bigraph_schema.schema import Tuple as _Tuple
    from v2ecoli.types.process import ProcessInstance

    @_resolve.dispatch
    def _resolve_tuple_shared(current: _Tuple, update: SharedProcess, path=None):
        return update

    @_resolve.dispatch
    def _resolve_shared_tuple(current: SharedProcess, update: _Tuple, path=None):
        return current

    @_resolve.dispatch
    def _resolve_pi_shared(current: ProcessInstance, update: SharedProcess, path=None):
        return update

    @_resolve.dispatch
    def _resolve_shared_pi(current: SharedProcess, update: ProcessInstance, path=None):
        return current

    _VECOLI.update(
        vecoli_dir=vecoli_dir, EcoliSim=EcoliSim,
        build_composite_native=build_composite_native,
        ECOLI_TYPES=ECOLI_TYPES, Composite=Composite, allocate_core=allocate_core,
    )
    return _VECOLI


def _build_vecoli(seed: int, condition: str):
    """Original vEcoli model as a process-bigraph composite (no Nextflow).

    Mirrors scripts/run_vecoli_composite.py: ``build_composite_native`` from the
    vEcoli composite-softfloor branch (carries the ppGpp soft-floor). The import
    + global registration happens once via ``_ensure_vecoli``; this builds a
    fresh per-seed composite on its own ``core``.
    """
    v = _ensure_vecoli()
    prev = os.getcwd()
    os.chdir(v["vecoli_dir"])
    try:
        core = v["allocate_core"]()
        core.register_types(v["ECOLI_TYPES"])
        sim = v["EcoliSim"].from_cli()
        # condition → media is set via the vEcoli sim config; lineage_seed=seed.
        sim.config["condition"] = condition
        sim.config["seed"] = seed
        sim.processes = sim._retrieve_processes(
            sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes)
        sim.topology = sim._retrieve_topology(
            sim.topology, sim.processes, sim.swap_processes, sim.log_updates)
        sim.process_configs = sim._retrieve_process_configs(sim.process_configs, sim.processes)
        state = v["build_composite_native"](core, sim.config)
        composite = v["Composite"](dict(schema=dict(), state=state), core=core)
        composite.to_run = []
        return composite
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
