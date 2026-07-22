"""BatchBaselineRunner — dispatch N independent baseline lineages in parallel.

A one-shot orchestrator Step for the ``batch_baseline`` composite. On its first
invocation it fans ``n_seeds`` whole-cell baseline runs (each an
``n_generations``-generation lineage) across Ray workers via
``run_seeds_parallel`` — the proven embarrassingly-parallel seed fan-out (one
worker PROCESS per seed, with a SAFE sequential fallback when Ray is absent).
Each seed persists its own multigen xarray-zarr store; the Step writes per-seed
store paths + summary observables into a top-level ``batch`` store.

Heavy by design: firing this Step launches full whole-cell simulations. It is
IDEMPOTENT — once ``batch.completed`` is set it no-ops, so running the composite
for any number of steps dispatches the workflow exactly once.

Building the ``batch_baseline`` document is cheap (no ParCa) because every
per-seed ``build_composite("baseline", ...)`` happens at RUN time, inside
``run_one`` — which is a module-level (picklable) function so Ray can ship it to
a fresh worker process.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


DEFAULT_N_SEEDS = 4
DEFAULT_N_GENERATIONS = 1
DEFAULT_BASE_SEED = 0
DEFAULT_CACHE_DIR = "out/cache"
DEFAULT_OUT_ROOT = "out/batch_baseline"
# Generous per-generation tick cap; a division ends a generation well before it
# (baseline divides at ~2700 ticks), so this only bounds a non-dividing run.
DEFAULT_MAX_STEPS_PER_GEN = 3000
DEFAULT_PARALLEL = "ray"


def _default_run_one(
    seed: int,
    *,
    n_generations: int,
    cache_dir: str,
    max_steps_per_gen: int,
    out_root: str,
) -> dict:
    """Build + run ONE baseline lineage for ``n_generations`` generations,
    persisting an xarray-zarr store. Top-level (picklable) so Ray can ship it to
    a worker; does its own per-worker setup and uses absolute-ish paths.

    Returns ``{"seed", "store_path", "summary"}`` where ``summary`` carries the
    generations reached, ticks run, and final cell mass (fg).
    """
    from process_bigraph import Composite  # noqa: F401 (kept for parity/debug)
    from v2ecoli import build_composite
    from v2ecoli.core import build_core
    from v2ecoli.library.xarray_run import run_multigen_xarray
    from v2ecoli.workflow.lineage import DEFAULT_XARRAY_VIEW

    out_dir = os.path.join(out_root, f"seed_{int(seed):02d}")
    os.makedirs(out_dir, exist_ok=True)
    store_path = os.path.join(out_dir, "store.zarr")

    core = build_core()
    composite = build_composite("baseline", core=core, seed=int(seed),
                                cache_dir=cache_dir)

    view = [dict(e, root=tuple(e["root"])) for e in DEFAULT_XARRAY_VIEW]
    metadata_base = {
        "experiment_id": f"batch_baseline_seed{int(seed):02d}",
        "variant": 0,
        "lineage_seed": int(seed),
        "time_step": 1.0,
        "max_duration": float(max_steps_per_gen),
    }
    result = run_multigen_xarray(
        composite,
        store_path=store_path,
        view=view,
        metadata_base=metadata_base,
        max_steps=int(max_steps_per_gen) * int(n_generations),
        max_generations=int(n_generations),
    )

    # Final cell mass (fg) from the followed cell's mass listener (best-effort —
    # a missing listener must not fail the whole batch). ``cell_mass`` is a pint
    # Quantity in femtograms, so take its magnitude (float(Quantity) raises).
    final_mass = None
    try:
        agents = (composite.state or {}).get("agents") or {}
        cell = next(iter(agents.values()))
        cell_mass = cell["listeners"]["mass"]["cell_mass"]
        final_mass = float(getattr(cell_mass, "magnitude", cell_mass))
    except Exception:
        pass

    gens = result.get("generations") or []
    return {
        "seed": int(seed),
        "store_path": store_path,
        "summary": {
            "generations_reached": len(gens),
            "steps": int(result.get("steps", 0)),
            "final_cell_mass_fg": final_mass,
        },
    }


def dispatch_batch(
    *,
    n_seeds: int,
    n_generations: int,
    base_seed: int,
    cache_dir: str,
    max_steps_per_gen: int,
    out_root: str,
    parallel: str | None,
    run_one: Callable[..., Any] | None = None,
) -> dict:
    """Fan ``n_seeds`` baseline lineages across Ray (or sequentially) and
    assemble the ``batch`` result dict.

    ``run_one`` is resolved at call time (defaults to the module's
    ``_default_run_one``) so tests can inject a lightweight stub or monkeypatch
    the module attribute without touching ParCa.
    """
    from v2ecoli.library.parallel_seeds import run_seeds_parallel

    if run_one is None:
        run_one = _default_run_one

    seeds = list(range(int(base_seed), int(base_seed) + int(n_seeds)))
    res = run_seeds_parallel(
        seeds,
        run_one,
        mode=parallel,
        run_kwargs={
            "n_generations": int(n_generations),
            "cache_dir": cache_dir,
            "max_steps_per_gen": int(max_steps_per_gen),
            "out_root": out_root,
        },
    )

    per_seed: dict[str, dict] = {}
    for seed, r in zip(seeds, res.results):
        key = f"{seed:02d}"
        if not isinstance(r, dict):
            per_seed[key] = {"error": "run produced no result"}
            continue
        per_seed[key] = {"store_path": r.get("store_path"), **(r.get("summary") or {})}

    return {
        "completed": True,
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "mode": res.mode,
        "wall_s": res.wall_s,
        "seeds": per_seed,
    }


class BatchBaselineRunner(Step):
    """One-shot Step that dispatches the parallel baseline batch (see module)."""

    config_schema = {
        "n_seeds": "integer",
        "n_generations": "integer",
        "base_seed": "integer",
        "cache_dir": "string",
        "max_steps_per_gen": "integer",
        "out_root": "string",
        "parallel": "string",  # "ray" (default) | "" for sequential
    }
    topology = {
        "batch": ("batch",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.n_seeds = int(cfg.get("n_seeds") or DEFAULT_N_SEEDS)
        self.n_generations = int(cfg.get("n_generations") or DEFAULT_N_GENERATIONS)
        self.base_seed = int(cfg.get("base_seed") or DEFAULT_BASE_SEED)
        self.cache_dir = cfg.get("cache_dir") or DEFAULT_CACHE_DIR
        self.max_steps_per_gen = int(cfg.get("max_steps_per_gen") or DEFAULT_MAX_STEPS_PER_GEN)
        self.out_root = cfg.get("out_root") or DEFAULT_OUT_ROOT
        # None => default "ray"; "" / "sequential" / "none" => sequential (None).
        p = cfg.get("parallel")
        if p is None:
            self.parallel: str | None = DEFAULT_PARALLEL
        else:
            self.parallel = p or None

    def inputs(self) -> dict[str, Any]:
        # Read `batch` so the idempotency guard is persistent (survives the
        # dashboard composite-runner rebuilding the Step from the document).
        return {"batch": InPlaceDict()}

    def outputs(self) -> dict[str, Any]:
        return {"batch": InPlaceDict()}

    def update(self, state, interval=None):
        batch = (state or {}).get("batch") or {}
        if isinstance(batch, dict) and batch.get("completed"):
            return {}  # already dispatched — no-op so the workflow fires once
        result = dispatch_batch(
            n_seeds=self.n_seeds,
            n_generations=self.n_generations,
            base_seed=self.base_seed,
            cache_dir=self.cache_dir,
            max_steps_per_gen=self.max_steps_per_gen,
            out_root=self.out_root,
            parallel=self.parallel,
        )
        return {"batch": result}
