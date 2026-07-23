"""batch_baseline — dispatch N baseline lineages in parallel on Ray.

A registered composite that, when run, fans ``n_seeds`` independent whole-cell
E. coli baseline runs (each an ``n_generations``-generation lineage) across Ray
workers and collects per-seed xarray-zarr store paths + summary observables into
a top-level ``batch`` store. The actual fan-out lives in the
``BatchBaselineRunner`` Step (``v2ecoli/steps/batch_baseline_runner.py``); this
module just declares the parameters and wires that Step into a document.

The document is CHEAP to build (no ParCa cache needed): every per-seed
``build_composite("baseline", ...)`` happens at RUN time inside the runner, not
at document-build time.
"""
from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import _make_instance, make_edge
from v2ecoli.core import build_core
from v2ecoli.steps.batch_baseline_runner import (
    DEFAULT_BASE_SEED,
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_STEPS_PER_GEN,
    DEFAULT_N_GENERATIONS,
    DEFAULT_N_SEEDS,
    DEFAULT_OUT_ROOT,
    DEFAULT_PARALLEL,
    BatchBaselineRunner,
)


BATCH_RUNNER_STEP_NAME = "batch_runner"


@composite_generator(
    name="batch_baseline",
    description=(
        "Dispatch N independent whole-cell E. coli baseline runs in parallel on "
        "Ray. Parameterized by n_seeds and n_generations: each seed runs an "
        "n_generations-generation lineage in its own Ray worker process (with a "
        "safe sequential fallback when Ray is absent), persists an xarray-zarr "
        "store, and reports its store path + summary observables into the "
        "top-level `batch` store."
    ),
    parameters={
        "n_seeds": {
            "type": "integer",
            "default": DEFAULT_N_SEEDS,
            "description": "Number of independent seeds (ecolis run in parallel).",
        },
        "n_generations": {
            "type": "integer",
            "default": DEFAULT_N_GENERATIONS,
            "description": "Cell-division generations to follow per seed lineage.",
        },
        "base_seed": {
            "type": "integer",
            "default": DEFAULT_BASE_SEED,
            "description": "First RNG seed; seeds are base_seed .. base_seed+n_seeds-1.",
        },
        "cache_dir": {
            "type": "string",
            "default": DEFAULT_CACHE_DIR,
            "description": "Path to the ParCa cache directory (read per seed at run).",
        },
        "max_steps_per_gen": {
            "type": "integer",
            "default": DEFAULT_MAX_STEPS_PER_GEN,
            "description": "Per-generation tick cap; division ends a generation sooner.",
        },
        "out_root": {
            "type": "string",
            "default": DEFAULT_OUT_ROOT,
            "description": "Directory root for the per-seed xarray-zarr stores.",
        },
        "parallel": {
            "type": "string",
            "default": DEFAULT_PARALLEL,
            "description": "'ray' to fan out across worker processes; '' for sequential.",
        },
    },
)
def batch_baseline(
    core: Any = None,
    *,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_generations: int = DEFAULT_N_GENERATIONS,
    base_seed: int = DEFAULT_BASE_SEED,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_steps_per_gen: int = DEFAULT_MAX_STEPS_PER_GEN,
    out_root: str = DEFAULT_OUT_ROOT,
    parallel: str = DEFAULT_PARALLEL,
) -> dict:
    """Build the batch_baseline composite document.

    Args:
        core: bigraph-schema core; a fresh one (``build_core()``) if None.
        n_seeds: number of parallel seeds.
        n_generations: generations per seed lineage.
        base_seed: first RNG seed.
        cache_dir: ParCa cache directory.
        max_steps_per_gen: per-generation tick cap.
        out_root: directory root for per-seed zarr stores.
        parallel: "ray" | "" (sequential).

    Returns:
        Process-bigraph document dict with a single ``state`` key: an empty
        ``batch`` store plus the ``BatchBaselineRunner`` Step.
    """
    if core is None:
        core = build_core()

    runner_config = {
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "base_seed": int(base_seed),
        "cache_dir": cache_dir,
        "max_steps_per_gen": int(max_steps_per_gen),
        "out_root": out_root,
        "parallel": parallel or "",
    }
    runner = _make_instance(BatchBaselineRunner, runner_config, core)

    state = {
        "batch": {},  # empty; the runner writes per-seed results here at run time
        "global_time": 0.0,
        BATCH_RUNNER_STEP_NAME: make_edge(
            runner,
            BatchBaselineRunner.topology,
            edge_type="step",
            config=runner_config,
        ),
    }
    return {"state": state}
