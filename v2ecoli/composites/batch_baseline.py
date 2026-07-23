"""batch_baseline — run N baseline lineages across seeds and generations.

A registered composite that, when run, dispatches v2ecoli's port of vEcoli's
lineage workflow: ``n_seeds`` independent seeds × ``n_generations`` generations,
one Ray worker process per seed (with a safe sequential fallback when Ray is
absent), all cells emitting into one shared hive-partitioned parquet sweep (plus
a per-lineage xarray-zarr store), followed by a single post-simulation flush that
runs the ported Analyses, Visualizations and report cards over that sweep.

The parameters mirror the knobs a vEcoli workflow config sets (``n_init_sims``,
``generations``, ``lineage_seed``, ``single_daughters``, ``time_step``,
``max_duration``, ``variants``, analysis selection); the mapping between the two
vocabularies lives in ``build_workflow_config``. The fan-out itself lives in the
``BatchBaselineRunner`` Step (``v2ecoli/steps/batch_baseline_runner.py``); this
module just declares the parameters and wires that Step into a document.

The document is CHEAP to build (no ParCa cache needed): every per-seed baseline
build happens at RUN time inside the workflow, not at document-build time.
"""
from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import (
    DEFAULT_BATCH_VISUALIZATIONS,
    _make_instance,
    make_edge,
)
from v2ecoli.core import build_core
from v2ecoli.steps.batch_baseline_runner import (
    DEFAULT_ANALYSES,
    DEFAULT_BASE_SEED,
    DEFAULT_CACHE_DIR,
    DEFAULT_EMITTER,
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_MAX_DURATION,
    DEFAULT_N_GENERATIONS,
    DEFAULT_N_SEEDS,
    DEFAULT_OUT_DIR,
    DEFAULT_PARALLEL,
    DEFAULT_SINGLE_DAUGHTERS,
    DEFAULT_TIME_STEP,
    BatchBaselineRunner,
)


BATCH_RUNNER_STEP_NAME = "batch_runner"


@composite_generator(
    name="batch_baseline",
    description=(
        "Run N independent whole-cell E. coli lineages across seeds and "
        "generations — v2ecoli's port of vEcoli's lineage workflow. Each of "
        "n_seeds seeds runs an n_generations-generation lineage in its own Ray "
        "worker process (safe sequential fallback when Ray is absent), emitting "
        "into one shared hive-partitioned parquet sweep plus a per-lineage "
        "xarray-zarr store. A post-simulation flush then runs the ported "
        "Analyses at every scale the batch covers (single, multigeneration, "
        "multiseed, multivariant) along with the Visualizations and report "
        "cards, placing each output in the owning study's report dir."
    ),
    parameters={
        "n_seeds": {
            "type": "integer",
            "default": DEFAULT_N_SEEDS,
            "description": "Number of independent seeds (vEcoli's n_init_sims); "
                           "each runs in its own worker process.",
        },
        "n_generations": {
            "type": "integer",
            "default": DEFAULT_N_GENERATIONS,
            "description": "Cell-division generations to follow per seed lineage "
                           "(vEcoli's generations).",
        },
        "base_seed": {
            "type": "integer",
            "default": DEFAULT_BASE_SEED,
            "description": "First RNG seed (vEcoli's lineage_seed); seeds are "
                           "base_seed .. base_seed+n_seeds-1.",
        },
        "single_daughters": {
            "type": "bool",
            "default": DEFAULT_SINGLE_DAUGHTERS,
            "description": "Follow ONE daughter per division (vEcoli's default). "
                           "Off = binary-tree lineage, which also enables the "
                           "multidaughter analyses.",
        },
        "time_step": {
            "type": "float",
            "default": DEFAULT_TIME_STEP,
            "description": "Simulation time step in seconds.",
        },
        "max_duration": {
            "type": "float",
            "default": DEFAULT_MAX_DURATION,
            "description": "Per-generation sim-time cap in seconds (vEcoli's "
                           "max_duration). A division ends a generation sooner; "
                           "this only bounds a non-dividing run.",
        },
        "variants": {
            "type": "map",
            "default": {},
            "description": "vEcoli-style variant grid ({name: {target, value}}); "
                           "crossed with the seed range. Non-empty also enables "
                           "the multivariant analyses.",
        },
        "cache_dir": {
            "type": "string",
            "default": DEFAULT_CACHE_DIR,
            "description": "Path to the ParCa cache directory (read per seed at run).",
        },
        "out_dir": {
            "type": "string",
            "default": "",
            "description": "Output root for the parquet sweep, the per-lineage "
                           "zarr stores and the analysis outputs. Empty = this "
                           f"run's own directory when launched from the "
                           f"workbench (so its Visualizations tab finds the "
                           f"sweep and runs don't overwrite each other), else "
                           f"{DEFAULT_OUT_DIR}.",
        },
        "experiment_id": {
            "type": "string",
            "default": DEFAULT_EXPERIMENT_ID,
            "description": "Experiment id stamped into the parquet partitions "
                           "and the zarr store names.",
        },
        "emitter": {
            "type": "string",
            "choices": ["both", "parquet", "xarray"],
            "default": DEFAULT_EMITTER,
            "description": "Observation sink per lineage: 'parquet' (what the "
                           "analyses read), 'xarray' (zarr store, what the "
                           "dashboard's per-run charts read), or 'both'.",
        },
        "analyses": {
            "type": "string",
            "choices": ["applicable", "none"],
            "default": DEFAULT_ANALYSES,
            "description": "'applicable' runs every ported analysis at the "
                           "scales this batch covers (single always; "
                           "multigeneration when n_generations>1; multiseed when "
                           "n_seeds>1; multidaughter/multivariant likewise). "
                           "'none' skips analyses; visualizations and report "
                           "cards still run.",
        },
        "study": {
            "type": "string",
            "default": "",
            "description": "Owning study slug for the flush's outputs. Empty = "
                           "infer from out_dir (a studies/<slug>/ segment).",
        },
        "parallel": {
            "type": "string",
            "default": DEFAULT_PARALLEL,
            "description": "'ray' to fan out across worker processes; '' for sequential.",
        },
    },
    # The runner is one-shot and idempotent: a single step fires the whole batch.
    default_n_steps=1,
    visualizations=DEFAULT_BATCH_VISUALIZATIONS,
)
def batch_baseline(
    core: Any = None,
    *,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_generations: int = DEFAULT_N_GENERATIONS,
    base_seed: int = DEFAULT_BASE_SEED,
    single_daughters: bool = DEFAULT_SINGLE_DAUGHTERS,
    time_step: float = DEFAULT_TIME_STEP,
    max_duration: float = DEFAULT_MAX_DURATION,
    variants: dict | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    out_dir: str = "",
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    emitter: str = DEFAULT_EMITTER,
    analyses: Any = DEFAULT_ANALYSES,
    study: str = "",
    parallel: str = DEFAULT_PARALLEL,
) -> dict:
    """Build the batch_baseline composite document.

    Args:
        core: bigraph-schema core; a fresh one (``build_core()``) if None.
        n_seeds: number of seeds (vEcoli's n_init_sims).
        n_generations: generations per seed lineage.
        base_seed: first RNG seed (vEcoli's lineage_seed).
        single_daughters: follow one daughter per division.
        time_step: simulation time step (seconds).
        max_duration: per-generation sim-time cap (seconds).
        variants: vEcoli-style variant grid crossed with the seed range.
        cache_dir: ParCa cache directory.
        out_dir: output root for the parquet sweep + zarr stores;
            empty resolves at run time (see resolve_out_dir).
        experiment_id: id stamped into partitions and store names.
        emitter: "both" | "parquet" | "xarray".
        analyses: "applicable" | "none" | explicit {scale: {name: params}}.
        study: owning study slug for flush placement ("" = infer).
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
        "single_daughters": bool(single_daughters),
        "time_step": float(time_step),
        "max_duration": float(max_duration),
        "variants": dict(variants or {}),
        "cache_dir": cache_dir,
        "out_dir": out_dir,
        "experiment_id": experiment_id,
        "emitter": emitter or DEFAULT_EMITTER,
        "analyses": analyses,
        "study": study,
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
