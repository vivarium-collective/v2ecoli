"""KO_batch_baseline — a seeds × generations batch with genes knocked out.

``batch_baseline`` with a genotype: every seed's lineage runs the same
translation-level knockout (design PR #341, Tranche A). The knockout is resolved
once against the ParCa cache into a ``translation_efficiencies`` config override,
then applied panel-wide — to every branch, across all seeds — via the workflow's
``base_config_overrides`` seam (not as a variant, which would fork one perturbed
arm against an unperturbed baseline). Everything else is ``batch_baseline``'s.

Use this to characterise a knockout across a seed ensemble: the same analyses,
visualizations and report cards a baseline batch produces, computed on the
perturbed genotype. See :mod:`v2ecoli.perturbations.translation` for the KO
mechanism and its caveats.
"""
from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import (
    DEFAULT_BATCH_VISUALIZATIONS,
    _make_instance,
    make_edge,
)
from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.perturbations import translation_efficiency_override
from v2ecoli.steps.batch_baseline_runner import (
    DEFAULT_ANALYSES,
    DEFAULT_BASE_SEED,
    DEFAULT_CACHE_DIR,
    DEFAULT_EMITTER,
    DEFAULT_MAX_DURATION,
    DEFAULT_N_GENERATIONS,
    DEFAULT_N_SEEDS,
    DEFAULT_PARALLEL,
    DEFAULT_SINGLE_DAUGHTERS,
    DEFAULT_TIME_STEP,
    BatchBaselineRunner,
)

BATCH_RUNNER_STEP_NAME = "batch_runner"
DEFAULT_EXPERIMENT_ID = "ko_batch_baseline"


@composite_generator(
    name="KO_batch_baseline",
    description=(
        "Seeds × generations batch of a knocked-out genotype. Every seed runs "
        "the same translation-level knockout (each gene in `knockouts` — an "
        "EcoCyc gene id or a monomer id — has its translation efficiency zeroed), "
        "applied panel-wide across all seeds. Otherwise identical to "
        "batch_baseline: one Ray worker per seed, the shared parquet sweep, and "
        "the full post-simulation analysis flush."
    ),
    parameters={
        "knockouts": {
            "type": "list",
            "default": [],
            "description": "Genes to knock out at the translation level across "
                           "every seed — EcoCyc gene ids (EG10526) or monomer "
                           "ids (LACY-MONOMER[c]). Empty = plain batch_baseline.",
        },
        "n_seeds": {
            "type": "integer",
            "default": DEFAULT_N_SEEDS,
            "description": "Number of independent seeds (vEcoli's n_init_sims).",
        },
        "n_generations": {
            "type": "integer",
            "default": DEFAULT_N_GENERATIONS,
            "description": "Cell-division generations per seed lineage.",
        },
        "base_seed": {
            "type": "integer",
            "default": DEFAULT_BASE_SEED,
            "description": "First RNG seed; seeds are base_seed .. base_seed+n_seeds-1.",
        },
        "single_daughters": {
            "type": "bool",
            "default": DEFAULT_SINGLE_DAUGHTERS,
            "description": "Follow ONE daughter per division (vEcoli's default).",
        },
        "time_step": {
            "type": "float",
            "default": DEFAULT_TIME_STEP,
            "description": "Simulation time step in seconds.",
        },
        "max_duration": {
            "type": "float",
            "default": DEFAULT_MAX_DURATION,
            "description": "Per-generation sim-time cap in seconds.",
        },
        "cache_dir": {
            "type": "string",
            "default": DEFAULT_CACHE_DIR,
            "description": "Path to the ParCa cache directory.",
        },
        "out_dir": {
            "type": "string",
            "default": "",
            "description": "Output root; empty = this run's own dir under the "
                           "workbench, else out/ko_batch_baseline.",
        },
        "experiment_id": {
            "type": "string",
            "default": DEFAULT_EXPERIMENT_ID,
            "description": "Experiment id stamped into partitions + store names.",
        },
        "emitter": {
            "type": "string",
            "choices": ["both", "parquet", "xarray"],
            "default": DEFAULT_EMITTER,
            "description": "Observation sink per lineage.",
        },
        "analyses": {
            "type": "string",
            "choices": ["applicable", "none"],
            "default": DEFAULT_ANALYSES,
            "description": "'applicable' runs every ported analysis at the scales "
                           "this batch covers; 'none' skips analyses.",
        },
        "study": {
            "type": "string",
            "default": "",
            "description": "Owning study slug for the flush's outputs.",
        },
        "parallel": {
            "type": "string",
            "default": DEFAULT_PARALLEL,
            "description": "'ray' to fan out across worker processes; '' sequential.",
        },
    },
    default_n_steps=1,
    visualizations=DEFAULT_BATCH_VISUALIZATIONS,
)
def KO_batch_baseline(
    core: Any = None,
    *,
    knockouts: list[str] | None = None,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_generations: int = DEFAULT_N_GENERATIONS,
    base_seed: int = DEFAULT_BASE_SEED,
    single_daughters: bool = DEFAULT_SINGLE_DAUGHTERS,
    time_step: float = DEFAULT_TIME_STEP,
    max_duration: float = DEFAULT_MAX_DURATION,
    cache_dir: str = DEFAULT_CACHE_DIR,
    out_dir: str = "",
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    emitter: str = DEFAULT_EMITTER,
    analyses: Any = DEFAULT_ANALYSES,
    study: str = "",
    parallel: str = DEFAULT_PARALLEL,
) -> dict:
    """Build the KO_batch_baseline document (see module docstring).

    Resolving the knockout here (document-build time) rather than deferring it to
    the runner is deliberate: an unknown or non-coding gene fails loudly when the
    composite is built, not silently inside a Ray worker mid-run.
    """
    if core is None:
        core = build_core()

    # Resolve the knockout ONCE against the cache; it depends only on cache_dir,
    # not on seed, so every branch shares the same override.
    base_config_overrides: dict = {}
    if knockouts:
        bundle = load_cache_bundle(cache_dir)
        base_config_overrides = translation_efficiency_override(
            bundle, list(knockouts))

    runner_config = {
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "base_seed": int(base_seed),
        "single_daughters": bool(single_daughters),
        "time_step": float(time_step),
        "max_duration": float(max_duration),
        "variants": {},
        "cache_dir": cache_dir,
        "out_dir": out_dir,
        "experiment_id": experiment_id,
        "emitter": emitter or DEFAULT_EMITTER,
        "analyses": analyses,
        "study": study,
        "parallel": parallel or "",
        "base_config_overrides": base_config_overrides,
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
