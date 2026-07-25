"""BatchBaselineRunner — dispatch N baseline lineages across seeds + generations.

A one-shot orchestrator Step for the ``batch_baseline`` composite. On its first
invocation it hands a vEcoli-shaped config to :func:`v2ecoli.workflow.run.run_workflow`,
which is v2ecoli's port of vEcoli's Nextflow lineage workflow: ``n_init_sims``
seeds × ``generations`` generations, one Ray worker process per seed (with a
SAFE sequential fallback when Ray is absent), every cell emitting into one
shared hive-partitioned parquet sweep, then a single post-simulation flush over
that sweep on the driver.

Delegating (rather than fanning out here) is what makes the batch produce the
same artifacts a vEcoli workflow run would: the flush dispatches the ported
Analyses at every scale the batch actually covers — ``single`` per cell,
``multigeneration`` across a lineage, ``multiseed`` across seeds,
``multivariant`` across a variant grid — plus the registered Visualizations and
report cards, placing each output in the owning study's report dir.

Heavy by design: firing this Step launches full whole-cell simulations. It is
IDEMPOTENT — once ``batch.completed`` is set it no-ops, so running the composite
for any number of steps dispatches the workflow exactly once.

Building the ``batch_baseline`` document stays cheap (no ParCa cache needed):
every per-seed ``baseline`` build happens at RUN time, inside the workflow's
``LineageProcess``.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from v2ecoli.steps.base import V2Step as Step


DEFAULT_N_SEEDS = 4
DEFAULT_N_GENERATIONS = 1
DEFAULT_BASE_SEED = 0
DEFAULT_CACHE_DIR = "out/cache"
DEFAULT_OUT_DIR = "out/batch_baseline"


def resolve_out_dir(out_dir: "str | None" = None) -> str:
    """Where this batch writes its sweep, stores and analysis outputs.

    An explicit ``out_dir`` always wins. Otherwise, when the composite is being
    run BY the workbench, use the run's own sweep dir
    (``$VIVARIUM_WORKBENCH_SWEEP_DIR`` = ``<run_dir>/parquet/<run_id>``, the
    exact path the run hands ParquetAnalysisView). Two things follow: the run's
    Visualizations tab finds the parquet history the declared analyses read
    (writing to the fixed workspace default instead left every panel saying "no
    parquet history under the run's sweep dir yet"), and each run keeps its own
    outputs instead of overwriting the previous run's. Outside a workbench run
    the fixed default stands.
    """
    if out_dir:
        return out_dir
    sweep = os.environ.get("VIVARIUM_WORKBENCH_SWEEP_DIR")
    if sweep:
        return sweep
    # On the sms-api compose / Ray-on-Batch path, run_pbg sets PBG_RESULTS_DIR to
    # the dir the entrypoint syncs to S3 (RAY_OUT_DIR). Land the sweep, stores and
    # analyses there so they're actually collected — the fixed workspace default
    # would write outside the synced dir and the results would never leave the
    # container.
    results = os.environ.get("PBG_RESULTS_DIR")
    if results:
        return os.path.join(results, "batch_baseline")
    return DEFAULT_OUT_DIR


DEFAULT_EXPERIMENT_ID = "batch_baseline"
# Per-GENERATION sim-time cap in seconds — vEcoli's ``max_duration``, which
# v2ecoli's LineageProcess applies per generation. A division ends a generation
# well before it (baseline divides at ~2700 s), so this only bounds a
# non-dividing run.
DEFAULT_MAX_DURATION = 3600.0
DEFAULT_TIME_STEP = 1.0
DEFAULT_SINGLE_DAUGHTERS = True
DEFAULT_PARALLEL = "ray"
# "both" = hive parquet (what the DuckDB analyses read) AND a per-lineage
# xarray-zarr store (what the workspace's xarray emitter + the dashboard's
# per-run charts read). "parquet" / "xarray" pick just one.
DEFAULT_EMITTER = "both"
DEFAULT_ANALYSES = "applicable"

# Registered but not a real analysis — a test fixture that would otherwise
# render an empty panel into every multivariant batch.
_ANALYSIS_DENYLIST = frozenset({"dummy"})


def applicable_analysis_scales(
    *,
    n_seeds: int,
    n_generations: int,
    single_daughters: bool = DEFAULT_SINGLE_DAUGHTERS,
    variants: "dict | None" = None,
) -> list[str]:
    """The analysis scales a batch of this shape actually has the cells for.

    Mirrors vEcoli's workflow, which only wires an analysis channel when the run
    produces the grouping it consumes: ``single`` always (one cell's timeseries),
    ``multigeneration`` once a lineage runs more than one generation,
    ``multiseed`` once there is more than one seed, ``multidaughter`` only when
    both daughters are simulated, ``multivariant`` only under a variant grid.
    """
    scales = ["single"]
    if int(n_generations) > 1:
        scales.append("multigeneration")
    if int(n_seeds) > 1:
        scales.append("multiseed")
    if not single_daughters:
        scales.append("multidaughter")
    if variants:
        scales.append("multivariant")
    return scales


def build_analysis_options(
    analyses: "str | dict | None" = DEFAULT_ANALYSES,
    *,
    n_seeds: int,
    n_generations: int,
    single_daughters: bool = DEFAULT_SINGLE_DAUGHTERS,
    variants: "dict | None" = None,
) -> dict:
    """Resolve the ``analyses`` parameter into a workflow ``analysis_options`` map.

    ``analyses`` is either an explicit ``{scale: {name: params}}`` mapping (used
    verbatim), the string ``"applicable"`` (every registered analysis at the
    scales this batch covers — see :func:`applicable_analysis_scales`), or
    ``"none"`` / empty (no analyses; the flush then only writes visualizations
    and report cards).
    """
    if isinstance(analyses, dict):
        return {k: dict(v or {}) for k, v in analyses.items() if v}
    choice = (analyses or "").strip().lower()
    if choice in ("", "none", "off", "false"):
        return {}
    if choice != "applicable":
        raise ValueError(
            f"analyses={analyses!r} — expected 'applicable', 'none', or an "
            "explicit {scale: {name: params}} mapping")

    # Import for the registration side effects: every ported analysis module
    # populates ANALYSIS_REGISTRY on import.
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    wanted = set(applicable_analysis_scales(
        n_seeds=n_seeds, n_generations=n_generations,
        single_daughters=single_daughters, variants=variants))
    options: dict[str, dict] = {}
    for name, cls in ANALYSIS_REGISTRY.items():
        if name in _ANALYSIS_DENYLIST or cls.scale not in wanted:
            continue
        options.setdefault(cls.scale, {})[name] = {}
    return options


def build_workflow_config(
    *,
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
    parallel: "str | None" = DEFAULT_PARALLEL,
    variants: "dict | None" = None,
    analyses: "str | dict | None" = DEFAULT_ANALYSES,
    study: str = "",
    base_config_overrides: "dict | None" = None,
) -> dict:
    """Translate the composite's parameters into a v2ecoli workflow config.

    The composite exposes the knobs under the names a batch user thinks in
    (``n_seeds``, ``n_generations``, ``base_seed``); the workflow consumes
    vEcoli's own key names. The mapping is deliberately explicit here so the two
    vocabularies stay traceable::

        n_seeds        -> n_init_sims          (vEcoli: sims per variant)
        n_generations  -> generations
        base_seed      -> lineage_seed         (first seed; seeds are contiguous)
        max_duration   -> max_duration_per_gen (LineageProcess caps per generation)
    """
    config: dict[str, Any] = {
        "experiment_id": experiment_id,
        "out_dir": resolve_out_dir(out_dir),
        "cache_dir": cache_dir,
        "n_init_sims": int(n_seeds),
        "generations": int(n_generations),
        "lineage_seed": int(base_seed),
        "single_daughters": bool(single_daughters),
        "time_step": float(time_step),
        "max_duration_per_gen": float(max_duration),
        "emitter": emitter or DEFAULT_EMITTER,
        "parallel": parallel or None,
        "variants": dict(variants or {}),
        "analysis_options": build_analysis_options(
            analyses, n_seeds=n_seeds, n_generations=n_generations,
            single_daughters=single_daughters, variants=variants),
    }
    if study:
        # Lets the flush place analyses/visualizations/report cards into this
        # study's report dir even when out_dir isn't under studies/<slug>/.
        config["study"] = study
    if base_config_overrides:
        # Applied to every branch by expand_branches (panel-wide, under the
        # variant grid) — how KO_batch_baseline knocks a gene out across all
        # seeds without forking a comparison arm.
        config["base_config_overrides"] = dict(base_config_overrides)
    return config


def link_sim_data(out_dir: str, cache_dir: str) -> "str | None":
    """Make the sweep self-describing by pairing it with its ParCa ``sim_data``.

    The analyses resolve sim_data by looking for a sweep-local
    ``simData*.cPickle`` FIRST — "the exact pairing, preferred"
    (:func:`v2ecoli.workflow.analysis_runner.resolve_sim_data`) — and only then
    fall back to ``$V2ECOLI_SIM_DATA`` or a global ``out/kb`` build. A batch runs
    every lineage from ``cache_dir``, so that cache's pickle IS the exact
    pairing; without this link the flush aborts with "no sim_data pickle under
    <out_dir>" even though the right one is sitting in the cache the run used.

    Symlinks (the pickle is ~100 MB); falls back to a copy where symlinks are
    unavailable. Returns the linked path, or None when there is nothing to link
    or the sweep already carries its own pickle.
    """
    src = os.path.join(cache_dir, "simData.cPickle")
    if not os.path.isfile(src):
        return None
    dest = os.path.join(out_dir, "simData.cPickle")
    if os.path.exists(dest):
        return dest
    os.makedirs(out_dir, exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dest)
    except OSError:
        import shutil
        shutil.copyfile(src, dest)
    return dest


def _lineage_store_path(out_dir: str, experiment_id: str, seed: int,
                        variant: int = 0) -> str:
    """Where LineageProcess writes a lineage's xarray-zarr store (mirrors its
    own naming) — reported per seed so downstream readers don't have to guess."""
    return os.path.join(out_dir, f"{experiment_id}_v{int(variant)}_s{int(seed)}.zarr")


def _per_seed_results(result: dict, *, seeds: list[int], out_dir: str,
                      experiment_id: str, emitter: str) -> dict:
    """Fold the workflow's ``branches`` (keyed ``variant=<v>/seed=<s>``) into the
    batch store's per-seed view, adding the zarr store path when one was written."""
    branches = result.get("branches") or {}
    per_seed: dict[str, dict] = {}
    for key, branch in branches.items():
        seed = None
        variant = 0
        for part in str(key).split("/"):
            field, _, value = part.partition("=")
            if field == "seed" and value.lstrip("-").isdigit():
                seed = int(value)
            elif field == "variant" and value.lstrip("-").isdigit():
                variant = int(value)
        if seed is None:
            continue
        gens = ((branch.get("summary") or {}).get("generations")) or []
        entry: dict[str, Any] = {
            "branch": key,
            "complete": bool(branch.get("complete")),
            "generations_reached": len(gens),
        }
        if emitter in ("xarray", "both"):
            store = _lineage_store_path(out_dir, experiment_id, seed, variant)
            entry["store_path"] = store
        per_seed[f"{seed:02d}"] = entry
    # Seeds the workflow never reported at all (worker died) still appear, so a
    # partial batch is visibly partial instead of silently short.
    for seed in seeds:
        per_seed.setdefault(f"{seed:02d}", {"error": "run produced no result"})
    return per_seed


def dispatch_batch(
    *,
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
    parallel: "str | None" = DEFAULT_PARALLEL,
    variants: "dict | None" = None,
    analyses: "str | dict | None" = DEFAULT_ANALYSES,
    study: str = "",
    base_config_overrides: "dict | None" = None,
    run_workflow_fn: "Callable[..., dict] | None" = None,
) -> dict:
    """Run the seeds × generations batch and assemble the ``batch`` result dict.

    ``run_workflow_fn`` is resolved at call time (defaults to
    :func:`v2ecoli.workflow.run.run_workflow`) so tests can inject a lightweight
    stub without touching ParCa or launching simulations.
    """
    if run_workflow_fn is None:
        from v2ecoli.workflow.run import run_workflow as run_workflow_fn

    out_dir = resolve_out_dir(out_dir)
    config = build_workflow_config(
        n_seeds=n_seeds, n_generations=n_generations, base_seed=base_seed,
        single_daughters=single_daughters, time_step=time_step,
        max_duration=max_duration, cache_dir=cache_dir, out_dir=out_dir,
        experiment_id=experiment_id, emitter=emitter, parallel=parallel,
        variants=variants, analyses=analyses, study=study,
        base_config_overrides=base_config_overrides)

    # Before the run, so the flush that follows it inside run_workflow can
    # resolve the sweep's sim_data (see link_sim_data).
    if config["analysis_options"]:
        try:
            link_sim_data(out_dir, cache_dir)
        except OSError as e:  # noqa: BLE001 — never fail a batch over a link
            print(f"batch_baseline: could not pair sim_data with the sweep: {e}")

    result = run_workflow_fn(config) or {}

    seeds = list(range(int(base_seed), int(base_seed) + int(n_seeds)))
    flush = result.get("flush") or {}
    return {
        "completed": True,
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "complete": bool(result.get("complete")),
        "mode": (result.get("parallel") or {}).get("mode") or "sequential",
        "wall_s": (result.get("parallel") or {}).get("wall_s") or result.get("elapsed"),
        "out_dir": out_dir,
        "emitter": config["emitter"],
        "analysis_scales": sorted(config["analysis_options"]),
        "seeds": _per_seed_results(
            result, seeds=seeds, out_dir=out_dir,
            experiment_id=experiment_id, emitter=config["emitter"]),
        # What the post-sim flush actually produced. `placed` lists the outputs
        # copied into the owning study's report dir — empty when no study owns
        # the run, in which case `viz_dir` is where the analyses and
        # visualizations still live (run_analyses always writes there first).
        "outputs": {
            "viz_dir": os.path.join(out_dir, "viz"),
            "placed": list(flush.get("placed") or []),
            "skipped": list(flush.get("skipped") or []),
            "error": flush.get("error"),
        },
    }


class BatchBaselineRunner(Step):
    """One-shot Step that dispatches the seeds × generations batch (see module)."""

    config_schema = {
        "n_seeds": "integer",
        "n_generations": "integer",
        "base_seed": "integer",
        "single_daughters": "boolean",
        "time_step": "float",
        "max_duration": "float",
        "cache_dir": "string",
        "out_dir": "string",
        "experiment_id": "string",
        "emitter": "string",          # "both" (default) | "parquet" | "xarray"
        "parallel": "string",         # "ray" (default) | "" for sequential
        "variants": "map",
        # Untyped-with-default (the config_overrides pattern): the value is
        # either a string choice or a {scale: {name: params}} mapping, which no
        # single bigraph-schema type covers.
        "analyses": {"_default": DEFAULT_ANALYSES},
        "study": "string",
        # Config overrides applied to every branch (panel-wide) — e.g. the
        # translation-efficiency patch KO_batch_baseline builds. Untyped: the
        # value is an arbitrary {'<proc>.<key>': value} map, and some values
        # (a numpy efficiencies array) no bigraph-schema type covers.
        "base_config_overrides": {"_default": {}},
    }
    topology = {
        "batch": ("batch",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.n_seeds = int(cfg.get("n_seeds") or DEFAULT_N_SEEDS)
        self.n_generations = int(cfg.get("n_generations") or DEFAULT_N_GENERATIONS)
        self.base_seed = int(cfg.get("base_seed") or DEFAULT_BASE_SEED)
        sd = cfg.get("single_daughters")
        self.single_daughters = DEFAULT_SINGLE_DAUGHTERS if sd is None else bool(sd)
        self.time_step = float(cfg.get("time_step") or DEFAULT_TIME_STEP)
        self.max_duration = float(cfg.get("max_duration") or DEFAULT_MAX_DURATION)
        self.cache_dir = cfg.get("cache_dir") or DEFAULT_CACHE_DIR
        # Left empty on purpose: resolve_out_dir picks the workbench run's own
        # sweep dir at RUN time, which isn't known when the document is built.
        self.out_dir = cfg.get("out_dir") or ""
        self.experiment_id = cfg.get("experiment_id") or DEFAULT_EXPERIMENT_ID
        self.emitter = cfg.get("emitter") or DEFAULT_EMITTER
        self.variants = dict(cfg.get("variants") or {})
        analyses = cfg.get("analyses")
        self.analyses = DEFAULT_ANALYSES if analyses is None else analyses
        self.study = cfg.get("study") or ""
        self.base_config_overrides = dict(cfg.get("base_config_overrides") or {})
        # None => default "ray"; "" / "sequential" / "none" => sequential (None).
        p = cfg.get("parallel")
        if p is None:
            self.parallel: str | None = DEFAULT_PARALLEL
        else:
            self.parallel = p or None

    def inputs(self) -> dict[str, Any]:
        # Read `batch` so the idempotency guard is persistent (survives the
        # dashboard composite-runner rebuilding the Step from the document).
        # Declare the port by its REGISTERED TYPE NAME ("inplace_dict" in
        # ECOLI_TYPES), not an `InPlaceDict()` instance: the instance serializes
        # to its repr in `to_document` (`"InPlaceDict(_default=None, ...)"`),
        # which the parser can't reparse when the .pbg is round-tripped through
        # remote dispatch (workbench export -> sms-api compose -> run_pbg on
        # Batch), failing Composite() with an IncompleteParseError. The name
        # string round-trips cleanly and resolves back to InPlaceDict via the core.
        return {"batch": "inplace_dict"}

    def triggers(self) -> dict[str, Any]:
        """No trigger ports — ``batch`` is a SILENT input.

        The Step still receives ``batch`` (the guard above reads it); it just
        stops being a *scheduling* input. That matters because
        ``build_step_network`` deliberately does not register a Step as the
        producer of a path it also consumes ("self-loops can't trigger"). With
        ``batch`` on both sides, nothing was recorded as producing ``batch``, so
        a downstream emitter's dependency on it counted as satisfied before this
        Step had run: the emitter landed in the SAME layer, read the store
        before the batch was dispatched, and the run's Results tab showed
        ``batch: {}`` no matter how many steps it ran. Declaring no triggers
        restores the edge — this Step runs in the first layer, the emitter in
        the next, and the emitted row carries the batch summary.
        """
        return {}

    def outputs(self) -> dict[str, Any]:
        # Registered type name, not an instance — see inputs() for the
        # serialization round-trip rationale.
        return {"batch": "inplace_dict"}

    def update(self, state, interval=None):
        batch = (state or {}).get("batch") or {}
        if isinstance(batch, dict) and batch.get("completed"):
            return {}  # already dispatched — no-op so the workflow fires once
        result = dispatch_batch(
            n_seeds=self.n_seeds,
            n_generations=self.n_generations,
            base_seed=self.base_seed,
            single_daughters=self.single_daughters,
            time_step=self.time_step,
            max_duration=self.max_duration,
            cache_dir=self.cache_dir,
            out_dir=self.out_dir,
            experiment_id=self.experiment_id,
            emitter=self.emitter,
            parallel=self.parallel,
            variants=self.variants,
            analyses=self.analyses,
            study=self.study,
            base_config_overrides=self.base_config_overrides,
        )
        return {"batch": result}
