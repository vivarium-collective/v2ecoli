"""Deep sim_data injection adapter for forward UQ (parameter-uq / param-uq-04).

Studies param-uq-00..02 varied only *config-reachable top-level scalars* — the
``config_overrides`` path on :func:`v2ecoli.composites.ecoli_baseline.baseline`
that patches ``configs[process][key]`` directly.  The open question for
param-uq-04 was how to reach the *deep physiological parameters* that live inside
the ``SimulationDataEcoli`` dataclass (``sim_data``) produced by ParCa —
transcription RNAP rates/fractions, FBA objective weights, mass fractions — which
the original uqEcoli analysis targets.

This module is the bridge from a UQ-sampled scalar to the in-memory ``sim_data``
used by a run.  pbg-uq drives the sampling (``CallableAdapter`` + ``ForwardUQ``);
this adapter turns each sampled row into a runnable v2ecoli composite.

Post-ParCa vs. rebuild — the decision, per parameter
----------------------------------------------------
The v2ecoli online composite consumes the **``configs`` dict** (per-process
config projected from ``sim_data`` at bundle-build time by
``LoadSimData.get_config_by_name``) plus a cached ``initial_state`` — it never
holds the raw ``sim_data`` object at run time.  So a deep param is reachable by
one of two mechanisms, and *which one* is an empirical property of the param:

* ``POST_PARCA`` — the parameter surfaces **verbatim as a runtime config key**
  (or a config value we can transform).  We inject it through ``config_overrides``
  on the *shared, already-loaded* bundle: no disk I/O, no rebuild (~0 ms
  overhead).  Preferred whenever correct.

* ``REBUILD`` — the parameter is consumed only at bundle-build time (it shapes
  ``initial_state`` and/or is projected into configs by a getter), so patching a
  runtime config key does nothing.  We deep-copy the fitted ``sim_data``, mutate
  the attribute, and regenerate the bundle with
  :func:`v2ecoli.core.save_sim_input` into a per-sample temp cache dir.  Crucially
  this is **not a full ParCa refit** — ``save_sim_input`` reuses the fitted
  ``sim_data`` and only re-projects configs + regenerates ``initial_state``, which
  measures ~2 s/sample on the Mac mini, ~75x cheaper than the ~2.5-min full ParCa
  the study originally budgeted for.  The caveat: this captures the *direct*
  (initial-state/config-projection) effect of the parameter, not any refit
  feedback loop through ParCa's expression/kinetics fitting — the standard
  forward-UQ-on-sim_data assumption (and what uqEcoli's ``sim_data_setattr``
  pattern does).

Empirical reachability findings (this worktree, minimal glucose, ppGpp on)
--------------------------------------------------------------------------
Established by the liveness smoke tests under ``scratch/`` before wiring the run:

* ``transcription.rnap_elongation_rate`` (scales
  ``configs['ecoli-transcript-elongation']['rnaPolymeraseElongationRateDict']``)
  — **POST_PARCA, LIVE**: instantaneous growth rate moves ~-2.0 % / +1.3 % across
  a 0.7x/1.3x scan over a 150-step window.  The one genuinely live deep lever
  reachable without a rebuild.

* ``mass.cell_dry_mass_fraction`` — **REBUILD, LIVE**: cell_mass swings ~1519→1135
  fg (−25 %) across [0.25, 0.35]; it sets the dry/wet mass partition read by
  ``generate_initial_state``, so a config override is inert and a bundle
  regeneration is required.  Strong, clean signal on the *mass* observable.

* ``metabolism.kinetic_objective_weight`` / ``secretion_penalty_coeff`` —
  **POST_PARCA, INERT**: present as float config keys on ``ecoli-metabolism`` and
  cleanly injectable, but growth/mass are byte-identical across [5e-8, 5e-7] /
  [5e-4, 5e-3] over 60–150 steps.  The FBA objective weighting does not move
  single-cell growth at this scale (kept in the design as a deep-param negative
  control — its Sobol index should be ~0).

* ``transcription.rnap_active_fraction`` (``fracActiveRnapDict``) — **MASKED**:
  reachable as a config key, but with ``ppgpp_regulation`` on (the baseline
  default) the active fraction is recomputed each tick from ppGpp via
  ``get_rnap_active_fraction_from_ppGpp``, so the static-dict override is inert.
  Reaching it would need ppGpp coupling disabled or a rebuild that re-derives the
  ppGpp map — documented, not used here.

The registry below encodes these decisions.  ``build_deep_param_evaluator``
returns a batch ``evaluate(X) -> Y`` ready for ``pbg_uq.CallableAdapter``.
"""
from __future__ import annotations

import copy
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Injection mode
# --------------------------------------------------------------------------- #
POST_PARCA = "post_parca"
REBUILD = "rebuild"


@dataclass
class DeepParam:
    """One deep sim_data parameter and how a sampled value reaches the run.

    Attributes:
        name: UQ parameter name (column in the sample matrix).
        sim_data_path: dotted path to the underlying ``sim_data`` attribute,
            for provenance/reporting (e.g. ``mass.cell_dry_mass_fraction``).
        mode: :data:`POST_PARCA` or :data:`REBUILD`.
        bounds: ``(low, high)`` sampling range for the UQ value.
        make_overrides: POST_PARCA only. ``f(value, base_configs) -> {config_key:
            value}`` — the ``config_overrides`` fragment this param contributes.
        mutate: REBUILD only. ``f(value, sim_data) -> None`` — mutate the
            (already deep-copied) ``sim_data`` in place.
        note: free-text reachability finding (LIVE / INERT / MASKED).
    """

    name: str
    sim_data_path: str
    mode: str
    bounds: tuple[float, float]
    make_overrides: Callable[[float, dict], dict] | None = None
    mutate: Callable[[float, Any], None] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.mode == POST_PARCA and self.make_overrides is None:
            raise ValueError(f"{self.name}: POST_PARCA param needs make_overrides")
        if self.mode == REBUILD and self.mutate is None:
            raise ValueError(f"{self.name}: REBUILD param needs a mutate fn")


# --------------------------------------------------------------------------- #
# Concrete deep-param constructors (encode the empirical decisions above)
# --------------------------------------------------------------------------- #
def rnap_elongation_rate(bounds: tuple[float, float] = (0.7, 1.3)) -> DeepParam:
    """RNAP (transcription) elongation rate, as a scale factor on the nominal.

    POST_PARCA: multiplies every condition entry of
    ``rnaPolymeraseElongationRateDict`` on the ``ecoli-transcript-elongation``
    config (the runtime transcription speed).  Live on instantaneous growth rate.
    """

    def _mk(value: float, configs: dict) -> dict:
        d = copy.deepcopy(configs["ecoli-transcript-elongation"]["rnaPolymeraseElongationRateDict"])
        for k in d:
            d[k] = d[k] * float(value)
        return {"ecoli-transcript-elongation.rnaPolymeraseElongationRateDict": d}

    return DeepParam(
        name="rnap_elongation_rate",
        sim_data_path="process.transcription.rnaPolymeraseElongationRateDict",
        mode=POST_PARCA,
        bounds=bounds,
        make_overrides=_mk,
        note="LIVE on growth rate; scale factor on nominal RNAP elongation speed.",
    )


def kinetic_objective_weight(bounds: tuple[float, float] = (5e-8, 5e-7)) -> DeepParam:
    """FBA kinetic-objective weight (metabolism). POST_PARCA; deep-param control.

    Present as a float config key; injects cleanly but is inert on single-cell
    growth/mass at this scale — kept as a negative control (Sobol should be ~0).
    """

    def _mk(value: float, configs: dict) -> dict:
        return {"ecoli-metabolism.kinetic_objective_weight": float(value)}

    return DeepParam(
        name="kinetic_objective_weight",
        sim_data_path="process.metabolism.kinetic_objective_weight",
        mode=POST_PARCA,
        bounds=bounds,
        make_overrides=_mk,
        note="INERT negative control; FBA objective weighting does not move growth.",
    )


def cell_dry_mass_fraction(bounds: tuple[float, float] = (0.25, 0.35)) -> DeepParam:
    """Dry-to-wet mass fraction (mass). REBUILD: sets the initial-state partition.

    Config override is inert (read only by ``generate_initial_state``), so a
    per-sample ``save_sim_input`` bundle regeneration is required.  Strong signal
    on the mass observable (cell_mass).
    """

    def _mut(value: float, sim_data: Any) -> None:
        sim_data.mass.cell_dry_mass_fraction = float(value)

    return DeepParam(
        name="cell_dry_mass_fraction",
        sim_data_path="mass.cell_dry_mass_fraction",
        mode=REBUILD,
        bounds=bounds,
        mutate=_mut,
        note="LIVE on mass; requires bundle regeneration (~2 s, no ParCa refit).",
    )


# --------------------------------------------------------------------------- #
# Injector — materialize a sample into (bundle, config_overrides)
# --------------------------------------------------------------------------- #
class SimDataInjector:
    """Turn a sampled deep-param dict into a runnable (bundle, config_overrides).

    Loads the fitted ``sim_data`` once (only if any REBUILD params are present)
    and the base bundle once.  Per sample:

    * REBUILD params  -> deep-copy sim_data, apply every ``mutate``, regenerate a
      temp bundle via ``save_sim_input`` (returned with a cleanup callback).
    * POST_PARCA params -> collect ``config_overrides`` against the active bundle.

    A sample with no REBUILD param reuses the shared base bundle (no temp dir).
    """

    def __init__(
        self,
        params: list[DeepParam],
        cache_dir: str = "out/cache",
        sim_data_path: str | None = None,
    ) -> None:
        from v2ecoli.core import load_cache_bundle

        self.params = params
        self.cache_dir = cache_dir
        self._needs_rebuild = any(p.mode == REBUILD for p in params)
        self._base_bundle = load_cache_bundle(cache_dir)
        self._base_sim_data: Any = None
        if self._needs_rebuild:
            sdp = sim_data_path or os.path.join(cache_dir, "simData.cPickle")
            with open(sdp, "rb") as f:
                self._base_sim_data = pickle.load(f)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def bounds(self) -> np.ndarray:
        return np.array([list(p.bounds) for p in self.params], dtype=float)

    def materialize(self, sample: dict[str, float], seed: int = 0):
        """Return ``(bundle, config_overrides, cleanup)`` for one sample."""
        rebuild_params = [p for p in self.params if p.mode == REBUILD]
        post_params = [p for p in self.params if p.mode == POST_PARCA]

        if rebuild_params:
            from v2ecoli.core import load_cache_bundle, save_sim_input

            sd = pickle.loads(pickle.dumps(self._base_sim_data))  # isolate per sample
            for p in rebuild_params:
                p.mutate(sample[p.name], sd)
            tmp = tempfile.mkdtemp(prefix="uq_deep_rebuild_")
            save_sim_input(sd, bundle_dir=tmp, seed=seed)
            bundle = load_cache_bundle(tmp)
            cleanup = lambda d=tmp: shutil.rmtree(d, ignore_errors=True)
        else:
            bundle = self._base_bundle
            cleanup = lambda: None

        overrides: dict = {}
        for p in post_params:
            overrides.update(p.make_overrides(sample[p.name], bundle["configs"]))
        return bundle, overrides, cleanup


# --------------------------------------------------------------------------- #
# Batch evaluator for pbg_uq.CallableAdapter
# --------------------------------------------------------------------------- #
def build_deep_param_evaluator(
    params: list[DeepParam],
    *,
    observables: list[str],
    cache_dir: str = "out/cache",
    n_steps: int = 150,
    chunk: int = 30,
    seed: int = 0,
    core: Any = None,
    log: Callable[[str], None] = print,
):
    """Build ``(evaluate, injector, core)`` for a deep-param forward-UQ run.

    ``evaluate(X) -> Y`` maps an ``(n, d)`` sample matrix (columns ordered as
    ``params``) to an ``(n, len(observables))`` observable matrix, running one
    v2ecoli composite per row at the fixed ``seed`` (so within a design the only
    variance is parametric; vary ``seed`` across PCRV designs for stability).

    Observables are ``listeners.mass.<name>`` leaves read back from a per-sample
    XArray/zarr store (time-mean), matching the studies 00–02 harness.
    """
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    from v2ecoli.composites._helpers import set_null_emitter_override
    from v2ecoli.library.xarray_run import view_from_emit_paths, run_multigen_xarray
    from process_bigraph import Composite
    from pbg_uq.emit import read_run

    if core is None:
        core = build_core()
    injector = SimDataInjector(params, cache_dir=cache_dir)
    view = view_from_emit_paths([f"listeners.mass.{o}" for o in observables])
    meta_base = {
        "experiment_id": "uq_deep",
        "variant": 0,
        "lineage_seed": seed,
        "time_step": 1.0,
        "max_duration": float(n_steps),
    }

    def _run_one(sample: dict[str, float]) -> np.ndarray:
        bundle, overrides, cleanup = injector.materialize(sample, seed=seed)
        set_null_emitter_override(True)
        try:
            doc = baseline(core=core, seed=seed, bundle=bundle,
                           config_overrides=(overrides or None))
        finally:
            set_null_emitter_override(False)
        comp = Composite(doc, core=core)
        tmp = tempfile.mkdtemp(prefix="uq_deep_run_")
        store = os.path.join(tmp, "run.zarr")
        try:
            run_multigen_xarray(comp, store_path=store, view=view,
                                metadata_base=meta_base, max_steps=n_steps,
                                max_generations=1, chunk=chunk)
            agg = read_run(store, observables)
            return np.array([agg[o] for o in observables], dtype=float)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            cleanup()

    names = injector.param_names

    def evaluate(X: np.ndarray) -> np.ndarray:
        import time
        n = X.shape[0]
        Y = np.zeros((n, len(observables)), dtype=float)
        last_good: np.ndarray | None = None
        for i, row in enumerate(X):
            t0 = time.perf_counter()
            sample = {names[j]: float(row[j]) for j in range(len(names))}
            try:
                y = _run_one(sample)
                Y[i] = y
                last_good = y.copy()
                log(f"  sample {i + 1}/{n} ok in {time.perf_counter() - t0:.0f}s  "
                    + " ".join(f"{o}={y[k]:.4g}" for k, o in enumerate(observables)))
            except Exception as exc:  # noqa: BLE001 — keep PCE fitting alive
                Y[i] = last_good if last_good is not None else np.zeros(len(observables))
                log(f"  sample {i + 1}/{n} FAILED ({exc!r}); using fallback")
        return Y

    return evaluate, injector, core
