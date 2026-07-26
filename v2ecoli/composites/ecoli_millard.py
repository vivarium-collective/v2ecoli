"""Millard whole-cell E. coli composite — one parameterized generator.

This module merges the former ``baseline_millard`` (LQR-free, env-responsive)
and the former ``millard-pdmp`` LQR composite (LQR setpoint-controlled) into a single
``baseline_millard(..., lqr=…)`` generator. Both drop v2ecoli's tFBA Metabolism
and run central-carbon metabolism through the Millard 2017 kinetic ODE
(``MillardPDMPMetabolism``) across all ~54 WCM processes; the ``lqr`` flag
selects between two wirings of that metabolism slot.

``lqr=False`` (DEFAULT) — bioreactor-oriented, environment-responsive
------------------------------------------------------------------------
  1. **No LQR controller.** The ``lqr-controller`` Process edge is dropped, and
     the Millard edge no longer reads ``lqr_control`` or writes
     ``control_applied``. With no LQR, the Millard kinetics are driven by the
     environment, not by a control law that pins fluxes to setpoints.
  2. **Env-driven kinetics.** The Millard step's ``external_concentrations``
     input port is wired to the cell's canonical environment store
     ``("environment", "external_concentrations")`` — the same path used by
     ``environment_driver`` / ``baseline_time_varying_env`` and the ``mbp-*``
     studies. External nutrient concentrations (e.g. GLCx) reach the ODE and
     modulate uptake fluxes each tick.

  For a standalone cell (no reactor yet) there is no live environment source, so
  ``external_concentrations`` resolves to an empty/default store and the Millard
  ODE runs on its internal glucose — expected and fine. The reactor coupling (a
  later study) supplies the real source through that store. Only the basico
  backend is supported: the env-responsive ``external_concentrations`` port was
  instrumented on ``MillardPDMPMetabolism`` (basico); the JAX port does not
  expose it.

``lqr=True`` — Millard ODE + multi-state LQR setpoint controller
----------------------------------------------------------------
  Reproduces the former ``millard-pdmp`` LQR composite (Phase-1 PDMP milestone):
  adds a ``['lqr-controller']`` execution layer, and the Millard edge instead
  reads ``lqr_control`` (``("shared", "lqr_control")``) and writes
  ``control_applied`` (``("shared", "control_applied")``). The ``backend`` knob
  selects the ODE integrator (``"basico"`` = COPASI, full LQR support;
  ``"jax"`` = JIT-compiled Diffrax port, faster at loose tols, no LQR yet), and
  the Phase-2/3 jump-process parameters (``with_ref_growth``,
  ``transcript_initiation_mode``, …) become active. Metabolism writeback to the
  WCM bulk is NOT wired (a separate ``central_metabolite_counts`` store).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle

from v2ecoli.composites._helpers import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    seed_mass_listener,
    _normalize_boundary_units,
    _expand_flushes,
    FLUSH,
    PARTITIONED_PROCESSES,  # noqa: F401 — kept for parity / discoverability
    ALL_PARTITIONED,
    CachedConfigLoader,
    DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)

# Reuse the shared step-config dispatcher, feature modules, and the env
# store registration hook. Only the metabolism edge wiring and the execution
# layers differ between the lqr=False / lqr=True paths.
from v2ecoli.composites._millard_helpers import (
    _get_step_config,
    _register_millard_pdmp_links,
    FEATURE_MODULES,
    DEFAULT_FEATURES,
)


# ---------------------------------------------------------------------------
# Execution layers — the LQR-free base. lqr=True inserts a ['lqr-controller']
# layer (+ its FLUSH) immediately after the metabolism FLUSH (see
# build_execution_layers).
# ---------------------------------------------------------------------------

# Only the Millard metabolism edge replaces 'ecoli-metabolism'.
MILLARD_EDGES = ['millard-pdmp-metabolism']

# Canonical environment store the cell exposes for external nutrient
# concentrations ({sbml_species_id: conc_mM}). Same path used by
# environment_driver / baseline_time_varying_env / the mbp-* studies.
EXTERNAL_CONCENTRATIONS_PATH = ("environment", "external_concentrations")

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: standalone
    ['ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation'], FLUSH,

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: protein deg + initiation/replication/complexation
    ['ecoli-protein-degradation'],
    ['ecoli-complexation', 'ecoli-chromosome-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],
    ['ecoli-rna-degradation_requester'],
    ['allocator_2'],
    ['ecoli-rna-degradation_evolver'], FLUSH,

    # Layer 5: elongation
    ['ecoli-polypeptide-elongation_requester', 'ecoli-transcript-elongation_requester'],
    ['allocator_3'],
    ['ecoli-polypeptide-elongation_evolver', 'ecoli-transcript-elongation_evolver'], FLUSH,

    # Layer 6: chromosome structure + Millard metabolism replacement.
    # Replaces the original ['ecoli-metabolism'] slot. When lqr=True a
    # ['lqr-controller'] layer (+FLUSH) is inserted right after the Millard
    # FLUSH; when lqr=False the Millard ODE is environment-driven (no
    # setpoint controller).
    ['ecoli-chromosome-structure'], FLUSH,
    ['millard-pdmp-metabolism'], FLUSH,

    # Layer 7: listeners
    ['counts_deriver', 'ecoli-mass-listener',
     'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener'], FLUSH,

    # Layer 7b: per-process likelihood collector (after the listener layer).
    ['likelihood_collector'], FLUSH,

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
    ['mark_d_period'], FLUSH,
    ['division'],
]


def build_execution_layers(features=None, *, lqr=False):
    """Build the flow layers, applying optional feature modules.

    When ``lqr=True`` a ``['lqr-controller']`` layer (and its FLUSH) is inserted
    immediately after the ``['millard-pdmp-metabolism'], FLUSH`` pair — exactly
    where the former ``millard-pdmp`` LQR composite placed it — so the built document
    is byte-identical to that legacy composite.
    """
    layers = copy.deepcopy(BASE_EXECUTION_LAYERS)
    if lqr:
        for i, layer in enumerate(layers):
            if isinstance(layer, list) and 'millard-pdmp-metabolism' in layer:
                # layers[i + 1] is the FLUSH following the Millard layer; put
                # the controller (+ its FLUSH) right after that FLUSH.
                layers.insert(i + 2, FLUSH)
                layers.insert(i + 2, ['lqr-controller'])
                break
    for feat_name in (features or []):
        feat = FEATURE_MODULES.get(feat_name)
        if feat is None:
            continue
        if 'insert_after' in feat:
            ref = feat['insert_after']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in feat.get('steps', []):
                        layers.insert(i + 1, [step_name])
                    break
        if 'insert_before' in feat:
            ref = feat['insert_before']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in reversed(feat.get('steps', [])):
                        layers.insert(i, [step_name])
                    break
        for listener in feat.get('listeners', []):
            for layer in layers:
                if isinstance(layer, list) and 'ecoli-mass-listener' in layer:
                    if listener not in layer:
                        layer.append(listener)
                    break
    return _expand_flushes(layers)


FLOW_ORDER = [step for layer in build_execution_layers(DEFAULT_FEATURES) for step in layer]


# ---------------------------------------------------------------------------
# Millard metabolism edge builders
# ---------------------------------------------------------------------------

def _build_millard_edge(core: Any, *, tick_s: float = 1.0):
    """Build the LQR-free, env-responsive ``MillardPDMPMetabolism`` edge (lqr=False).

    Differences from the lqr=True edge (``_build_millard_lqr_edge``):
      - ``lqr_control`` is NOT wired into ``in_topo`` (no LQR drive — the
        step's empty-lqr path is a no-op control, u=0).
      - ``external_concentrations`` IS wired to the environment store so the
        bioreactor's external nutrient levels reach the ODE each tick.
      - ``control_applied`` is NOT wired into ``out_topo`` (no LQR diagnostics).

    Only the basico backend is supported (the env-responsive port lives on
    ``MillardPDMPMetabolism``; the JAX port has no ``external_concentrations``).
    """
    from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism
    cfg = {
        "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
        "tick_s": tick_s,
        "intervals": 10,
        "control_reaction": "PTS_4",
        "control_parameter": "kF",
        "u_clip": 0.5,
    }
    instance = MillardPDMPMetabolism(config=cfg, core=core)
    in_topo = {
        "bulk": ("bulk",),
        # listeners.mass provides cell_mass_fg used to compute live cell
        # volume for the mM->count translation.
        "listeners_mass": ("listeners", "mass"),
        # Bioreactor drive: external nutrient concentrations (mM) keyed by
        # SBML species id. Empty/default store here (no live reactor yet) =>
        # Millard runs on its internal glucose; the reactor study supplies
        # the real source through this same path.
        "external_concentrations": EXTERNAL_CONCENTRATIONS_PATH,
    }
    out_topo = {
        "species_concentrations": ("shared", "central_metabolites"),
        # Per-reaction fluxes (mM/s) published to the agent-root
        # ('central_fluxes',) store.
        "central_fluxes": ("central_fluxes",),
        "bulk": ("bulk",),
        # Per-tick signed medium-exchange counts (O2/glucose/acetate) routed to
        # the cell's ('environment', 'exchange') store — the same store the WCM
        # metabolism writes. Drives the reactor coupler's O2 consumption and
        # makes the mass-conservation deriver evaluable on the Millard cell.
        "environment": ("environment",),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


def _build_millard_lqr_edge(core: Any, *, tick_s: float = 1.0,
                            backend: str = "basico"):
    """Build the LQR-driven Millard Process edge (lqr=True).

    Reads ``lqr_control`` (``("shared", "lqr_control")``) and writes
    ``control_applied``; there is no ``external_concentrations`` input.

    ``backend="basico"`` uses COPASI/basico (default; supports LQR control
    parameter modulation). ``backend="jax"`` uses the JIT-compiled
    JAX/Diffrax port — measured 1.8x faster at loose tolerances, but omits
    LQR control until the SBML->JAX translator is extended to expose
    runtime-settable parameters (task #19).
    """
    if backend == "jax":
        from v2ecoli.steps.millard_pdmp_metabolism_jax import (
            MillardPDMPMetabolismJAX as _Cls,
        )
        cfg = {
            "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
            "tick_s": tick_s,
            # Tight tol matches basico's LSODA accuracy; the loose-tol
            # variant produced slightly different bulk deltas that drove
            # Equilibrium's reconciler to fail. Tight tol costs ~2x basico
            # standalone but stays within ~10⁻¹⁰ of basico's trajectory.
            "rtol": 1e-6,
            "atol": 1e-9,
        }
    elif backend == "basico":
        from v2ecoli.steps.millard_pdmp_metabolism import (
            MillardPDMPMetabolism as _Cls,
        )
        cfg = {
            "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
            "tick_s": tick_s,
            "intervals": 10,
            "control_reaction": "PTS_4",
            "control_parameter": "kF",
            "u_clip": 0.5,
        }
    else:
        raise ValueError(f"Unknown PDMP backend: {backend!r}. "
                         "Expected 'basico' or 'jax'.")
    instance = _Cls(config=cfg, core=core)
    in_topo = {
        "lqr_control": ("shared", "lqr_control"),
        "bulk": ("bulk",),
        # listeners.mass provides cell_mass_fg used to compute live cell
        # volume for the mM->count translation (task #15).
        "listeners_mass": ("listeners", "mass"),
    }
    out_topo = {
        "species_concentrations": ("shared", "central_metabolites"),
        # Per-reaction fluxes (mM/s) published to the agent-root
        # ('central_fluxes',) store that FBAFluxCoupler reads (its topology
        # declares ('central_fluxes',)).
        "central_fluxes": ("central_fluxes",),
        "control_applied": ("shared", "control_applied"),
        "bulk": ("bulk",),
        # Per-tick signed medium-exchange counts (O2/glucose/acetate) -> the
        # cell's ('environment', 'exchange') store (WCM convention).
        "environment": ("environment",),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


def _build_lqr_controller_edge(core: Any, *, tick_s: float = 1.0):
    """Build the multi-state LQR controller Process edge (lqr=True only)."""
    from v2ecoli.steps.lqr_controller_multistate import LQRControllerMultiState
    cfg = {
        "linearization_npz": "v2ecoli/data/millard_linearization.npz",
        "Q_diag_weight": 1.0,
        "R": 0.1,
        "tick_s": tick_s,
    }
    instance = LQRControllerMultiState(config=cfg, core=core)
    in_topo = {"central_metabolites_millard": ("shared", "central_metabolites")}
    out_topo = {
        "lqr_control": ("shared", "lqr_control"),
        "lqr_diagnostics": ("shared", "lqr_diagnostics"),
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


@composite_generator(
    name="ecoli_millard",
    description=(
        "Whole-cell E. coli composite with v2ecoli's tFBA Metabolism replaced "
        "by the Millard 2017 kinetic ODE (MillardPDMPMetabolism). The `lqr` "
        "flag selects the metabolism wiring: lqr=False (default) drops the LQR "
        "setpoint controller and wires the Millard step's external_concentrations "
        "input to ('environment', 'external_concentrations') so a bioreactor "
        "environment can drive central-carbon kinetics (basico only); lqr=True "
        "adds the multi-state LQR controller layer and the setpoint control "
        "wiring (lqr_control input, control_applied output) — the former "
        "former millard-pdmp LQR composite. Standalone (no live reactor) => external "
        "concentrations default to empty and the ODE runs on internal glucose."
    ),
    parameters={
        "seed": {
            "type": "integer", "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string", "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
        "tick_s": {
            "type": "float", "default": 1.0,
            "description": "Millard metabolism update interval in seconds",
        },
        "features": {
            "type": "list",
            "default": [],
            "description": "Opt-in feature-module names to insert in addition to "
                           "the defaults (e.g. ['mass_conservation']). Each "
                           "must be a key in FEATURE_MODULES. (lqr=False path.)",
        },
        "lqr": {
            "type": "boolean", "default": False,
            "description": (
                "False (default): LQR-free, environment-responsive Millard "
                "cell. True: add the multi-state LQR setpoint controller layer "
                "and the setpoint control wiring — reproduces the former "
                "former millard-pdmp LQR composite."
            ),
        },
        "backend": {
            "type": "string", "default": "basico",
            "description": (
                "ODE integrator backend for the Millard substep (lqr=True "
                "only). 'basico' uses COPASI (full LQR support); 'jax' uses "
                "the JIT-compiled Diffrax port (faster at loose tols; no LQR "
                "yet). Ignored when lqr=False (basico only)."
            ),
        },
        "with_ref_growth": {
            "type": "boolean", "default": False,
            "description": (
                "(lqr=True) Enable the reference-growth driver — scaffold that "
                "drives precursor pools to compensate for the Millard ODE's "
                "missing biomass equation. See `ref_growth_flux_source` "
                "for the two flux modes."
            ),
        },
        "ref_growth_flux_source": {
            "type": "string", "default": "proportional",
            "description": (
                "(lqr=True) Driver flux mode (only used when "
                "with_ref_growth=True). 'proportional' scales pools at "
                "μ=2.44e-4/s — teleonomic but moves cm_final only ~2 fg of the "
                "187 fg gap because precursor turnover (~1.8M ATP/s) is ~1000× "
                "larger. 'measured_kfba' injects at constant per-second rates "
                "measured from a 600 s kFBA-baseline run "
                "(scripts/sample_kfba_precursor_fluxes.py → "
                ".pbg/runs/kfba-precursor-fluxes.json); top rates: "
                "GLT 5413/s, ATP 1640/s, UTP 803/s, TTP 787/s."
            ),
        },
        "transcript_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "(lqr=True) Phase-2 jump-process opt-in for transcription "
                "initiation. 'discrete' (default, legacy): multinomial event "
                "distribution with exact Σ N_i = n_target — bit-identical to "
                "baseline. 'poisson': per-promoter Poisson(n_target · p_i) "
                "tau-leap; each promoter becomes an independent continuous-time "
                "jump process whose per-tick marginal Phase-3 inference can "
                "integrate against. Resource cap is the actual inactive-RNAP "
                "pool, not the discrete-time target (avoids 12% asymmetric-"
                "truncation undercount)."
            ),
        },
        "polypeptide_initiation_mode": {
            "type": "string", "default": "discrete",
            "description": (
                "(lqr=True) Phase-2 jump-process opt-in for translation "
                "initiation. Same dispatch as transcript_initiation_mode but "
                "for PolypeptideInitiation; ribosome activation per protein "
                "becomes per-protein Poisson(n_target · p_i) tau-leap "
                "with the resource cap pinned to min(30S, 50S) instead "
                "of the discrete-time target."
            ),
        },
        "ref_growth_feedback_tau_s": {
            "type": "number", "default": 1.0,
            "description": (
                "(lqr=True) Feedback smoothing time-constant for the "
                "consumption_matched ref-growth driver. ``1.0`` (default) "
                "keeps the legacy tight per-tick controller — the driver "
                "fully compensates last tick's variance, so the PDMP "
                "ensemble's per-tick jump-process variance is invisible "
                "at cell_mass. Larger values smooth the consumption "
                "estimate (EMA) so per-tick stochasticity manifests in "
                "pool counts and downstream in mass; the long-run mean "
                "still tracks the kFBA-measured growth rate. Try 60 s "
                "for a 1-minute-window controller."
            ),
        },
        "ref_growth_feedback_period_ticks": {
            "type": "integer", "default": 1,
            "description": (
                "(lqr=True) Sprint-8 sparse-injection knob for the "
                "consumption_matched ref-growth driver. ``1`` (default) "
                "acts every tick. Larger values decimate the controller — "
                "it acts every Nth tick, compensating the cumulative "
                "consumption with a period-scaled target injection. "
                "Pools drift open-loop between corrections, letting per-"
                "tick Poisson variance manifest in the PDMP ensemble. "
                "Bypasses the EMA when > 1 (one knob at a time)."
            ),
        },
        "transcript_init_prob_scale": {
            "type": "number", "default": 1.0,
            "description": (
                "(lqr=True) Phase-3 sprint-7 ABC-SMC knob. In poisson mode, "
                "multiplies the per-promoter initiation rate by this "
                "scalar before sampling. Default 1.0 reproduces the "
                "unperturbed sampler; values away from 1.0 produce "
                "ensembles at distinguishable parameter settings for "
                "the ABC-SMC inference stub. Only effective when "
                "transcript_initiation_mode='poisson'."
            ),
        },
        "polypeptide_init_prob_scale": {
            "type": "number", "default": 1.0,
            "description": (
                "(lqr=True) Phase-3 sprint-10 ABC-SMC knob, mirror of "
                "transcript_init_prob_scale on the translation side. "
                "Only effective when polypeptide_initiation_mode="
                "'poisson'."
            ),
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    core_extensions=[_register_millard_pdmp_links],
)
def baseline_millard(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    tick_s: float = 1.0,
    features: list | None = None,
    lqr: bool = False,
    backend: str = "basico",
    with_ref_growth: bool = False,
    ref_growth_flux_source: str = "proportional",
    transcript_initiation_mode: str = "discrete",
    polypeptide_initiation_mode: str = "discrete",
    ref_growth_feedback_tau_s: float = 1.0,
    ref_growth_feedback_period_ticks: int = 1,
    transcript_init_prob_scale: float = 1.0,
    polypeptide_init_prob_scale: float = 1.0,
) -> dict:
    """Build the process-bigraph state document for the Millard cell composite.

    ``lqr=False`` (default) reproduces the env-responsive Millard cell;
    ``lqr=True`` reproduces the former ``millard-pdmp`` LQR composite (Millard ODE +
    multi-state LQR controller).
    """
    if core is None:
        core = build_core()
    _register_millard_pdmp_links(core)

    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

    # Feature selection differs by path (preserves each legacy composite's
    # exact behavior): lqr=False honors the caller's `features`; lqr=True
    # only opts in the ref-growth driver via `with_ref_growth`.
    if lqr:
        _features = list(DEFAULT_FEATURES)
        if with_ref_growth and 'ref_growth_driver' not in _features:
            _features.append('ref_growth_driver')
    else:
        _features = list(DEFAULT_FEATURES)
        for f in (features or []):
            if f not in _features:
                _features.append(f)
    features = _features

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    for store in ['listeners', 'process',
                  'allocator_rng', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault('mass', {'dry_mass': 0.0, 'cell_mass': 0.0})
    cell_state.setdefault('allocator_rng', np.random.RandomState(seed=seed))

    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'enabled': False,
    })

    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    cell_state.setdefault('request', {})
    cell_state.setdefault('allocate', {})
    for proc_name in ALL_PARTITIONED:
        cell_state['request'].setdefault(proc_name, {'bulk': {}})
        cell_state['allocate'].setdefault(proc_name, {'bulk': {}})
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Agent-root central_fluxes store: Millard publishes per-reaction fluxes
    # (mM/s) here.
    cell_state.setdefault('central_fluxes', {})

    # Millard shared stores.
    cell_state.setdefault('shared', {})
    cell_state['shared'].setdefault('central_metabolites', {})
    cell_state.setdefault('environment', {})
    if lqr:
        # LQR / bridge stores (former millard-pdmp LQR composite).
        cell_state['shared'].setdefault('central_fluxes', {})
        cell_state['shared'].setdefault('lqr_control', {'u': 0.0, 'u_dict': {}})
        cell_state['shared'].setdefault('lqr_diagnostics', {})
        cell_state['shared'].setdefault('control_applied', {})
        cell_state['shared'].setdefault('bridge_diagnostics', {})
        # FBABridge writeback target — kept under 'shared/' (same parent as
        # 'central_metabolites') because top-level dict stores don't accept
        # dict-merge updates in process-bigraph the way nested map stores do.
        cell_state['shared'].setdefault('central_metabolite_counts', {})
    else:
        # Canonical environment store the Millard edge reads its external
        # nutrient concentrations from. Declared so the topology resolves even
        # when no live reactor source is attached (empty store => no-op drive).
        cell_state['environment'].setdefault('external_concentrations', {})
    # Medium-exchange store the Millard edge writes (signed per-tick counts) and
    # the mass-conservation deriver reads. map[float] accumulates per-tick.
    cell_state['environment'].setdefault('exchange', {})

    # Mock loader: cache configs + minimal sim_data.
    loader = CachedConfigLoader(configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)

    execution_layers = build_execution_layers(features, lqr=lqr)
    flow_order = [step for layer in execution_layers for step in layer]

    _process_cache = {}
    for step_name in flow_order:
        if step_name == 'millard-pdmp-metabolism':
            if lqr:
                cell_state[step_name] = _build_millard_lqr_edge(
                    core, tick_s=tick_s, backend=backend)
            else:
                cell_state[step_name] = _build_millard_edge(core, tick_s=tick_s)
            continue
        if step_name == 'lqr-controller':
            cell_state[step_name] = _build_lqr_controller_edge(core, tick_s=tick_s)
            continue
        if lqr:
            config = _get_step_config(
                loader, step_name, core, _process_cache, master_seed=seed,
                ref_growth_flux_source=ref_growth_flux_source,
                ref_growth_feedback_tau_s=ref_growth_feedback_tau_s,
                ref_growth_feedback_period_ticks=ref_growth_feedback_period_ticks,
                transcript_initiation_mode=transcript_initiation_mode,
                transcript_init_prob_scale=transcript_init_prob_scale,
                polypeptide_initiation_mode=polypeptide_initiation_mode,
                polypeptide_init_prob_scale=polypeptide_init_prob_scale,
            )
        else:
            config = _get_step_config(
                loader, step_name, core, _process_cache, master_seed=seed,
            )
        if config is not None:
            if len(config) == 5:
                instance, topology, edge_type, in_topo, out_topo = config
                cell_state[step_name] = make_edge(
                    instance, topology, input_topology=in_topo,
                    output_topology=out_topo, edge_type=edge_type)
            else:
                instance, topology, edge_type = config
                cell_state[step_name] = make_edge(
                    instance, topology, edge_type=edge_type)

    for proc_name, proc_instance in _process_cache.items():
        cell_state['process'][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    seed_mass_listener(cell_state, core)

    inject_flow_dependencies(
        cell_state, flow_order, layers=execution_layers)

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': flow_order,
    }
