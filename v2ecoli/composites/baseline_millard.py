"""Bioreactor-oriented Millard cell composite — the LQR-free, env-responsive
sibling of ``millard_pdmp_baseline``.

This composite makes the same structural swap as ``millard_pdmp_baseline``
(v2ecoli's tFBA Metabolism is removed; the Millard 2017 kinetic ODE
``MillardPDMPMetabolism`` runs central-carbon metabolism across all ~54 WCM
processes), but is aimed at *bioreactor* coupling rather than setpoint control:

  1. **No LQR controller.** The ``lqr-controller`` Process edge is dropped, and
     the Millard edge no longer reads ``lqr_control`` or writes
     ``control_applied``. With no LQR, the Millard kinetics are driven by the
     environment, not by a control law that pins fluxes to setpoints.
  2. **Env-driven kinetics.** The Millard step's ``external_concentrations``
     input port (added in the prior task) is wired to the cell's canonical
     environment store ``("environment", "external_concentrations")`` — the
     same path used by ``environment_driver`` / ``baseline_time_varying_env``
     and the ``mbp-*`` studies. External nutrient concentrations (e.g. GLCx)
     reach the ODE and modulate uptake fluxes each tick.

For *this* study (a standalone cell, no reactor yet) there is no live
environment source, so ``external_concentrations`` resolves to an empty/default
store and the Millard ODE runs on its internal glucose — expected and fine. The
reactor coupling (a later study) supplies the real source through that store.

Only the basico backend is supported: the env-responsive
``external_concentrations`` port was instrumented on
``MillardPDMPMetabolism`` (basico); the JAX port does not expose it.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

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

# Reuse the parent's step-config dispatcher, feature modules, and the env
# store registration hook. Only the metabolism edge wiring and the execution
# layers differ here (no LQR controller).
from v2ecoli.composites.millard_pdmp_baseline import (
    _get_step_config,
    _register_millard_pdmp_links,
    FEATURE_MODULES,
    DEFAULT_FEATURES,
)


# ---------------------------------------------------------------------------
# Execution layers — identical to millard_pdmp_baseline but WITHOUT the
# ['lqr-controller'] layer (and its FLUSH). No LQR controller is instantiated.
# ---------------------------------------------------------------------------

# Only the Millard metabolism edge replaces 'ecoli-metabolism'; no lqr-controller.
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
    # Replaces the original ['ecoli-metabolism'] slot. Unlike
    # millard_pdmp_baseline there is NO ['lqr-controller'] layer — the
    # Millard ODE is environment-driven, not setpoint-controlled.
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


def build_execution_layers(features=None):
    """Build the flow layers, applying optional feature modules.

    Mirrors ``millard_pdmp_baseline.build_execution_layers`` but over this
    module's LQR-free ``BASE_EXECUTION_LAYERS``.
    """
    layers = copy.deepcopy(BASE_EXECUTION_LAYERS)
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
# Millard metabolism edge — LQR-free, env-responsive.
# ---------------------------------------------------------------------------

def _build_millard_edge(core: Any, *, tick_s: float = 1.0):
    """Build the LQR-free, env-responsive ``MillardPDMPMetabolism`` edge.

    Differences from ``millard_pdmp_baseline._build_millard_pdmp_edge``:
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
    }
    edge = make_edge(
        instance, in_topo,
        input_topology=in_topo, output_topology=out_topo,
        edge_type='process', config=cfg,
    )
    edge['interval'] = tick_s
    return edge


@composite_generator(
    name="baseline_millard",
    description=(
        "LQR-free, environment-responsive whole-cell E. coli composite with "
        "v2ecoli's tFBA Metabolism replaced by the Millard 2017 kinetic ODE "
        "(MillardPDMPMetabolism, basico backend). Sibling of "
        "millard_pdmp_baseline: drops the LQR setpoint controller and wires "
        "the Millard step's external_concentrations input to the cell's "
        "('environment', 'external_concentrations') store so a bioreactor "
        "environment can drive central-carbon kinetics. Standalone (no live "
        "reactor) => external concentrations default to empty and the ODE runs "
        "on its internal glucose."
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
                           "must be a key in FEATURE_MODULES.",
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
) -> dict:
    """Build the process-bigraph state document for the env-responsive,
    LQR-free Millard cell composite."""
    if core is None:
        core = build_core()
    _register_millard_pdmp_links(core)

    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

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

    # Millard shared store (no LQR / bridge stores needed here).
    cell_state.setdefault('shared', {})
    cell_state['shared'].setdefault('central_metabolites', {})

    # Canonical environment store the Millard edge reads its external nutrient
    # concentrations from. Declared so the topology resolves even when no live
    # reactor source is attached (empty store => no-op env drive).
    cell_state.setdefault('environment', {})
    cell_state['environment'].setdefault('external_concentrations', {})

    # Mock loader: cache configs + minimal sim_data.
    loader = CachedConfigLoader(configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)

    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    _process_cache = {}
    for step_name in flow_order:
        if step_name == 'millard-pdmp-metabolism':
            cell_state[step_name] = _build_millard_edge(core, tick_s=tick_s)
            continue
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
