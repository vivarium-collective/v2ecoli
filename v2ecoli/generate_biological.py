"""
Document generation for v2ecoli (biological architecture — pilot).

Fourth architecture variant organized by biological function rather than
resource-competition allocator layers. Purpose: reduce layer count and
concentrate arbitration to the one bulk pool that actually competes
(charged tRNAs, shared between polypeptide_elongation and rna_degradation).

Design choices vs reconciled:
- The three partitioned processes (rna_degradation, transcript_elongation,
  polypeptide_elongation) run in ONE ReconciledStep, not two. Reconciliation
  handles non-overlapping pools correctly, so merging the two historical
  allocator layers (rna_degradation alone, and the two elongations) into
  one group produces the same arbitration result with fewer scheduler
  layers.
- Flushes are emitted at biological-subsystem boundaries, not after every
  unique-molecule mutation. We retain flushes where downstream steps read
  unique-state changes committed upstream.
- Standalone processes keep running as plain Steps exactly like reconciled.
- No semantic behavior change vs reconciled is expected; this is purely a
  layout reorganization. We ship it as a 4th comparison column so deltas
  are measured, not assumed.
"""

import copy
import numpy as np
from bigraph_schema import allocate_core

from v2ecoli.types import ECOLI_TYPES

# Same partitioned process classes (reused, not re-imported)
from v2ecoli.generate import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    _seed_mass_listener,
    _normalize_boundary_units,
    _make_instance,
    _get_special_step,
    _expand_flushes,
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
)

# Reuse reconciled's standalone dispatch; sequential layer is our own.
from v2ecoli.generate_reconciled import _instantiate_standalone_step
from v2ecoli.steps.sequential_core import SequentialCoreStep


# ---------------------------------------------------------------------------
# Biological grouping: one reconciled group for all pool-contending processes
# ---------------------------------------------------------------------------

BIOLOGICAL_LAYERS = {
    # Two reconciled groups, kept separate (rather than merged into one
    # scheduler step) because rna_degradation writes unique-state RNA
    # deletions that must flush via UniqueUpdate before the elongation
    # group's calculate_request reads active_RNAPs / RNAs. Internal flush
    # isn't available as a library primitive, so we keep the FLUSH
    # between bio_degradation and bio_elongation.
    'bio_degradation': [['ecoli-rna-degradation']],
    'bio_elongation': [
        ['ecoli-polypeptide-elongation', 'ecoli-transcript-elongation'],
    ],
}


# ---------------------------------------------------------------------------
# Feature modules — same registry as reconciled
# ---------------------------------------------------------------------------

FEATURE_MODULES = {
    'supercoiling': {
        'insert_after': 'ecoli-chromosome-structure',
        'steps': ['dna-supercoiling-step'],
        'listeners': ['dna_supercoiling_listener'],
    },
    'ppgpp_regulation': {
        'insert_before': 'ecoli-transcript-initiation',
        'steps': ['ppgpp-initiation'],
    },
    'trna_attenuation': {
        'insert_before': 'bio_elongation',
        'steps': ['trna-attenuation-config'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']


# ---------------------------------------------------------------------------
# Execution layers — biologically organized, flushes only at subsystem edges
# ---------------------------------------------------------------------------

BASE_EXECUTION_LAYERS = [
    # --- Cell cycle start: post-division bookkeeping -----------------------
    ['post-division-mass-listener'], FLUSH,

    # --- Environment & media ------------------------------------------------
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # --- Gene regulation: signalling + RNA maturation ----------------------
    # Equilibrium and TCS are local signalling; RNA maturation acts on
    # transcripts already produced in prior timestep.
    ['ecoli-equilibrium', 'ecoli-two-component-system',
     'ecoli-rna-maturation'], FLUSH,

    # --- Gene regulation: TF binding ----------------------------------------
    ['ecoli-tf-binding'], FLUSH,

    # --- Proteostasis: protein degradation ---------------------------------
    ['ecoli-protein-degradation'],

    # --- Biogenesis initiation + evolve-only complexation -------------------
    # Non-competing production: complexation, chromosome replication, and the
    # two initiation steps share no consumable pool.
    ['ecoli-complexation', 'ecoli-chromosome-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],

    # --- Core pool arbitration: degradation then elongation -----------------
    # Two reconciled sub-groups with an intervening flush so elongation
    # sees post-degradation unique-state (RNAs marked deleted, etc.).
    ['bio_degradation'], FLUSH,
    ['bio_elongation'], FLUSH,

    # --- Chromosome structure & metabolism ----------------------------------
    ['ecoli-chromosome-structure'], FLUSH,
    ['ecoli-metabolism'], FLUSH,

    # --- Listeners (parallel; all read-only, no mutual dep) -----------------
    ['RNA_counts_listener', 'dna_supercoiling_listener', 'ecoli-mass-listener',
     'monomer_counts_listener', 'replication_data_listener',
     'ribosome_data_listener', 'rna_synth_prob_listener',
     'rnap_data_listener', 'unique_molecule_counts'], FLUSH,

    # --- Emitter + clock ----------------------------------------------------
    ['emitter'],
    ['global_clock'],

    # --- Cell cycle end: division check -------------------------------------
    ['mark_d_period'], FLUSH,
    ['division'],
]


def build_execution_layers(features=None):
    """Build EXECUTION_LAYERS from base + enabled feature modules."""
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


EXECUTION_LAYERS = build_execution_layers(DEFAULT_FEATURES)
FLOW_ORDER = [step for layer in EXECUTION_LAYERS for step in layer]


# ---------------------------------------------------------------------------
# Step instantiation — reuse reconciled paths, add bio_core dispatch
# ---------------------------------------------------------------------------

def _instantiate_sequential_layer(layer_name, subgroup_names, loader, core, seed=0):
    """Instantiate a SequentialCoreStep wrapping named sub-groups in order.

    subgroup_names is a list-of-lists of process names. Each inner list
    becomes one reconciled sub-group inside the single step.
    """
    from v2ecoli.library.ecoli_step import set_current_core
    from v2ecoli.library.config_resolver import resolve_config

    subgroups = []
    all_topologies = {}

    for group in subgroup_names:
        subgroup = []
        for proc_name in group:
            proc_cls = PARTITIONED_PROCESSES[proc_name]
            try:
                config = loader.get_config_by_name(proc_name)
            except (KeyError, AttributeError):
                config = {}
            config = resolve_config(config) if config else config

            set_current_core(core)
            process = proc_cls(config)
            set_current_core(None)

            topology = dict(config.get('topology', {}) or {})
            if not topology:
                topology = getattr(process, 'topology',
                                   getattr(proc_cls, 'topology', {}))
                if callable(topology):
                    topology = topology()
                topology = dict(topology)

            subgroup.append(process)
            all_topologies[proc_name] = topology
        subgroups.append(subgroup)

    instance = SequentialCoreStep({
        'subgroups': subgroups,
        'seed': seed,
    })

    # Union of all process topologies, with control ports added.
    unified_topo = {}
    for topo in all_topologies.values():
        for port, path in topo.items():
            if port not in unified_topo:
                unified_topo[port] = path

    in_topo = dict(unified_topo)
    in_topo['global_time'] = ('global_time',)
    in_topo.setdefault('timestep', ('timestep',))
    in_topo['next_update_time'] = ('next_update_time',)

    out_topo = {'next_update_time': ('next_update_time',)}
    out_ports = set(instance.outputs().keys())
    for port in out_ports:
        if port in ('next_update_time', 'global_time', 'timestep'):
            continue
        if port in unified_topo:
            out_topo[port] = unified_topo[port]
        elif port == 'listeners':
            out_topo['listeners'] = unified_topo.get(
                'listeners', ('listeners',))

    return instance, unified_topo, 'step', in_topo, out_topo


def _get_step_config(loader, step_name, core, seed=0):
    """Get step config for biological architecture."""
    if step_name.startswith('allocator'):
        return None

    if step_name in BIOLOGICAL_LAYERS:
        proc_names = BIOLOGICAL_LAYERS[step_name]
        return _instantiate_sequential_layer(
            step_name, proc_names, loader, core, seed=seed)

    try:
        config = loader.get_config_by_name(step_name)
    except (KeyError, AttributeError):
        return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    return _instantiate_standalone_step(step_name, config, loader, core)


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_biological_document(initial_state, configs, unique_names,
                              dry_mass_inc_dict=None, core=None, seed=0):
    """Build a biological-architecture document from pre-loaded configs.

    Same interface as generate.build_document but uses one merged
    ReconciledStep for all three bulk-pool-competing processes.
    """
    if core is None:
        core = allocate_core()
        core.register_types(ECOLI_TYPES)

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    for store in ['listeners', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault(
        'mass', {'dry_mass': 0.0, 'cell_mass': 0.0})

    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })

    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'attenuation_probability': [],
    })

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Mock loader — identical to reconciled/departitioned
    class _CachedLoader:
        def __init__(self, configs, unique_names, dry_mass_inc_dict):
            self._configs = configs
            self.unique_names = unique_names

            class _SimData:
                class _InternalState:
                    class _UniqueMolecule:
                        def __init__(self, names):
                            self.unique_molecule_definitions = {
                                n: {} for n in names}
                    unique_molecule = None
                    def __init__(self, names):
                        self.unique_molecule = self._UniqueMolecule(names)
                internal_state = None
                expectedDryMassIncreaseDict = {}

            self.sim_data = _SimData()
            self.sim_data.internal_state = _SimData._InternalState(
                unique_names)
            self.sim_data.expectedDryMassIncreaseDict = (
                dry_mass_inc_dict or {})

        def get_config_by_name(self, name):
            if name in self._configs:
                return self._configs[name]
            raise KeyError(f'Unknown: {name}')

    loader = _CachedLoader(configs, unique_names, dry_mass_inc_dict)

    for step_name in FLOW_ORDER:
        config = _get_step_config(loader, step_name, core, seed=seed)
        if config is None:
            raise RuntimeError(
                f"Biological: step {step_name!r} appears in FLOW_ORDER "
                f"but could not be instantiated. Register it in "
                f"BIOLOGICAL_LAYERS or generate_reconciled._instantiate_"
                f"standalone_step.")
        if len(config) == 5:
            instance, topology, edge_type, in_topo, out_topo = config
            cell_state[step_name] = make_edge(
                instance, topology, input_topology=in_topo,
                output_topology=out_topo, edge_type=edge_type)
        else:
            instance, topology, edge_type = config
            cell_state[step_name] = make_edge(
                instance, topology, edge_type=edge_type)

    _seed_state_from_defaults(cell_state)
    _seed_mass_listener(cell_state, core)

    inject_flow_dependencies(
        cell_state, FLOW_ORDER, layers=EXECUTION_LAYERS)

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': FLOW_ORDER,
    }
