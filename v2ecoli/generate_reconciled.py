"""
Document generation for v2ecoli (reconciled architecture).

Uses ReconciledStep wrappers that group processes by allocator layer
and reconcile their bulk requests proportionally — matching the
Allocator's fairness logic without the Requester/Allocator/Evolver
split.

Each allocator layer becomes a single ReconciledStep that:
  1. Collects calculate_request from all processes in the layer
  2. Reconciles requests against available bulk supply
  3. Runs evolve_state with reconciled allocations
"""

import copy
import numpy as np
from bigraph_schema import allocate_core

from v2ecoli.types import ECOLI_TYPES

# Same partitioned process classes
from v2ecoli.processes.equilibrium import Equilibrium
from v2ecoli.processes.two_component_system import TwoComponentSystem
from v2ecoli.processes.rna_maturation import RnaMaturation
from v2ecoli.processes.complexation import Complexation
from v2ecoli.processes.protein_degradation import ProteinDegradation
from v2ecoli.processes.transcript_initiation import TranscriptInitiation
from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from v2ecoli.processes.chromosome_replication import ChromosomeReplication

from v2ecoli.steps.reconciled import ReconciledStep

# Reuse helpers from generate.py
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
    ALLOCATOR_LAYERS,
)


# ---------------------------------------------------------------------------
# Reconciled layer definitions
# ---------------------------------------------------------------------------

# Map allocator layer names to their step names in the execution graph
RECONCILED_LAYERS = {
    name.replace('allocator_', 'reconciled_'): procs
    for name, procs in ALLOCATOR_LAYERS.items()
}


# ---------------------------------------------------------------------------
# Feature modules (adapted for reconciled step names)
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
        'insert_before': 'reconciled_3',
        'steps': ['trna-attenuation-config'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']  # trna_attenuation disabled to match v1 default


# ---------------------------------------------------------------------------
# Execution layers (reconciled — one step per allocator layer)
# ---------------------------------------------------------------------------

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: standalone Steps (formerly in allocator_1, now non-partitioned)
    ['ecoli-equilibrium', 'ecoli-two-component-system',
     'ecoli-rna-maturation'], FLUSH,

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4a: protein degradation (standalone)
    ['ecoli-protein-degradation'],

    # Layer 4b: standalone init/replication/complexation
    # (formerly in allocator_2 alongside rna_degradation)
    ['ecoli-complexation', 'ecoli-chromosome-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],

    # Layer 4c: reconciled rna_degradation (still shares water with elongation)
    ['reconciled_2'], FLUSH,

    # Layer 5: reconciled elongation layer
    ['reconciled_3'], FLUSH,

    # Layer 6: chromosome structure + metabolism
    ['ecoli-chromosome-structure'], FLUSH,
    ['ecoli-metabolism'], FLUSH,

    # Layer 7: listeners (parallel)
    ['RNA_counts_listener', 'dna_supercoiling_listener', 'ecoli-mass-listener',
     'monomer_counts_listener', 'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener', 'unique_molecule_counts'], FLUSH,

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
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
# Step instantiation (reconciled)
# ---------------------------------------------------------------------------

def _instantiate_reconciled_layer(layer_name, proc_names, loader, core, seed=0):
    """Instantiate a ReconciledStep wrapping all processes in one allocator layer."""
    from v2ecoli.library.ecoli_step import set_current_core
    from v2ecoli.library.config_resolver import resolve_config

    processes = []
    all_topologies = {}

    for proc_name in proc_names:
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

        processes.append(process)
        all_topologies[proc_name] = topology

    # Processes whose evolve_state is self-contained: they read bulk
    # directly and produce deltas without needing a prior request phase.
    # Skipping calculate_request avoids redundant computation.
    EVOLVE_ONLY = {
        'ecoli-rna-maturation',       # stoichiometry recomputed in evolve
        'ecoli-complexation',         # stochastic system.evolve() in evolve
    }
    evolve_only_names = [p.name for p in processes if p.name in EVOLVE_ONLY]

    instance = ReconciledStep({
        'processes': processes,
        'seed': seed,
        'evolve_only': evolve_only_names,
    })

    # Build unified topology: union of all process topologies
    unified_topo = {}
    for proc_name, topo in all_topologies.items():
        for port, path in topo.items():
            if port not in unified_topo:
                unified_topo[port] = path

    # Add control ports
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


def _instantiate_standalone_step(step_name, config, loader, core):
    """Instantiate a standalone (non-partitioned) step."""
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.processes.metabolism import Metabolism
    from v2ecoli.steps.listeners.mass_listener import (
        MassListener, PostDivisionMassListener)
    from v2ecoli.steps.listeners.rna_counts import RNACounts
    from v2ecoli.steps.listeners.rna_synth_prob import RnaSynthProb
    from v2ecoli.steps.listeners.monomer_counts import MonomerCounts
    from v2ecoli.steps.listeners.dna_supercoiling import DnaSupercoiling
    from v2ecoli.steps.listeners.replication_data import ReplicationData
    from v2ecoli.steps.listeners.rnap_data import RnapData
    from v2ecoli.steps.listeners.unique_molecule_counts import UniqueMoleculeCounts
    from v2ecoli.steps.listeners.ribosome_data import RibosomeData
    from v2ecoli.steps.media_update import MediaUpdate
    from v2ecoli.steps.exchange_data import ExchangeData

    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-metabolism': Metabolism,
        # Processes promoted from PartitionedProcess to plain Step in baseline
        # (commits 1a5c0ab and 0e282f5) — must run as standalone here too,
        # otherwise they are silently dropped and the cell stops growing.
        'ecoli-equilibrium': Equilibrium,
        'ecoli-two-component-system': TwoComponentSystem,
        'ecoli-rna-maturation': RnaMaturation,
        'ecoli-complexation': Complexation,
        'ecoli-protein-degradation': ProteinDegradation,
        'ecoli-transcript-initiation': TranscriptInitiation,
        'ecoli-polypeptide-initiation': PolypeptideInitiation,
        'ecoli-chromosome-replication': ChromosomeReplication,
    }

    SIMPLE_STEPS = {
        'ecoli-mass-listener': MassListener,
        'post-division-mass-listener': PostDivisionMassListener,
        'RNA_counts_listener': RNACounts,
        'rna_synth_prob_listener': RnaSynthProb,
        'monomer_counts_listener': MonomerCounts,
        'dna_supercoiling_listener': DnaSupercoiling,
        'replication_data_listener': ReplicationData,
        'rnap_data_listener': RnapData,
        'unique_molecule_counts': UniqueMoleculeCounts,
        'ribosome_data_listener': RibosomeData,
        'media_update': MediaUpdate,
        'exchange_data': ExchangeData,
    }

    from v2ecoli.library.config_resolver import resolve_config
    config = resolve_config(config) if config else config

    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = _make_instance(step_cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    if step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = _make_instance(cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    return None


def _get_step_config(loader, step_name, core, seed=0):
    """Get step config for reconciled architecture."""
    # Skip allocators (not used)
    if step_name.startswith('allocator'):
        return None

    # Handle reconciled layers
    if step_name in RECONCILED_LAYERS:
        proc_names = RECONCILED_LAYERS[step_name]
        return _instantiate_reconciled_layer(
            step_name, proc_names, loader, core, seed=seed)

    # Handle standalone / listener / special steps
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

def build_reconciled_document(initial_state, configs, unique_names,
                              dry_mass_inc_dict=None, core=None, seed=0):
    """Build a reconciled-architecture document from pre-loaded configs.

    Same interface as generate.build_document but uses ReconciledStep
    instead of Requester/Allocator/Evolver.

    Steps: 11 partitioned processes -> 3 ReconciledSteps (one per layer)
    + same listeners, infra, and standalone steps.
    """
    if core is None:
        core = allocate_core()
        core.register_types(ECOLI_TYPES)

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    # Pre-create virtual stores (no request/allocate needed)
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

    # Initialize next_update_time for all processes
    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    # Listener sub-stores
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })

    # Feature module stores
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

    # Mock loader
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
            # Match baseline behavior: silently skip steps without a resolvable
            # config (e.g. exchange_data, which has no LoadSimData entry).
            continue
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
