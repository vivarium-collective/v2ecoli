"""
Partitioned document generation for v2ecoli.

Builds a simulation document using the old partitioned architecture
(requester/allocator/evolver pattern) from commit 5d03dea.

This module provides ``make_partitioned_composite`` which creates a
document compatible with ``Composite()`` using the frozen partitioned
process classes from ``v2ecoli.partitioned.processes``.
"""

import os
import copy
import numpy as np
from bigraph_schema import allocate_core

from v2ecoli.library.units import units
from v2ecoli.types import ECOLI_TYPES
from v2ecoli.generate import (
    make_edge,
    inject_flow_dependencies,
    list_paths,
    _seed_mass_listener,
    _get_special_step,
)

# Import all partitioned process classes from the frozen snapshot
from v2ecoli.partitioned.processes import (
    EquilibriumLogic, EquilibriumRequester, EquilibriumEvolver,
    TwoComponentSystemLogic, TwoComponentSystemRequester, TwoComponentSystemEvolver,
    RnaMaturationLogic, RnaMaturationRequester, RnaMaturationEvolver,
    ComplexationLogic, ComplexationRequester, ComplexationEvolver,
    ProteinDegradationLogic, ProteinDegradationRequester, ProteinDegradationEvolver,
    RnaDegradationLogic, RnaDegradationRequester, RnaDegradationEvolver,
    TranscriptInitiationLogic, TranscriptInitiationRequester, TranscriptInitiationEvolver,
    TranscriptElongationLogic, TranscriptElongationRequester, TranscriptElongationEvolver,
    PolypeptideInitiationLogic, PolypeptideInitiationRequester, PolypeptideInitiationEvolver,
    PolypeptideElongationLogic, PolypeptideElongationRequester, PolypeptideElongationEvolver,
    ChromosomeReplicationLogic, ChromosomeReplicationRequester, ChromosomeReplicationEvolver,
)


# ---------------------------------------------------------------------------
# Partitioned execution layers (from 5d03dea generate.py)
# ---------------------------------------------------------------------------

PARTITIONED_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener', 'unique_update_1'],

    # Layer 1: media/environment (sequential sub-steps)
    ['media_update'],
    ['unique_update_2'],
    ['ecoli-tf-unbinding'],
    ['exchange_data'],
    ['unique_update_3'],

    # Layer 2: partition layer 1 -- requesters (parallel)
    ['ecoli-equilibrium_requester', 'ecoli-rna-maturation_requester',
     'ecoli-two-component-system_requester'],
    ['allocator_1'],
    # Layer 2: partition layer 1 -- evolvers (parallel)
    ['ecoli-equilibrium_evolver', 'ecoli-rna-maturation_evolver',
     'ecoli-two-component-system_evolver'],
    ['unique_update_4'],

    # Layer 3: TF binding
    ['ecoli-tf-binding'],
    ['unique_update_5'],

    # Layer 4: partition layer 2 -- requesters (parallel)
    ['ecoli-chromosome-replication_requester', 'ecoli-complexation_requester',
     'ecoli-polypeptide-initiation_requester', 'ecoli-protein-degradation_requester',
     'ecoli-rna-degradation_requester', 'ecoli-transcript-initiation_requester'],
    ['allocator_2'],
    # Layer 4: partition layer 2 -- evolvers (parallel)
    ['ecoli-chromosome-replication_evolver', 'ecoli-complexation_evolver',
     'ecoli-polypeptide-initiation_evolver', 'ecoli-protein-degradation_evolver',
     'ecoli-rna-degradation_evolver', 'ecoli-transcript-initiation_evolver'],
    ['unique_update_6'],

    # Layer 5: partition layer 3 -- elongation requesters (parallel)
    ['ecoli-polypeptide-elongation_requester', 'ecoli-transcript-elongation_requester'],
    ['allocator_3'],
    # Layer 5: partition layer 3 -- elongation evolvers (parallel)
    ['ecoli-polypeptide-elongation_evolver', 'ecoli-transcript-elongation_evolver'],
    ['unique_update_7'],

    # Layer 6: chromosome structure + metabolism (sequential)
    ['ecoli-chromosome-structure'],
    ['unique_update_8'],
    ['ecoli-metabolism'],
    ['unique_update_9'],

    # Layer 7: listeners (parallel)
    ['RNA_counts_listener', 'dna_supercoiling_listener', 'ecoli-mass-listener',
     'monomer_counts_listener', 'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener', 'unique_molecule_counts'],
    ['unique_update_10'],

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
    ['mark_d_period'],
    ['unique_update_11'],
    ['division'],
]

PARTITIONED_FLOW = [step for layer in PARTITIONED_EXECUTION_LAYERS for step in layer]


# ---------------------------------------------------------------------------
# Partitioned EXPLICIT_STEPS mapping
# ---------------------------------------------------------------------------

EXPLICIT_STEPS = {
    'ecoli-protein-degradation': {
        'class': ProteinDegradationLogic,
        'requester_class': ProteinDegradationRequester,
        'evolver_class': ProteinDegradationEvolver,
    },
    'ecoli-equilibrium': {
        'class': EquilibriumLogic,
        'requester_class': EquilibriumRequester,
        'evolver_class': EquilibriumEvolver,
    },
    'ecoli-two-component-system': {
        'class': TwoComponentSystemLogic,
        'requester_class': TwoComponentSystemRequester,
        'evolver_class': TwoComponentSystemEvolver,
    },
    'ecoli-rna-maturation': {
        'class': RnaMaturationLogic,
        'requester_class': RnaMaturationRequester,
        'evolver_class': RnaMaturationEvolver,
    },
    'ecoli-complexation': {
        'class': ComplexationLogic,
        'requester_class': ComplexationRequester,
        'evolver_class': ComplexationEvolver,
    },
    'ecoli-polypeptide-initiation': {
        'class': PolypeptideInitiationLogic,
        'requester_class': PolypeptideInitiationRequester,
        'evolver_class': PolypeptideInitiationEvolver,
    },
    'ecoli-transcript-initiation': {
        'class': TranscriptInitiationLogic,
        'requester_class': TranscriptInitiationRequester,
        'evolver_class': TranscriptInitiationEvolver,
    },
    'ecoli-rna-degradation': {
        'class': RnaDegradationLogic,
        'requester_class': RnaDegradationRequester,
        'evolver_class': RnaDegradationEvolver,
    },
    'ecoli-polypeptide-elongation': {
        'class': PolypeptideElongationLogic,
        'requester_class': PolypeptideElongationRequester,
        'evolver_class': PolypeptideElongationEvolver,
    },
    'ecoli-transcript-elongation': {
        'class': TranscriptElongationLogic,
        'requester_class': TranscriptElongationRequester,
        'evolver_class': TranscriptElongationEvolver,
    },
    'ecoli-chromosome-replication': {
        'class': ChromosomeReplicationLogic,
        'requester_class': ChromosomeReplicationRequester,
        'evolver_class': ChromosomeReplicationEvolver,
    },
}

_ALL_PARTITIONED = [
    'ecoli-chromosome-replication', 'ecoli-complexation',
    'ecoli-equilibrium', 'ecoli-polypeptide-elongation',
    'ecoli-polypeptide-initiation', 'ecoli-protein-degradation',
    'ecoli-rna-degradation', 'ecoli-rna-maturation',
    'ecoli-transcript-elongation', 'ecoli-transcript-initiation',
    'ecoli-two-component-system',
]


# ---------------------------------------------------------------------------
# _instantiate_step (from 5d03dea generate.py, using partitioned classes)
# ---------------------------------------------------------------------------

def _instantiate_partitioned_step(step_name, config, loader, core,
                                   process_cache=None):
    """Instantiate a partitioned process step from its config.

    Uses the frozen partitioned process classes from
    ``v2ecoli.partitioned.processes``.
    """
    if process_cache is None:
        process_cache = {}

    # Standalone and simple steps are shared with the current architecture
    from v2ecoli.processes.tf_binding import TfBindingLogic, TfBindingStep
    from v2ecoli.processes.tf_unbinding import TfUnbindingLogic, TfUnbindingStep
    from v2ecoli.processes.chromosome_structure import (
        ChromosomeStructureLogic, ChromosomeStructureStep)
    from v2ecoli.processes.metabolism import MetabolismLogic, MetabolismStep
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

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBindingStep,
        'ecoli-tf-unbinding': TfUnbindingStep,
        'ecoli-chromosome-structure': ChromosomeStructureStep,
        'ecoli-metabolism': MetabolismStep,
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

    if base_name in EXPLICIT_STEPS:
        spec = EXPLICIT_STEPS[base_name]
        if base_name in process_cache:
            process = process_cache[base_name]
        else:
            process = spec['class'](parameters=config)
            process_cache[base_name] = process
        topology = process.topology

        if step_name.endswith('_requester'):
            req_config = {'process': process, 'process_name': base_name}
            instance = spec['requester_class'](config=req_config, core=core)
            in_topo = dict(topology)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {'next_update_time': ('next_update_time', base_name)}
            if 'request' in out_ports:
                out_topo['request'] = (f'request_{base_name}',)
            if 'listeners' in out_ports:
                out_topo['listeners'] = topology.get('listeners', ('listeners',))
            return instance, topology, 'step', in_topo, out_topo

        elif step_name.endswith('_evolver'):
            ev_config = {'process': process}
            instance = spec['evolver_class'](config=ev_config, core=core)
            in_topo = dict(topology)
            in_topo['allocate'] = (f'allocate_{base_name}',)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {'next_update_time': ('next_update_time', base_name)}
            for port in out_ports:
                if port == 'next_update_time':
                    continue
                if port in topology:
                    out_topo[port] = topology[port]
                elif port == 'listeners':
                    out_topo['listeners'] = ('listeners',)
            return instance, topology, 'step', in_topo, out_topo

    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = step_cls(config=config, core=core)
        topology = instance.topology
        return instance, topology, 'step'

    elif step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = cls(config=config, core=core)
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    return None


def _get_partitioned_step_config(loader, step_name, core, process_cache=None):
    """Get (instance, topology, edge_type) for a step.

    Mirrors ``_get_step_config`` from generate.py but uses the frozen
    partitioned process classes.
    """
    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Handle allocators first (not in loader configs)
    if step_name.startswith('allocator'):
        from v2ecoli.steps.allocator import Allocator
        all_partitioned = list(EXPLICIT_STEPS.keys())
        ALLOCATOR_LAYERS = {
            'allocator_1': ['ecoli-equilibrium', 'ecoli-rna-maturation',
                            'ecoli-two-component-system'],
            'allocator_2': ['ecoli-chromosome-replication', 'ecoli-complexation',
                            'ecoli-polypeptide-initiation', 'ecoli-protein-degradation',
                            'ecoli-rna-degradation', 'ecoli-transcript-initiation'],
            'allocator_3': ['ecoli-polypeptide-elongation',
                            'ecoli-transcript-elongation'],
        }
        try:
            alloc_config = loader.get_config_by_name('allocator')
        except Exception:
            alloc_config = {}
        layer_procs = ALLOCATOR_LAYERS.get(step_name, all_partitioned)
        if not alloc_config.get('process_names'):
            alloc_config['process_names'] = all_partitioned
        alloc_config['layer_processes'] = layer_procs
        instance = Allocator(config=alloc_config, core=core)
        topo = instance.topology
        return instance, topo, 'step'

    try:
        config = loader.get_config_by_name(base_name)
    except (KeyError, AttributeError):
        try:
            config = loader.get_config_by_name(step_name)
        except (KeyError, AttributeError):
            return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    return _instantiate_partitioned_step(
        step_name, config, loader, core, process_cache)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_partitioned_document_from_configs(initial_state, configs, unique_names,
                                            dry_mass_inc_dict=None, core=None,
                                            seed=0):
    """Build a partitioned-architecture document from pre-loaded configs.

    This mirrors ``build_document_from_configs`` from ``v2ecoli.generate``
    but uses the frozen partitioned process classes and the old
    requester/allocator/evolver execution layers.

    Args:
        initial_state: Dict with bulk, unique, environment, boundary.
        configs: Dict mapping step names to config dicts.
        unique_names: List of unique molecule names.
        dry_mass_inc_dict: Optional dict of expected dry mass increases.
        core: bigraph-schema core. If None, creates one.
        seed: Random seed.

    Returns:
        Document dict for Composite().
    """
    if core is None:
        core = allocate_core()
        core.register_types(ECOLI_TYPES)

    cell_state = {}
    cell_state.update(initial_state)

    # Pre-create virtual stores
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

    # Pre-create flat per-process request/allocate stores
    for proc_name in _ALL_PARTITIONED:
        cell_state[f'request_{proc_name}'] = {'bulk': []}
        cell_state[f'allocate_{proc_name}'] = {'bulk': {}}

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21) * units.mmol / units.L / units.s,
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Create a mock loader that returns configs from the cache
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
            self.sim_data.internal_state = _SimData._InternalState(unique_names)
            self.sim_data.expectedDryMassIncreaseDict = dry_mass_inc_dict or {}

        def get_config_by_name(self, name):
            if name in self._configs:
                return self._configs[name]
            raise KeyError(f'Unknown: {name}')

    loader = _CachedLoader(configs, unique_names, dry_mass_inc_dict)

    _process_cache = {}
    for step_name in PARTITIONED_FLOW:
        config = _get_partitioned_step_config(
            loader, step_name, core, _process_cache)
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

    _seed_mass_listener(cell_state, core)
    inject_flow_dependencies(
        cell_state, PARTITIONED_FLOW, layers=PARTITIONED_EXECUTION_LAYERS)

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': PARTITIONED_FLOW,
    }


def make_partitioned_composite(cache_dir=None, initial_state=None, configs=None,
                               unique_names=None, dry_mass_inc_dict=None,
                               core=None, seed=0):
    """Build and return a partitioned-architecture Composite.

    Args:
        cache_dir: Path to cache directory (loads initial_state + configs from cache).
        initial_state: Dict with bulk, unique, environment, boundary.
        configs: Dict mapping step names to config dicts.
        unique_names: List of unique molecule names.
        dry_mass_inc_dict: Optional dict of expected dry mass increases.
        core: bigraph-schema core. If None, creates one.
        seed: Random seed.

    Returns:
        process_bigraph.Composite instance.
    """
    import dill
    from process_bigraph import Composite
    from v2ecoli.cache import load_initial_state

    if core is None:
        core = allocate_core()
        core.register_types(ECOLI_TYPES)

    if cache_dir is not None and initial_state is None:
        initial_state = load_initial_state(
            os.path.join(cache_dir, 'initial_state.json'))
        with open(os.path.join(cache_dir, 'sim_data_cache.dill'), 'rb') as f:
            cache = dill.load(f)
        configs = cache['configs']
        unique_names = cache.get('unique_names', [])
        dry_mass_inc_dict = cache.get('dry_mass_inc_dict', {})

    document = build_partitioned_document_from_configs(
        initial_state, configs, unique_names,
        dry_mass_inc_dict=dry_mass_inc_dict,
        core=core, seed=seed)

    return Composite(document, core=core)
