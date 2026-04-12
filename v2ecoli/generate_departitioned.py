"""
Document generation for v2ecoli (departitioned architecture).

Builds the E. coli simulation document using DepartitionedStep wrappers
instead of the Requester/Allocator/Evolver pattern. Each partitioned
process runs as a single step that computes its own request and
immediately applies it.

Uses the same process classes and configs as the partitioned architecture,
just wired differently.
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
from v2ecoli.processes.rna_degradation import RnaDegradation
from v2ecoli.processes.transcript_initiation import TranscriptInitiation
from v2ecoli.processes.transcript_elongation import TranscriptElongation
from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from v2ecoli.processes.polypeptide_elongation import PolypeptideElongation
from v2ecoli.processes.chromosome_replication import ChromosomeReplication

# DepartitionedStep wrapper
from v2ecoli.steps.departitioned import DepartitionedStep

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
)


# ---------------------------------------------------------------------------
# Feature modules (same as baseline, adapted for departitioned step names)
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
        'insert_before': 'ecoli-transcript-elongation',
        'steps': ['trna-attenuation-config'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']  # trna_attenuation disabled to match v1 default


# ---------------------------------------------------------------------------
# Execution layers (departitioned — no allocators)
# ---------------------------------------------------------------------------

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: partition layer 1 (sequential standalone)
    ['ecoli-equilibrium'],
    ['ecoli-rna-maturation'],
    ['ecoli-two-component-system'], FLUSH,

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: partition layer 2 (sequential standalone)
    ['ecoli-complexation'],
    ['ecoli-protein-degradation'],
    ['ecoli-rna-degradation'],
    ['ecoli-transcript-initiation'],
    ['ecoli-polypeptide-initiation'],
    ['ecoli-chromosome-replication'], FLUSH,

    # Layer 5: partition layer 3 (sequential standalone)
    ['ecoli-transcript-elongation'],
    ['ecoli-polypeptide-elongation'], FLUSH,

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
# Step instantiation (departitioned)
# ---------------------------------------------------------------------------

def _instantiate_departitioned_step(step_name, config, loader, core):
    """Instantiate a step for the departitioned architecture."""
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

    # Processes whose evolve_state is self-contained and can skip
    # calculate_request entirely (avoids redundant computation)
    EVOLVE_ONLY = {
        'ecoli-rna-maturation',   # stoichiometry recomputed in evolve
        'ecoli-complexation',     # stochastic system.evolve() in evolve
    }

    # Departitioned: wrap PartitionedProcess in DepartitionedStep
    if step_name in PARTITIONED_PROCESSES:
        proc_cls = PARTITIONED_PROCESSES[step_name]
        from v2ecoli.library.ecoli_step import set_current_core
        set_current_core(core)
        process = proc_cls(config)
        set_current_core(None)

        # Flag evolve-only processes to skip calculate_request in _do_update
        if step_name in EVOLVE_ONLY:
            process.evolve_only = True

        topology = dict(config.get('topology', {}) or {})
        if not topology:
            topology = getattr(process, 'topology',
                               getattr(proc_cls, 'topology', {}))
            if callable(topology):
                topology = topology()
            topology = dict(topology)

        instance = DepartitionedStep({
            'time_step': config.get('time_step', 1),
            'process': process,
        })

        in_topo = dict(topology)
        in_topo['global_time'] = ('global_time',)
        in_topo.setdefault('timestep', ('timestep',))
        in_topo['next_update_time'] = ('next_update_time', step_name)

        out_topo = {
            'next_update_time': ('next_update_time', step_name),
        }
        out_ports = set(instance.outputs().keys())
        for port in out_ports:
            if port in ('next_update_time', 'global_time', 'timestep'):
                continue
            if port in topology:
                out_topo[port] = topology[port]
            elif port == 'listeners':
                out_topo['listeners'] = topology.get(
                    'listeners', ('listeners',))

        return instance, topology, 'step', in_topo, out_topo

    # Standalone steps (unchanged)
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


def _get_step_config(loader, step_name, core):
    """Get step config — skip allocators."""
    if step_name.startswith('allocator'):
        return None

    try:
        config = loader.get_config_by_name(step_name)
    except (KeyError, AttributeError):
        return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    return _instantiate_departitioned_step(step_name, config, loader, core)


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_departitioned_document(initial_state, configs, unique_names,
                                 dry_mass_inc_dict=None, core=None, seed=0):
    """Build a departitioned-architecture document from pre-loaded configs.

    Same interface as generate.build_document but without allocators.
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
        config = _get_step_config(loader, step_name, core)
        if config is None:
            raise RuntimeError(
                f"Departitioned: step {step_name!r} appears in FLOW_ORDER "
                f"but could not be instantiated. This usually means the "
                f"step was promoted from PartitionedProcess to Step in "
                f"baseline generate.py and needs to be registered in "
                f"STANDALONE_STEPS here."
            )
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
