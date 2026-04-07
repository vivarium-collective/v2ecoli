"""
Document generation for v2ecoli.

Builds the E. coli simulation document from raw data through the ParCa
pipeline, entirely within v2ecoli — no vEcoli dependency required.

Pipeline: raw data (TSV) → ParCa → simData → LoadSimData → process configs
→ initial state → document (pickle)
"""

import os
import copy
import pickle
import time

import dill
import numpy as np
from bigraph_schema import allocate_core

from v2ecoli.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from v2ecoli.reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
from v2ecoli.library.sim_data import LoadSimData
from v2ecoli.library.filepath import ROOT_PATH
from v2ecoli.library.units import units
from v2ecoli.types import ECOLI_TYPES


# ---------------------------------------------------------------------------
# Flow ordering (mirrors the default vEcoli execution layers)
# ---------------------------------------------------------------------------

# The flow defines the execution order of all steps. Partitioned processes
# are split into requester → allocator → evolver layers. UniqueUpdate steps
# are inserted between layers to flush accumulated unique molecule updates.

# Layer-based execution order. Steps within the same layer share a flow
# token and can potentially execute in parallel when sequential_steps is
# disabled. Layers execute strictly in order.
EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener', 'unique_update_1'],

    # Layer 1: media/environment (sequential sub-steps)
    ['media_update'],
    ['unique_update_2'],
    ['ecoli-tf-unbinding'],
    ['exchange_data'],
    ['unique_update_3'],

    # Layer 2: standalone processes (sequential)
    ['ecoli-rna-maturation'],
    ['ecoli-equilibrium'],
    ['ecoli-two-component-system'],
    ['unique_update_4'],

    # Layer 3: TF binding
    ['ecoli-tf-binding'],
    ['unique_update_5'],

    # Layer 4: standalone processes (sequential)
    ['ecoli-complexation'],
    ['ecoli-protein-degradation'],
    ['ecoli-rna-degradation'],
    ['ecoli-transcript-initiation'],
    ['ecoli-polypeptide-initiation'],
    ['ecoli-chromosome-replication'],
    ['unique_update_6'],

    # Layer 5: elongation (sequential)
    ['ecoli-transcript-elongation'],
    ['ecoli-polypeptide-elongation'],
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

# Flat list for backward compatibility (preserves original ordering within layers)
DEFAULT_FLOW = [step for layer in EXECUTION_LAYERS for step in layer]


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _seed_mass_listener(cell_state, core):
    """Run mass listener once to populate initial mass values."""

    # Get mass listener config from LoadSimData would be complex,
    # so just find the instance if it's already been added to cell_state
    for name in ['post-division-mass-listener', 'ecoli-mass-listener']:
        edge = cell_state.get(name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'next_update'):
            continue

        # Build a simple view from cell_state
        view = {}
        wires = edge.get('inputs', {})
        for port, wire in wires.items():
            if isinstance(wire, list) and wire:
                target = cell_state
                for seg in wire:
                    if isinstance(target, dict):
                        target = target.get(seg)
                    else:
                        target = None
                        break
                if target is not None:
                    view[port] = target

        # Ensure all arrays are writeable
        for key in ['bulk']:
            arr = view.get(key)
            if arr is not None and hasattr(arr, 'flags'):
                try:
                    arr.flags.writeable = True
                except ValueError:
                    view[key] = arr.copy()
                    view[key].flags.writeable = True
        for uname, uarr in view.get('unique', {}).items():
            if hasattr(uarr, 'flags'):
                try:
                    uarr.flags.writeable = True
                except ValueError:
                    pass

        try:
            delta = instance.next_update(1.0, view)
            if delta and 'listeners' in delta:
                mass = delta['listeners'].get('mass', {})
                cell_state['listeners']['mass'].update(mass)
        except Exception:
            pass
        break  # Only need to run one mass listener


def list_paths(path):
    """Convert tuple paths to list paths. Flatten _path dicts."""
    if isinstance(path, tuple):
        return list(path)
    elif isinstance(path, dict):
        if '_path' in path:
            # Flatten: _path is the base, other keys are overrides
            # For Composite compatibility, split into separate ports
            result = {}
            for key, subpath in path.items():
                if key == '_path':
                    continue
                result[key] = list_paths(subpath)
            return result
        return {key: list_paths(subpath) for key, subpath in path.items()}
    return path


def inject_flow_dependencies(cell_state, flow_order, layers=None):
    """Add flow token wiring and priorities to enforce execution order.

    When layers is provided, uses layer-based tokens: all steps in a layer
    depend on the previous layer's token and produce the current layer's
    token. Steps within a layer can potentially run in parallel.

    When layers is None, falls back to per-step chain tokens.
    """
    if layers is None:
        # Legacy per-step chain
        n = len(flow_order)
        for i, step_name in enumerate(flow_order):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                continue
            edge['priority'] = float(n - i)
            if i == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']
            if i > 0:
                edge.setdefault('inputs', {})[f'_flow_in_{i}'] = [f'_flow_token_{i-1}']
            if i < n - 1:
                edge.setdefault('outputs', {})[f'_flow_out_{i}'] = [f'_flow_token_{i}']
        return

    # Layer-based tokens
    n_layers = len(layers)
    # Global step index for priority (earlier steps get higher priority)
    step_idx = 0
    total_steps = sum(len(layer) for layer in layers)

    for layer_idx, layer in enumerate(layers):
        for j, step_name in enumerate(layer):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                step_idx += 1
                continue

            # Priority: earlier layers and earlier within-layer get higher
            edge['priority'] = float(total_steps - step_idx)

            # All layer-0 steps depend on global_time (the simulation trigger)
            if layer_idx == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']

            # All steps in this layer depend on previous layer's token
            if layer_idx > 0:
                edge.setdefault('inputs', {})[f'_layer_in_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx - 1}']

            # All steps in the layer produce the output token.
            # Within-layer parallelism requires process-bigraph to
            # support barrier tokens (W/W on same value = sync point).
            if layer_idx < n_layers - 1:
                edge.setdefault('outputs', {})[f'_layer_out_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx}']

            step_idx += 1


def make_edge(instance, topology, input_topology=None, output_topology=None,
              edge_type='step', config=None):
    """Create an edge dict for a process/step instance.

    Includes the instance directly and its input/output schemas.
    When input_topology/output_topology are provided, they override
    topology for the respective wiring direction.
    """
    wires = list_paths(topology)
    in_wires = list_paths(input_topology) if input_topology is not None else wires
    out_wires = list_paths(output_topology) if output_topology is not None else wires
    state = {'priority': 1.0} if edge_type == 'step' else {'interval': 1.0}

    # Get port schemas from instance
    inputs_schema = {}
    outputs_schema = {}
    if hasattr(instance, 'inputs'):
        try:
            inputs_schema = instance.inputs()
        except Exception:
            pass
    if hasattr(instance, 'outputs'):
        try:
            outputs_schema = instance.outputs()
        except Exception:
            pass

    # Store address and config for .pbg serialization
    cls = type(instance)
    address = f'local:{cls.__module__}.{cls.__qualname__}'
    raw_config = config or getattr(instance, '_raw_config', {})

    state.update({
        '_type': edge_type,
        'address': address,
        'config': raw_config,
        'instance': instance,
        '_inputs': inputs_schema,
        '_outputs': outputs_schema,
        'inputs': copy.deepcopy(in_wires),
        'outputs': copy.deepcopy(out_wires),
    })
    return state


# ---------------------------------------------------------------------------
# Run ParCa
# ---------------------------------------------------------------------------

def run_parca(outdir='out/kb', cpus=1):
    """Run the ParCa pipeline: raw data → simData.

    Args:
        outdir: Directory to save simData pickle.
        cpus: Number of CPUs for parallel fitting.

    Returns:
        Path to the simData pickle file.
    """
    os.makedirs(outdir, exist_ok=True)
    sim_data_path = os.path.join(outdir, 'simData.cPickle')

    if os.path.exists(sim_data_path):
        print(f"simData already exists at {sim_data_path}")
        return sim_data_path

    print(f"{time.ctime()}: Loading raw data...")
    raw_data = KnowledgeBaseEcoli(
        operons_on=True,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False)

    print(f"{time.ctime()}: Running ParCa (fitSimData_1)...")
    sim_data = fitSimData_1(raw_data=raw_data, cpus=cpus)

    print(f"{time.ctime()}: Saving simData to {sim_data_path}")
    with open(sim_data_path, 'wb') as f:
        pickle.dump(sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return sim_data_path


# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------

def build_document(sim_data_path=None, seed=0):
    """Build a complete E. coli simulation document.

    If sim_data_path is not provided, runs ParCa first.

    Args:
        sim_data_path: Path to simData pickle. If None, runs ParCa.
        seed: Random seed for initial state generation.

    Returns:
        Document dict with 'state', 'flow_order' keys.
    """
    if sim_data_path is None:
        sim_data_path = run_parca()

    core = allocate_core()
    core.register_types(ECOLI_TYPES)

    # Load simData and generate process configs + initial state
    loader = LoadSimData(sim_data_path=sim_data_path, seed=seed)
    initial_state = loader.generate_initial_state()

    # Build cell state: initial state + process edges
    cell_state = {}
    cell_state.update(initial_state)

    # Pre-create virtual stores that steps will read/write
    for store in ['listeners', 'process',
                  'allocator_rng', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')

    # Pre-populate listeners.mass with defaults so mass listener can run
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault('mass', {'dry_mass': 0.0, 'cell_mass': 0.0})

    # Seed random state for allocator
    cell_state.setdefault('allocator_rng', np.random.RandomState(seed=seed))

    # Pre-populate process_state with defaults that metabolism needs
    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21) * units.mmol / units.L / units.s,
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Cache for shared Logic instances (requester + evolver share one)
    _process_cache = {}

    # Add all process/step edges with their configs and topologies
    for step_name in DEFAULT_FLOW:
        config = _get_step_config(loader, step_name, core, _process_cache)
        if config is not None:
            if len(config) == 5:
                instance, topology, edge_type, in_topo, out_topo = config
                cell_state[step_name] = make_edge(
                    instance, topology, input_topology=in_topo,
                    output_topology=out_topo, edge_type=edge_type)
            else:
                instance, topology, edge_type = config
                cell_state[step_name] = make_edge(instance, topology, edge_type=edge_type)

    # Seed mass listener after edges are created
    _seed_mass_listener(cell_state, core)

    # Add flow dependencies (synthetic wiring for execution order)
    inject_flow_dependencies(cell_state, DEFAULT_FLOW, layers=EXECUTION_LAYERS)

    # Wrap in agent container
    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': DEFAULT_FLOW,
    }


def build_document_from_configs(initial_state, configs, unique_names,
                                dry_mass_inc_dict=None, core=None, seed=0):
    """Build document from pre-loaded configs and initial state (from cache).

    Args:
        initial_state: Dict with bulk, unique, environment, boundary.
        configs: Dict mapping step names to config dicts.
        unique_names: List of unique molecule names.
        core: bigraph-schema core. If None, creates one.
        seed: Random seed.

    Returns:
        Document dict for Composite.
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
                            self.unique_molecule_definitions = {n: {} for n in names}
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
    for step_name in DEFAULT_FLOW:
        config = _get_step_config(loader, step_name, core, _process_cache)
        if config is not None:
            if len(config) == 5:
                instance, topology, edge_type, in_topo, out_topo = config
                cell_state[step_name] = make_edge(
                    instance, topology, input_topology=in_topo,
                    output_topology=out_topo, edge_type=edge_type)
            else:
                instance, topology, edge_type = config
                cell_state[step_name] = make_edge(instance, topology, edge_type=edge_type)

    _seed_mass_listener(cell_state, core)
    inject_flow_dependencies(cell_state, DEFAULT_FLOW, layers=EXECUTION_LAYERS)

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': DEFAULT_FLOW,
    }


def _get_step_config(loader, step_name, core, process_cache=None):
    """Get (instance, topology, edge_type) for a step from LoadSimData.

    Returns None if the step can't be configured (e.g. missing config).
    """
    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    try:
        config = loader.get_config_by_name(base_name)
    except (KeyError, AttributeError):
        try:
            config = loader.get_config_by_name(step_name)
        except (KeyError, AttributeError):
            return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    return _instantiate_step(step_name, config, loader, core, process_cache)


def _instantiate_step(step_name, config, loader, core, process_cache=None):
    """Instantiate a v2ecoli process/step from its config."""
    if process_cache is None:
        process_cache = {}
    from v2ecoli.processes.equilibrium import EquilibriumStep
    from v2ecoli.processes.two_component_system import TwoComponentSystemStep
    from v2ecoli.processes.rna_maturation import RnaMaturationStep
    from v2ecoli.processes.tf_binding import TfBindingLogic, TfBindingStep
    from v2ecoli.processes.tf_unbinding import TfUnbindingLogic, TfUnbindingStep
    from v2ecoli.processes.transcript_initiation import TranscriptInitiationStep
    from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiationStep
    from v2ecoli.processes.chromosome_replication import ChromosomeReplicationStep
    from v2ecoli.processes.protein_degradation import ProteinDegradationStep
    from v2ecoli.processes.rna_degradation import RnaDegradationStep
    from v2ecoli.processes.complexation import ComplexationStep
    from v2ecoli.processes.transcript_elongation import TranscriptElongationStep
    from v2ecoli.processes.polypeptide_elongation import PolypeptideElongationStep
    from v2ecoli.processes.chromosome_structure import ChromosomeStructureLogic, ChromosomeStructureStep
    from v2ecoli.processes.metabolism import MetabolismLogic, MetabolismStep
    from v2ecoli.steps.listeners.mass_listener import MassListener, PostDivisionMassListener
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
    # Map step names to classes and their topologies
    # Partitioned processes have _requester/_evolver suffixes
    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Standalone steps (no requester/evolver split)
    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBindingStep,
        'ecoli-tf-unbinding': TfUnbindingStep,
        'ecoli-chromosome-structure': ChromosomeStructureStep,
        'ecoli-metabolism': MetabolismStep,
        'ecoli-complexation': ComplexationStep,
        'ecoli-protein-degradation': ProteinDegradationStep,
        'ecoli-rna-maturation': RnaMaturationStep,
        'ecoli-equilibrium': EquilibriumStep,
        'ecoli-two-component-system': TwoComponentSystemStep,
        'ecoli-rna-degradation': RnaDegradationStep,
        'ecoli-transcript-initiation': TranscriptInitiationStep,
        'ecoli-polypeptide-initiation': PolypeptideInitiationStep,
        'ecoli-chromosome-replication': ChromosomeReplicationStep,
        'ecoli-transcript-elongation': TranscriptElongationStep,
        'ecoli-polypeptide-elongation': PolypeptideElongationStep,
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

    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = step_cls(config=config, core=core)
        instance._raw_config = config
        topology = instance.topology
        return instance, topology, 'step'

    elif step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = cls(config=config, core=core)
        instance._raw_config = config
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    return None


def _get_special_step(loader, step_name, core):
    """Handle steps that aren't in LoadSimData's config map."""
    from v2ecoli.steps.unique_update import UniqueUpdate

    unique_names = list(loader.sim_data.internal_state.unique_molecule.unique_molecule_definitions.keys())
    unique_topo = {name: (name,) for name in unique_names}

    if step_name.startswith('unique_update'):
        # v1 uses plural names mapping to ('unique', singular_name)
        UNIQUE_PLURAL = {
            'full_chromosome': 'full_chromosomes',
            'chromosome_domain': 'chromosome_domains',
            'active_replisome': 'active_replisomes',
            'oriC': 'oriCs',
            'promoter': 'promoters',
            'chromosomal_segment': 'chromosomal_segments',
            'DnaA_box': 'DnaA_boxes',
            'active_RNAP': 'active_RNAPs',
            'RNA': 'RNAs',
            'gene': 'genes',
            'active_ribosome': 'active_ribosome',  # no plural change
        }
        unique_topo_v1 = {}
        unique_names_v1 = []
        for name in unique_names:
            plural = UNIQUE_PLURAL.get(name, name)
            unique_topo_v1[plural] = ('unique', name)
            unique_names_v1.append(plural)
        config = {'unique_names': unique_names_v1, 'unique_topo': unique_topo_v1}
        instance = UniqueUpdate(config=config, core=core)
        return instance, unique_topo_v1, 'step'

    if step_name == 'global_clock':
        from v2ecoli.steps.global_clock import GlobalClock
        instance = GlobalClock(config={}, core=core)
        topo = {
            'global_time': ('global_time',),
            'next_update_time': ('next_update_time',),
        }
        return instance, topo, 'process'  # NOTE: process, not step

    if step_name == 'emitter':
        from process_bigraph.emitter import RAMEmitter
        emit_schema = {
            'global_time': 'float',
            'bulk': 'array',
            'listeners': {'mass': {
                'cell_mass': 'float',
                'water_mass': 'float',
                'dry_mass': 'float',
                'protein_mass': 'float',
                'rna_mass': 'float',
                'rRna_mass': 'float',
                'tRna_mass': 'float',
                'mRna_mass': 'float',
                'dna_mass': 'float',
                'smallMolecule_mass': 'float',
                'instantaneous_growth_rate': 'float',
                'volume': 'float',
            }},
            # Unique molecules for chromosome state visualization
            'full_chromosome': 'any',
            'active_replisome': 'any',
            'active_RNAP': 'any',
            'chromosome_domain': 'any',
        }
        instance = RAMEmitter({'emit': emit_schema}, core)
        topo = {
            'global_time': ('global_time',),
            'bulk': ('bulk',),
            'listeners': ('listeners',),
            'full_chromosome': ('unique', 'full_chromosome'),
            'active_replisome': ('unique', 'active_replisome'),
            'active_RNAP': ('unique', 'active_RNAP'),
            'chromosome_domain': ('unique', 'chromosome_domain'),
        }
        return instance, topo, 'step'

    if step_name == 'replication_data_listener':
        from v2ecoli.steps.listeners.replication_data import ReplicationData
        config = {'time_step': 1}
        instance = ReplicationData(config=config, core=core)
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    if step_name == 'mark_d_period':
        from v2ecoli.steps.division import MarkDPeriod
        instance = MarkDPeriod(config={}, core=core)
        topo = {
            'full_chromosome': ('unique', 'full_chromosome'),
            'global_time': ('global_time',),
            'divide': ('divide',),
        }
        return instance, topo, 'step'

    if step_name == 'division':
        from v2ecoli.steps.division import Division
        try:
            div_config = loader.get_config_by_name('division')
        except Exception:
            div_config = {}
        div_config.setdefault('agent_id', '0')
        div_config.setdefault('division_threshold', 'mass_distribution')
        dry_mass_inc = getattr(getattr(loader, 'sim_data', None),
                               'expectedDryMassIncreaseDict', {})
        div_config.setdefault('dry_mass_inc_dict', dry_mass_inc)
        # Pass configs for building daughter cell states
        if hasattr(loader, '_configs'):
            div_config['configs'] = loader._configs
        div_config.setdefault('unique_names', getattr(loader, 'unique_names', []))
        instance = Division(config=div_config, core=core)
        topo = {
            'bulk': ('bulk',),
            'unique': ('unique',),
            'listeners': ('listeners',),
            'environment': ('environment',),
            'boundary': ('boundary',),
            'global_time': ('global_time',),
            'division_threshold': ('division_threshold',),
            'media_id': ('environment', 'media_id'),
            'agents': ('..',),
        }
        return instance, topo, 'step'

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_document(outpath='out/ecoli.pickle', sim_data_path=None, seed=0):
    """Build and save the E. coli simulation document.

    Runs ParCa if simData doesn't exist, then builds the document
    with all process instances and initial state.

    Args:
        outpath: Path for the output document pickle.
        sim_data_path: Path to simData. If None, runs ParCa.
        seed: Random seed.

    Returns:
        The path to the saved document.
    """
    document = build_document(sim_data_path=sim_data_path, seed=seed)

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'wb') as f:
        dill.dump(document, f)

    print(f"Saved document to {outpath}")
    return outpath
