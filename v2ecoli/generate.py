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

from v2ecoli.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from v2ecoli.reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
from v2ecoli.library.sim_data import LoadSimData
from v2ecoli.library.filepath import ROOT_PATH


# ---------------------------------------------------------------------------
# Flow ordering (mirrors the default vEcoli execution layers)
# ---------------------------------------------------------------------------

# The flow defines the execution order of all steps. Partitioned processes
# are split into requester → allocator → evolver layers. UniqueUpdate steps
# are inserted between layers to flush accumulated unique molecule updates.

DEFAULT_FLOW = [
    # Layer 0: post-division mass
    'post-division-mass-listener',
    'unique_update_1',

    # Layer 1: media/environment
    'media_update',
    'unique_update_2',
    'ecoli-tf-unbinding',
    'exchange_data',
    'unique_update_3',

    # Layer 2: partition layer 1 (requesters)
    'ecoli-equilibrium_requester',
    'ecoli-rna-maturation_requester',
    'ecoli-two-component-system_requester',
    'allocator_1',
    'ecoli-equilibrium_evolver',
    'ecoli-rna-maturation_evolver',
    'ecoli-two-component-system_evolver',
    'unique_update_4',

    # Layer 3: TF binding
    'ecoli-tf-binding',
    'unique_update_5',

    # Layer 4: partition layer 2 (requesters)
    'ecoli-chromosome-replication_requester',
    'ecoli-complexation_requester',
    'ecoli-polypeptide-initiation_requester',
    'ecoli-protein-degradation_requester',
    'ecoli-rna-degradation_requester',
    'ecoli-transcript-initiation_requester',
    'allocator_2',
    'ecoli-chromosome-replication_evolver',
    'ecoli-complexation_evolver',
    'ecoli-polypeptide-initiation_evolver',
    'ecoli-protein-degradation_evolver',
    'ecoli-rna-degradation_evolver',
    'ecoli-transcript-initiation_evolver',
    'unique_update_6',

    # Layer 5: partition layer 3 (elongation)
    'ecoli-polypeptide-elongation_requester',
    'ecoli-transcript-elongation_requester',
    'allocator_3',
    'ecoli-polypeptide-elongation_evolver',
    'ecoli-transcript-elongation_evolver',
    'unique_update_7',

    # Layer 6: chromosome structure + metabolism
    'ecoli-chromosome-structure',
    'unique_update_8',
    'ecoli-metabolism',
    'unique_update_9',

    # Layer 7: listeners
    'RNA_counts_listener',
    'dna_supercoiling_listener',
    'ecoli-mass-listener',
    'monomer_counts_listener',
    'replication_data_listener',
    'ribosome_data_listener',
    'rna_synth_prob_listener',
    'rnap_data_listener',
    'unique_molecule_counts',
    'unique_update_10',

    # Emitter: collect data after all listeners
    'emitter',
    # Clock process: drives time and triggers step cascades
    'global_clock',

    # Layer 8: division check
    'mark_d_period',
    'unique_update_11',
    'division',
]


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _seed_mass_listener(cell_state, core):
    """Run mass listener once to populate initial mass values."""
    import numpy as np
    from v2ecoli.steps.listeners.mass_listener import MassListener

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


def inject_flow_dependencies(cell_state, flow_order):
    """Add synthetic wiring and priorities to enforce execution order."""
    n = len(flow_order)
    for i, step_name in enumerate(flow_order):
        edge = cell_state.get(step_name)
        if not isinstance(edge, dict):
            continue
        # Set priority: earlier steps get higher priority
        edge['priority'] = float(n - i)
        if i == 0:
            edge.setdefault('inputs', {}).setdefault('global_time', ['global_time'])
        if i > 0:
            edge.setdefault('inputs', {})[f'_flow_in_{i}'] = [f'_flow_token_{i-1}']
        if i < len(flow_order) - 1:
            edge.setdefault('outputs', {})[f'_flow_out_{i}'] = [f'_flow_token_{i}']


def make_edge(instance, topology, edge_type='step'):
    """Create an edge dict for a process/step instance.

    Includes the instance directly and its input/output schemas.
    """
    wires = list_paths(topology)
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

    state.update({
        '_type': edge_type,
        'instance': instance,
        '_inputs': inputs_schema,
        '_outputs': outputs_schema,
        'inputs': copy.deepcopy(wires),
        'outputs': copy.deepcopy(wires),
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
    raw_data = KnowledgeBaseEcoli()

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
    from bigraph_schema import allocate_core
    from v2ecoli.types import ECOLI_TYPES

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
    for store in ['listeners', 'request', 'allocate', 'process',
                  'allocator_rng', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)

    # Pre-populate listeners.mass with defaults so mass listener can run
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault('mass', {'dry_mass': 0.0, 'cell_mass': 0.0})

    # Seed random state for allocator
    import numpy as np
    cell_state.setdefault('allocator_rng', np.random.RandomState(seed=seed))

    # Cache for shared PartitionedProcess instances (requester + evolver share one)
    _process_cache = {}

    # Add all process/step edges with their configs and topologies
    for step_name in DEFAULT_FLOW:
        config = _get_step_config(loader, step_name, core, _process_cache)
        if config is not None:
            instance, topology, edge_type = config
            cell_state[step_name] = make_edge(instance, topology, edge_type)

    # Seed mass listener after edges are created
    _seed_mass_listener(cell_state, core)

    # Add flow dependencies (synthetic wiring for execution order)
    inject_flow_dependencies(cell_state, DEFAULT_FLOW)

    # Wrap in agent container
    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
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
    from v2ecoli.processes.equilibrium import Equilibrium
    from v2ecoli.processes.two_component_system import TwoComponentSystem
    from v2ecoli.processes.rna_maturation import RnaMaturation
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.transcript_initiation import TranscriptInitiation
    from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
    from v2ecoli.processes.chromosome_replication import ChromosomeReplication
    from v2ecoli.processes.protein_degradation import ProteinDegradation
    from v2ecoli.processes.rna_degradation import RnaDegradation
    from v2ecoli.processes.complexation import Complexation
    from v2ecoli.processes.transcript_elongation import TranscriptElongation
    from v2ecoli.processes.polypeptide_elongation import PolypeptideElongation
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.processes.metabolism import Metabolism
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
    from v2ecoli.steps.partition import Requester, Evolver

    # Map step names to classes and their topologies
    # Partitioned processes have _requester/_evolver suffixes
    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    PARTITIONED = {
        'ecoli-equilibrium': Equilibrium,
        'ecoli-two-component-system': TwoComponentSystem,
        'ecoli-rna-maturation': RnaMaturation,
        'ecoli-tf-binding': TfBinding,
        'ecoli-transcript-initiation': TranscriptInitiation,
        'ecoli-polypeptide-initiation': PolypeptideInitiation,
        'ecoli-chromosome-replication': ChromosomeReplication,
        'ecoli-protein-degradation': ProteinDegradation,
        'ecoli-rna-degradation': RnaDegradation,
        'ecoli-complexation': Complexation,
        'ecoli-transcript-elongation': TranscriptElongation,
        'ecoli-polypeptide-elongation': PolypeptideElongation,
    }

    # PartitionedProcesses that run as single steps (no requester/evolver split)
    STANDALONE_PARTITIONED = {
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-metabolism': Metabolism,
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

    if base_name in PARTITIONED:
        # Share the same PartitionedProcess between requester and evolver
        if base_name in process_cache:
            process = process_cache[base_name]
        else:
            proc_cls = PARTITIONED[base_name]
            process = proc_cls(parameters=config, core=core)
            process_cache[base_name] = process
        topology = process.topology

        if step_name.endswith('_requester'):
            req_config = {'process': process, 'name': step_name}
            instance = Requester(config=req_config, core=core)
            # Requester topology extends process topology
            req_topo = dict(topology)
            req_topo['request'] = ('request', base_name)
            req_topo['process'] = ('process', base_name)
            req_topo['global_time'] = ('global_time',)
            req_topo['timestep'] = ('timestep',)
            req_topo['next_update_time'] = ('next_update_time', base_name)
            return instance, req_topo, 'step'

        elif step_name.endswith('_evolver'):
            ev_config = {'process': process, 'name': step_name}
            instance = Evolver(config=ev_config, core=core)
            ev_topo = dict(topology)
            ev_topo['allocate'] = ('allocate', base_name)
            ev_topo['process'] = ('process', base_name)
            ev_topo['global_time'] = ('global_time',)
            ev_topo['timestep'] = ('timestep',)
            ev_topo['next_update_time'] = ('next_update_time', base_name)
            return instance, ev_topo, 'step'

        else:
            # Standalone use of a PartitionedProcess (e.g. tf-binding)
            process = proc_cls(parameters=config, core=core)
            return process, topology, 'step'

    elif step_name in STANDALONE_PARTITIONED:
        cls = STANDALONE_PARTITIONED[step_name]
        instance = cls(parameters=config, core=core)
        topology = instance.topology
        return instance, topology, 'step'

    elif step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = cls(config=config, core=core)
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    return None


def _get_special_step(loader, step_name, core):
    """Handle steps that aren't in LoadSimData's config map."""
    from v2ecoli.steps.unique_update import UniqueUpdate
    from v2ecoli.steps.allocator import Allocator

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

    if step_name.startswith('allocator'):
        try:
            config = loader.get_config_by_name('allocator')
        except Exception:
            config = {}
        all_partitioned = [
            'ecoli-chromosome-replication', 'ecoli-complexation',
            'ecoli-equilibrium', 'ecoli-polypeptide-elongation',
            'ecoli-polypeptide-initiation', 'ecoli-protein-degradation',
            'ecoli-rna-degradation', 'ecoli-rna-maturation',
            'ecoli-transcript-elongation', 'ecoli-transcript-initiation',
            'ecoli-two-component-system',
        ]
        if not config.get('process_names'):
            config['process_names'] = all_partitioned
        if config:
            instance = Allocator(config=config, core=core)
            topo = instance.topology
            return instance, topo, 'step'

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
            }},
        }
        instance = RAMEmitter({'emit': emit_schema}, core)
        topo = {
            'global_time': ('global_time',),
            'listeners': ('listeners',),
        }
        return instance, topo, 'step'

    if step_name == 'replication_data_listener':
        from v2ecoli.steps.listeners.replication_data import ReplicationData
        config = {'time_step': 1}
        instance = ReplicationData(config=config, core=core)
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    # mark_d_period, division — optional, skip for now
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
