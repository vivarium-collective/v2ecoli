"""
Document generation for v2ecoli (partitioned architecture).

Builds the E. coli simulation document using the partitioned architecture
(requester/allocator/evolver pattern). The document is a nested dict
suitable for ``Composite(document, core=core)``.

Pipeline: ParCa (vEcoli) -> simData -> LoadSimData -> process configs
-> initial state -> document dict -> Composite
"""

import os
import copy
import numpy as np
from bigraph_schema import allocate_core

from v2ecoli.types.quantity import ureg as units
from v2ecoli.types import ECOLI_TYPES

# Import partitioned process classes (vEcoli-style: single class per process)
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
from v2ecoli.processes.plasmid_replication import PlasmidReplication

# Generic Requester/Evolver wrappers from partition.py
from v2ecoli.steps.partition import Requester, Evolver, PartitionedProcess

# Additional processes and listeners used in _instantiate_step
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


# ---------------------------------------------------------------------------
# Execution layers (partitioned architecture)
# ---------------------------------------------------------------------------

# FLUSH marks a position where the UniqueNumpyUpdater buffer should be
# drained before the next layer runs (see UniqueUpdate docstring). The
# sentinel is expanded to a real step by _expand_flushes() at build time
# so the declarations below read as biology instead of plumbing.
FLUSH = '__unique_flush__'

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment (sequential sub-steps)
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: standalone (no partitioning needed)
    ['ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation'], FLUSH,

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: protein degradation (standalone — no resource competition)
    ['ecoli-protein-degradation'],

    # Layer 4b: standalone initiation/complexation/plasmid-replication
    # Chromosome replication moved into allocator_2 alongside rna-degradation
    # (matching vEcoli's flow — both depend on ecoli-tf-binding at the same
    # dependency depth, so they share an allocator layer there).
    ['ecoli-complexation', 'ecoli-plasmid-replication',
     'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation'],
    # Allocator_2: rna-degradation + chromosome-replication
    ['ecoli-rna-degradation_requester', 'ecoli-chromosome-replication_requester'],
    ['allocator_2'],
    ['ecoli-rna-degradation_evolver', 'ecoli-chromosome-replication_evolver'], FLUSH,

    # Layer 5: partition layer 3 -- elongation requesters (parallel)
    ['ecoli-polypeptide-elongation_requester', 'ecoli-transcript-elongation_requester'],
    ['allocator_3'],
    # Layer 5: partition layer 3 -- elongation evolvers (parallel)
    ['ecoli-polypeptide-elongation_evolver', 'ecoli-transcript-elongation_evolver'], FLUSH,

    # Layer 6: chromosome structure + metabolism (sequential)
    ['ecoli-chromosome-structure'], FLUSH,
    ['ecoli-metabolism'], FLUSH,

    # Layer 7: listeners (parallel)
    ['RNA_counts_listener', 'ecoli-mass-listener',
     'monomer_counts_listener', 'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener', 'unique_molecule_counts'], FLUSH,

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
    ['mark_d_period'], FLUSH,
    ['division'],
]


def _expand_flushes(layers):
    """Replace each FLUSH sentinel with a real [unique_update_N] sub-layer.

    N is assigned in declaration order so state keys stay stable across
    architecture variants (baseline, departitioned, reconciled). The
    resulting step names are handled by _get_special_step via the
    'unique_update' prefix.
    """
    out, n = [], 0
    for layer in layers:
        if layer == FLUSH:
            n += 1
            out.append([f'unique_update_{n}'])
        else:
            out.append(layer)
    return out


# ---------------------------------------------------------------------------
# Feature modules — optional steps added to the execution layers
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
        'insert_before': 'ecoli-transcript-elongation_requester',
        'steps': ['trna-attenuation-config'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']  # trna_attenuation disabled to match v1 default


def build_execution_layers(features=None):
    """Build EXECUTION_LAYERS from base + enabled feature modules."""
    layers = copy.deepcopy(BASE_EXECUTION_LAYERS)
    for feat_name in (features or []):
        feat = FEATURE_MODULES.get(feat_name)
        if feat is None:
            continue
        # Insert steps after a reference step
        if 'insert_after' in feat:
            ref = feat['insert_after']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in feat.get('steps', []):
                        layers.insert(i + 1, [step_name])
                    break
        # Insert steps before a reference step
        if 'insert_before' in feat:
            ref = feat['insert_before']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in reversed(feat.get('steps', [])):
                        layers.insert(i, [step_name])
                    break
        # Add listeners to the listener layer
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
# Partitioned process registry
# ---------------------------------------------------------------------------

PARTITIONED_PROCESSES = {
    'ecoli-rna-degradation': RnaDegradation,
    'ecoli-chromosome-replication': ChromosomeReplication,
    'ecoli-transcript-elongation': TranscriptElongation,
    'ecoli-polypeptide-elongation': PolypeptideElongation,
}

# Promoted to standalone (no resource competition):
# - ProteinDegradation, Equilibrium, TwoComponentSystem, Complexation,
#   RnaMaturation, TranscriptInitiation, PolypeptideInitiation
# Chromosome replication lives in allocator_2 alongside rna-degradation,
# matching vEcoli's flow (both depend on ecoli-tf-binding at the same depth).

ALL_PARTITIONED = list(PARTITIONED_PROCESSES.keys())

ALLOCATOR_LAYERS = {
    # RNA degradation shares water with polymerizations; chromosome
    # replication is in this layer per vEcoli's flow (same tf-binding
    # dependency depth).
    'allocator_2': ['ecoli-rna-degradation', 'ecoli-chromosome-replication'],
    # Elongation processes compete for NTPs / charged tRNAs
    'allocator_3': ['ecoli-polypeptide-elongation',
                    'ecoli-transcript-elongation'],
}


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _seed_state_from_defaults(cell_state):
    """Walk each edge's port_defaults and inject values into cell_state.

    Each step instance provides port_defaults() which returns a nested dict
    of default values for ports that need pre-population. This replaces the
    vivarium ports_schema-based seeding.
    """
    for edge in list(cell_state.values()):
        if not (isinstance(edge, dict) and 'instance' in edge):
            continue
        instance = edge['instance']
        try:
            defaults = instance.port_defaults()
        except (AttributeError, Exception):
            continue
        if not defaults:
            continue
        for port_name, wire_path in edge.get('inputs', {}).items():
            port_default = defaults.get(port_name)
            if port_default is None or not isinstance(wire_path, list):
                continue
            if isinstance(port_default, dict):
                # Nested defaults — wrap each leaf as {'_default': value}
                _inject_nested_defaults(cell_state, wire_path, port_default)
            else:
                # Scalar default
                _inject_port_default(cell_state, wire_path, {'_default': port_default})


def _inject_nested_defaults(state, wire_path, defaults_dict):
    """Recursively inject nested default values into state."""
    target = state
    for segment in wire_path:
        if not isinstance(target, dict):
            return
        target = target.setdefault(segment, {})
    if not isinstance(target, dict):
        return
    for key, val in defaults_dict.items():
        if isinstance(val, dict):
            sub = target.setdefault(key, {})
            if isinstance(sub, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, dict):
                        sub2 = sub.setdefault(k2, {})
                        if isinstance(sub2, dict):
                            for k3, v3 in v2.items():
                                sub2.setdefault(k3, v3)
                    else:
                        sub.setdefault(k2, v2)
        else:
            target.setdefault(key, val)


def _inject_port_default(state, wire_path, port_schema):
    """Inject _default values along wire_path into state."""
    if '_default' in port_schema:
        default = port_schema['_default']
        target = state
        for segment in wire_path[:-1]:
            if not isinstance(target, dict):
                return
            target = target.setdefault(segment, {})
        if isinstance(target, dict) and wire_path:
            key = wire_path[-1]
            current = target.get(key)
            if current is None or (
                    isinstance(current, (list, dict, tuple))
                    and len(current) == 0):
                target[key] = default
        return

    target = state
    for segment in wire_path:
        if not isinstance(target, dict):
            return
        target = target.setdefault(segment, {})
    if not isinstance(target, dict):
        return
    for key, subport in port_schema.items():
        if key.startswith('_') or key == '*' or not isinstance(subport, dict):
            continue
        _inject_port_default(target, [key], subport)


def _seed_mass_listener(cell_state, core):
    """Run mass listener once to populate initial mass values."""
    for name in ['post-division-mass-listener', 'ecoli-mass-listener']:
        edge = cell_state.get(name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'next_update'):
            continue

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
        break


def list_paths(path):
    """Convert tuple paths to list paths. Flatten _path dicts."""
    if isinstance(path, tuple):
        return list(path)
    elif isinstance(path, dict):
        if '_path' in path:
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
    """
    if layers is None:
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

    n_layers = len(layers)
    step_idx = 0
    total_steps = sum(len(layer) for layer in layers)

    for layer_idx, layer in enumerate(layers):
        for j, step_name in enumerate(layer):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                step_idx += 1
                continue

            edge['priority'] = float(total_steps - step_idx)

            if layer_idx == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']

            if layer_idx > 0:
                edge.setdefault('inputs', {})[f'_layer_in_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx - 1}']

            if layer_idx < n_layers - 1:
                edge.setdefault('outputs', {})[f'_layer_out_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx}']

            step_idx += 1


def make_edge(instance, topology, input_topology=None, output_topology=None,
              edge_type='step', config=None):
    """Create an edge dict for a process/step instance."""
    wires = list_paths(topology)
    in_wires = list_paths(input_topology) if input_topology is not None else wires
    out_wires = list_paths(output_topology) if output_topology is not None else wires
    state = {'priority': 1.0} if edge_type == 'step' else {'interval': 1.0}

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

    cls = type(instance)
    address = f'local:{cls.__module__}.{cls.__qualname__}'
    raw_config = config or getattr(instance, '_raw_config', {})

    state.update({
        '_type': edge_type,
        'address': address,
        'config': raw_config,
        '_inputs': inputs_schema,
        '_outputs': outputs_schema,
        'instance': instance,
        'inputs': copy.deepcopy(in_wires),
        'outputs': copy.deepcopy(out_wires),
    })
    return state


def _normalize_boundary_units(cell_state):
    """Re-create pint Quantities in boundary.external with the current registry."""
    boundary = cell_state.get('boundary', {})
    external = boundary.get('external', {})
    if not isinstance(external, dict):
        return
    from v2ecoli.library.unit_bridge import unum_to_pint
    for key, val in external.items():
        q = unum_to_pint(val)
        if hasattr(q, 'magnitude') and hasattr(q, 'units'):
            external[key] = float(q.magnitude)


# ---------------------------------------------------------------------------
# Step instantiation
# ---------------------------------------------------------------------------

def _make_instance(cls, config, core):
    """Instantiate a Step/Process class, trying multiple signatures."""
    from v2ecoli.library.ecoli_step import set_current_core
    set_current_core(core)
    try:
        # Try vEcoli-style (parameters=)
        return cls(parameters=config)
    except TypeError:
        try:
            # Try PBG-style (config=, core=)
            return cls(config=config, core=core)
        except TypeError:
            # Try positional
            return cls(config)
    finally:
        set_current_core(None)


def _instantiate_step(step_name, config, loader, core, process_cache=None):
    """Instantiate a partitioned process step from its config."""
    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-metabolism': Metabolism,
        'ecoli-protein-degradation': ProteinDegradation,
        'ecoli-equilibrium': Equilibrium,
        'ecoli-two-component-system': TwoComponentSystem,
        'ecoli-complexation': Complexation,
        'ecoli-rna-maturation': RnaMaturation,
        'ecoli-transcript-initiation': TranscriptInitiation,
        'ecoli-polypeptide-initiation': PolypeptideInitiation,
        'ecoli-plasmid-replication': PlasmidReplication,
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

    # Partitioned processes: wrap with generic Requester/Evolver
    if base_name in PARTITIONED_PROCESSES:
        proc_cls = PARTITIONED_PROCESSES[base_name]
        if base_name in process_cache:
            process = process_cache[base_name]
        else:
            from v2ecoli.library.ecoli_step import set_current_core
            set_current_core(core)
            process = proc_cls(config)
            set_current_core(None)
            process_cache[base_name] = process
        topology = dict(config.get('topology', {}) or {})
        if not topology:
            topology = getattr(process, 'topology',
                               getattr(proc_cls, 'topology', {}))
            if callable(topology):
                topology = topology()
            topology = dict(topology)

        if step_name.endswith('_requester'):
            instance = Requester({
                'time_step': config.get('time_step', 1),
                'process': process,
            })
            in_topo = dict(topology)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            in_topo['process'] = ('process', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {
                'next_update_time': ('next_update_time', base_name),
                'process': ('process', base_name),
            }
            if 'request' in out_ports:
                out_topo['request'] = ('request', base_name)
            if 'listeners' in out_ports:
                out_topo['listeners'] = topology.get('listeners', ('listeners',))
            return instance, topology, 'step', in_topo, out_topo

        elif step_name.endswith('_evolver'):
            instance = Evolver({
                'time_step': config.get('time_step', 1),
                'process': process,
            })
            in_topo = dict(topology)
            in_topo['allocate'] = ('allocate', base_name)
            in_topo['global_time'] = ('global_time',)
            in_topo.setdefault('timestep', ('timestep',))
            in_topo['next_update_time'] = ('next_update_time', base_name)
            in_topo['process'] = ('process', base_name)
            out_ports = set(instance.outputs().keys())
            out_topo = {
                'next_update_time': ('next_update_time', base_name),
                'process': ('process', base_name),
            }
            for port in out_ports:
                if port in ('next_update_time', 'process', 'allocate',
                            'global_time', 'timestep'):
                    continue
                if port in topology:
                    out_topo[port] = topology[port]
                elif port == 'listeners':
                    out_topo['listeners'] = ('listeners',)
            return instance, topology, 'step', in_topo, out_topo

    # Standalone steps (non-partitioned)
    # vEcoli classes take (parameters=) but PBG Step requires (config, core=)
    # Try the vEcoli signature first, fall back to PBG
    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = _make_instance(step_cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    elif step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = _make_instance(cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    return None


def _get_step_config(loader, step_name, core, process_cache=None):
    """Get (instance, topology, edge_type[, in_topo, out_topo]) for a step."""
    if process_cache is None:
        process_cache = {}

    base_name = step_name.replace('_requester', '').replace('_evolver', '')

    # Handle allocators (not in loader configs)
    if step_name.startswith('allocator'):
        from v2ecoli.steps.allocator import Allocator
        try:
            alloc_config = loader.get_config_by_name('allocator')
        except Exception:
            alloc_config = {}
        layer_procs = ALLOCATOR_LAYERS.get(step_name, ALL_PARTITIONED)
        if not alloc_config.get('process_names'):
            alloc_config['process_names'] = ALL_PARTITIONED
        alloc_config['layer_processes'] = layer_procs
        instance = _make_instance(Allocator, alloc_config, core)
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

    return _instantiate_step(
        step_name, config, loader, core, process_cache)


def _get_special_step(loader, step_name, core):
    """Handle steps that aren't in LoadSimData's config map."""
    from v2ecoli.steps.unique_update import UniqueUpdate

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule
        .unique_molecule_definitions.keys())

    if step_name.startswith('unique_update'):
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
            'active_ribosome': 'active_ribosome',
        }
        unique_topo_v1 = {}
        unique_names_v1 = []
        for name in unique_names:
            plural = UNIQUE_PLURAL.get(name, name)
            unique_topo_v1[plural] = ('unique', name)
            unique_names_v1.append(plural)
        config = {'unique_names': unique_names_v1, 'unique_topo': unique_topo_v1}
        instance = _make_instance(UniqueUpdate, config, core)
        return instance, unique_topo_v1, 'step'

    if step_name == 'global_clock':
        from v2ecoli.steps.global_clock import GlobalClock
        instance = GlobalClock(config={}, core=core)
        topo = {
            'global_time': ('global_time',),
            'next_update_time': ('next_update_time',),
        }
        return instance, topo, 'process'

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
            'full_chromosome': 'node',
            'active_replisome': 'node',
            'active_RNAP': 'node',
            'chromosome_domain': 'node',
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

    if step_name == 'ppgpp-initiation':
        from v2ecoli.steps.ppgpp_initiation import PpgppInitiation
        # Pull ppGpp-related config from transcript-initiation's config
        try:
            ti_config = loader.get_config_by_name('ecoli-transcript-initiation')
        except (KeyError, AttributeError):
            ti_config = {}
        ppgpp_config = {
            'ppgpp': ti_config.get('ppgpp', ''),
            'synth_prob': ti_config.get('synth_prob'),
            'copy_number': ti_config.get('copy_number', 1),
            'n_avogadro': ti_config.get('n_avogadro', 0),
            'cell_density': ti_config.get('cell_density', 0),
            'get_rnap_active_fraction_from_ppGpp': ti_config.get(
                'get_rnap_active_fraction_from_ppGpp'),
            'trna_attenuation': ti_config.get('trna_attenuation', False),
            'attenuated_rna_indices': ti_config.get('attenuated_rna_indices', []),
            'attenuation_adjustments': ti_config.get('attenuation_adjustments', []),
        }
        from v2ecoli.library.config_resolver import resolve_config
        ppgpp_config = resolve_config(ppgpp_config)
        instance = _make_instance(PpgppInitiation, ppgpp_config, core)
        topo = {
            'bulk': ('bulk',),
            'listeners': ('listeners',),
            'ppgpp_state': ('ppgpp_state',),
        }
        return instance, topo, 'step'

    if step_name == 'trna-attenuation-config':
        from v2ecoli.steps.trna_attenuation import TrnaAttenuationConfig
        # Pull attenuation config from transcript-elongation's config
        try:
            te_config = loader.get_config_by_name('ecoli-transcript-elongation')
        except (KeyError, AttributeError):
            te_config = {}
        att_config = {
            'get_attenuation_stop_probabilities': te_config.get(
                'get_attenuation_stop_probabilities'),
            'attenuated_rna_indices': te_config.get(
                'attenuated_rna_indices', []),
            'location_lookup': te_config.get('location_lookup', {}),
            'cell_density': te_config.get('cell_density', 0),
            'n_avogadro': te_config.get('n_avogadro', 0),
            'charged_trnas': te_config.get('charged_trnas', []),
        }
        instance = _make_instance(TrnaAttenuationConfig, att_config, core)
        topo = {
            'attenuation_config': ('attenuation_config',),
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
        from v2ecoli.library.unit_bridge import unum_to_pint
        dry_mass_inc = {k: unum_to_pint(v) for k, v in dry_mass_inc.items()}
        div_config.setdefault('dry_mass_inc_dict', dry_mass_inc)
        if hasattr(loader, '_configs'):
            div_config['configs'] = loader._configs
        div_config.setdefault('unique_names', getattr(loader, 'unique_names', []))
        instance = _make_instance(Division, div_config, core)
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
# Document builder
# ---------------------------------------------------------------------------

def build_document(initial_state, configs, unique_names,
                   dry_mass_inc_dict=None, core=None, seed=0,
                   features=None):
    """Build a partitioned-architecture document from pre-loaded configs.

    Args:
        initial_state: Dict with bulk, unique, environment, boundary.
        configs: Dict mapping step names to config dicts.
        unique_names: List of unique molecule names.
        dry_mass_inc_dict: Optional dict of expected dry mass increases.
        core: bigraph-schema core. If None, creates one.
        seed: Random seed.
        features: List of feature module names to enable (default: DEFAULT_FEATURES).

    Returns:
        Document dict for Composite().
    """
    if core is None:
        core = allocate_core()
        core.register_types(ECOLI_TYPES)

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

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

    # Pre-create feature module stores
    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'enabled': False,
    })

    # Initialize next_update_time for all partitioned processes
    # (requesters/evolvers check this to decide whether to run)
    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)  # Run on first tick

    # Pre-create shared request/allocate/process stores with per-process sub-keys
    cell_state.setdefault('request', {})
    cell_state.setdefault('allocate', {})
    proc_store = cell_state.setdefault('process', {})
    for proc_name in ALL_PARTITIONED:
        cell_state['request'].setdefault(proc_name, {'bulk': {}})
        cell_state['allocate'].setdefault(proc_name, {'bulk': {}})
        # process store starts empty — populated when requester runs
    # Pre-create listener sub-stores expected by various steps
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })
    # Listener sub-stores are populated by _seed_state_from_ports below

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
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

    # Build execution layers for the requested feature set
    if features is None:
        features = DEFAULT_FEATURES
    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    _process_cache = {}
    for step_name in flow_order:
        config = _get_step_config(
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

    # Place shared PartitionedProcess instances in the process store
    # (Requester/Evolver read them from state via the 'process' port)
    for proc_name, proc_instance in _process_cache.items():
        cell_state['process'][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    _seed_mass_listener(cell_state, core)

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
