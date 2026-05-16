"""Shared helpers for the v2ecoli composite generators.

These were previously defined in ``v2ecoli/generate.py`` and re-imported by
``generate_departitioned.py`` and ``generate_reconciled.py``.  Task 14 moves
them here so the legacy generate*.py files can be deleted.

Exported names (all are considered semi-private implementation details):
  - make_edge
  - inject_flow_dependencies
  - _seed_state_from_defaults
  - seed_mass_listener
  - _normalize_boundary_units
  - _make_instance
  - _get_special_step
  - _expand_flushes
  - FLUSH
  - PARTITIONED_PROCESSES
  - ALL_PARTITIONED
  - ALLOCATOR_LAYERS
"""

from __future__ import annotations

import copy

import numpy as np

# ---------------------------------------------------------------------------
# Process imports (needed for PARTITIONED_PROCESSES and _instantiate_step)
# ---------------------------------------------------------------------------
from v2ecoli.processes.rna_degradation import RnaDegradation
from v2ecoli.processes.transcript_elongation import TranscriptElongation
from v2ecoli.processes.polypeptide_elongation import PolypeptideElongation


# ---------------------------------------------------------------------------
# FLUSH sentinel and helper
# ---------------------------------------------------------------------------

# FLUSH marks a position where the UniqueNumpyUpdater buffer should be
# drained before the next layer runs.  Expanded to a real step by
# _expand_flushes() at build time.
FLUSH = '__unique_flush__'


def _expand_flushes(layers):
    """Replace each FLUSH sentinel with a real [unique_update_N] sub-layer.

    N is assigned in declaration order so state keys stay stable across
    architecture variants (baseline, departitioned, reconciled).
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
# Partitioned process registry
# ---------------------------------------------------------------------------

PARTITIONED_PROCESSES = {
    'ecoli-rna-degradation': RnaDegradation,
    'ecoli-transcript-elongation': TranscriptElongation,
    'ecoli-polypeptide-elongation': PolypeptideElongation,
}

ALL_PARTITIONED = list(PARTITIONED_PROCESSES.keys())


# ---------------------------------------------------------------------------
# Canonical visualization set for single-cell architectures.
# ---------------------------------------------------------------------------
# Shared by baseline / departitioned / reconciled — all three resolve to the
# same observables.mass / unique-molecule layout so the same viz tiles apply.
# Surfaced via ``@composite_generator(visualizations=...)`` so any Study
# built on one of these architectures inherits the v2ecoli simulation report
# panels without having to hand-author them in spec.yaml.
#
# Coverage:
#   - workflow: integrated WorkflowVisualization (chromosome state + replication
#     forks + ppGpp dynamics + mass fold change + division — the full legacy
#     simulation report)
#   - topology: NetworkVisualization of the process wiring
#
# Note: the cell-mass / cell-volume / growth-rate / absolute-mass-components /
# mass-fold-change TimeSeriesPlots that previously lived here used a
# `config.observable: '<short-name>'` convention that the dashboard's
# build_viz_composite (vivarium-dashboard, lib/investigations.py) does not yet
# understand — it only honors `inputs_map` for port→observable wiring, so
# those plots came out with empty y-data even though the emitter recorded
# them. They will land back here once the dashboard grows a short-name /
# leaf-name resolver for canonical viz wiring.
DEFAULT_SINGLE_CELL_VISUALIZATIONS = [
    {
        'name': 'workflow',
        'address': 'local:WorkflowVisualization',
        'config': {'title': 'v2ecoli — single-cell lifecycle'},
    },
    {
        'name': 'topology',
        'address': 'local:NetworkVisualization',
        'config': {'title': 'Process topology'},
    },
]

ALLOCATOR_LAYERS = {
    # RNA degradation shares water with polymerizations
    'allocator_2': ['ecoli-rna-degradation'],
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
    of default values for ports that need pre-population.
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
                _inject_nested_defaults(cell_state, wire_path, port_default)
            else:
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


def seed_mass_listener(cell_state, core):
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
    token.
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
# Step instantiation helpers
# ---------------------------------------------------------------------------

def _make_instance(cls, config, core):
    """Instantiate a Step/Process class, trying multiple signatures."""
    from v2ecoli.library.ecoli_step import set_current_core
    set_current_core(core)
    try:
        return cls(parameters=config)
    except TypeError:
        try:
            return cls(config=config, core=core)
        except TypeError:
            return cls(config)
    finally:
        set_current_core(None)


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
        div_config.setdefault('cache_dir', getattr(loader, 'cache_dir', 'out/cache'))
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
