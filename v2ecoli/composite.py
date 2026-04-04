"""
Document generation and simulation engine for v2ecoli.

Uses EcoliSim from vEcoli to generate initial state and process configs,
then runs the simulation using v2ecoli processes with in-place state mutation.
"""

import os
import copy
import functools
import time

import dill
import numpy as np

from contextlib import chdir

from bigraph_schema import deep_merge, get_path

from wholecell.utils.filepath import ROOT_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

import ecoli.library.schema as ecoli_schema_mod
from ecoli.library.schema import UniqueNumpyUpdater, bulk_numpy_updater


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _make_arrays_writeable(state):
    """Recursively make all numpy arrays in the state writeable."""
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                if not value.flags.writeable:
                    state[key] = value.copy()
                    state[key].flags.writeable = True
            elif hasattr(value, 'struct_array'):
                arr = value.struct_array
                if isinstance(arr, np.ndarray) and not arr.flags.writeable:
                    value.struct_array = arr.copy()
                    value.struct_array.flags.writeable = True
            elif isinstance(value, dict):
                _make_arrays_writeable(value)


def _disable_readonly_arrays():
    """Monkey-patch bulk_numpy_updater to not set arrays read-only."""
    if hasattr(ecoli_schema_mod.bulk_numpy_updater, '_patched'):
        return

    @functools.wraps(bulk_numpy_updater)
    def writeable_updater(current, update):
        current.flags.writeable = True
        for idx, value in update:
            current["count"][idx] += value
        return current

    writeable_updater._patched = True
    ecoli_schema_mod.bulk_numpy_updater = writeable_updater


def _deep_update(target, source):
    """Recursively update target dict with source dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ---------------------------------------------------------------------------
# View / update helpers (for in-place execution)
# ---------------------------------------------------------------------------

def _resolve_wire(cell_state, wire_path):
    """Resolve a wire path to a value in cell_state."""
    if isinstance(wire_path, list) and wire_path:
        current = cell_state
        for segment in wire_path:
            if isinstance(current, dict):
                current = current.get(segment)
            else:
                return None
        return current
    elif isinstance(wire_path, dict):
        base_path = wire_path.get('_path')
        if base_path:
            result = _resolve_wire(cell_state, base_path)
            if result is not None and isinstance(result, dict):
                result = copy.copy(result)
            else:
                result = {}
        else:
            result = {}
        for sub_key, sub_path in wire_path.items():
            if sub_key == '_path':
                continue
            sub_val = _resolve_wire(cell_state, sub_path)
            if sub_val is not None:
                result[sub_key] = sub_val
        return result
    return None


def _build_view(cell_state, edge, instance):
    """Build a state view for a step by following its input wires."""
    try:
        ports = instance.ports_schema()
    except AttributeError:
        ports = {}
    view = {}
    wires = edge.get('inputs', {})
    for port_name, wire_path in wires.items():
        resolved = _resolve_wire(cell_state, wire_path)
        if resolved is not None:
            view[port_name] = resolved
        elif port_name in ports and isinstance(ports[port_name], dict) and '_default' in ports[port_name]:
            view[port_name] = ports[port_name]['_default']
    return view


def _apply_nested_wire_update(cell_state, wire_dict, update_value):
    """Apply an update through a nested wire dict."""
    if not isinstance(update_value, dict):
        base_path = wire_dict.get('_path')
        if base_path and isinstance(base_path, list):
            _set_at_path(cell_state, base_path, update_value)
        return
    base_path = wire_dict.get('_path')
    for sub_key, sub_value in update_value.items():
        sub_wire = wire_dict.get(sub_key)
        if sub_wire is not None:
            if isinstance(sub_wire, list):
                _set_at_path(cell_state, sub_wire, sub_value)
            elif isinstance(sub_wire, dict):
                _apply_nested_wire_update(cell_state, sub_wire, sub_value)
        elif base_path and isinstance(base_path, list):
            _set_at_path(cell_state, base_path + [sub_key], sub_value)


def _set_at_path(state, path, value):
    """Set a value at a path in a nested dict."""
    target = state
    for segment in path[:-1]:
        if isinstance(target, dict):
            if segment not in target:
                target[segment] = {}
            target = target[segment]
        else:
            return
    if isinstance(target, dict) and path:
        key = path[-1]
        current = target.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            _deep_update(current, value)
        else:
            target[key] = value


def apply_step_update(cell_state, edge, instance, delta, unique_updaters=None):
    """Apply a step's delta update to cell_state by following output wire paths."""
    try:
        ports = instance.ports_schema()
    except Exception:
        return

    output_wires = edge.get('outputs', {})

    for port_name, update_value in delta.items():
        if port_name.startswith('_flow_'):
            continue

        wire_path = output_wires.get(port_name)
        if wire_path is None:
            continue

        if isinstance(wire_path, dict):
            _apply_nested_wire_update(cell_state, wire_path, update_value)
            continue

        if not isinstance(wire_path, list) or not wire_path:
            continue

        port = ports.get(port_name, {})
        updater = port.get('_updater') if isinstance(port, dict) else None

        target = cell_state
        for segment in wire_path[:-1]:
            if isinstance(target, dict):
                if segment not in target:
                    target[segment] = {}
                target = target[segment]
            else:
                target = None
                break

        if not isinstance(target, dict):
            continue

        key = wire_path[-1]

        if updater == 'set' or port_name in ('next_update_time', 'process'):
            target[key] = update_value
        elif callable(updater):
            current = target.get(key)
            if current is not None:
                if isinstance(current, np.ndarray):
                    try:
                        current.flags.writeable = True
                    except ValueError:
                        current = current.copy()
                        current.flags.writeable = True
                        target[key] = current
                wire_key = tuple(wire_path) if isinstance(wire_path, list) else None
                if (unique_updaters and wire_key and wire_key in unique_updaters
                        and hasattr(updater, '__self__')
                        and isinstance(updater.__self__, UniqueNumpyUpdater)):
                    shared_updater = unique_updaters[wire_key]
                    result = shared_updater.updater(current, update_value)
                    if result is not current:
                        target[key] = result
                else:
                    updater(current, update_value)
        elif isinstance(update_value, dict):
            current = target.get(key)
            if isinstance(current, dict):
                _deep_update(current, update_value)
            else:
                target[key] = update_value
        elif isinstance(update_value, (int, float)):
            current = target.get(key)
            if isinstance(current, (int, float)):
                target[key] = current + update_value
            else:
                target[key] = update_value
        elif update_value is not None:
            target[key] = update_value


def fill_missing_state(state, process):
    """Fill in missing state keys with defaults from ports_schema."""
    try:
        ports = process.ports_schema()
    except Exception:
        return state
    _fill_defaults_recursive(state, ports)
    return state


def _fill_defaults_recursive(state, schema):
    if not isinstance(schema, dict) or not isinstance(state, dict):
        return
    for key, port in schema.items():
        if key.startswith('_'):
            continue
        if not isinstance(port, dict):
            continue
        if key not in state:
            if '_default' in port:
                state[key] = port['_default']
        elif isinstance(state[key], dict):
            _fill_defaults_recursive(state[key], port)


# ---------------------------------------------------------------------------
# Shared unique updater registry
# ---------------------------------------------------------------------------

def _share_unique_updaters(cell_state, step_order):
    shared = {}
    for step_name in step_order:
        edge = cell_state.get(step_name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'ports_schema'):
            continue
        try:
            ports = instance.ports_schema()
        except Exception:
            continue
        output_wires = edge.get('outputs', {})
        for port_name, port in ports.items():
            if not isinstance(port, dict):
                continue
            updater = port.get('_updater')
            if updater is None or not hasattr(updater, '__self__'):
                continue
            if not isinstance(updater.__self__, UniqueNumpyUpdater):
                continue
            wire_path = output_wires.get(port_name)
            if not isinstance(wire_path, list):
                continue
            wire_key = tuple(wire_path)
            if wire_key not in shared:
                shared[wire_key] = UniqueNumpyUpdater()
    return shared


# ---------------------------------------------------------------------------
# Seed listeners
# ---------------------------------------------------------------------------

LISTENERS_TO_SEED = ['post-division-mass-listener', 'ecoli-mass-listener']
SCALAR_STATE_KEYS = {'global_time', 'timestep', 'next_update_time'}


def _ensure_wired_paths(cell_state, edge):
    """Ensure output paths that will receive dict updates exist as empty dicts."""
    wires = edge.get('outputs', {})
    for port_name, wire_path in wires.items():
        if isinstance(wire_path, list) and len(wire_path) == 1:
            key = wire_path[0]
            if key in SCALAR_STATE_KEYS:
                continue
            if key not in cell_state or cell_state[key] is None:
                cell_state[key] = {}


def _populate_port_defaults(cell_state, edge, instance):
    """Populate port defaults into the state along wired paths."""
    try:
        ports = instance.ports_schema()
    except Exception:
        return
    wires = edge.get('inputs', {})
    for port_name, wire_path in wires.items():
        if not isinstance(wire_path, list) or not wire_path:
            continue
        port = ports.get(port_name)
        if not isinstance(port, dict):
            continue
        if '_default' in port:
            target = cell_state
            for segment in wire_path[:-1]:
                if isinstance(target, dict):
                    if segment not in target or target[segment] is None:
                        target[segment] = {}
                    target = target[segment]
                else:
                    break
            if isinstance(target, dict):
                last = wire_path[-1]
                if last not in target or target[last] is None:
                    target[last] = port['_default']
        else:
            _inject_nested_defaults(cell_state, wire_path, port)


def _inject_nested_defaults(state, wire_path, port_schema):
    """Inject nested port defaults into state at the given wire path."""
    target = state
    for segment in wire_path:
        if isinstance(target, dict):
            if segment not in target or target[segment] is None:
                target[segment] = {}
            target = target[segment]
        else:
            return
    if not isinstance(target, dict):
        return
    for key, value in port_schema.items():
        if key.startswith('_'):
            continue
        if isinstance(value, dict):
            if '_default' in value and key not in target:
                target[key] = value['_default']
            elif key not in target:
                target[key] = {}
                _inject_nested_defaults(target, [key], value)


def _apply_dict_updates(cell_state, output_wires, update):
    """Apply only dict-valued updates from a step back into the state."""
    if not update:
        return
    for port_name, value in update.items():
        if not isinstance(value, dict):
            continue
        wire_path = output_wires.get(port_name)
        if not isinstance(wire_path, list) or not wire_path:
            continue
        target = cell_state
        for segment in wire_path[:-1]:
            if isinstance(target, dict):
                if segment not in target:
                    target[segment] = {}
                target = target[segment]
            else:
                break
        if isinstance(target, dict):
            last = wire_path[-1]
            if last not in target:
                target[last] = {}
            if isinstance(target[last], dict):
                target[last].update(value)


def _seed_listeners(cell_state):
    for step_name in LISTENERS_TO_SEED:
        edge = cell_state.get(step_name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'next_update'):
            continue
        _ensure_wired_paths(cell_state, edge)
        _populate_port_defaults(cell_state, edge, instance)
        try:
            view = _build_view(cell_state, edge, instance)
            timestep = instance.parameters.get('timestep', 1.0)
            update = instance.next_update(timestep, view)
            _apply_dict_updates(cell_state, edge.get('outputs', {}), update)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Flow ordering
# ---------------------------------------------------------------------------

def extract_flow_priorities(flow):
    order = list(flow.keys())
    n = len(order)
    return {step_name: float(n - i) for i, step_name in enumerate(order)}


def inject_flow_dependencies(cell_state, flow):
    order = list(flow.keys())
    for i, step_name in enumerate(order):
        edge = cell_state.get(step_name)
        if not isinstance(edge, dict):
            continue
        if i == 0:
            edge.setdefault('inputs', {}).setdefault('global_time', ['global_time'])
        if i > 0:
            edge.setdefault('inputs', {})[f'_flow_in_{i}'] = [f'_flow_token_{i-1}']
        if i < len(order) - 1:
            edge.setdefault('outputs', {})[f'_flow_out_{i}'] = [f'_flow_token_{i}']


def list_paths(path):
    if isinstance(path, tuple):
        return list(path)
    elif isinstance(path, dict):
        return {key: list_paths(subpath) for key, subpath in path.items()}


# ---------------------------------------------------------------------------
# Migration from EcoliSim
# ---------------------------------------------------------------------------

def translate_processes(tree, topology=None, edge_type=None):
    """Translate v1 process/step instances into edge dicts."""
    from vivarium.core.process import Process as VivariumProcess, Step as VivariumStep
    from bigraph_schema import Edge as BigraphEdge

    if isinstance(tree, (VivariumProcess, VivariumStep, BigraphEdge)):
        # Prefer the instance's own topology over the composite-level one
        instance_topology = getattr(tree, 'topology', None)
        if instance_topology:
            topology = instance_topology
        elif topology is None:
            topology = {}
        wires = list_paths(topology)

        if edge_type == 'process':
            state = {'interval': 1.0}
        else:
            state = {'priority': 1.0}

        state.update({
            '_type': 'step' if edge_type != 'process' else 'process',
            'instance': tree,
            'inputs': copy.deepcopy(wires),
            'outputs': copy.deepcopy(wires),
        })
        return state
    elif isinstance(tree, dict):
        return {key: translate_processes(subtree,
                    topology[key] if topology else None,
                    edge_type=edge_type)
                for key, subtree in tree.items()}
    else:
        return tree


def migrate_composite(sim):
    """Build the composite state from EcoliSim."""
    processes = translate_processes(sim.ecoli.processes, sim.ecoli.topology, edge_type='process')
    steps = translate_processes(sim.ecoli.steps, sim.ecoli.topology, edge_type='step')

    state = deep_merge(processes, steps)
    state = deep_merge(state, sim.generated_initial_state)

    flow = sim.ecoli.flow
    for path_key, substates in state.items():
        if isinstance(substates, dict):
            subflow = flow.get(path_key, {})
            for subkey, subsubstates in substates.items():
                if isinstance(subsubstates, dict):
                    inner_flow = subflow.get(subkey, {})
                    if inner_flow:
                        priorities = extract_flow_priorities(inner_flow)
                        for step_name, priority in priorities.items():
                            if isinstance(subsubstates.get(step_name), dict):
                                subsubstates[step_name]['priority'] = priority
                        inject_flow_dependencies(subsubstates, inner_flow)
    return state


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _populate_all_defaults(cell_state):
    """Run _populate_port_defaults for every step to ensure nested defaults exist."""
    for name, edge in list(cell_state.items()):
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        _ensure_wired_paths(cell_state, edge)
        _populate_port_defaults(cell_state, edge, instance)


def _initialize_virtual_stores(cell_state):
    """Pre-create stores that steps will write to but don't exist initially.

    Scans all step edges for input/output wires and ensures the target
    paths exist as empty dicts in cell_state.
    """
    for name, edge in list(cell_state.items()):
        if not isinstance(edge, dict):
            continue
        for ports_key in ('inputs', 'outputs'):
            wires = edge.get(ports_key, {})
            for port_name, wire_path in wires.items():
                if port_name.startswith('_flow'):
                    continue
                if isinstance(wire_path, list) and len(wire_path) >= 1:
                    key = wire_path[0]
                    if key in SCALAR_STATE_KEYS:
                        continue
                    if key not in cell_state or cell_state[key] is None:
                        cell_state[key] = {}
                elif isinstance(wire_path, dict):
                    base = wire_path.get('_path')
                    if isinstance(base, list) and base:
                        key = base[0]
                        if key not in cell_state or cell_state[key] is None:
                            cell_state[key] = {}


class EcoliSimulation:
    """Runs the E. coli simulation using in-place state mutation.

    Steps execute in flow order, applying updates directly to the shared
    cell state via v1-compatible updaters.
    """

    def __init__(self, state, flow_order):
        self.state = state
        self.flow_order = flow_order  # list of step names in execution order
        self._cell_path = None
        self._unique_updaters = None

        # Find cell state path
        for path_key, substates in state.items():
            if isinstance(substates, dict) and path_key != 'global_time':
                for subkey in substates:
                    if isinstance(substates[subkey], dict) and len(substates[subkey]) > 10:
                        self._cell_path = (path_key, subkey)
                        break
            if self._cell_path:
                break

        if self._cell_path:
            cell_state = get_path(state, self._cell_path)
            _make_arrays_writeable(cell_state)
            _disable_readonly_arrays()
            _initialize_virtual_stores(cell_state)
            _populate_all_defaults(cell_state)
            _seed_listeners(cell_state)

            step_names = [k for k, v in cell_state.items()
                          if isinstance(v, dict) and 'instance' in v]
            self._unique_updaters = _share_unique_updaters(cell_state, step_names)

        if 'global_time' not in state:
            state['global_time'] = 0.0

    def run(self, duration):
        """Run the simulation for the given duration."""
        cell_state = get_path(self.state, self._cell_path)
        end_time = self.state['global_time'] + duration
        timestep = 1.0

        while self.state['global_time'] < end_time:
            _make_arrays_writeable(cell_state)

            for step_name in self.flow_order:
                edge = cell_state.get(step_name)
                if not isinstance(edge, dict) or 'instance' not in edge:
                    continue

                instance = edge['instance']
                try:
                    view = _build_view(cell_state, edge, instance)
                    view = fill_missing_state(view, instance)

                    if hasattr(instance, 'next_update'):
                        ts = instance.parameters.get('timestep', timestep)
                        delta = instance.next_update(ts, view)
                    else:
                        delta = {}

                    if delta:
                        apply_step_update(cell_state, edge, instance, delta,
                                          unique_updaters=self._unique_updaters)
                except Exception:
                    pass

            self.state['global_time'] += timestep
            cell_state['global_time'] = self.state['global_time']


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_document(outpath='out/ecoli.pickle'):
    """Build the E. coli composite from EcoliSim and save as a document.

    Args:
        outpath: Path for the output file.

    Returns:
        The path to the saved file.
    """
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file(CONFIG_DIR_PATH + "default.json")
        sim.build_ecoli()

    state = migrate_composite(sim)

    # Extract flow order
    flow = sim.ecoli.flow
    flow_order = []
    for path_key in state:
        if isinstance(state[path_key], dict):
            subflow = flow.get(path_key, {})
            for subkey in state[path_key]:
                inner_flow = subflow.get(subkey, {})
                if inner_flow:
                    flow_order = list(inner_flow.keys())
                    break

    document = {'state': state, 'flow_order': flow_order}

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'wb') as f:
        dill.dump(document, f)

    print(f"Saved document to {outpath}")
    return outpath


def load_simulation(path='out/ecoli.pickle'):
    """Load a saved document and return an EcoliSimulation ready to run.

    Args:
        path: Path to the document produced by generate_document.

    Returns:
        An EcoliSimulation ready for .run(interval).
    """
    with open(path, 'rb') as f:
        document = dill.load(f)

    return EcoliSimulation(document['state'], document['flow_order'])
