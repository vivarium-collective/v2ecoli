"""
Self-contained simulation engine for v2ecoli.

No dependencies on vEcoli, genEcoli, vivarium, or wholecell at runtime.
Loads a pre-generated pickle document and runs the simulation using
in-place state mutation with v1-compatible updaters.
"""

import copy
import functools

import dill
import numpy as np

from bigraph_schema import get_path

from v2ecoli.library.schema import UniqueNumpyUpdater


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


def _is_unique_numpy_updater(updater):
    """Check if an updater is a UniqueNumpyUpdater bound method (duck-typed)."""
    if not hasattr(updater, '__self__'):
        return False
    obj = updater.__self__
    return (hasattr(obj, 'add_updates')
            and hasattr(obj, 'set_updates')
            and hasattr(obj, 'delete_updates'))


def _is_bulk_numpy_updater(updater):
    """Check if an updater is a bulk_numpy_updater function (duck-typed)."""
    name = getattr(updater, '__name__', '')
    wrapped = getattr(updater, '__wrapped__', None)
    wrapped_name = getattr(wrapped, '__name__', '') if wrapped else ''
    return name in ('bulk_numpy_updater', 'writeable_updater') or \
           wrapped_name == 'bulk_numpy_updater'


def _disable_readonly_updaters(cell_state):
    """Patch bulk_numpy_updater instances on process ports to not set read-only.

    Finds all callable updaters on process instances that look like
    bulk_numpy_updater and replaces them with a writeable version.
    """
    for name, edge in list(cell_state.items()):
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'ports_schema'):
            continue
        try:
            ports = instance.ports_schema()
        except Exception:
            continue
        for port_name, port in ports.items():
            if not isinstance(port, dict):
                continue
            updater = port.get('_updater')
            if callable(updater) and _is_bulk_numpy_updater(updater) \
                    and not getattr(updater, '_patched', False):
                @functools.wraps(updater)
                def writeable_updater(current, update, _orig=updater):
                    current.flags.writeable = True
                    for idx, value in update:
                        current["count"][idx] += value
                    return current
                writeable_updater._patched = True
                port['_updater'] = writeable_updater


def _deep_update(target, source):
    """Recursively update target dict with source dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ---------------------------------------------------------------------------
# View / update helpers
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
        elif port_name in ports and isinstance(ports[port_name], dict) \
                and '_default' in ports[port_name]:
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
    except (Exception, AttributeError):
        ports = {}

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
        elif callable(updater) or _is_bulk_numpy_updater(updater):
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
                        and _is_unique_numpy_updater(updater)):
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
        elif isinstance(update_value, list) and len(update_value) > 0:
            # Check if this is a bulk numpy update (list of (idx, val) tuples)
            current = target.get(key)
            if hasattr(current, 'dtype') and 'count' in getattr(current, 'dtype', {}).names or []:
                try:
                    current.flags.writeable = True
                except ValueError:
                    current = current.copy()
                    current.flags.writeable = True
                    target[key] = current
                for idx, value in update_value:
                    current["count"][idx] += value
            else:
                target[key] = update_value
        elif update_value is not None:
            if isinstance(update_value, list) and len(update_value) == 0:
                continue
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
            if updater is None:
                continue
            if not _is_unique_numpy_updater(updater):
                continue
            wire_path = output_wires.get(port_name)
            if not isinstance(wire_path, list):
                continue
            wire_key = tuple(wire_path)
            if wire_key not in shared:
                shared[wire_key] = UniqueNumpyUpdater()
    return shared


# ---------------------------------------------------------------------------
# State initialization
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


def _populate_all_defaults(cell_state):
    for name, edge in list(cell_state.items()):
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        _ensure_wired_paths(cell_state, edge)
        _populate_port_defaults(cell_state, edge, instance)


def _initialize_virtual_stores(cell_state):
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


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class EcoliSimulation:
    """Runs the E. coli simulation using in-place state mutation.

    Self-contained — requires no vEcoli imports at runtime.
    Loads from a pickle document produced by generate.py.
    """

    def __init__(self, state, flow_order):
        self.state = state
        self.flow_order = flow_order
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
            _disable_readonly_updaters(cell_state)
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

                    params = getattr(instance, 'parameters', {})
                    ts = params.get('timestep', timestep) if isinstance(params, dict) else timestep

                    if hasattr(instance, 'next_update'):
                        delta = instance.next_update(ts, view)
                    elif hasattr(instance, 'update'):
                        delta = instance.update(view)
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

def load_simulation(path='out/ecoli.pickle'):
    """Load a saved document and return an EcoliSimulation ready to run.

    Args:
        path: Path to the document produced by generate.py.

    Returns:
        An EcoliSimulation ready for .run(interval).
    """
    with open(path, 'rb') as f:
        document = dill.load(f)

    return EcoliSimulation(document['state'], document['flow_order'])
