"""
JSON cache for v2ecoli simulation states.

Provides save/load for two checkpoint levels:
1. sim_data cache — ParCa output (process configs + initial state parameters)
2. ecoli_wcm cache — fully wired simulation document ready for Composite

Uses bigraph-schema's serialize/realize for numpy array round-tripping.
"""

import os
import json

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Handle structured arrays specially
            if obj.dtype.names:
                # Serialize dtype preserving sub-array shapes
                dtype_list = []
                for name in obj.dtype.names:
                    field_dtype = obj.dtype[name]
                    if field_dtype.shape:
                        dtype_list.append((name, str(field_dtype.base), list(field_dtype.shape)))
                    else:
                        dtype_list.append((name, str(field_dtype)))
                return {
                    '__numpy_structured__': True,
                    'dtype': dtype_list,
                    'shape': list(obj.shape),
                    'data': [list(row) for row in obj.tolist()],
                }
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'shape': list(obj.shape),
                'data': obj.tolist(),
            }
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return {'__set__': True, 'data': sorted(list(obj))}
        if isinstance(obj, bytes):
            return {'__bytes__': True, 'data': obj.hex()}
        if hasattr(obj, 'asNumber') and hasattr(obj, '_unit'):
            # unum quantity
            val = obj.asNumber()
            if isinstance(val, np.ndarray):
                return {'__unum_array__': True, 'value': val.tolist(), 'unit': str(obj._unit)}
            return {'__unum__': True, 'value': float(val), 'unit': str(obj._unit)}
        if hasattr(obj, 'magnitude') and hasattr(obj, 'units'):
            # pint Quantity
            return {'__pint__': True, 'magnitude': float(obj.magnitude), 'units': str(obj.units)}
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'data': list(obj)}
        if isinstance(obj, type):
            return {'__type__': True, 'name': obj.__name__, 'module': obj.__module__}
        if callable(obj) and not isinstance(obj, type):
            name = getattr(obj, '__name__', str(obj))
            module = getattr(obj, '__module__', '')
            return {'__callable__': True, 'name': name, 'module': module}
        if hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
            return {'__object__': True, 'class': type(obj).__name__, 'data': str(obj)[:200]}
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def numpy_json_hook(obj):
    """JSON object hook that reconstructs numpy arrays and other types."""
    if isinstance(obj, dict):
        if obj.get('__numpy_structured__'):
            dtype = np.dtype([tuple(field) for field in obj['dtype']])
            return np.array([tuple(row) for row in obj['data']], dtype=dtype).reshape(obj['shape'])
        if obj.get('__numpy__'):
            return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        if obj.get('__set__'):
            return set(obj['data'])
        if obj.get('__bytes__'):
            return bytes.fromhex(obj['data'])
        if obj.get('__pint__'):
            from v2ecoli.library.units import units
            return obj['magnitude'] * getattr(units, obj['units'].split()[-1], 1)
        if obj.get('__unum__'):
            from v2ecoli.library.units import units
            return obj['value']  # Just return the number for now
    return obj


def _stringify_keys(obj):
    """Convert non-string dict keys to strings for JSON compatibility."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_stringify_keys(v) for v in obj]
    return obj


def save_json(data, path):
    """Save data to JSON with numpy support."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = _stringify_keys(data)
    with open(path, 'w') as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=1)
    print(f"Saved {path} ({os.path.getsize(path) // 1024}KB)")


def load_json(path):
    """Load data from JSON with numpy reconstruction."""
    with open(path) as f:
        return json.load(f, object_hook=numpy_json_hook)


def save_initial_state(initial_state, path='out/initial_state.json'):
    """Save the E. coli initial state (bulk, unique, environment, boundary) as JSON."""
    # Convert MetadataArray objects to regular arrays with metadata preserved
    state = {}
    for key, value in initial_state.items():
        if isinstance(value, np.ndarray):
            state[key] = value
        elif isinstance(value, dict):
            sub = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    entry = {'array': v}
                    if hasattr(v, 'metadata'):
                        entry['metadata'] = v.metadata
                    sub[k] = entry
                else:
                    sub[k] = v
            state[key] = sub
        else:
            state[key] = value
    save_json(state, path)


def load_initial_state(path='out/initial_state.json'):
    """Load E. coli initial state from JSON."""
    from v2ecoli.library.schema import MetadataArray
    state = load_json(path)

    # Reconstruct MetadataArray objects for unique molecules
    if 'unique' in state and isinstance(state['unique'], dict):
        unique = {}
        for name, entry in state['unique'].items():
            if isinstance(entry, dict) and 'array' in entry:
                arr = entry['array']
                metadata = entry.get('metadata')
                if isinstance(arr, np.ndarray) and 'unique_index' in arr.dtype.names:
                    unique[name] = MetadataArray(arr, metadata=metadata)
                else:
                    unique[name] = arr
            elif isinstance(entry, np.ndarray):
                unique[name] = entry
            else:
                unique[name] = entry
        state['unique'] = unique

    return state
