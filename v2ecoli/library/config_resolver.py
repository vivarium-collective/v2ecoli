"""Config resolution utilities for serializable process configs.

Converts between runtime configs (with callables) and serialized configs
(with function references). Enables JSON round-tripping of process configs.
"""

from v2ecoli.library.function_registry import get_function


FUNCTION_MARKER = '_function'
DATA_MARKER = '_data'


def resolve_config(config):
    """Walk a config dict and resolve all function references.

    Replaces dicts like {"_function": "name", "_data": {...}} with
    the result of calling the registered factory with the data.

    Args:
        config: Config dict, possibly containing function references.

    Returns:
        New dict with function references resolved to callables.
    """
    if not isinstance(config, dict):
        return config

    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict) and FUNCTION_MARKER in value:
            func_name = value[FUNCTION_MARKER]
            data = value.get(DATA_MARKER, {})
            factory = get_function(func_name)
            if data:
                resolved[key] = factory(**data)
            else:
                resolved[key] = factory
        elif isinstance(value, dict):
            resolved[key] = resolve_config(value)
        else:
            resolved[key] = value
    return resolved


def make_function_ref(name, **data):
    """Create a serializable function reference dict.

    Args:
        name: Registered function name (e.g. "equilibrium.ode_solver").
        **data: Keyword arguments to pass to the factory.

    Returns:
        Dict like {"_function": "name", "_data": {...}}.
    """
    ref = {FUNCTION_MARKER: name}
    if data:
        ref[DATA_MARKER] = data
    return ref


def is_function_ref(value):
    """Check if a value is a function reference dict."""
    return isinstance(value, dict) and FUNCTION_MARKER in value
