"""Function registry for serializable process configs.

Maps string keys to callable functions so that configs can store
function references as strings instead of bound methods. This enables
JSON serialization of process configs for .pbg files.

Usage:
    # Registration (at module level):
    @register("equilibrium.ode_solver")
    def equilibrium_ode_solver(stoich_matrix, rates_fwd, rates_rev):
        def solver(counts, volume, avogadro, random, jit=False):
            ...
        return solver

    # In config:
    config = {"solver": {"_function": "equilibrium.ode_solver", "_data": {...}}}

    # Resolution (at init time):
    from v2ecoli.library.config_resolver import resolve_config
    resolved = resolve_config(config)
    # resolved["solver"] is now a callable
"""

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {}


def register(name):
    """Decorator to register a function/factory in the registry."""
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_function(name):
    """Look up a registered function by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown function: {name!r}. "
                       f"Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_functions():
    """Return sorted list of all registered function names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Tier A: Pure stateless functions
# ---------------------------------------------------------------------------

@register("replication.make_elongation_rates")
def replication_make_elongation_rates(random_state, n_sequences,
                                      base_rate, time_step):
    """Replication fork elongation rates (deterministic for now)."""
    return np.full(n_sequences, base_rate * time_step)


@register("transcription.make_elongation_rates")
def transcription_make_elongation_rates(random_state, base_rate,
                                         time_step, variable_elongation):
    """RNAP elongation rates."""
    if variable_elongation:
        return np.array([max(1, int(random_state.normal(
            base_rate * time_step, base_rate * time_step * 0.1)))])
    return np.array([int(base_rate * time_step)])


@register("translation.make_elongation_rates")
def translation_make_elongation_rates(random_state, base_rate,
                                       time_step, variable_elongation):
    """Ribosome elongation rates."""
    if variable_elongation:
        return np.array([max(1, int(random_state.normal(
            base_rate * time_step, base_rate * time_step * 0.1)))])
    return np.array([int(base_rate * time_step)])
