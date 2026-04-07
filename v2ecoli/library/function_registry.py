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
from v2ecoli.library.random import stochasticRound

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
# Underlying elongation rate function (from wholecell.utils.random)
# ---------------------------------------------------------------------------

def _make_elongation_rates_core(random, size, base, amplified_indexes,
                                ceiling, time_step, variable_elongation=False):
    """Create elongation rate array with optional amplification.

    Standalone version of wholecell.utils.random.make_elongation_rates.

    Args:
        random: numpy RandomState
        size: number of elements
        base: base elongation rate (per second)
        amplified_indexes: indexes to set to ceiling rate
        ceiling: max rate for amplified indexes
        time_step: simulation timestep
        variable_elongation: if True, add gaussian noise
    """
    rates = np.full(size, base * time_step)
    if variable_elongation:
        rates = stochasticRound(random, rates).astype(np.int64)
    else:
        rates = rates.astype(np.int64)

    if len(amplified_indexes) > 0:
        rates[amplified_indexes] = int(ceiling * time_step)

    return rates


# ---------------------------------------------------------------------------
# Tier A: Pure stateless functions
# ---------------------------------------------------------------------------

@register("replication.make_elongation_rates")
def replication_make_elongation_rates(random, replisomes, base, time_step):
    """Replication fork elongation rates.

    Pure function — no closure data needed.
    """
    return np.full(replisomes, stochasticRound(random, base * time_step),
                   dtype=np.int64)


# ---------------------------------------------------------------------------
# Tier A+: Stateless-ish functions (need closure data extracted at config time)
# ---------------------------------------------------------------------------

@register("transcription.make_elongation_rates")
def transcription_make_elongation_rates_factory(n_TUs, rRNA_indexes,
                                                 stable_RNA_elongation_rate):
    """Factory: creates transcription elongation rate function.

    Closure data (extracted from sim_data.process.transcription):
        n_TUs: number of transcription units
        rRNA_indexes: numpy array of rRNA TU indexes
        stable_RNA_elongation_rate: max rate for stable RNAs
    """
    rRNA_indexes = np.asarray(rRNA_indexes)

    def make_elongation_rates(random, base, time_step, variable_elongation=False):
        return _make_elongation_rates_core(
            random, n_TUs, base, rRNA_indexes,
            stable_RNA_elongation_rate, time_step, variable_elongation)

    return make_elongation_rates


@register("translation.make_elongation_rates")
def translation_make_elongation_rates_factory(n_monomers,
                                               ribosomal_protein_indexes,
                                               max_elongation_rate):
    """Factory: creates translation elongation rate function.

    Closure data (extracted from sim_data.process.translation):
        n_monomers: number of monomers
        ribosomal_protein_indexes: numpy array of ribosomal protein indexes
        max_elongation_rate: max rate for ribosomal proteins
    """
    ribosomal_protein_indexes = np.asarray(ribosomal_protein_indexes)

    def make_elongation_rates(random, base, time_step, variable_elongation=False):
        return _make_elongation_rates_core(
            random, n_monomers, base, ribosomal_protein_indexes,
            max_elongation_rate, time_step, variable_elongation)

    return make_elongation_rates
