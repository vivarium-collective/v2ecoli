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


# ---------------------------------------------------------------------------
# Tier B: Factory functions (need closure data)
# ---------------------------------------------------------------------------

@register("replication.get_average_copy_number")
def get_average_copy_number_factory(replichore_lengths, c_period_in_mins,
                                     d_period_in_mins):
    """Factory: gene copy number based on replication fork position.

    Closure data from sim_data.process.replication:
        replichore_lengths: array of 2 ints (forward, reverse)
        c_period_in_mins: float
        d_period_in_mins: float
    """
    replichore_lengths = np.asarray(replichore_lengths)

    def get_average_copy_number(tau, coords):
        right_replichore_length = replichore_lengths[0]
        left_replichore_length = replichore_lengths[1]
        relative_pos = np.array(coords, float)
        relative_pos[coords > 0] = relative_pos[coords > 0] / right_replichore_length
        relative_pos[coords < 0] = -relative_pos[coords < 0] / left_replichore_length
        return 2 ** (
            ((1 - relative_pos) * c_period_in_mins + d_period_in_mins) / tau
        )

    return get_average_copy_number


@register("transcription.get_rnap_active_fraction_from_ppGpp")
def get_rnap_active_fraction_factory(fraction_active_rnap_bound,
                                      fraction_active_rnap_free,
                                      ppgpp_km_squared):
    """Factory: RNAP active fraction as function of ppGpp concentration.

    Closure data from sim_data.process.transcription:
        fraction_active_rnap_bound: float
        fraction_active_rnap_free: float
        ppgpp_km_squared: float (squared KM for ppGpp binding)
    """
    def fraction_rnap_bound_ppgpp(ppgpp):
        try:
            ppgpp_val = ppgpp.asNumber()
        except AttributeError:
            ppgpp_val = float(ppgpp)
        return ppgpp_val ** 2 / (ppgpp_km_squared + ppgpp_val ** 2)

    def get_rnap_active_fraction_from_ppGpp(ppgpp):
        f_ppgpp = fraction_rnap_bound_ppgpp(ppgpp)
        return (
            fraction_active_rnap_bound * f_ppgpp
            + fraction_active_rnap_free * (1 - f_ppgpp)
        )

    return get_rnap_active_fraction_from_ppGpp


@register("mass.get_dna_critical_mass")
def get_dna_critical_mass_factory(dry_mass_params, cell_dry_mass_fraction):
    """Factory: critical initiation mass for DNA replication.

    Closure data from sim_data.mass (GrowthRateParameters):
        dry_mass_params: array of 2 floats (slope, intercept for 1/dryMass vs tau)
        cell_dry_mass_fraction: float
    """
    from v2ecoli.library.units import units
    NORMAL_CRITICAL_MASS = 975 * units.fg
    SLOW_GROWTH_FACTOR = 1.2

    dry_mass_params = np.asarray(dry_mass_params)

    def get_dna_critical_mass(doubling_time):
        try:
            tau = doubling_time.asNumber(units.min)
        except AttributeError:
            tau = float(doubling_time)
        inverse_mass = dry_mass_params[0] * tau + dry_mass_params[1]
        if inverse_mass < 0:
            raise ValueError(f"Negative inverse mass at tau={tau}")
        avg_dry_mass = units.fg / inverse_mass
        mass = avg_dry_mass / cell_dry_mass_fraction
        return min(mass * SLOW_GROWTH_FACTOR, NORMAL_CRITICAL_MASS)

    return get_dna_critical_mass


@register("growth_rate.get_ribosome_elongation_rate_by_ppgpp")
def get_ribosome_elongation_rate_by_ppgpp_factory(
        ppgpp_units_str, rate_units_str, fit_vmax, KI, H,
        charging_fraction_of_max_elong_rate):
    """Factory: ribosome elongation rate as Hill function of ppGpp.

    Closure data from sim_data.growth_rate_parameters:
        ppgpp_units_str: string representation of ppGpp concentration units
        rate_units_str: string representation of rate units
        fit_vmax: float (fitted Vmax)
        KI: float (Hill inhibition constant)
        H: float (Hill coefficient)
        charging_fraction_of_max_elong_rate: float (default 0.9)
    """
    from v2ecoli.library.units import units
    # Reconstruct unit objects
    ppgpp_units = units.umol / units.L
    rate_units = units.aa / units.s

    def get_ribosome_elongation_rate_by_ppgpp(ppgpp, max_rate=None):
        vmax = fit_vmax if max_rate is None else max_rate
        try:
            ppgpp_val = ppgpp.asNumber(ppgpp_units)
        except AttributeError:
            ppgpp_val = float(ppgpp)
        return (
            rate_units * vmax
            / (1 + (ppgpp_val / KI) ** H)
            / charging_fraction_of_max_elong_rate
        )

    return get_ribosome_elongation_rate_by_ppgpp


@register("transcription.get_attenuation_stop_probabilities")
def get_attenuation_stop_probabilities_factory(aa_from_trna, attenuation_k):
    """Factory: tRNA attenuation stop probabilities.

    Closure data from sim_data.process.transcription:
        aa_from_trna: 2D array (n_amino_acids × n_trnas), binary mapping
        attenuation_k: 2D array (n_amino_acids × n_attenuated_genes)
    """
    from v2ecoli.library.units import units
    aa_from_trna = np.asarray(aa_from_trna)
    attenuation_k = np.asarray(attenuation_k)

    def get_attenuation_stop_probabilities(trna_conc):
        trna_by_aa = units.matmul(aa_from_trna, trna_conc)
        return 1 - np.exp(units.strip_empty_units(trna_by_aa @ attenuation_k))

    return get_attenuation_stop_probabilities
