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
    from v2ecoli.library.units import units as _units
    PPGPP_CONC_UNITS = _units.umol / _units.L

    def fraction_rnap_bound_ppgpp(ppgpp):
        try:
            ppgpp_val = ppgpp.asNumber(PPGPP_CONC_UNITS)
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
        attenuation_k: 2D array (n_amino_acids × n_attenuated_genes), in L/umol
    """
    from v2ecoli.library.units import units
    aa_from_trna = np.asarray(aa_from_trna)
    # attenuation_k stored as plain floats in L/umol;
    # trna_conc comes in as umol/L, so product is unitless
    _attenuation_k = np.asarray(attenuation_k)

    def get_attenuation_stop_probabilities(trna_conc):
        # Strip units and do plain numpy math
        # trna_conc is in umol/L, attenuation_k is in L/umol, product is unitless
        trna_vals = np.array([
            float(x.asNumber(units.umol / units.L)) if hasattr(x, 'asNumber')
            else float(x) for x in trna_conc
        ])
        trna_by_aa = aa_from_trna @ trna_vals
        exponent = trna_by_aa @ _attenuation_k
        return 1 - np.exp(exponent)

    return get_attenuation_stop_probabilities


@register("equilibrium.ode_solver")
def equilibrium_ode_solver_factory(stoich_matrix, rates_fwd, rates_rev,
                                    mets_to_rxn_fluxes, Rp, Pp,
                                    rates_fn_dill, rates_jac_fn_dill):
    """Factory: equilibrium ODE solver to steady state.

    Closure data from sim_data.process.equilibrium:
        stoich_matrix: array (n_mets × n_rxns)
        rates_fwd, rates_rev: arrays (n_rxns,)
        mets_to_rxn_fluxes: array (n_rxns × n_mets)
        Rp, Pp: arrays (n_mets × n_rxns)
        rates_fn_dill: base64-encoded dill of the (non_jit, jit) rate functions
        rates_jac_fn_dill: base64-encoded dill of the jacobian functions
    """
    import base64, dill
    from scipy import integrate

    _stoichMatrix = np.asarray(stoich_matrix)
    _rates_fwd = np.asarray(rates_fwd)
    _rates_rev = np.asarray(rates_rev)
    _mets_to_rxn_fluxes = np.asarray(mets_to_rxn_fluxes)
    _Rp = np.asarray(Rp)
    _Pp = np.asarray(Pp)

    _rates = dill.loads(base64.b64decode(rates_fn_dill))
    _rates_jac = dill.loads(base64.b64decode(rates_jac_fn_dill))

    def derivatives(t, y):
        return _stoichMatrix.dot(_rates[0](t, y, _rates_fwd, _rates_rev))

    def derivatives_jit(t, y):
        return _stoichMatrix.dot(_rates[1](t, y, _rates_fwd, _rates_rev))

    def derivatives_jacobian(t, y):
        return _stoichMatrix.dot(_rates_jac[0](t, y, _rates_fwd, _rates_rev))

    def derivatives_jacobian_jit(t, y):
        return _stoichMatrix.dot(_rates_jac[1](t, y, _rates_fwd, _rates_rev))

    def fluxes_and_molecules_to_SS(moleculeCounts, cellVolume, nAvogadro,
                                    random_state, time_limit=1e20,
                                    max_iter=100, jit=True):
        y_init = moleculeCounts / (cellVolume * nAvogadro)

        deriv = derivatives_jit if jit else derivatives
        jac = derivatives_jacobian_jit if jit else derivatives_jacobian

        for method in ["LSODA", "BDF"]:
            try:
                sol = integrate.solve_ivp(
                    deriv, [0, time_limit], y_init,
                    method=method, t_eval=[0, time_limit], jac=jac)
                break
            except ValueError as e:
                print(f"Warning: switching solver method in equilibrium, {e!r}")
        else:
            raise RuntimeError("Could not solve ODEs in equilibrium to SS.")

        y = sol.y.T
        if np.any(y[-1, :] * (cellVolume * nAvogadro) <= -1):
            raise ValueError("Negative values at equilibrium steady state.")
        if np.linalg.norm(deriv(0, y[-1, :]), np.inf) * (cellVolume * nAvogadro) > 1:
            raise RuntimeError("Did not reach steady state for equilibrium.")

        y[y < 0] = 0
        yMolecules = y * (cellVolume * nAvogadro)
        dYMolecules = yMolecules[-1, :] - yMolecules[0, :]

        for i in range(max_iter):
            rxnFluxes = stochasticRound(
                random_state, np.dot(_mets_to_rxn_fluxes, dYMolecules))
            if np.all(moleculeCounts + _stoichMatrix.dot(rxnFluxes) >= 0):
                break
        else:
            raise ValueError("Negative counts in equilibrium steady state.")

        rxnFluxesN = -1.0 * (rxnFluxes < 0) * rxnFluxes
        rxnFluxesP = 1.0 * (rxnFluxes > 0) * rxnFluxes
        moleculesNeeded = np.dot(_Rp, rxnFluxesP) + np.dot(_Pp, rxnFluxesN)

        return rxnFluxes, moleculesNeeded

    return fluxes_and_molecules_to_SS


@register("two_component_system.ode_solver")
def two_component_system_ode_solver_factory(
        stoich_matrix_I, stoich_matrix_J, stoich_matrix_V,
        rates_fwd, rates_rev,
        independent_molecule_indexes, atp_reaction_reactant_mask,
        independent_molecules_atp_index, dependency_matrix,
        rates_fn_dill, rates_jac_fn_dill):
    """Factory: two-component system ODE solver.

    Closure data from sim_data.process.two_component_system.
    """
    import base64, dill
    from scipy import integrate

    _stoichMatrixI = np.asarray(stoich_matrix_I)
    _stoichMatrixJ = np.asarray(stoich_matrix_J)
    _stoichMatrixV = np.asarray(stoich_matrix_V)
    _rates_fwd = np.asarray(rates_fwd)
    _rates_rev = np.asarray(rates_rev)
    _independent_molecule_indexes = np.asarray(independent_molecule_indexes)
    _atp_reaction_reactant_mask = np.asarray(atp_reaction_reactant_mask)
    _independent_molecules_atp_index = int(independent_molecules_atp_index)
    _dependency_matrix = np.asarray(dependency_matrix)

    _rates = dill.loads(base64.b64decode(rates_fn_dill))
    _rates_jac = dill.loads(base64.b64decode(rates_jac_fn_dill))

    IVP_METHODS = ["LSODA", "BDF", "Radau", "RK45", "RK23", "DOP853"]

    def _build_stoich():
        shape = (_stoichMatrixI.max() + 1, _stoichMatrixJ.max() + 1)
        out = np.zeros(shape, np.float64)
        out[_stoichMatrixI, _stoichMatrixJ] = _stoichMatrixV
        return out

    _stoich_full = _build_stoich()

    def derivatives(t, y):
        return _stoich_full.dot(_rates[0](y, t))

    def derivatives_jit(t, y):
        return _stoich_full.dot(_rates[1](y, t))

    def derivatives_jacobian(t, y):
        return _stoich_full.dot(_rates_jac[0](y, t))

    def derivatives_jacobian_jit(t, y):
        return _stoich_full.dot(_rates_jac[1](y, t))

    def molecules_to_next_time_step(
            moleculeCounts, cellVolume, nAvogadro, timeStepSec,
            random_state, method="LSODA", min_time_step=None,
            jit=True, methods_tried=None):

        y_init = moleculeCounts / (cellVolume * nAvogadro)
        deriv = derivatives_jit if jit else derivatives
        jac = derivatives_jacobian_jit if jit else derivatives_jacobian

        sol = integrate.solve_ivp(
            deriv, [0, timeStepSec], y_init,
            method=method, t_eval=[0, timeStepSec],
            atol=1e-8, jac=jac)
        y = sol.y.T

        if np.any(y[-1, :] * (cellVolume * nAvogadro) <= -1e-3):
            if min_time_step and timeStepSec > min_time_step:
                return molecules_to_next_time_step(
                    moleculeCounts, cellVolume, nAvogadro,
                    timeStepSec / 2, random_state, method=method,
                    min_time_step=min_time_step, jit=jit)

            if methods_tried is None:
                methods_tried = set()
            methods_tried.add(method)
            for new_method in IVP_METHODS:
                if new_method in methods_tried:
                    continue
                print(f"Warning: switching to {new_method} method in TCS")
                return molecules_to_next_time_step(
                    moleculeCounts, cellVolume, nAvogadro,
                    timeStepSec, random_state, method=new_method,
                    min_time_step=min_time_step, jit=jit,
                    methods_tried=methods_tried)
            else:
                raise Exception("ODE for two-component systems has negative values.")

        y[y < 0] = 0
        yMolecules = y * (cellVolume * nAvogadro)
        dYMolecules = yMolecules[-1, :] - yMolecules[0, :]

        independentMoleculesCounts = np.round(
            dYMolecules[_independent_molecule_indexes])

        max_atp_rxns = moleculeCounts[_atp_reaction_reactant_mask].min()
        independentMoleculesCounts[_independent_molecules_atp_index] = np.fmin(
            independentMoleculesCounts[:_independent_molecules_atp_index].sum()
            + independentMoleculesCounts[(_independent_molecules_atp_index + 1):].sum(),
            max_atp_rxns)

        allMoleculesChanges = _dependency_matrix.dot(independentMoleculesCounts)

        negative = independentMoleculesCounts.copy()
        negative[negative > 0] = 0
        negative[_independent_molecules_atp_index] = (
            negative[:_independent_molecules_atp_index].sum()
            + negative[(_independent_molecules_atp_index + 1):].sum())
        moleculesNeeded = _dependency_matrix.dot(-negative).clip(min=0)
        positive = independentMoleculesCounts.copy()
        positive[positive < 0] = 0
        moleculesNeeded += _dependency_matrix.dot(-positive).clip(min=0)

        return moleculesNeeded, allMoleculesChanges

    return molecules_to_next_time_step


@register("external_state.get_import_constraints")
def get_import_constraints_factory(all_external_exchange_molecules):
    """Factory: import constraints from media composition."""
    def get_import_constraints(unconstrained, constrained, constraint_units):
        unconstrained_molecules = [
            mol_id in unconstrained
            for mol_id in all_external_exchange_molecules
        ]
        constrained_molecules = [
            mol_id in constrained
            for mol_id in all_external_exchange_molecules
        ]
        constraints = [
            constrained.get(mol_id, np.nan * constraint_units).asNumber(constraint_units)
            for mol_id in all_external_exchange_molecules
        ]
        return unconstrained_molecules, constrained_molecules, constraints
    return get_import_constraints


@register("external_state.exchange_data_from_media")
def exchange_data_from_media_factory(saved_media, env_to_exchange_map,
                                      secretion_exchange_molecules,
                                      import_constraint_threshold,
                                      carbon_sources):
    """Factory: exchange data from media label."""
    from v2ecoli.library.units import units
    secretion_set = set(secretion_exchange_molecules)

    def exchange_data_from_concentrations(molecules):
        exchange_molecules = {
            env_to_exchange_map[mol]: conc for mol, conc in molecules.items()
        }
        importUnconstrainedExchangeMolecules = {
            mol_id for mol_id, conc in exchange_molecules.items()
            if conc >= import_constraint_threshold
        }
        externalExchangeMolecules = set(importUnconstrainedExchangeMolecules)
        importExchangeMolecules = set(importUnconstrainedExchangeMolecules)

        importConstrainedExchangeMolecules = {}
        oxygen_id = "OXYGEN-MOLECULE[p]"
        for cs_id in carbon_sources:
            if cs_id in importUnconstrainedExchangeMolecules:
                if oxygen_id in importUnconstrainedExchangeMolecules:
                    importConstrainedExchangeMolecules[cs_id] = 20.0 * (
                        units.mmol / units.g / units.h)
                else:
                    importConstrainedExchangeMolecules[cs_id] = 100.0 * (
                        units.mmol / units.g / units.h)
                importUnconstrainedExchangeMolecules.remove(cs_id)

        externalExchangeMolecules.update(secretion_set)
        return {
            "externalExchangeMolecules": externalExchangeMolecules,
            "importExchangeMolecules": importExchangeMolecules,
            "importConstrainedExchangeMolecules": importConstrainedExchangeMolecules,
            "importUnconstrainedExchangeMolecules": importUnconstrainedExchangeMolecules,
            "secretionExchangeMolecules": secretion_set,
        }

    def exchange_data_from_media(media_label):
        return exchange_data_from_concentrations(saved_media[media_label])

    return exchange_data_from_media


@register("growth_rate.get_ppGpp_conc")
def get_ppGpp_conc_factory(x_units_str, y_units_str, fit_params):
    """Factory: ppGpp concentration from doubling time."""
    from v2ecoli.library.units import units
    from wholecell.utils.fitting import interpolate_linearized_fit

    x_units = units.min
    y_units = units.pmol / units.L

    def get_ppGpp_conc(doubling_time):
        try:
            x = doubling_time.asNumber(x_units)
        except AttributeError:
            x = float(doubling_time)
        return y_units * interpolate_linearized_fit(x, *fit_params)

    return get_ppGpp_conc


@register("getter.get_masses")
def get_masses_factory(all_total_masses, mass_units_value):
    """Factory: get molecular masses by ID."""
    from v2ecoli.library.units import units
    import re
    _compartment_tag = re.compile(r'\[.*\]')
    _mass_units = mass_units_value * units.g / units.mol

    def get_masses(mol_ids):
        masses = [
            all_total_masses[_compartment_tag.sub("", mol_id)]
            for mol_id in mol_ids
        ]
        return _mass_units * np.array(masses)

    return get_masses


@register("getter.get_mass")
def get_mass_factory(all_total_masses, mass_units_value):
    """Factory: get single molecular mass by ID."""
    from v2ecoli.library.units import units
    import re
    _compartment_tag = re.compile(r'\[.*\]')
    _mass_units = mass_units_value * units.g / units.mol

    def get_mass(mol_id):
        return _mass_units * all_total_masses[_compartment_tag.sub("", mol_id)]

    return get_mass


@register("metabolism.concentration_updates")
def concentration_updates_factory(default_concentrations_dict, exchange_fluxes,
                                   relative_changes, molecule_set_amounts,
                                   molecule_scale_factors, linked_metabolites):
    """Factory: concentration updates object for metabolism.

    Returns an object with concentrations_based_on_nutrients() method
    and linked_metabolites attribute.
    """
    from v2ecoli.library.units import units as u

    mol_per_L = u.mol / u.L

    # Reconstruct Unum values for molecule_set_amounts
    _molecule_set_amounts = {
        k: v * mol_per_L if not hasattr(v, 'asNumber') else v
        for k, v in molecule_set_amounts.items()
    }

    class _ConcentrationUpdates:
        def __init__(self):
            self.default_concentrations_dict = default_concentrations_dict
            self.exchange_fluxes = exchange_fluxes
            self.relative_changes = relative_changes
            self.molecule_set_amounts = _molecule_set_amounts
            self.molecule_scale_factors = molecule_scale_factors
            self.linked_metabolites = linked_metabolites
            self.units = mol_per_L

        def concentrations_based_on_nutrients(self, media_id=None,
                                               imports=None,
                                               conversion_units=None):
            if conversion_units:
                conversion = self.units.asNumber(conversion_units)
            else:
                conversion = self.units

            if imports is None and media_id is not None:
                imports = self.exchange_fluxes.get(media_id, set())

            concDict = self.default_concentrations_dict.copy()
            metaboliteTargetIds = sorted(concDict.keys())
            concentrations = conversion * np.array(
                [concDict[k] for k in metaboliteTargetIds])
            concDict = dict(zip(metaboliteTargetIds, concentrations))

            if imports is not None:
                if conversion_units:
                    conversion_to_no_units = conversion_units.asUnit(self.units)

                if media_id in self.relative_changes:
                    for mol_id, conc_change in self.relative_changes[media_id].items():
                        if mol_id in concDict:
                            concDict[mol_id] *= conc_change

                for moleculeName, setAmount in self.molecule_set_amounts.items():
                    if (moleculeName in imports
                        and (moleculeName[:-3] + "[c]" not in self.molecule_scale_factors
                             or moleculeName == "L-SELENOCYSTEINE[c]")
                    ) or (moleculeName in self.molecule_scale_factors
                           and moleculeName[:-3] + "[p]" in imports):
                        if conversion_units:
                            setAmount = (setAmount / conversion_to_no_units).asNumber()
                        concDict[moleculeName] = setAmount

            for met, linked in self.linked_metabolites.items():
                concDict[met] = concDict[linked["lead"]] * linked["ratio"]

            return concDict

    return _ConcentrationUpdates()


@register("metabolism.exchange_constraints")
def exchange_constraints_factory(concentration_updates_obj):
    """Factory: exchange constraints for FBA.

    Takes a concentration_updates object (from the concentration_updates factory).
    """
    from v2ecoli.library.units import units

    def exchange_constraints(exchangeIDs, coefficient, targetUnits, media_id,
                             unconstrained, constrained,
                             concModificationsBasedOnCondition=None):
        newObjective = concentration_updates_obj.concentrations_based_on_nutrients(
            imports=unconstrained.union(constrained),
            media_id=media_id,
            conversion_units=targetUnits)

        if concModificationsBasedOnCondition is not None:
            newObjective.update(concModificationsBasedOnCondition)

        externalMoleculeLevels = np.zeros(len(exchangeIDs), np.float64)
        for index, moleculeID in enumerate(exchangeIDs):
            if moleculeID in unconstrained:
                externalMoleculeLevels[index] = np.inf
            elif moleculeID in constrained:
                externalMoleculeLevels[index] = (
                    constrained[moleculeID].asNumber(targetUnits) * coefficient)

        return newObjective, externalMoleculeLevels

    return exchange_constraints


@register("metabolism.get_kinetic_constraints")
def get_kinetic_constraints_factory(enzymes_expr, saturations_expr, kcats):
    """Factory: kinetic constraints from enzyme/substrate concentrations.

    Closure data from sim_data.process.metabolism:
        enzymes_expr: string expression for enzyme indexing
        saturations_expr: string expression for saturation calculation
        kcats: array (n_reactions × 3) of kcat values
    """
    from v2ecoli.library.units import units
    CONC_UNITS = units.umol / units.L

    _kcats = np.asarray(kcats)
    _compiled_enzymes = eval(f"lambda e: {enzymes_expr}")
    _compiled_saturation = eval(f"lambda s: {saturations_expr}")

    def get_kinetic_constraints(enzymes, substrates):
        enzs = enzymes.asNumber(CONC_UNITS)
        subs = substrates.asNumber(CONC_UNITS)

        capacity = np.array(_compiled_enzymes(enzs))[:, None] * _kcats
        saturation = np.array(
            [[min(v), sum(v) / len(v), max(v)] for v in _compiled_saturation(subs)]
        )
        return CONC_UNITS / units.s * capacity * saturation

    return get_kinetic_constraints


@register("mass.get_biomass_as_concentrations")
def get_biomass_as_concentrations_factory(precomputed):
    """Factory: biomass concentrations from doubling time.

    precomputed: dict mapping doubling_time_minutes (float) to
    dict of {metabolite_id: concentration_mol_per_L}
    """
    from v2ecoli.library.units import units

    # Convert string keys back to floats (JSON serialization)
    lookup = {float(k): v for k, v in precomputed.items()}
    sorted_dts = sorted(lookup.keys())

    def get_biomass_as_concentrations(doubling_time, **kwargs):
        try:
            dt_min = doubling_time.asNumber(units.min)
        except AttributeError:
            dt_min = float(doubling_time)

        # Find closest precomputed doubling time
        closest = min(sorted_dts, key=lambda x: abs(x - dt_min))
        conc_dict = lookup[closest]

        # Return with units (mol/L to match original)
        return {k: v * units.mol / units.L for k, v in conc_dict.items()}

    return get_biomass_as_concentrations


@register("metabolism.get_pathway_enzyme_counts_per_aa")
def get_pathway_enzyme_counts_per_aa_factory(enzyme_to_amino_acid_fwd,
                                              enzyme_to_amino_acid_rev):
    """Factory: enzyme counts per amino acid for supply calculation.

    Closure data from sim_data.process.metabolism:
        enzyme_to_amino_acid_fwd: array (n_enzymes × n_amino_acids)
        enzyme_to_amino_acid_rev: array (n_enzymes × n_amino_acids)
    """
    fwd = np.asarray(enzyme_to_amino_acid_fwd)
    rev = np.asarray(enzyme_to_amino_acid_rev)

    def get_pathway_enzyme_counts_per_aa(enzyme_counts):
        return enzyme_counts @ fwd, enzyme_counts @ rev

    return get_pathway_enzyme_counts_per_aa


@register("transcription.synth_prob_from_ppgpp")
def synth_prob_from_ppgpp_factory(exp_free, exp_ppgpp,
                                   ppgpp_growth_parameters,
                                   rna_deg_rates, wt_replication_coordinates,
                                   is_rRNA, ppgpp_km_squared):
    """Factory: synthesis probability as function of ppGpp concentration.

    Closure data from sim_data.process.transcription:
        exp_free: array (n_TUs,) — expression without ppGpp
        exp_ppgpp: array (n_TUs,) — expression with ppGpp
        ppgpp_growth_parameters: tuple (x_transform, y_transform, slope, intercept)
        rna_deg_rates: array (n_TUs,) — degradation rates in 1/s
        wt_replication_coordinates: array (n_TUs,) — WT replication coordinates
        is_rRNA: boolean array (n_TUs,)
        ppgpp_km_squared: float — squared KM for ppGpp binding
    """
    from wholecell.utils.fitting import interpolate_linearized_fit
    from v2ecoli.library.units import units

    exp_free = np.asarray(exp_free, dtype=float)
    exp_ppgpp = np.asarray(exp_ppgpp, dtype=float)
    rna_deg_rates = np.asarray(rna_deg_rates, dtype=float)
    wt_replication_coordinates = np.asarray(wt_replication_coordinates)
    is_rRNA = np.asarray(is_rRNA, dtype=bool)
    x_transform, y_transform, slope, intercept = ppgpp_growth_parameters

    def fraction_rnap_bound_ppgpp(ppgpp_val):
        return ppgpp_val ** 2 / (ppgpp_km_squared + ppgpp_val ** 2)

    def normalize(x):
        total = x.sum()
        if total > 0:
            return x / total
        return x

    PPGPP_CONC_UNITS = units.umol / units.L

    def synth_prob_from_ppgpp(ppgpp, copy_number, balanced_rRNA_prob=True):
        try:
            ppgpp_val = ppgpp.asNumber(PPGPP_CONC_UNITS)
        except AttributeError:
            ppgpp_val = float(ppgpp)

        f_ppgpp = fraction_rnap_bound_ppgpp(ppgpp_val)

        y = interpolate_linearized_fit(
            ppgpp_val, x_transform, y_transform, slope, intercept)
        growth = max(float(y), 0.0)
        tau = np.log(2) / growth / 60 if growth > 0 else 1e12

        loss = growth + rna_deg_rates
        n_avg_copy = copy_number(tau, wt_replication_coordinates)

        factor = loss / n_avg_copy
        prob = normalize(
            (exp_free * (1 - f_ppgpp) + exp_ppgpp * f_ppgpp) * factor
        )

        if balanced_rRNA_prob:
            prob[is_rRNA] = prob[is_rRNA].mean()

        return prob, factor

    return synth_prob_from_ppgpp
