"""
Polypeptide elongation kinetics: ppGpp synthesis/degradation, tRNA
charging, and amino acid supply functions.

These module-level free functions are used by the SteadyStateElongationModel
to implement the ppGpp-regulated translation model with explicit
aminoacyl-tRNA synthetase kinetics. Separated from the main
``polypeptide_elongation`` module purely for file-size manageability;
behavior is unchanged.
"""

from typing import Any, Callable, Optional, Tuple

from numba import njit
import numpy as np
import numpy.typing as npt
import pint
from scipy.integrate import solve_ivp

from wholecell.utils.random import stochasticRound

from v2ecoli.library.unit_bridge import unum_to_pint
from v2ecoli.processes.polypeptide.common import MICROMOLAR_UNITS

# Function bodies below normalize every unit-bearing input through
# unum_to_pint, then convert to MICROMOLAR_UNITS for the numerical math,
# so callers can pass Unum or pint Quantities interchangeably.


def ppgpp_metabolite_changes(
    uncharged_trna_conc: pint.Quantity,
    charged_trna_conc: pint.Quantity,
    ribosome_conc: pint.Quantity,
    f: npt.NDArray[np.float64],
    rela_conc: pint.Quantity,
    spot_conc: pint.Quantity,
    ppgpp_conc: pint.Quantity,
    counts_to_molar: pint.Quantity,
    v_rib: pint.Quantity,
    charging_params: dict[str, Any],
    ppgpp_params: dict[str, Any],
    time_step: float,
    request: bool = False,
    limits: Optional[npt.NDArray[np.float64]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> tuple[npt.NDArray[np.int64], int, int, pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity]:
    """
    Calculates the changes in metabolite counts based on ppGpp synthesis and
    degradation reactions.

    Args:
        uncharged_trna_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of uncharged tRNA associated with each amino acid
        charged_trna_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of charged tRNA associated with each amino acid
        ribosome_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of active ribosomes
        f: fraction of each amino acid to be incorporated
            to total amino acids incorporated
        rela_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of RelA
        spot_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of SpoT
        ppgpp_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of ppGpp
        counts_to_molar: conversion factor
            from counts to molarity (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
        v_rib: rate of amino acid incorporation at the ribosome (units of uM/s)
        charging_params: parameters used in charging equations
        ppgpp_params: parameters used in ppGpp reactions
        time_step: length of the current time step
        request: if True, only considers reactant stoichiometry,
            otherwise considers reactants and products. For use in
            calculateRequest. GDP appears as both a reactant and product
            and the request can be off the actual use if not handled in this
            manner.
        limits: counts of molecules that are available to prevent
            negative total counts as a result of delta_metabolites.
            If None, no limits are placed on molecule changes.
        random_state: random state for the process
    Returns:
        7-element tuple containing

        - **delta_metabolites**: the change in counts of each metabolite
          involved in ppGpp reactions
        - **n_syn_reactions**: the number of ppGpp synthesis reactions
        - **n_deg_reactions**: the number of ppGpp degradation reactions
        - **v_rela_syn**: rate of synthesis from RelA per amino
          acid tRNA species
        - **v_spot_syn**: rate of synthesis from SpoT
        - **v_deg**: rate of degradation from SpoT
        - **v_deg_inhibited**: rate of degradation from SpoT per
          amino acid tRNA species
    """

    if random_state is None:
        random_state = np.random.RandomState()

    uncharged_trna_conc = unum_to_pint(uncharged_trna_conc).to(MICROMOLAR_UNITS).magnitude
    charged_trna_conc = unum_to_pint(charged_trna_conc).to(MICROMOLAR_UNITS).magnitude
    ribosome_conc = unum_to_pint(ribosome_conc).to(MICROMOLAR_UNITS).magnitude
    rela_conc = unum_to_pint(rela_conc).to(MICROMOLAR_UNITS).magnitude
    spot_conc = unum_to_pint(spot_conc).to(MICROMOLAR_UNITS).magnitude
    ppgpp_conc = unum_to_pint(ppgpp_conc).to(MICROMOLAR_UNITS).magnitude
    counts_to_micromolar = unum_to_pint(counts_to_molar).to(MICROMOLAR_UNITS).magnitude

    numerator = (
        1
        + charged_trna_conc / charging_params["krta"]
        + uncharged_trna_conc / charging_params["krtf"]
    )
    saturated_charged = charged_trna_conc / charging_params["krta"] / numerator
    saturated_uncharged = uncharged_trna_conc / charging_params["krtf"] / numerator
    if v_rib == 0:
        ribosome_conc_a_site = f * ribosome_conc
    else:
        ribosome_conc_a_site = (
            f * v_rib / (saturated_charged * charging_params["max_elong_rate"])
        )
    ribosomes_bound_to_uncharged = ribosome_conc_a_site * saturated_uncharged

    # Handle rare cases when tRNA concentrations are 0
    # Can result in inf and nan so assume a fraction of ribosomes
    # bind to the uncharged tRNA if any tRNA are present or 0 if not
    mask = ~np.isfinite(ribosomes_bound_to_uncharged)
    ribosomes_bound_to_uncharged[mask] = (
        ribosome_conc
        * f[mask]
        * np.array(uncharged_trna_conc[mask] + charged_trna_conc[mask] > 0)
    )

    # Calculate active fraction of RelA
    competitive_inhibition = 1 + ribosomes_bound_to_uncharged / ppgpp_params["KD_RelA"]
    inhibition_product = np.prod(competitive_inhibition)
    with np.errstate(divide="ignore"):
        frac_rela = 1 / (
            ppgpp_params["KD_RelA"]
            / ribosomes_bound_to_uncharged
            * inhibition_product
            / competitive_inhibition
            + 1
        )

    # Calculate rates for synthesis and degradation
    v_rela_syn = ppgpp_params["k_RelA"] * rela_conc * frac_rela
    v_spot_syn = ppgpp_params["k_SpoT_syn"] * spot_conc
    v_syn = v_rela_syn.sum() + v_spot_syn
    max_deg = ppgpp_params["k_SpoT_deg"] * spot_conc * ppgpp_conc
    fractions = uncharged_trna_conc / ppgpp_params["KI_SpoT"]
    v_deg = max_deg / (1 + fractions.sum())
    v_deg_inhibited = (max_deg - v_deg) * fractions / fractions.sum()

    # Convert to discrete reactions
    n_syn_reactions = stochasticRound(
        random_state, v_syn * time_step / counts_to_micromolar
    )[0]
    n_deg_reactions = stochasticRound(
        random_state, v_deg * time_step / counts_to_micromolar
    )[0]

    # Only look at reactant stoichiometry if requesting molecules to use
    if request:
        ppgpp_reaction_stoich = np.zeros_like(ppgpp_params["ppgpp_reaction_stoich"])
        reactants = ppgpp_params["ppgpp_reaction_stoich"] < 0
        ppgpp_reaction_stoich[reactants] = ppgpp_params["ppgpp_reaction_stoich"][
            reactants
        ]
    else:
        ppgpp_reaction_stoich = ppgpp_params["ppgpp_reaction_stoich"]

    # Calculate the change in metabolites and adjust to limits if provided
    # Possible reactions are adjusted down to limits if the change in any
    # metabolites would result in negative counts
    max_iterations = int(n_deg_reactions + n_syn_reactions + 1)
    old_counts = None
    for it in range(max_iterations):
        delta_metabolites = (
            ppgpp_reaction_stoich[:, ppgpp_params["synthesis_index"]] * n_syn_reactions
            + ppgpp_reaction_stoich[:, ppgpp_params["degradation_index"]]
            * n_deg_reactions
        )

        if limits is None:
            break
        else:
            final_counts = delta_metabolites + limits

            if np.all(final_counts >= 0) or (
                old_counts is not None and np.all(final_counts == old_counts)
            ):
                break

            limited_index = np.argmin(final_counts)
            if (
                ppgpp_reaction_stoich[limited_index, ppgpp_params["synthesis_index"]]
                < 0
            ):
                limited = np.ceil(
                    final_counts[limited_index]
                    / ppgpp_reaction_stoich[
                        limited_index, ppgpp_params["synthesis_index"]
                    ]
                )
                n_syn_reactions -= min(limited, n_syn_reactions)
            if (
                ppgpp_reaction_stoich[limited_index, ppgpp_params["degradation_index"]]
                < 0
            ):
                limited = np.ceil(
                    final_counts[limited_index]
                    / ppgpp_reaction_stoich[
                        limited_index, ppgpp_params["degradation_index"]
                    ]
                )
                n_deg_reactions -= min(limited, n_deg_reactions)

            old_counts = final_counts
    else:
        raise ValueError("Failed to meet molecule limits with ppGpp reactions.")

    return (
        delta_metabolites,
        n_syn_reactions,
        n_deg_reactions,
        v_rela_syn,
        v_spot_syn,
        v_deg,
        v_deg_inhibited,
    )


def calculate_trna_charging(
    synthetase_conc: pint.Quantity,
    uncharged_trna_conc: pint.Quantity,
    charged_trna_conc: pint.Quantity,
    aa_conc: pint.Quantity,
    ribosome_conc: pint.Quantity,
    f: pint.Quantity,
    params: dict[str, Any],
    supply: Optional[Callable] = None,
    time_limit: float = 1000,
    limit_v_rib: bool = False,
    use_disabled_aas: bool = False,
) -> tuple[pint.Quantity, float, pint.Quantity, pint.Quantity, pint.Quantity]:
    """
    Calculates the steady state value of tRNA based on charging and
    incorporation through polypeptide elongation. The fraction of
    charged/uncharged is also used to determine how quickly the
    ribosome is elongating. All concentrations are given in units of
    :py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`.

    Args:
        synthetase_conc: concentration of synthetases associated with
            each amino acid
        uncharged_trna_conc: concentration of uncharged tRNA associated
            with each amino acid
        charged_trna_conc: concentration of charged tRNA associated with
            each amino acid
        aa_conc: concentration of each amino acid
        ribosome_conc: concentration of active ribosomes
        f: fraction of each amino acid to be incorporated to total amino
            acids incorporated
        params: parameters used in charging equations
        supply: function to get the rate of amino acid supply (synthesis
            and import) based on amino acid concentrations. If None, amino
            acid concentrations remain constant during charging
        time_limit: time limit to reach steady state
        limit_v_rib: if True, v_rib is limited to the number of amino acids
            that are available
        use_disabled_aas: if False, amino acids in
            :py:data:`~ecoli.processes.polypeptide_elongation.REMOVED_FROM_CHARGING`
            are excluded from charging

    Returns:
        5-element tuple containing

        - **new_fraction_charged**: fraction of total tRNA that is charged for each
          amino acid species
        - **v_rib**: ribosomal elongation rate in units of uM/s
        - **total_synthesis**: the total amount of amino acids synthesized during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
        - **total_import**: the total amount of amino acids imported during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
        - **total_export**: the total amount of amino acids exported during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
    """

    def negative_check(trna1: npt.NDArray[np.float64], trna2: npt.NDArray[np.float64]):
        """
        Check for floating point precision issues that can lead to small
        negative numbers instead of 0. Adjusts both species of tRNA to
        bring concentration of trna1 to 0 and keep the same total concentration.

        Args:
            trna1: concentration of one tRNA species (charged or uncharged)
            trna2: concentration of another tRNA species (charged or uncharged)
        """

        mask = trna1 < 0
        trna2[mask] = trna1[mask] + trna2[mask]
        trna1[mask] = 0

    def dcdt(t: float, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Function for solve_ivp to integrate

        Args:
            c: 1D array of concentrations of uncharged and charged tRNAs
                dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
            t: time of integration step

        Returns:
            Array of dc/dt for tRNA concentrations
                dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
        """

        v_charging, dtrna, daa = dcdt_jit(
            t,
            c,
            n_aas_masked,
            n_aas,
            mask,
            params["kS"],
            synthetase_conc,
            params["KMaa"],
            params["KMtf"],
            f,
            params["krta"],
            params["krtf"],
            params["max_elong_rate"],
            ribosome_conc,
            limit_v_rib,
            aa_rate_limit,
            v_rib_max,
        )

        if supply is None:
            v_synthesis = np.zeros(n_aas)
            v_import = np.zeros(n_aas)
            v_export = np.zeros(n_aas)
        else:
            aa_conc = c[2 * n_aas_masked : 2 * n_aas_masked + n_aas]
            v_synthesis, v_import, v_export = supply(unit_conversion * aa_conc)
            v_supply = v_synthesis + v_import - v_export
            daa[mask] = v_supply[mask] - v_charging

        return np.hstack((-dtrna, dtrna, daa, v_synthesis, v_import, v_export))

    # Convert inputs for integration
    synthetase_conc = unum_to_pint(synthetase_conc).to(MICROMOLAR_UNITS).magnitude
    uncharged_trna_conc = unum_to_pint(uncharged_trna_conc).to(MICROMOLAR_UNITS).magnitude
    charged_trna_conc = unum_to_pint(charged_trna_conc).to(MICROMOLAR_UNITS).magnitude
    aa_conc = unum_to_pint(aa_conc).to(MICROMOLAR_UNITS).magnitude
    ribosome_conc = unum_to_pint(ribosome_conc).to(MICROMOLAR_UNITS).magnitude
    unit_conversion = params["unit_conversion"]

    # Remove disabled amino acids from calculations
    n_total_aas = len(aa_conc)
    if use_disabled_aas:
        mask = np.ones(n_total_aas, bool)
    else:
        mask = params["charging_mask"]
    synthetase_conc = synthetase_conc[mask]
    original_uncharged_trna_conc = uncharged_trna_conc[mask]
    original_charged_trna_conc = charged_trna_conc[mask]
    original_aa_conc = aa_conc[mask]
    f = f[mask]

    n_aas = len(aa_conc)
    n_aas_masked = len(original_aa_conc)

    # Limits for integration
    aa_rate_limit = original_aa_conc / time_limit
    trna_rate_limit = original_charged_trna_conc / time_limit
    v_rib_max = max(0, ((aa_rate_limit + trna_rate_limit) / f).min())

    # Integrate rates of charging and elongation
    c_init = np.hstack(
        (
            original_uncharged_trna_conc,
            original_charged_trna_conc,
            aa_conc,
            np.zeros(n_aas),
            np.zeros(n_aas),
            np.zeros(n_aas),
        )
    )
    sol = solve_ivp(dcdt, [0, time_limit], c_init, method="BDF")
    c_sol = sol.y.T

    # Determine new values from integration results
    final_uncharged_trna_conc = c_sol[-1, :n_aas_masked]
    final_charged_trna_conc = c_sol[-1, n_aas_masked : 2 * n_aas_masked]
    total_synthesis = c_sol[-1, 2 * n_aas_masked + n_aas : 2 * n_aas_masked + 2 * n_aas]
    total_import = c_sol[
        -1, 2 * n_aas_masked + 2 * n_aas : 2 * n_aas_masked + 3 * n_aas
    ]
    total_export = c_sol[
        -1, 2 * n_aas_masked + 3 * n_aas : 2 * n_aas_masked + 4 * n_aas
    ]

    negative_check(final_uncharged_trna_conc, final_charged_trna_conc)
    negative_check(final_charged_trna_conc, final_uncharged_trna_conc)

    fraction_charged = final_charged_trna_conc / (
        final_uncharged_trna_conc + final_charged_trna_conc
    )
    numerator_ribosome = 1 + np.sum(
        f
        * (
            params["krta"] / final_charged_trna_conc
            + final_uncharged_trna_conc
            / final_charged_trna_conc
            * params["krta"]
            / params["krtf"]
        )
    )
    v_rib = params["max_elong_rate"] * ribosome_conc / numerator_ribosome
    if limit_v_rib:
        v_rib_max = max(
            0,
            (
                (
                    original_aa_conc
                    + (original_charged_trna_conc - final_charged_trna_conc)
                )
                / time_limit
                / f
            ).min(),
        )
        v_rib = min(v_rib, v_rib_max)

    # Replace SEL fraction charged with average
    new_fraction_charged = np.zeros(n_total_aas)
    new_fraction_charged[mask] = fraction_charged
    new_fraction_charged[~mask] = fraction_charged.mean()

    return new_fraction_charged, v_rib, total_synthesis, total_import, total_export


@njit(error_model="numpy")
def dcdt_jit(
    t,
    c,
    n_aas_masked,
    n_aas,
    mask,
    kS,
    synthetase_conc,
    KMaa,
    KMtf,
    f,
    krta,
    krtf,
    max_elong_rate,
    ribosome_conc,
    limit_v_rib,
    aa_rate_limit,
    v_rib_max,
):
    uncharged_trna_conc = c[:n_aas_masked]
    charged_trna_conc = c[n_aas_masked : 2 * n_aas_masked]
    aa_conc = c[2 * n_aas_masked : 2 * n_aas_masked + n_aas]
    masked_aa_conc = aa_conc[mask]

    v_charging = (
        kS
        * synthetase_conc
        * uncharged_trna_conc
        * masked_aa_conc
        / (KMaa[mask] * KMtf[mask])
        / (
            1
            + uncharged_trna_conc / KMtf[mask]
            + masked_aa_conc / KMaa[mask]
            + uncharged_trna_conc * masked_aa_conc / KMtf[mask] / KMaa[mask]
        )
    )
    numerator_ribosome = 1 + np.sum(
        f
        * (
            krta / charged_trna_conc
            + uncharged_trna_conc / charged_trna_conc * krta / krtf
        )
    )
    v_rib = max_elong_rate * ribosome_conc / numerator_ribosome

    # Handle case when f is 0 and charged_trna_conc is 0
    if not np.isfinite(v_rib):
        v_rib = 0

    # Limit v_rib and v_charging to the amount of available amino acids
    if limit_v_rib:
        v_charging = np.fmin(v_charging, aa_rate_limit)
        v_rib = min(v_rib, v_rib_max)

    dtrna = v_charging - v_rib * f
    daa = np.zeros(n_aas)

    return v_charging, dtrna, daa


def get_charging_supply_function(
    supply_in_charging: bool,
    mechanistic_supply: bool,
    mechanistic_aa_transport: bool,
    amino_acid_synthesis: Callable,
    amino_acid_import: Callable,
    amino_acid_export: Callable,
    aa_supply_scaling: Callable,
    counts_to_molar: pint.Quantity,
    aa_supply: npt.NDArray[np.float64],
    fwd_enzyme_counts: npt.NDArray[np.int64],
    rev_enzyme_counts: npt.NDArray[np.int64],
    dry_mass: pint.Quantity,
    importer_counts: npt.NDArray[np.int64],
    exporter_counts: npt.NDArray[np.int64],
    aa_in_media: npt.NDArray[np.bool_],
) -> Optional[Callable[[npt.NDArray[np.float64]], Tuple[pint.Quantity, pint.Quantity, pint.Quantity]]]:
    """
    Get a function mapping internal amino acid concentrations to the amount of
    amino acid supply expected.

    Args:
        supply_in_charging: True if using aa_supply_in_charging option
        mechanistic_supply: True if using mechanistic_translation_supply option
        mechanistic_aa_transport: True if using mechanistic_aa_transport option
        amino_acid_synthesis: function to provide rates of synthesis for amino
            acids based on the internal state
        amino_acid_import: function to provide import rates for amino
            acids based on the internal and external state
        amino_acid_export: function to provide export rates for amino
            acids based on the internal state
        aa_supply_scaling: function to scale the amino acid supply based
            on the internal state
        counts_to_molar: conversion factor for counts to molar
            (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
        aa_supply: rate of amino acid supply expected
        fwd_enzyme_counts: enzyme counts in forward reactions for each amino acid
        rev_enzyme_counts: enzyme counts in loss reactions for each amino acid
        dry_mass: dry mass of the cell with mass units
        importer_counts: counts for amino acid importers
        exporter_counts: counts for amino acid exporters
        aa_in_media: True for each amino acid that is present in the media
    Returns:
        Function that provides the amount of supply (synthesis, import, export)
        for each amino acid based on the internal state of the cell
    """

    # Create functions that are only dependent on amino acid concentrations for more stable
    # charging and amino acid concentrations.  If supply_in_charging is not set, then
    # setting None will maintain constant amino acid concentrations throughout charging.
    supply_function = None
    if supply_in_charging:
        counts_to_molar = unum_to_pint(counts_to_molar).to(MICROMOLAR_UNITS).magnitude
        zeros = counts_to_molar * np.zeros_like(aa_supply)
        if mechanistic_supply:
            if mechanistic_aa_transport:

                def supply_function(aa_conc):
                    return (
                        counts_to_molar
                        * amino_acid_synthesis(
                            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
                        )[0],
                        counts_to_molar
                        * amino_acid_import(
                            aa_in_media,
                            dry_mass,
                            aa_conc,
                            importer_counts,
                            mechanistic_aa_transport,
                        ),
                        counts_to_molar
                        * amino_acid_export(
                            exporter_counts, aa_conc, mechanistic_aa_transport
                        ),
                    )
            else:

                def supply_function(aa_conc):
                    return (
                        counts_to_molar
                        * amino_acid_synthesis(
                            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
                        )[0],
                        counts_to_molar
                        * amino_acid_import(
                            aa_in_media,
                            dry_mass,
                            aa_conc,
                            importer_counts,
                            mechanistic_aa_transport,
                        ),
                        zeros,
                    )
        else:

            def supply_function(aa_conc):
                return (
                    counts_to_molar
                    * aa_supply
                    * aa_supply_scaling(aa_conc, aa_in_media),
                    zeros,
                    zeros,
                )

    return supply_function
