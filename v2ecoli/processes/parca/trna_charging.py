"""Minimal tRNA-charging helper extracted from the original vivarium-ecoli
``ecoli.processes.polypeptide_elongation`` module.

The ParCa's ``create_bulk_container()`` (in ``ecoli.library.initial_conditions``)
calls ``calculate_trna_charging`` and uses the constants ``REMOVED_FROM_CHARGING``
and ``MICROMOLAR_UNITS``. The surrounding process class, partition/metabolism
imports, and vivarium-core machinery of the upstream module have no role in
the ParCa and are deliberately not vendored.

Contains:
  MICROMOLAR_UNITS, REMOVED_FROM_CHARGING
  calculate_trna_charging(...)
  dcdt_jit(...) — @njit helper called inside calculate_trna_charging
"""

from typing import Any, Callable, Optional

from numba import njit
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from unum import Unum

from v2ecoli.processes.parca.wholecell.utils import units


MICROMOLAR_UNITS = units.umol / units.L
REMOVED_FROM_CHARGING = {"L-SELENOCYSTEINE[c]"}


def calculate_trna_charging(
    synthetase_conc: Unum,
    uncharged_trna_conc: Unum,
    charged_trna_conc: Unum,
    aa_conc: Unum,
    ribosome_conc: Unum,
    f: Unum,
    params: dict[str, Any],
    supply: Optional[Callable] = None,
    time_limit: float = 1000,
    limit_v_rib: bool = False,
    use_disabled_aas: bool = False,
) -> tuple[Unum, float, Unum, Unum, Unum]:
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
    synthetase_conc = synthetase_conc.asNumber(MICROMOLAR_UNITS)
    uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
    charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
    aa_conc = aa_conc.asNumber(MICROMOLAR_UNITS)
    ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)
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
