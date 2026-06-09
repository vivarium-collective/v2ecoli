"""Batched extraction of FBA solver results — bit-equivalent to the upstream
``FluxBalanceAnalysis`` per-quantity getters but with the wrapper-per-call
overhead collapsed into pure-numpy + a single shared col_primals lookup.

Why this exists
---------------

On every tick, ``v2ecoli/processes/metabolism.py:_do_update`` calls a sequence
of ``fba.get…()`` methods. The upstream implementations in
``wholecell.utils.modular_fba`` and ``wholecell.utils._netflow.nf_glpk``
expose per-quantity, per-name lookups — each wraps a list-comp + dict
lookup + ``np.array`` construction. Concretely on a typical kFBA tick:

* ``fba.getOutputMoleculeLevelsChange()`` fans out to ~520 separate
  ``solver.getFlowRates(stoich.keys())`` calls — one per output molecule —
  even though every call hits the same already-cached primal solution.
* ``fba.getReducedCosts(fba.getReactionIDs())`` does a single ~5 000-entry
  list comp + ``np.array``.
* ``fba.getReactionFluxes()`` / ``getExternalExchangeFluxes()`` /
  ``getShadowPrices(...)`` each do their own list-comp.

This module pre-resolves the stable name → solver-index mappings once at
Process ``__init__`` and exposes ``extract_results(fba, idx)`` which reads
the raw GLPK column-primal / column-dual / row-dual arrays once and slices
for the reaction/exchange/shadow paths.

The per-output-molecule sum uses a tight Python loop over the *current*
``_outputMoleculeCoeffs`` — it is mutated every tick by
``update_homeostatic_targets`` (see modular_fba.py:1230), so the
stoichiometry cannot be precomputed. But the loop reuses one shared
``col_primals`` view; the 520 upstream ``solver.getFlowRates`` calls
collapse to 520 plain-Python iterations with direct ``col_primals[idx]``
reads — no wrapper invocation, no list-comp + ``np.array`` per molecule.

All outputs preserve the exact ordering, dtype, and float64 values of the
upstream getters.

Tested via ``scripts/bench_equiv.py`` (mass/elongation fingerprints) and
``pytest -m sim tests/test_architectures_grow.py`` (ppgpp + AA-supply paths).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class FBAExtractIndices:
    """Stable name → solver-index mappings resolved once per Process build.

    All fields are derived from the FBA object's stable post-build state
    (``_reactionIDs``, ``_externalExchangeIDs``,
    ``_solver._flows``, ``_solver._materialIdxLookup``).
    """
    reaction_flux_idx: np.ndarray
    external_exchange_idx: np.ndarray
    kinetics_constrained_idx: np.ndarray
    shadow_idx: np.ndarray


def precompute_indices(
    fba,
    shadow_molecule_ids,
    kinetics_constrained_reactions,
) -> FBAExtractIndices:
    """Resolve stable name → solver-index mappings up front.

    Called once per Metabolism Process __init__ after the FBA object is built.

    Args:
        fba: a ``FluxBalanceAnalysis`` whose construction is complete
            (equality constraints built, all reactions/molecules registered).
        shadow_molecule_ids: ordered iterable of material IDs that
            ``getShadowPrices`` will be asked for each tick (typically
            ``self.model.metaboliteNamesFromNutrients``).
        kinetics_constrained_reactions: ordered iterable of reaction IDs
            for the kinetics-targeted subset
            (``self.model.kinetics_constrained_reactions``).
    """
    solver = fba._solver
    flows = solver._flows
    materials = solver._materialIdxLookup

    reaction_ids = fba._reactionIDs
    reaction_flux_idx = np.fromiter(
        (flows[r] for r in reaction_ids),
        dtype=np.int64,
        count=len(reaction_ids),
    )

    exchange_ids = fba._externalExchangeIDs
    external_exchange_idx = np.fromiter(
        (flows[e] for e in exchange_ids),
        dtype=np.int64,
        count=len(exchange_ids),
    )

    kinetics_constrained_idx = np.fromiter(
        (flows[r] for r in kinetics_constrained_reactions),
        dtype=np.int64,
        count=len(kinetics_constrained_reactions),
    )

    shadow_ids_tuple = tuple(shadow_molecule_ids)
    shadow_idx = np.fromiter(
        (materials[m] for m in shadow_ids_tuple),
        dtype=np.int64,
        count=len(shadow_ids_tuple),
    )

    return FBAExtractIndices(
        reaction_flux_idx=reaction_flux_idx,
        external_exchange_idx=external_exchange_idx,
        kinetics_constrained_idx=kinetics_constrained_idx,
        shadow_idx=shadow_idx,
    )


def extract_results(fba, idx: FBAExtractIndices) -> dict:
    """Pull all per-tick FBA result arrays from one solve call.

    Replaces a per-tick sequence of ``fba.getReactionFluxes()``,
    ``fba.getExternalExchangeFluxes()``, ``fba.getOutputMoleculeLevelsChange()``,
    ``fba.getReducedCosts(reactionIDs)``, ``fba.getShadowPrices(metaboliteNames)``,
    and ``fba.getReactionFluxes(kinetics_constrained_reactions)``.

    ``fba.solve()`` must have been called this tick before this function;
    the upstream solver caches ``_col_primals`` / ``_col_duals`` / ``_row_duals``
    on the solve, and we read those directly.
    """
    solver = fba._solver
    flows = solver._flows
    col_primals = np.asarray(solver._col_primals, dtype=np.float64)
    col_duals = np.asarray(solver._col_duals, dtype=np.float64)
    row_duals = np.asarray(solver._row_duals, dtype=np.float64)

    # Per-output-molecule sum — pure-Python loop over the *current* stoich
    # (mutated each tick by update_homeostatic_targets), but reusing one
    # shared col_primals array. Replaces the 520 wrapper-per-call upstream
    # path with 520 direct dict-lookup + numpy-scalar multiplies.
    output_coeffs = fba._outputMoleculeCoeffs
    n_out = len(output_coeffs)
    output_molecule_changes = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        total = 0.0
        for flow_name, coeff in output_coeffs[i].items():
            total += coeff * col_primals[flows[flow_name]]
        output_molecule_changes[i] = -total

    return {
        "reaction_fluxes": col_primals[idx.reaction_flux_idx],
        "external_exchange_fluxes": col_primals[idx.external_exchange_idx],
        "kinetics_constrained_fluxes": col_primals[idx.kinetics_constrained_idx],
        "output_molecule_changes": output_molecule_changes,
        "reduced_costs": col_duals[idx.reaction_flux_idx],
        "shadow_prices": row_duals[idx.shadow_idx],
    }
