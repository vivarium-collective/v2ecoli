"""Regression test: polypeptide/kinetics functions accept Unum or pint
inputs interchangeably and produce the same numeric outputs.

The hot loops in kinetics.py (ppgpp_metabolite_changes,
calculate_trna_charging) historically required Unum-typed concentrations.
After migration they accept pint Quantities too. This test asserts that
the two input forms produce identical numeric output (within 1e-9
relative tolerance), which is the regression guard the migration spec
mandates for hot-loop edits.
"""

from __future__ import annotations

import numpy as np
import pytest

from wholecell.utils import units as wc_units
from v2ecoli.types.quantity import ureg
from v2ecoli.processes.polypeptide.kinetics import (
    ppgpp_metabolite_changes,
)


def _ppgpp_inputs(builder):
    """Build representative inputs for ppgpp_metabolite_changes using the
    given unit builder (wc_units for Unum, ureg for pint). The numeric
    values are arbitrary but match shape/scale of the production system."""
    n_aa = 21
    rng = np.random.RandomState(seed=42)
    uM = builder.umol / builder.L if builder is wc_units else builder.umol / builder.L

    # Unit-on-the-left so numpy doesn't take operator precedence.
    uncharged = uM * rng.uniform(0.5, 5.0, n_aa)
    charged = uM * rng.uniform(0.5, 5.0, n_aa)
    f = rng.dirichlet(np.ones(n_aa))
    rela = 0.5 * uM
    spot = 0.5 * uM
    ppgpp = 50.0 * uM
    ribosome = 30.0 * uM
    counts_to_molar = 1e-3 * uM  # arbitrary
    v_rib = 10.0  # plain scalar; matches production usage

    charging_params = {
        "krta": 1.0,
        "krtf": 0.5,
        "max_elong_rate": 22.0,
    }
    n_reactions = 4
    reaction_stoich = np.array(
        [
            [-1, 1],
            [+1, -1],
            [-1, 0],
            [0, +1],
        ],
        dtype=float,
    )
    ppgpp_params = {
        "KD_RelA": 0.1,
        "KI_SpoT": 0.1,
        "k_RelA": 75.0,
        "k_SpoT_syn": 0.1,
        "k_SpoT_deg": 0.001,
        "ppgpp_reaction_stoich": reaction_stoich,
        "synthesis_index": 0,
        "degradation_index": 1,
    }
    return dict(
        uncharged_trna_conc=uncharged,
        charged_trna_conc=charged,
        ribosome_conc=ribosome,
        f=f,
        rela_conc=rela,
        spot_conc=spot,
        ppgpp_conc=ppgpp,
        counts_to_molar=counts_to_molar,
        v_rib=v_rib,
        charging_params=charging_params,
        ppgpp_params=ppgpp_params,
        time_step=1.0,
        random_state=np.random.RandomState(seed=0),
    )


def _close(a, b, rtol=1e-9):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol)


def test_ppgpp_metabolite_changes_unum_pint_match():
    unum_inputs = _ppgpp_inputs(wc_units)
    pint_inputs = _ppgpp_inputs(ureg)

    out_unum = ppgpp_metabolite_changes(**unum_inputs)
    out_pint = ppgpp_metabolite_changes(**pint_inputs)

    assert len(out_unum) == len(out_pint)
    for a, b in zip(out_unum, out_pint):
        assert _close(a, b), f"output mismatch: {a} vs {b}"
