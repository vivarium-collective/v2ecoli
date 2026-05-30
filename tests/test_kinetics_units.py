"""Regression test: ``ppgpp_metabolite_changes`` runs on the current
plain-float (µM-magnitude) input contract.

History: kinetics' hot loops once accepted unit-bearing (Unum/pint)
concentrations, and this test guarded Unum↔pint interchangeability. That
contract was **removed** — per-call ``.to(MICROMOLAR_UNITS).magnitude``
pre-stripping moved to the caller (see ``elongation_models.py`` ~line 441 and
the ``ppgpp_metabolite_changes`` docstring), so the hot loops now take plain
numpy/float magnitudes and ``counts_to_molar: float``. Feeding unit-bearing
values now (correctly) raises. This test guards the current contract: the
function runs on plain-float inputs and returns finite metabolite changes.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.processes.polypeptide.kinetics import ppgpp_metabolite_changes


def _ppgpp_inputs():
    """Representative plain-float (µM-magnitude) inputs — the production
    contract (caller strips units before calling)."""
    n_aa = 21
    rng = np.random.RandomState(seed=42)
    reaction_stoich = np.array(
        [[-1, 1], [+1, -1], [-1, 0], [0, +1]], dtype=float)
    return dict(
        uncharged_trna_conc=rng.uniform(0.5, 5.0, n_aa),   # µM magnitudes
        charged_trna_conc=rng.uniform(0.5, 5.0, n_aa),
        ribosome_conc=30.0,
        f=rng.dirichlet(np.ones(n_aa)),
        rela_conc=0.5,
        spot_conc=0.5,
        ppgpp_conc=50.0,
        counts_to_molar=1e-3,
        v_rib=10.0,
        charging_params={"krta": 1.0, "krtf": 0.5, "max_elong_rate": 22.0},
        ppgpp_params={
            "KD_RelA": 0.1,
            "KI_SpoT": 0.1,
            "k_RelA": 75.0,
            "k_SpoT_syn": 0.1,
            "k_SpoT_deg": 0.001,
            "ppgpp_reaction_stoich": reaction_stoich,
            "synthesis_index": 0,
            "degradation_index": 1,
        },
        time_step=1.0,
        random_state=np.random.RandomState(seed=0),
    )


def test_ppgpp_metabolite_changes_runs_on_plain_floats():
    out = ppgpp_metabolite_changes(**_ppgpp_inputs())
    # Returns a tuple; the first element is the metabolite-count delta vector.
    assert len(out) >= 1
    delta = np.asarray(out[0])
    assert delta.size > 0
    assert np.all(np.isfinite(delta)), f"non-finite metabolite changes: {delta}"
