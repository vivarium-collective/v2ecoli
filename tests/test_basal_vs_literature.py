"""Tests for the basal report card's ``vs_literature`` reference mode.

Two layers:

  1. **The ``literature`` criterion** (``card_criteria.grade_axis``) — the new
     scalar-vs-experimental-band criterion, including the differentiated
     first-principles ``theoretical_max`` violation. Pure unit tests, no bundle.
  2. **The card** (``scripts/render_basal_vs_literature.build``) — the blessed
     baseline graded against the ecoli-sources ``validation_data`` bundle.
     Pins the physiology story: μ within the measured band, glucose uptake far
     below it, biomass yield above the stoichiometric ceiling.
"""
import os
import sys

import pytest

from v2ecoli.library.card_criteria import grade_axis
from v2ecoli.library.report_card import grade_card

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import render_basal_vs_literature as rvl  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the ``literature`` criterion
# ---------------------------------------------------------------------------

def _crit(measured, tmax=None, tol=0.10):
    return {"type": "literature", "measured": measured,
            "theoretical_max": tmax, "tol_rel": tol}


def test_literature_within_measured_band():
    g = grade_axis({"mean": 0.42}, _crit([0.355, 0.44, 0.444, 0.42], tmax=0.538))
    assert g["verdict"] == "within_tol"
    assert g["detail"]["first_principles_violation"] is False


def test_literature_exceeds_theoretical_max_is_flagged_mismatch():
    g = grade_axis({"mean": 0.891}, _crit([0.355, 0.44, 0.444, 0.42], tmax=0.538))
    assert g["verdict"] == "mismatch"
    assert g["detail"]["first_principles_violation"] is True
    assert "first-principles" in g["meter"]


def test_literature_below_band_no_ceiling_is_plain_mismatch():
    g = grade_axis({"mean": 5.145}, _crit([8.46, 8.59, 10.5, 10.58]))
    assert g["verdict"] == "mismatch"
    assert g["detail"]["first_principles_violation"] is False


def test_literature_just_outside_band_is_drift():
    # band [0.4,0.4]; within = ±10% -> [0.36,0.44]; drift = ±20% -> [0.32,0.48]
    assert grade_axis({"mean": 0.46}, _crit([0.40]))["verdict"] == "drift"


def test_literature_below_ceiling_but_below_band_is_unflagged_mismatch():
    g = grade_axis({"mean": 0.20}, _crit([0.355, 0.44], tmax=0.538))
    assert g["verdict"] == "mismatch"
    assert g["detail"]["first_principles_violation"] is False


def test_literature_no_reference_is_ungraded():
    assert grade_axis({"mean": 0.5}, _crit([]))["verdict"] == "ungraded"


def test_literature_takes_bare_scalar_or_node():
    c = _crit([0.4], tmax=0.538)
    assert grade_axis(0.42, c)["verdict"] == grade_axis({"mean": 0.42}, c)["verdict"]


# ---------------------------------------------------------------------------
# 2. the card (needs the installed ecoli-sources validation bundle + fixture)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graded():
    card, reference, model = rvl.build()
    return grade_card(card, reference), model


def test_card_overall_mismatch(graded):
    report, _ = graded
    assert report["overall"] == "mismatch"


def test_card_growth_rate_within_band(graded):
    # μ ≈ 0.83 1/h sits inside the measured 0.68–0.81 band (+tol) — the growth
    # rate is right; the defect is elsewhere.
    report, _ = graded
    assert report["axes"]["physiology.growth_rate"]["verdict"] in ("within_tol", "drift")


def test_card_glucose_uptake_mismatch(graded):
    # q_glc ≈ 5.1 mmol/gDW/h vs measured 8.5–10.6 — the model under-consumes glucose.
    report, _ = graded
    assert report["axes"]["physiology.glucose_uptake"]["verdict"] == "mismatch"


def test_card_biomass_yield_is_first_principles_violation(graded):
    # Yxs ≈ 0.89 gDW/g exceeds the 0.538 stoichiometric ceiling — the headline.
    report, _ = graded
    ax = report["axes"]["physiology.biomass_yield"]
    assert ax["verdict"] == "mismatch"
    assert ax["detail"]["first_principles_violation"] is True


def test_card_yield_is_direct_with_percell_spread(graded):
    # The direct mass-balance method gives yield a per-cell distribution (the
    # ratio-of-means did not) — so the yield axis can carry a sim violin.
    report, model = graded
    assert len(model.get("biomass_yield_cells", [])) >= 3
    assert report["axes"]["physiology.biomass_yield"]["measured"].get("values")


def test_model_carbon_conserves(graded):
    # The implied biomass carbon content is physically plausible -> mass IS
    # conserved; the yield violation is energetic (under-respiration), not
    # carbon creation.
    _, model = graded
    assert 0.40 < model["implied_biomass_C_gC_per_gDW"] < 0.55


# ---------------------------------------------------------------------------
# 3. Metabolism + Proteome sections (present when the baked fixtures exist)
# ---------------------------------------------------------------------------

def test_card_glycolysis_routing_is_correct(graded):
    # The differentiated finding: the model routes central carbon CORRECTLY —
    # its EMP/oxPPP/ED split matches Crown's 13C-MFA composition (low TV).
    report, _ = graded
    ax = report["axes"].get("metabolism.glycolysis_split")
    assert ax is not None and ax["verdict"] == "within_tol"
    assert ax["detail"]["tv"] < 0.05


def test_card_under_respires(graded):
    # ...while respiration FAILS: O2 and CO2 far below the measured bands.
    report, _ = graded
    assert report["axes"]["metabolism.o2_uptake"]["verdict"] == "mismatch"
    assert report["axes"]["metabolism.co2_evolution"]["verdict"] == "mismatch"


def test_card_no_overflow(graded):
    # No acetate overflow (model ~0 vs measured 3.9–7).
    report, _ = graded
    assert report["axes"]["metabolism.acetate_secretion"]["verdict"] == "mismatch"


def test_card_proteome_concordant_but_not_tight(graded):
    # Proteome correlates with Schmidt (r ~ 0.75, matching Eran's showcase) but
    # is not within the tight 0.9 band -> drift; graded by Pearson r, not
    # identity-R².
    report, _ = graded
    ax = report["axes"].get("proteome.abundance")
    assert ax is not None and ax["verdict"] in ("within_tol", "drift")
    assert ax["value"] > 0.7
