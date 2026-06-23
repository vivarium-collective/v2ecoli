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


# ---------------------------------------------------------------------------
# 4. Central-carbon flux scatter (signed reaction-set matcher vs Crown 2015)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cc_reactions():
    import json
    met = json.load(open(rvl._MET_JSON, encoding="utf-8"))
    cc = met["central_carbon"]
    assert cc["normalized_to_glucose_100"] is True
    return {r["label"]: r for r in cc["reactions"]}


def test_central_carbon_scatter_graded_mismatch(graded):
    # The vector grades mismatch: routing matches but the TCA collapse +
    # reductive lower branch sit far off the identity line.
    report, _ = graded
    ax = report["axes"].get("metabolism.central_carbon_flux")
    assert ax is not None and ax["verdict"] == "mismatch"


def test_central_carbon_clean_reactions_on_identity(cc_reactions):
    # Glycolysis + oxidative-PPP routing matches Crown — these sit on the
    # identity line (carbon routing into central metabolism is right).
    for lbl, lo, hi in [("Pgi", 60, 80), ("GAPDH", 130, 160), ("Eno", 130, 160),
                        ("Zwf", 20, 30), ("6PGDH", 20, 30)]:
        assert lo < cc_reactions[lbl]["model"] < hi, lbl


def test_central_carbon_lower_tca_is_reductive_reverse(cc_reactions):
    # The model runs the lower TCA backwards (OAC->Mal->Fum): Fum/MDH carry a
    # NEGATIVE flux vs Crown's positive, and are flagged reductive_reverse. This
    # is the signed-matcher payoff — a magnitude comparison would hide it.
    for lbl in ("Fum", "MDH"):
        r = cc_reactions[lbl]
        assert r["model"] < 0, lbl
        assert r["flag"] == "reductive_reverse"


def test_central_carbon_oxidative_tca_collapsed(cc_reactions):
    # PDH well below Crown's ~114, and aKGDH / malate synthase ~0 — the
    # oxidative cycle isn't turning (the under-respiration at reaction level).
    assert cc_reactions["PDH"]["model"] < 20
    assert abs(cc_reactions["aKGDH"]["model"]) < 1
    assert abs(cc_reactions["MS"]["model"]) < 1


def test_central_carbon_flags(cc_reactions):
    # The annotation flags the matcher carries through to the scatter.
    assert cc_reactions["Pyk"]["flag"] == "pts_coupled"
    assert cc_reactions["Pfk"]["flag"] == "aldolase_bypass"
    assert cc_reactions["Fba"]["flag"] == "aldolase_bypass"


def test_central_carbon_glucose_entry_excluded(cc_reactions):
    # Glucose entry (Crown's PTS, crown_f1) is ¹³C-non-identifiable vs the
    # model's glucokinase+PYK route -> intentionally omitted from the set.
    assert all(r["crown_fid"] != "crown_f1" for r in cc_reactions.values())


def test_central_carbon_ed_branch_tracked_but_off(cc_reactions):
    # The Entner-Doudoroff branch is tracked explicitly (EDD/EDA) even though
    # it's off in the model (~0) — visible-but-negligible, not dropped.
    r = cc_reactions["EDD / EDA"]
    assert abs(r["model"]) < 1 and r["group"] == "ED"


# ---------------------------------------------------------------------------
# 5. TCA branch-point fate nodes (isocitrate, acetyl-CoA) — composition bars
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fate_nodes():
    import json
    return json.load(open(rvl._MET_JSON, encoding="utf-8"))["nodes"]


def test_isocitrate_node_oxidative_dominant(graded, fate_nodes):
    # ICit routes ~96% oxidative (ICDH→α-KG) vs Crown 87% — a modest drift.
    report, _ = graded
    ax = report["axes"].get("metabolism.isocitrate_split")
    assert ax is not None and ax["verdict"] in ("within_tol", "drift")
    assert fate_nodes["isocitrate"]["fractions"]["oxidative_TCA"] > 0.85


def test_accoa_node_no_overflow_is_mismatch(graded, fate_nodes):
    # AcCoA fate: the model sends ~78% to biosynthesis with NO acetate overflow,
    # vs Crown's 59% acetate — the AcCoA-node face of the overflow defect.
    report, _ = graded
    ax = report["axes"].get("metabolism.accoa_split")
    assert ax is not None and ax["verdict"] == "mismatch"
    fr = fate_nodes["accoa"]["fractions"]
    assert fr["acetate"] < 0.02          # no overflow
    assert fr["biosynthesis"] > 0.6      # carbon dumped into biomass instead


# ---------------------------------------------------------------------------
# 6. Macromolecular composition (% dry weight) vs Bremer & Dennis 2008
# ---------------------------------------------------------------------------

def test_composition_reference_interpolates_to_model_td():
    # B&D is a growth-rate series; the reference is interpolated to the model's
    # doubling time, and the four fractions are a composition (sum ~ 1).
    import json
    comp = json.load(open(rvl._COMP_JSON, encoding="utf-8"))
    ref = rvl.composition_reference(comp["doubling_time_min"])
    assert 0.50 < ref["protein"] < 0.60   # ~55% at td ~52 min
    assert 0.10 < ref["rna"] < 0.15
    assert abs(sum(ref.values()) - 1.0) < 1e-6


def test_composition_rna_matches_protein_low_pool_high(graded):
    # The finding: RNA fraction matches B&D, but the model under-represents
    # protein and over-represents the small-molecule pool ('other') — graded
    # drift. An independent reference (not ParCa fit-to).
    report, _ = graded
    ax = report["axes"].get("composition.macromolecular")
    assert ax is not None and ax["verdict"] in ("within_tol", "drift")
    import json
    fr = json.load(open(rvl._COMP_JSON, encoding="utf-8"))["fractions"]
    assert fr["protein"] < 0.50           # model protein low vs B&D ~0.55
    assert fr["other"] > 0.35             # small-molecule pool inflated vs B&D ~0.30


def test_composition_rna_is_total_rna(graded):
    # RNA is total RNA (rRNA+tRNA+mRNA); its fraction (~0.13) is in the B&D band,
    # not mRNA-only (which would be ~0.006 and a ~20x mismatch).
    import json
    fr = json.load(open(rvl._COMP_JSON, encoding="utf-8"))["fractions"]
    assert 0.10 < fr["rna"] < 0.15


# ---------------------------------------------------------------------------
# 7. Metabolite pools (aggregate) vs Bennett 2009 — fit-to consistency
# ---------------------------------------------------------------------------

def test_metabolite_pool_total_under_bennett(graded):
    # The model's realized aggregate metabolite pool is ~0.54x its Bennett-derived
    # target total — it under-fills the metabolome in aggregate. A fit-to
    # consistency check (the model is calibrated to these targets), graded mismatch.
    report, _ = graded
    ax = report["axes"].get("pools.total")
    assert ax is not None and ax["verdict"] == "mismatch"
    import json
    pools = json.load(open(rvl._POOLS_JSON, encoding="utf-8"))
    assert 0.4 < pools["ratio"] < 0.7          # model under-fills (~0.54x)
    assert pools["n_matched"] > 50             # a substantial mapped set


def test_metabolite_pool_concentrations_plausible(graded):
    # Sanity on units (mol/L via bulk count / volume[fL]·Na): the per-metabolite
    # realized concentrations are in a physiological mM range, not off by 1e3.
    import json
    per = json.load(open(rvl._POOLS_JSON, encoding="utf-8"))["per_metabolite"]
    # glutamate (GLT) is the abundant pool — model ~tens of mM (Bennett 96 mM)
    glt = per.get("GLT")
    assert glt is not None and 0.005 < glt["model"] < 0.10   # 5-100 mM


# ---------------------------------------------------------------------------
# 8. Metabolite pools (per-metabolite scatter) vs Bennett 2009 — fit-to
# ---------------------------------------------------------------------------

def test_metabolite_per_metabolite_concordant(graded):
    # Per-metabolite model vs the genuine Bennett concentrations, log-log Pearson
    # r. A fit-to consistency check (the model is calibrated to these targets), so
    # it tracks them across ~5 orders of magnitude (r ~ 0.93) -> within_tol/drift.
    report, _ = graded
    ax = report["axes"].get("pools.per_metabolite")
    assert ax is not None and ax["verdict"] in ("within_tol", "drift")
    assert ax["value"] > 0.85


def test_metabolite_scatter_uses_genuine_bennett_with_ci():
    # The reference side is the genuine Bennett slot value + 95% CI (joined by the
    # slot's ecocyc_id), not the rounded reconstruction column: a substantial
    # mapped set, asymmetric CI error bars paired to each point, and glutamate's
    # published 0.096 mol/L present in the reference vector.
    _, reference, _ = rvl.build()
    crit = reference["axes"]["pools.per_metabolite"]["criterion"]
    n = len(crit["ref_vector"])
    assert n > 50
    assert len(crit["ref_err"]) == 2 and len(crit["ref_err"][0]) == n
    assert len(crit["ids"]) == n                         # one label per point
    assert any(abs(v - 0.096) < 1e-4 for v in crit["ref_vector"])  # GLT genuine


def test_bennett_pools_mapping_coverage():
    # The slot's ecocyc_id column resolves to the model ids: every metabolite the
    # model realizes (the baked per_metabolite set) has a genuine Bennett value.
    import json
    per = set(json.load(open(rvl._POOLS_JSON, encoding="utf-8"))["per_metabolite"])
    ref = rvl.bennett_pools()
    assert per <= set(ref)                              # full coverage of the model set
    assert ref["GLT"]["name"].lower().startswith("glutamate")
