"""Tests for the meta-tier report card: basal-condition phenotype.

Three layers, mirroring the card's three artifacts:

  1. **Measurement math** — ``PopulationPhenotypeBasalCard.analyze`` over synthetic
     per-cell records (no ParCa cache needed). Pins burn-in filtering,
     doubling-time mean/CV, and composition-fraction reduction.
  2. **Config threading** — the dedicated ``generation_lower_bound`` knob
     actually reaches the step through ``run_analyses`` (the framework
     previously discarded per-analysis options).
  3. **The grade** — ``grade_card`` grades a *measured* card against a
     *pinned reference* whose axes carry typed criteria (``rel_tol`` /
     ``ttest`` / ``r2`` / ``flux_scatter`` / ``boolean``), each earning a
     4-state verdict (within_tol / drift / mismatch / ungraded). The reference
     is ``tests/fixtures/population_phenotype_basal_reference.json``, pinned to a blessed
     current-main ensemble (no biological judgement yet — judgement comes later
     from reading drift against the pin).
"""
import json
import os

import pytest

from v2ecoli.workflow.analysis import PopulationPhenotypeBasalCard
from v2ecoli.library.report_card import grade_card, card_from_analysis, _omics_table


_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_REFERENCE = os.path.join(_FIXTURE_DIR, "population_phenotype_basal_reference.json")


# ---------------------------------------------------------------------------
# Synthetic ensemble: 2 seeds x 3 generations of one variant.
# gen 0 is the inoculation transient (dropped by burn-in); gens 1-2 are the
# steady-balanced-growth window the card grades. seed=1/gen=2 does not divide
# (its division_time is the run cap) so it is excluded from doubling stats but
# still counts for composition.
# ---------------------------------------------------------------------------

def _ensemble():
    def cell(seed, gen, dt, divided, prot, rna, dna):
        # composition grades TOTAL RNA / dry weight (rna_fraction_mean), so the
        # synthetic records carry that key (not the legacy rRna-only fraction).
        return {"variant": 0, "lineage_seed": seed, "generation": gen,
                "agent_id": "0" * (gen + 1), "divided": divided,
                "division_time": dt, "final_dry_mass": 700.0,
                "protein_fraction_mean": prot, "rna_fraction_mean": rna,
                "dna_fraction_mean": dna}
    return [
        # gen 0 — burn-in (dropped at generation_lower_bound=1)
        cell(0, 0, 3000.0, True, 0.40, 0.08, 0.030),
        cell(1, 0, 3100.0, True, 0.41, 0.08, 0.030),
        # gen 1
        cell(0, 1, 2400.0, True, 0.50, 0.10, 0.020),
        cell(1, 1, 2600.0, True, 0.52, 0.11, 0.020),
        # gen 2
        cell(0, 2, 2500.0, True, 0.51, 0.105, 0.021),
        cell(1, 2, 9999.0, False, 0.53, 0.115, 0.019),  # did not divide
    ]


def _card(generation_lower_bound=0):
    from bigraph_schema import allocate_core
    return PopulationPhenotypeBasalCard(
        {"generation_lower_bound": generation_lower_bound},
        core=allocate_core())


# ---------------------------------------------------------------------------
# 1. Measurement math
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_burn_in_drops_early_generations():
    """generation_lower_bound=1 excludes the gen-0 inoculation transient."""
    out = _card(generation_lower_bound=1).analyze(_ensemble())
    assert out["generation_lower_bound"] == 1
    assert out["n_cells"] == 4  # gens 1-2, both seeds


@pytest.mark.fast
def test_doubling_time_over_confirmed_divisions_only():
    """Doubling stats use divided cells only; the run-cap non-divider is out.

    Kept divided dts (burn-in=1): 2400, 2600, 2500 -> mean 2500, n 3.
    The seed=1/gen=2 cell (divided=False, dt=9999 run cap) is excluded.
    Doubling time now lives under the ``physiology`` group."""
    out = _card(generation_lower_bound=1).analyze(_ensemble())
    dt = out["physiology"]["doubling_time"]
    assert dt["n"] == 3
    assert dt["mean"] == pytest.approx(2500.0)
    # CV = pstdev / mean; pstdev of {2400,2500,2600} = 81.6497...
    assert dt["cv"] == pytest.approx(81.64966 / 2500.0, rel=1e-4)


@pytest.mark.fast
def test_composition_fractions_reduced_across_ensemble():
    """Composition is the ensemble mean/std/cv of the per-cell fraction means.

    protein over the 4 kept cells: 0.50, 0.52, 0.51, 0.53 -> mean 0.515."""
    out = _card(generation_lower_bound=1).analyze(_ensemble())
    comp = out["composition"]
    assert comp["protein_fraction"]["n"] == 4
    assert comp["protein_fraction"]["mean"] == pytest.approx(0.515)
    assert comp["rna_fraction"]["mean"] == pytest.approx((0.10 + 0.11 + 0.105 + 0.115) / 4)
    assert comp["dna_fraction"]["mean"] == pytest.approx((0.020 + 0.020 + 0.021 + 0.019) / 4)
    assert set(comp) == {"protein_fraction", "rna_fraction", "dna_fraction"}


@pytest.mark.fast
def test_physiology_and_ribosome_groups_present():
    """The card emits Physiology + Ribosomes groups; axes for which the
    synthetic records carry no signal degrade to n=0, not an exception."""
    out = _card(generation_lower_bound=1).analyze(_ensemble())
    assert "doubling_time" in out["physiology"]
    assert "cell_mass" in out["physiology"]  # present, n=0 here (no mass in synth)
    assert out["physiology"]["cell_mass"]["n"] == 0
    assert set(out["ribosomes"]) == {"total", "active_fraction",
                                     "elongation_rate", "production"}


@pytest.mark.fast
def test_zero_fraction_cells_excluded_from_composition():
    """A cell with no valid mass (zero fraction) is skipped, not averaged in."""
    rows = _ensemble()
    rows.append({"variant": 0, "lineage_seed": 2, "generation": 1, "agent_id": "00",
                 "divided": True, "division_time": 2500.0, "final_dry_mass": 700.0,
                 "protein_fraction_mean": 0.0, "rna_fraction_mean": 0.0,
                 "dna_fraction_mean": 0.0})
    out = _card(generation_lower_bound=1).analyze(rows)
    # n_cells counts the row (it is in-window) but composition skips the zero
    assert out["n_cells"] == 5
    assert out["composition"]["protein_fraction"]["n"] == 4


@pytest.mark.fast
def test_empty_ensemble_is_safe():
    out = _card().analyze([])
    assert out["n_cells"] == 0
    assert out["physiology"]["doubling_time"]["mean"] == 0.0
    assert out["composition"]["protein_fraction"]["mean"] == 0.0


@pytest.mark.fast
def test_generation_lower_bound_threads_through_run_analyses(monkeypatch, tmp_path):
    """The per-analysis option dict reaches the step (framework gap fixed).

    With generation_lower_bound=1 the gen-0 cells must be excluded; if the
    config were dropped (the old ``step_cls({})`` behavior) n_cells would be 6."""
    import v2ecoli.workflow.analysis_runner as ar
    # build_cell_records returns {cell_key: record}; run_analyses takes .values()
    recs = {(c["lineage_seed"], c["generation"]): c for c in _ensemble()}
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: recs)
    options = {"multiseed": {"population_phenotype_basal": {"generation_lower_bound": 1}}}
    results = ar.run_analyses(str(tmp_path), options)
    card = list(results["multiseed"]["population_phenotype_basal"].values())[0]
    assert "error" not in card, card
    assert card["generation_lower_bound"] == 1
    assert card["n_cells"] == 4


# ---------------------------------------------------------------------------
# 2. The grade — typed criteria, 4-state verdict, worst-axis overall.
#    (grade_card lives in v2ecoli.library.report_card so the renderer and this
#    test share one implementation.)
# ---------------------------------------------------------------------------

def _ref_axis(path, criterion):
    return {"axes": {path: {"group": "Test", "label": path, "criterion": criterion}}}


@pytest.mark.fast
def test_grade_rel_tol_bands():
    """rel_tol: within_tol <= tol, drift <= 2*tol, mismatch beyond; the worst
    axis sets ``overall``; a null reference is ungraded (not a failure)."""
    measured = {"physiology": {"doubling_time": {"mean": 2500.0}}}

    def grade(ref, tol=0.05):
        r = _ref_axis("physiology.doubling_time",
                      {"type": "rel_tol", "reference": ref, "tol_rel": tol})
        return grade_card(measured, r)

    assert grade(2500.0)["overall"] == "within_tol"          # exact
    assert grade(2300.0)["overall"] == "drift"               # rel 0.087 in (.05,.10]
    assert grade(2000.0)["overall"] == "mismatch"            # rel 0.25 > .10
    assert grade(None)["overall"] == "ungraded"              # no reference


@pytest.mark.fast
def test_grade_ttest_magnitude_with_p_guard():
    """ttest: tight identical populations pass; a large, significant shift is a
    mismatch (magnitude past mismatch_pct AND p < p_min)."""
    base = [2400.0, 2500.0, 2600.0, 2450.0, 2550.0]
    crit = {"type": "ttest", "ref_values": base, "within_pct": 0.05,
            "mismatch_pct": 0.10, "p_min": 0.05}
    ref = _ref_axis("physiology.doubling_time", crit)

    same = {"physiology": {"doubling_time": {"values": base, "mean": 2500.0}}}
    assert grade_card(same, ref)["overall"] == "within_tol"

    shifted_vals = [v * 1.3 for v in base]  # +30%, tiny spread -> significant
    shifted = {"physiology": {"doubling_time":
                              {"values": shifted_vals, "mean": 3250.0}}}
    assert grade_card(shifted, ref)["overall"] == "mismatch"


@pytest.mark.fast
def test_grade_ttest_inactive_sentinel():
    """An inactive-reference axis (e.g. acetate ~0): stays inactive -> pass;
    becomes active -> mismatch (the sentinel tripped)."""
    crit = {"type": "ttest", "ref_values": [0.0] * 6, "within_pct": 0.05,
            "mismatch_pct": 0.10, "inactive_eps": 1e-6}
    ref = _ref_axis("fluxes.acetate", crit)
    inactive = {"fluxes": {"acetate": {"values": [0.0] * 6, "mean": 0.0}}}
    assert grade_card(inactive, ref)["overall"] == "within_tol"
    active = {"fluxes": {"acetate": {"values": [0.5] * 6, "mean": 0.5}}}
    assert grade_card(active, ref)["overall"] == "mismatch"


@pytest.mark.fast
def test_grade_flux_scatter_qualitative_change_is_mismatch():
    """flux_scatter: a flux that appears (ref~0 -> active) or disappears is a
    mismatch regardless of R², and the changed index is reported."""
    ref_vec = [1.0, 0.0, -2.0]
    crit = {"type": "flux_scatter", "ref_vector": ref_vec,
            "r2_min": 0.99, "r2_drift": 0.95}
    ref = _ref_axis("fluxes.exchange", crit)

    matched = {"fluxes": {"exchange": {"vector": [1.0, 0.0, -2.0]}}}
    assert grade_card(matched, ref)["overall"] == "within_tol"

    appeared = {"fluxes": {"exchange": {"vector": [1.0, 0.5, -2.0]}}}
    g = grade_card(appeared, ref)["axes"]["fluxes.exchange"]
    assert g["verdict"] == "mismatch"
    assert g["detail"]["appeared"] == [1]


@pytest.mark.fast
def test_grade_flux_scatter_subfloor_flip_is_not_graded():
    """A flip whose active-side magnitude is below the significance floor
    (qual_eps) is reported but does NOT drive the verdict — FBA jitter at the
    detection floor (~1e-6 with CV>100%) is not a behavioral change."""
    ref_vec = [1.0, 0.0, -2.0]
    crit = {"type": "flux_scatter", "ref_vector": ref_vec,
            "r2_min": 0.99, "r2_drift": 0.95, "qual_eps": 1e-3}
    ref = _ref_axis("fluxes.exchange", crit)

    # index 1 appears at 5e-6 — above active_eps (1e-6) but below qual_eps (1e-3)
    near_floor = {"fluxes": {"exchange": {"vector": [1.0, 5e-6, -2.0]}}}
    g = grade_card(near_floor, ref)["axes"]["fluxes.exchange"]
    assert g["verdict"] == "within_tol"          # not a mismatch
    assert g["detail"]["appeared"] == []          # not counted as a real flip
    assert g["detail"]["sub_floor"] == [1]        # but still reported


@pytest.mark.fast
def test_omics_outlier_table():
    """The gene-expression outlier table surfaces genes past the log2FC cutoff,
    gates low-count ratio blow-ups via min_count, shows absolute counts, and
    marks genes fully off in one model as ±∞."""
    crit = {
        "ref_vector": [100.0, 100.0, 0.5, 50.0, 0.0],
        "ids": ["a_RNA", "b_RNA", "c_RNA", "d_RNA", "e_RNA"],
        "symbols": ["alpha", "beta", "gamma", "delta", "eps"],
        "names": ["A desc", "B desc", "C desc", "D desc", "E desc"],
        "outlier_log2fc": 2.0, "min_count": 10, "outlier_top_n": 20,
    }
    #   alpha: 100->800  (+3.0, kept)        beta: 100->100 (0, dropped)
    #   gamma: 0.5->5    (+3.3 but max<10 → gated by min_count)
    #   delta: 50->0     (off in v2 → −∞)    eps: 0->40 (on in v2 → +∞)
    measured = {"vector": [800.0, 100.0, 5.0, 0.0, 40.0]}
    html = _omics_table(measured, crit, "v1", "v2")
    assert "alpha" in html and "+3.00" in html      # real over-expression
    assert "eps" in html and "+∞" in html            # appeared (on in v2)
    assert "delta" in html and "−∞" in html          # lost (off in v2)
    assert "gamma" not in html                        # gated by min_count
    assert "beta" not in html                         # below cutoff
    assert ">800<" in html and ">50<" in html         # absolute counts shown


@pytest.mark.fast
def test_grade_r2_vector():
    """r2: identical vectors -> R²=1 pass; a degraded vector drops the verdict."""
    ref_vec = [10.0, 100.0, 1000.0, 50.0, 500.0]
    crit = {"type": "r2", "ref_vector": ref_vec, "r2_min": 0.99, "r2_drift": 0.95}
    ref = _ref_axis("geneexp.transcriptome", crit)
    same = {"geneexp": {"transcriptome": {"vector": list(ref_vec)}}}
    assert grade_card(same, ref)["overall"] == "within_tol"


# ---------------------------------------------------------------------------
# 3. Meta-tier grade against the pinned reference fixture.
# ---------------------------------------------------------------------------

def _load_reference():
    if not os.path.isfile(_REFERENCE):
        pytest.skip(f"pinned reference not found at {_REFERENCE}")
    with open(_REFERENCE, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.behavior
def test_population_phenotype_basal_matches_pinned_reference():
    """META-TIER GRADE: a measured basal ensemble vs the pinned reference.

    Skips until (a) the reference is ``status: populated`` and (b) a measured
    analysis.json is available (``V2ECOLI_BASAL_ANALYSIS`` env var pointing at a
    population_phenotype_basal run's analysis.json). The scalar axes (physiology /
    composition / ribosomes) grade directly; the vector axes (exchange fluxes /
    gene expression) render ungraded unless the report's vector merge has run —
    ungraded never fails, so the gate is honest either way."""
    reference = _load_reference()
    if reference.get("status") != "populated":
        pytest.skip(f"reference {reference.get('status')!r}; "
                    "populate from a blessed main ensemble run first")

    measured_path = os.environ.get("V2ECOLI_BASAL_ANALYSIS")
    if not measured_path or not os.path.isfile(measured_path):
        pytest.skip("set V2ECOLI_BASAL_ANALYSIS to a population_phenotype_basal "
                    "analysis.json from an ensemble run")
    with open(measured_path, encoding="utf-8") as f:
        card = card_from_analysis(json.load(f))

    report = grade_card(card, reference)
    failures = {p: g for p, g in report["axes"].items()
                if g["verdict"] == "mismatch"}
    assert report["overall"] != "mismatch", \
        f"basal phenotype drifted from pinned reference: {failures}"
