"""Tests for the Beulig comparison harness (mbp-09 req-3 / mbp-05 core).

Unit tests cover the CSV batch-phase loader + unit conversion + axis/group
assembly with NO simulation. Integration tests (marked slow/sim) run the coupled
arms for a SMALL number of steps (smoke) and assert a well-formed
report_card_verdict.json lands at the exact card paths the studies reference, and
that the workspace report_card_axis evaluator RESOLVES (non-ungraded) for the
gradeable groups.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.beulig_comparison_harness as h

WS_ROOT = h.WS_ROOT


# --- unit: CSV loader + reference scalar + conversion -----------------------

def test_load_od_csv_batch_phase_only():
    """The OD CSV loader keeps only batch-phase (t<0) rows, averages reactor
    columns, and returns a sorted time series."""
    times, means = h.load_wide_csv_batch_phase(h.BEULIG_DIR / "reactor_OD_data.csv")
    assert times, "no batch-phase OD rows loaded"
    assert all(t < 0 for t in times), "fed-batch (t>=0) rows leaked into batch phase"
    assert times == sorted(times), "times not sorted ascending"
    assert all(m > 0 for m in means), "non-positive OD means"


def test_od_peak_to_gdw_matches_documented_batch_peak():
    """Beulig batch OD peak x 0.34 reproduces the study's documented ~9.6 g/L
    batch peak (OD ~28 x 0.34), exercising the OD->gDW conversion."""
    ref, meta = h._pub_od_gdw()
    assert ref is not None
    # OD peak ~28 per the study note; gDW peak ~9.5 g/L.
    assert 20 < meta["od_peak"] < 40, f"OD peak {meta['od_peak']} off expected ~28"
    assert ref == pytest.approx(meta["od_peak"] * h.OD_TO_GDW)
    assert 6.0 < ref < 14.0, f"gDW peak {ref} not near the documented ~9.6 g/L"


def test_batch_reference_scalar_modes():
    assert h.batch_reference_scalar([1.0, 5.0, 3.0], "peak") == 5.0
    assert h.batch_reference_scalar([1.0, 5.0, 3.0], "endpoint") == 3.0
    assert h.batch_reference_scalar([], "peak") is None


def test_gur_per_biomass_reference_is_plausible():
    """Published GUR per-biomass derived from process_summary is a plausible
    positive metabolic rate (mmol/(gDW.h))."""
    gur = h.load_process_summary_gur_per_biomass()
    assert gur is not None and gur > 0, "no GUR reference derived"


def test_build_axes_group_mapping_covers_all_five_groups():
    """The overlays map onto the canonical groups; growth/substrate/gas_transfer/
    summary carry a GRADED axis, byproducts/ph_control are present-but-ungraded."""
    sim = {"sim_biomass_gL": 0.02, "sim_gur": 1.0, "sim_otr": 0.5, "sim_ctr": 0.0}
    reference, card = h.build_axes(sim, tol_rel=0.25)
    from v2ecoli.library.report_card import grade_card, verdict_json
    verdict = verdict_json(grade_card(card, reference))
    groups = verdict["groups"]
    for g in ("growth", "substrate", "byproducts", "gas_transfer", "ph_control",
              "summary"):
        assert g in groups, f"group {g} missing"
        assert groups[g]["axes"], f"group {g} has no axes"
    # growth has a graded (non-ungraded) axis; byproducts is all ungraded.
    assert groups["growth"]["verdict"] != "ungraded"
    assert groups["summary"]["verdict"] == "within_tol"
    assert groups["byproducts"]["verdict"] == "ungraded"


# --- integration: run both arms (smoke) ------------------------------------

@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("arm,card_dir", [("wcm", "vs_beulig"),
                                          ("millard", "millard_vs_beulig")])
def test_arm_writes_wellformed_verdict_and_resolves(arm, card_dir):
    """Running an arm for a few steps writes a well-formed report_card_verdict.json
    at the study-referenced card path, with the 5 study groups present and each
    group's axes non-empty; and the workspace evaluator returns a non-ungraded
    result for at least one group."""
    verdict = h.run_harness(arm, steps=2, render=False)

    vpath = h.CARD_ROOT / card_dir / "report_card_verdict.json"
    assert vpath.is_file(), f"verdict not written at {vpath}"
    on_disk = json.loads(vpath.read_text())
    assert on_disk["schema"] == "report_card_verdict/v1"
    groups = on_disk["groups"]
    for g in ("growth", "substrate", "byproducts", "gas_transfer", "summary"):
        assert g in groups, f"group {g} missing from {arm} card"
        assert groups[g]["axes"], f"group {g} empty in {arm} card"

    # The workspace report_card_axis evaluator resolves (non-ungraded) for growth.
    from pbg_v2ecoli.evaluators import evaluate_report_card_group
    test = {"measure": {"kind": "report_card_axis",
                        "card": f"docs/report_cards/beulig_batch/{card_dir}",
                        "group": "growth"}}
    res = evaluate_report_card_group(test, None, str(WS_ROOT))
    assert res["result"] in ("PASS", "FAIL"), (
        f"growth group did not resolve: {res}")


@pytest.mark.sim
@pytest.mark.slow
def test_report_card_axis_tests_resolve_via_study_evaluator():
    """The mbp-05 + mbp-09 report_card_axis tests now RESOLVE (not 'ungraded')
    against the produced cards, through the public study_evaluator seam."""
    # Ensure both cards exist (cheap re-run; smoke).
    h.run_harness("wcm", steps=2, render=False)
    h.run_harness("millard", steps=2, render=False)

    from pbg_superpowers.study_evaluator import evaluate_test
    cases = [
        ("docs/report_cards/beulig_batch/vs_beulig", "growth"),       # mbp-05
        ("docs/report_cards/beulig_batch/vs_beulig", "gas_transfer"),
        ("docs/report_cards/beulig_batch/millard_vs_beulig", "growth"),  # mbp-09
        ("docs/report_cards/beulig_batch/millard_vs_beulig", "summary"),
    ]
    for card, group in cases:
        test = {"measure": {"kind": "report_card_axis", "card": card, "group": group},
                "pass_if": {"op": "report_card_group_within_tol"}}
        res = evaluate_test(test, None, str(WS_ROOT))
        assert res.get("result") in ("PASS", "FAIL"), (
            f"{card} group={group} did not resolve: {res}")
