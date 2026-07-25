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


# --- unit: multi-gen production derivation (no sim) -------------------------

def test_cell_mass_value_fg_coercions():
    """The read-back cell_mass coercer handles bare float (fg magnitude), the
    serialized-quantity dict form, and None."""
    assert h._cell_mass_value_fg(123.0) == 123.0
    assert h._cell_mass_value_fg({"magnitude": 55.0, "units": "fg"}) == 55.0
    assert h._cell_mass_value_fg({"value": 7.0}) == 7.0
    assert h._cell_mass_value_fg(None) is None


def test_derive_sim_scalars_matches_inline_derivation():
    """_derive_sim_scalars reduces per-step series to the graded peaks, with
    dt_s inferred from the global_time progression (so the GUR per-hour scale is
    correct). This is the SHARED reducer for the inline smoke + multigen paths."""
    # 3 ticks at dt=1s. Glucose uptake (negative counts) + a cell mass in fg;
    # transport deltas climb then the peak is taken.
    biomass_gL = [0.0, 0.01, 0.02]
    glc_counts = [0.0, -1.0e12, -2.0e12]   # counts/step, <0 = uptake
    cell_mass = [300.0, 300.0, 300.0]      # fg
    o2_delta = [0.0, 0.5, 0.3]             # mg/L per 1s transport interval
    co2_delta = [0.0, -0.4, -0.2]
    gtime = [1.0, 2.0, 3.0]
    sim = h._derive_sim_scalars(
        "reactor_bird_coupled_millard", 3, 1.0,
        biomass_gL, glc_counts, cell_mass, o2_delta, co2_delta, gtime)

    assert sim["composite"] == "reactor_bird_coupled_millard"
    assert sim["dt_s"] == 1.0
    assert sim["sim_biomass_gL"] == 0.02
    # GUR peak: from the -2e12 counts at 300 fg, per hour.
    expected_gur = (2.0e12 / h.AVOGADRO * 1000.0 * (h.SECONDS_PER_HOUR / 1.0)) / (
        300.0 * h.FG_PER_GRAM)
    assert sim["sim_gur"] == pytest.approx(expected_gur)
    # OTR peak from the +0.5 mg/L delta; CTR peak from |−0.4|.
    assert sim["sim_otr"] == pytest.approx(0.5 / h.MW_O2 * h.SECONDS_PER_HOUR)
    assert sim["sim_ctr"] == pytest.approx(0.4 / h.MW_CO2 * h.SECONDS_PER_HOUR)
    # The derived scalars flow straight into the existing axis builder.
    reference, card = h.build_axes(sim, tol_rel=0.25)
    assert card["growth"]["biomass_gDW_per_L"]["mean"] == 0.02

    # dt_override forces the per-tick basis (multigen samples every `chunk`
    # ticks but the counts are per-tick): GUR scales by SECONDS_PER_HOUR/dt.
    sim_dt = h._derive_sim_scalars(
        "reactor_bird_coupled", 3, 1.0,
        biomass_gL, glc_counts, cell_mass, o2_delta, co2_delta,
        [60.0, 120.0, 180.0], dt_override=1.0)
    assert sim_dt["dt_s"] == 1.0
    assert sim_dt["sim_gur"] == pytest.approx(expected_gur)


def test_multigen_path_constants_cover_harness_observables():
    """The production runner's emit-path sets cover exactly the stores the
    inline path reads: agent cell_mass + glucose exchange, root population +
    reactor transport/volume."""
    assert "listeners/mass/cell_mass" in h.MULTIGEN_AGENT_PATHS
    assert "environment/exchange/GLC" in h.MULTIGEN_AGENT_PATHS
    assert "population/biomass_concentration_gL" in h.MULTIGEN_ROOT_PATHS
    assert "reactor/o2_transport_delta" in h.MULTIGEN_ROOT_PATHS
    assert "reactor/co2_transport_delta" in h.MULTIGEN_ROOT_PATHS
    assert "reactor/volume_L" in h.MULTIGEN_ROOT_PATHS


# --- unit: iML1515 arm (predicted-vs-measured μ, graded via r2) -------------

@pytest.mark.sim
@pytest.mark.slow
def test_iml1515_arm_grades_growth_via_r2():
    """The iml1515 arm loads the Beulig batch-phase samples, runs iML1515 FBA at
    the measured uptake bounds, and grades growth (predicted-vs-measured μ) via
    an r2 criterion — the growth group resolves (NON-ungraded). Inputs the FBA
    BOUNDS are set from (glucose / O2 uptake) are honestly ungraded, and the
    summary boolean records that iML1515 ran."""
    pairs = h.run_iml1515_arm(max_samples=8)
    assert pairs, "no iML1515 pairs produced"
    assert all("mu_measured" in p and "mu_predicted" in p for p in pairs)

    reference, card = h.build_iml1515_axes(pairs)
    from v2ecoli.library.report_card import grade_card, verdict_json
    verdict = verdict_json(grade_card(card, reference))
    groups = verdict["groups"]

    # growth is GRADED (r2), not ungraded.
    assert "growth" in groups, "growth group missing"
    assert groups["growth"]["verdict"] != "ungraded", (
        f"growth not graded: {groups['growth']}")
    assert groups["growth"]["axes"][0]["meter"].startswith("R²")
    # inputs the bounds are set from must NOT be graded as predictions.
    assert groups["substrate"]["verdict"] == "ungraded"
    assert groups["gas_transfer"]["verdict"] == "ungraded"
    # iML1515 ran -> summary passes.
    assert groups["summary"]["verdict"] == "within_tol"


@pytest.mark.sim
@pytest.mark.slow
def test_iml1515_arm_writes_verdict_and_resolves():
    """run_harness('iml1515') writes report_card_verdict.json at the card path,
    and the workspace evaluator resolves (PASS/FAIL, not ungraded) for growth."""
    verdict = h.run_harness("iml1515", max_samples=8, render=False)
    vpath = h.CARD_ROOT / "iml1515_vs_beulig" / "report_card_verdict.json"
    assert vpath.is_file(), f"verdict not written at {vpath}"
    on_disk = json.loads(vpath.read_text())
    assert on_disk["schema"] == "report_card_verdict/v1"
    assert "iML1515" in (on_disk.get("model_ref") or "")
    assert "not apples-to-apples" in (on_disk.get("scope") or "").lower()

    from pbg_v2ecoli.evaluators import evaluate_report_card_group
    test = {"measure": {"kind": "report_card_axis",
                        "card": "docs/report_cards/beulig_batch/iml1515_vs_beulig",
                        "group": "growth"}}
    res = evaluate_report_card_group(test, None, str(WS_ROOT))
    assert res["result"] in ("PASS", "FAIL"), (
        f"growth group did not resolve: {res}")


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

    from viva_superpowers.study_evaluator import evaluate_test
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
