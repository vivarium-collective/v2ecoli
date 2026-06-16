"""Unit tests for the ``curve_response`` report-card criterion (card_criteria).

Grades a swept dose-response (e.g. acetate-vs-growth-rate) against Basan's
2-parameter overflow line — onset growth rate ``λac`` + ``slope`` — with a
reference-aware qualitative gate. Pure function; no simulation.

Reference = the Basan 2015 titratable-PtsG glucose series (Fig. 1 source data):
λ = [0.58, 0.64, 0.82, 0.95], Jac = [0, 0.02, 0.62, 2.06] (mM/OD600/h). With a
0.05 floor the supra-floor fit gives λac_ref ≈ 0.764 h⁻¹, slope_ref ≈ 11.1.
"""
import pytest

from v2ecoli.library.card_criteria import grade_axis, _fit_overflow_line

REF_X = [0.58, 0.64, 0.82, 0.95]
REF_Y = [0.0, 0.02, 0.62, 2.06]
FLOOR = 0.05


def _crit(**over):
    c = {"type": "curve_response", "ref_x": REF_X, "ref_y": REF_Y, "active_eps": FLOOR}
    c.update(over)
    return c


def test_reference_fit_recovers_basan_features():
    fit = _fit_overflow_line(REF_X, REF_Y, FLOOR)
    assert fit is not None
    assert fit["lac"] == pytest.approx(0.764, abs=0.01)
    assert fit["slope"] == pytest.approx(11.08, rel=0.02)
    assert fit["n_supra"] == 2


def test_match_is_within_tol():
    # Model reproduces the curve exactly → both features land → within_tol.
    res = grade_axis({"x": REF_X, "y": REF_Y}, _crit())
    assert res["verdict"] == "within_tol"
    assert res["value"]["lac"] == pytest.approx(0.764, abs=0.01)
    assert res["detail"]["onset_verdict"] == "within_tol"
    assert res["detail"]["slope_verdict"] == "within_tol"


def test_no_overflow_is_mismatch_via_gate():
    # Baseline-FBA-like: sweep spans past λac_ref but acetate never crosses the
    # floor → the model lacks the overflow phenomenon → mismatch (gate 2).
    res = grade_axis({"x": [0.60, 0.75, 0.85, 0.95], "y": [0.0, 0.0, 0.0, 0.0]}, _crit())
    assert res["verdict"] == "mismatch"
    assert res["detail"]["reason"] == "no overflow trend in model"


def test_shifted_onset_is_drift():
    # Overflow present but onset shifted ~0.19 h⁻¹ late (within the drift band
    # 0.1–0.2); slope close → worst-of = drift.
    res = grade_axis({"x": [0.85, 0.92, 1.0, 1.1], "y": [0.0, 0.0, 0.5, 1.5]}, _crit())
    assert res["detail"]["onset_verdict"] == "drift"
    assert res["verdict"] == "drift"


def test_badly_shifted_onset_is_mismatch():
    # Onset shifted > 0.2 h⁻¹ late → onset mismatch dominates.
    res = grade_axis({"x": [0.90, 0.98, 1.05, 1.15], "y": [0.0, 0.0, 0.5, 1.5]}, _crit())
    assert res["detail"]["onset_verdict"] == "mismatch"
    assert res["verdict"] == "mismatch"


def test_sweep_below_onset_is_ungraded():
    # Sweep never reaches λac_ref → can't assess onset you never crossed; this
    # gate (range) fires BEFORE the no-overflow gate.
    res = grade_axis({"x": [0.40, 0.50, 0.60, 0.70], "y": [0.0, 0.0, 0.0, 0.02]}, _crit())
    assert res["verdict"] == "ungraded"
    assert res["detail"]["reason"] == "sweep below reference onset"


def test_slope_not_graded_when_units_unresolved():
    # grade_slope=False (OD600→gDCW unresolved): onset is unit-invariant and
    # still grades; slope is reported but not graded.
    res = grade_axis({"x": REF_X, "y": REF_Y}, _crit(grade_slope=False))
    assert res["verdict"] == "within_tol"          # from onset alone
    assert res["detail"]["slope_verdict"] == "ungraded"
    assert res["detail"]["onset_verdict"] == "within_tol"


def test_missing_inputs_ungraded():
    assert grade_axis({"x": REF_X, "y": REF_Y}, {"type": "curve_response"})["verdict"] == "ungraded"
    assert grade_axis(None, _crit())["verdict"] == "ungraded"
