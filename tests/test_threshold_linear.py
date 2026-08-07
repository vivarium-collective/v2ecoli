"""Unit tests for the ``threshold_linear`` report-card criterion (card_criteria).

Grades a swept response curve **shape-first**: the response is ~0 below a
threshold, then rises ~linearly. Two PRIMARY readouts — linearity (R² of the
supra-threshold rise) and the rise slope (relative) — with a reference-aware
gate (absence of a rise above the reference threshold = mismatch). The threshold
position is SECONDARY: graded for the record but NOT gating. Pure function; no
simulation.

The reference used throughout is the criterion's worked example: the Vemuri 2006
MG1655 chemostat acetate-overflow curve, as a dimensionless acetate-carbon yield
vs glucose uptake rate (`Y_ac = 2·qAc/(6·GUR)`) — GUR up to ~7.9 mmol/gDW/h,
Y_ac 0 → 0.25. With a 0.005 yield floor the supra-floor fit gives threshold
GUR ≈ 4.61, slope ≈ 0.077, R² ≈ 1. Nothing in the criterion is specific to it.
"""
import pytest

from v2ecoli.library.card_criteria import grade_axis, _fit_threshold_linear

# Vemuri 2006 (via Zhuang SI) — GUR and the derived acetate-carbon yield.
REF_X = [0.684, 1.209, 2.166, 3.358, 4.712, 5.960, 7.890]   # driver, mmol/gDW/h
REF_Y = [0.0, 0.0, 0.0, 0.0, 0.009692, 0.101734, 0.253992]  # response (dimensionless)
FLOOR = 0.005


def _crit(**over):
    c = {"type": "threshold_linear", "ref_x": REF_X, "ref_y": REF_Y, "active_eps": FLOOR}
    c.update(over)
    return c


def test_reference_fit_recovers_curve():
    fit = _fit_threshold_linear(REF_X, REF_Y, FLOOR)
    assert fit is not None
    assert fit["onset"] == pytest.approx(4.61, abs=0.05)    # threshold in driver units
    assert fit["slope"] == pytest.approx(0.077, rel=0.05)   # dimensionless rise slope
    assert fit["r2"] == pytest.approx(1.0, abs=0.01)        # the rise is linear
    assert fit["n_supra"] == 3


def test_match_is_within_tol():
    # Model reproduces the curve exactly → linear rise + slope land → within_tol.
    res = grade_axis({"x": REF_X, "y": REF_Y}, _crit())
    assert res["verdict"] == "within_tol"
    assert res["detail"]["lin_verdict"] == "within_tol"
    assert res["detail"]["slope_verdict"] == "within_tol"


def test_no_rise_is_mismatch_via_gate():
    # Sweep spans past the reference threshold but the response never crosses the
    # floor → wrong shape (flat) → mismatch (gate 2). Absence of an expected
    # response is a mismatch, not a pass.
    res = grade_axis({"x": [3.0, 5.0, 6.5, 8.0], "y": [0.0, 0.0, 0.0, 0.0]}, _crit())
    assert res["verdict"] == "mismatch"
    assert res["detail"]["reason"] == "no rise in model (flat) — wrong shape"


def test_sweep_below_threshold_is_ungraded():
    # Sweep never reaches the reference threshold → can't assess a shape you never
    # entered; this range gate fires before the no-rise gate.
    res = grade_axis({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 0.0, 0.0, 0.001]}, _crit())
    assert res["verdict"] == "ungraded"
    assert res["detail"]["reason"] == "sweep below reference threshold"


def test_shifted_threshold_still_passes_on_shape():
    # Rise present, LINEAR, matching slope (~0.077), but the threshold sits at 6.0
    # vs the reference 4.61 — a relative shift of 1.39/4.61 ≈ 30%, i.e. past
    # onset_tol (25%) but inside onset_warn (50%) → drift. The threshold is
    # secondary, so it does NOT gate: shape carries the axis to within_tol.
    res = grade_axis({"x": [5.0, 6.0, 7.0, 8.0, 9.0],
                      "y": [0.0, 0.0, 0.077, 0.154, 0.231]}, _crit())
    assert res["verdict"] == "within_tol"
    assert res["detail"]["onset"] == pytest.approx(6.0, abs=0.05)
    assert res["detail"]["onset_verdict"] == "drift"      # off...
    assert res["detail"]["onset_gating"] is False         # ...but not gating
    assert res["detail"]["lin_verdict"] == "within_tol"


def test_threshold_band_is_relative_not_absolute():
    # The threshold band is relative to the reference threshold, so rescaling the
    # driver axis (here mmol/gDW/h -> umol/gDW/h) must leave every verdict alone.
    # An absolute band would silently reclassify the same curve.
    measured = {"x": [5.0, 6.0, 7.0, 8.0, 9.0], "y": [0.0, 0.0, 0.077, 0.154, 0.231]}
    base = grade_axis(measured, _crit())
    scaled = grade_axis(
        {"x": [v * 1000 for v in measured["x"]], "y": measured["y"]},
        _crit(ref_x=[v * 1000 for v in REF_X]),
    )
    assert scaled["verdict"] == base["verdict"]
    assert scaled["detail"]["onset_verdict"] == base["detail"]["onset_verdict"]
    assert scaled["detail"]["d_onset_rel"] == pytest.approx(base["detail"]["d_onset_rel"])
    assert scaled["detail"]["d_slope_rel"] == pytest.approx(base["detail"]["d_slope_rel"])


def test_slope_off_is_graded_down():
    # Linear rise (high R²) but the slope is ~4× the reference → slope drives the
    # verdict (the magnitude of the response is wrong), linearity still passes.
    res = grade_axis({"x": [5.0, 6.0, 7.0, 8.0], "y": [0.0, 0.3, 0.6, 0.9]}, _crit())
    assert res["detail"]["lin_verdict"] == "within_tol"
    assert res["detail"]["slope_verdict"] == "mismatch"
    assert res["verdict"] == "mismatch"


def test_nonlinear_rise_is_graded_down():
    # Rise present but it saturates (not linear) → linearity degrades the verdict
    # below within_tol (the 0→linear shape is the thing being graded).
    res = grade_axis({"x": [5.0, 6.0, 7.0, 8.0, 9.0],
                      "y": [0.0, 0.10, 0.15, 0.17, 0.18]}, _crit())
    assert res["detail"]["lin_verdict"] != "within_tol"
    assert res["verdict"] in ("drift", "mismatch")


def test_missing_inputs_ungraded():
    assert grade_axis({"x": REF_X, "y": REF_Y},
                      {"type": "threshold_linear"})["verdict"] == "ungraded"
    assert grade_axis(None, _crit())["verdict"] == "ungraded"
