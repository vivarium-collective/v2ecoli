"""Unit tests for the ``curve_response`` report-card criterion (card_criteria).

Grades a swept overflow response **shape-first**, as a dimensionless yield curve
(acetate-carbon yield Y_ac vs the carbon-influx / growth driver): the yield is ~0
below an onset, then rises ~linearly. Two PRIMARY, strain-robust readouts —
linearity (R² of the supra-onset rise) and the yield slope (relative) — with a
reference-aware gate (absence of overflow above the reference onset = mismatch).
The onset is SECONDARY: graded for the record but NOT gating (it shifts with
strain/condition). Pure function; no simulation.

Reference = the Vemuri 2006 MG1655 chemostat overflow curve as Y_ac vs GUR
(`Y_ac = 2·qAc/(6·GUR)`): GUR up to ~7.9 mmol/gDW/h, Y_ac 0 → 0.25. With a 0.005
yield floor the supra-floor fit gives onset GUR ≈ 4.61, slope ≈ 0.077, R² ≈ 1.
"""
import pytest

from v2ecoli.library.card_criteria import grade_axis, _fit_overflow_line

# Vemuri 2006 (via Zhuang SI) — GUR and the derived acetate-carbon yield.
REF_X = [0.684, 1.209, 2.166, 3.358, 4.712, 5.960, 7.890]   # GUR, mmol/gDW/h
REF_Y = [0.0, 0.0, 0.0, 0.0, 0.009692, 0.101734, 0.253992]  # Y_ac (dimensionless)
FLOOR = 0.005


def _crit(**over):
    c = {"type": "curve_response", "ref_x": REF_X, "ref_y": REF_Y, "active_eps": FLOOR}
    c.update(over)
    return c


def test_reference_fit_recovers_yield_curve():
    fit = _fit_overflow_line(REF_X, REF_Y, FLOOR)
    assert fit is not None
    assert fit["lac"] == pytest.approx(4.61, abs=0.05)      # onset in GUR units
    assert fit["slope"] == pytest.approx(0.077, rel=0.05)   # dimensionless yield slope
    assert fit["r2"] == pytest.approx(1.0, abs=0.01)        # the rise is linear
    assert fit["n_supra"] == 3


def test_match_is_within_tol():
    # Model reproduces the yield curve exactly → linear rise + slope land → within_tol.
    res = grade_axis({"x": REF_X, "y": REF_Y}, _crit())
    assert res["verdict"] == "within_tol"
    assert res["detail"]["lin_verdict"] == "within_tol"
    assert res["detail"]["slope_verdict"] == "within_tol"


def test_no_overflow_is_mismatch_via_gate():
    # Baseline-FBA-like: sweep spans past the reference onset but the yield never
    # crosses the floor → wrong shape (flat) → mismatch (gate 2).
    res = grade_axis({"x": [3.0, 5.0, 6.5, 8.0], "y": [0.0, 0.0, 0.0, 0.0]}, _crit())
    assert res["verdict"] == "mismatch"
    assert "no overflow" in res["detail"]["reason"]


def test_sweep_below_onset_is_ungraded():
    # Sweep never reaches the reference onset → can't assess a shape you never
    # entered; this range gate fires before the no-overflow gate.
    res = grade_axis({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 0.0, 0.0, 0.001]}, _crit())
    assert res["verdict"] == "ungraded"
    assert res["detail"]["reason"] == "sweep below reference onset"


def test_shifted_onset_still_passes_on_shape():
    # Overflow present, LINEAR, matching slope (~0.077), but onset shifted ~1.4
    # GUR late. Onset is secondary → it does NOT gate: shape carries within_tol
    # even though the onset verdict is mismatch.
    res = grade_axis({"x": [5.0, 6.0, 7.0, 8.0, 9.0],
                      "y": [0.0, 0.0, 0.077, 0.154, 0.231]}, _crit())
    assert res["verdict"] == "within_tol"
    assert res["detail"]["onset_verdict"] == "mismatch"   # far off...
    assert res["detail"]["onset_gating"] is False         # ...but not gating
    assert res["detail"]["lin_verdict"] == "within_tol"


def test_slope_off_is_graded_down():
    # Linear rise (high R²) but the yield slope is ~4× the reference → slope drives
    # the verdict (the magnitude of overflow is wrong), linearity still passes.
    res = grade_axis({"x": [5.0, 6.0, 7.0, 8.0], "y": [0.0, 0.3, 0.6, 0.9]}, _crit())
    assert res["detail"]["lin_verdict"] == "within_tol"
    assert res["detail"]["slope_verdict"] == "mismatch"
    assert res["verdict"] == "mismatch"


def test_nonlinear_rise_is_graded_down():
    # Overflow present but the rise saturates (not linear) → linearity degrades the
    # verdict below within_tol (the 0→linear shape is the thing being graded).
    res = grade_axis({"x": [5.0, 6.0, 7.0, 8.0, 9.0],
                      "y": [0.0, 0.10, 0.15, 0.17, 0.18]}, _crit())
    assert res["detail"]["lin_verdict"] != "within_tol"
    assert res["verdict"] in ("drift", "mismatch")


def test_missing_inputs_ungraded():
    assert grade_axis({"x": REF_X, "y": REF_Y}, {"type": "curve_response"})["verdict"] == "ungraded"
    assert grade_axis(None, _crit())["verdict"] == "ungraded"
