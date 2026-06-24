"""Tests for the metabolic-perturbation overflow card render
(``scripts/render_overflow_card``).

Reference side reads the installed ecoli-sources
``perturbation__overflow__acetate_vs_growth`` slot (needs the pinned bundle);
grading uses synthetic model arms. Acetate-carbon yield Y_ac = (2·acetate)/(6·glucose),
graded vs growth rate (Vemuri primary).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import render_overflow_card as roc  # noqa: E402
from v2ecoli.library.report_card import grade_card, render_html  # noqa: E402


# --- reference side (reads the slot) ----------------------------------------

def test_vemuri_curve_is_raw_yield():
    v = roc.vemuri_curve()
    assert v["x"] == sorted(v["x"])               # ascending growth rate
    assert min(v["y"]) == pytest.approx(0.0, abs=1e-9)   # flat below onset
    assert max(v["y"]) == pytest.approx(0.25, abs=0.03)  # ~25% C->acetate at top
    assert len(v["err"]) == 2 and len(v["err"][0]) == len(v["x"])


def test_basan_context_via_si():
    b = roc.basan_curve()
    # SI-derived dimensionless yield for the ptsG titration: 0 -> ~0.10
    assert b["y"][0] == pytest.approx(0.0, abs=0.01)
    assert max(b["y"]) == pytest.approx(0.10, abs=0.03)


def test_mg1655_anchor_point():
    a = roc.mg1655_anchor()
    # raw 13C anchor(s): ~15-22% of carbon to acetate at the basal growth rate
    assert a["x"] and all(0.6 < x < 0.75 for x in a["x"])
    assert all(0.10 < y < 0.25 for y in a["y"])


# --- grading (synthetic model arms) -----------------------------------------

def _model_from_points(pts):
    # pts: list of (growth, glucose_uptake, acetate) -> runner-style rows
    return {"knob": "test", "n_seeds": 3, "generations": 3,
            "rows": [{"growth_rate_per_h": g, "glucose": -gur, "acetate": ac}
                     for g, gur, ac in pts]}


def test_model_matching_vemuri_is_within_tol():
    # Reproduce the Vemuri yield curve exactly (acetate = 3·Y_ac·GUR) -> within_tol.
    v = roc.vemuri_curve()
    pts = [(x, 1.0, 3.0 * y * 1.0) for x, y in zip(v["x"], v["y"])]  # GUR=1 -> Y_ac=2·ac/6=ac/3
    card, ref, _ = roc.build(model_json="/nonexistent", bundle_path=None)
    card["metabolism"]["acetate_overflow"] = {  # inject the synthetic model curve
        "x": [p[0] for p in pts], "y": [v["y"][i] for i in range(len(v["y"]))]}
    report = grade_card(card, ref)
    assert report["axes"]["metabolism.acetate_overflow"]["verdict"] == "within_tol"


def test_flat_model_is_mismatch():
    # FBA-like: no overflow across a sweep spanning past the Vemuri onset (~0.41/h).
    card, ref, _ = roc.build(model_json="/nonexistent", bundle_path=None)
    card["metabolism"]["acetate_overflow"] = {"x": [0.3, 0.5, 0.6, 0.7], "y": [0.0, 0.0, 0.0, 0.0]}
    report = grade_card(card, ref)
    ax = report["axes"]["metabolism.acetate_overflow"]
    assert ax["verdict"] == "mismatch"
    assert "no overflow" in ax["detail"]["reason"]


# --- end-to-end render -------------------------------------------------------

def test_card_renders_without_ploterr():
    card, ref, _ = roc.build(model_json="/nonexistent", bundle_path=None)
    card["metabolism"]["acetate_overflow"] = {"x": [0.4, 0.5, 0.6], "y": [0.0, 0.10, 0.25]}
    html = render_html(card, ref, model_ref="test", generated="test")
    assert "plot unavailable" not in html
    assert "acetate-carbon yield" in html        # the axis rendered
