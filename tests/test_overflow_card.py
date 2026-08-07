"""Tests for the aerobic acetate-overflow report card.

Reference side reads the installed ecoli-sources
``perturbation__overflow__acetate_vs_growth`` slot (needs the pinned bundle);
grading uses synthetic model arms. Acetate-carbon yield Y_ac = (2·acetate)/(6·glucose),
graded vs growth rate (Vemuri primary) by the ``threshold_linear`` criterion.

Covers both layers: the science core (``v2ecoli.library.overflow``) and the Gen-2
Step (``AcetateOverflowCard``), mirroring ``tests/test_vs_literature_card.py``.
"""
from pathlib import Path

import pytest
import yaml

from v2ecoli.library import overflow as ovf
from v2ecoli.library.report_card import grade_card, render_html
from v2ecoli.workflow.post_sim import REPORT_CARD_REGISTRY
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.acetate_overflow_card import AcetateOverflowCard

REPO = Path(__file__).resolve().parents[1]

# The verdict the 2026-07-22 baseline-FBA run produced, and the bar for the
# Gen-1 -> Gen-2 migration: renaming curve_response -> threshold_linear must not
# move the science.
_GOLDEN_OVERALL = "mismatch"
_GOLDEN_AXIS = "metabolism.acetate_overflow"


# --- reference side (reads the slot) ----------------------------------------

def test_vemuri_curve_is_raw_yield():
    v = ovf.vemuri_curve()
    assert v["x"] == sorted(v["x"])               # ascending growth rate
    assert min(v["y"]) == pytest.approx(0.0, abs=1e-9)   # flat below onset
    assert max(v["y"]) == pytest.approx(0.25, abs=0.03)  # ~25% C->acetate at top
    assert len(v["err"]) == 2 and len(v["err"][0]) == len(v["x"])


def test_basan_context_via_si():
    b = ovf.basan_curve()
    # SI-derived dimensionless yield for the ptsG titration: 0 -> ~0.10
    assert b["y"][0] == pytest.approx(0.0, abs=0.01)
    assert max(b["y"]) == pytest.approx(0.10, abs=0.03)


def test_mg1655_anchor_point():
    a = ovf.mg1655_anchor()
    # raw 13C anchor(s): ~15-22% of carbon to acetate at the basal growth rate
    assert a["x"] and all(0.6 < x < 0.75 for x in a["x"])
    assert all(0.10 < y < 0.25 for y in a["y"])


# --- grading (synthetic model arms) -----------------------------------------

def test_model_matching_vemuri_is_within_tol():
    # Reproduce the Vemuri yield curve exactly -> within_tol.
    v = ovf.vemuri_curve()
    card, ref, _ = ovf.build(model_json=Path("/nonexistent"))
    card["metabolism"]["acetate_overflow"] = {"x": list(v["x"]), "y": list(v["y"])}
    report = grade_card(card, ref)
    assert report["axes"][_GOLDEN_AXIS]["verdict"] == "within_tol"


def test_flat_model_is_mismatch():
    # FBA-like: no overflow across a sweep spanning past the Vemuri onset (~0.41/h).
    card, ref, _ = ovf.build(model_json=Path("/nonexistent"))
    card["metabolism"]["acetate_overflow"] = {"x": [0.3, 0.5, 0.6, 0.7], "y": [0.0, 0.0, 0.0, 0.0]}
    report = grade_card(card, ref)
    ax = report["axes"][_GOLDEN_AXIS]
    assert ax["verdict"] == "mismatch"
    assert "no rise in model" in ax["detail"]["reason"]


# --- end-to-end render -------------------------------------------------------

def test_card_renders_without_ploterr():
    card, ref, _ = ovf.build(model_json=Path("/nonexistent"))
    card["metabolism"]["acetate_overflow"] = {"x": [0.4, 0.5, 0.6], "y": [0.0, 0.10, 0.25]}
    html = render_html(card, ref, model_ref="test", generated="test")
    assert "plot unavailable" not in html
    assert "acetate-carbon yield" in html        # the axis rendered


# --- the Gen-2 Step ----------------------------------------------------------

def _ctx(tmp_path, refs=None):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    spec = {"name": "Demo"}
    if refs:
        spec["report_card_refs"] = refs
    (sd / "study.yaml").write_text(yaml.safe_dump(spec))
    # ws_root must be the real repo for the fixtures pointer to resolve.
    return StudyContext(study_name="demo", study_dir=sd, spec=spec, ws_root=REPO)


def test_registered():
    assert REPORT_CARD_REGISTRY.get("acetate_overflow") is AcetateOverflowCard


def test_absent_without_ref(tmp_path, core):
    step = AcetateOverflowCard({}, core=core)
    assert step.applies(_ctx(tmp_path)) is False


def test_absent_when_ref_dir_missing_fixture(tmp_path, core):
    step = AcetateOverflowCard({}, core=core)
    ctx = _ctx(tmp_path, {"acetate_overflow": "workspace/studies"})
    assert step.applies(ctx) is False          # dir exists, fixture doesn't — no throw


def test_applies_for_overflow_fixtures(tmp_path, core):
    step = AcetateOverflowCard({}, core=core)
    ctx = _ctx(tmp_path, {"acetate_overflow": "tests/fixtures/overflow_acetate_vs_growth"})
    assert step.applies(ctx) is True


def test_build_returns_valid_verdict(tmp_path, core):
    step = AcetateOverflowCard({}, core=core)
    ctx = _ctx(tmp_path, {"acetate_overflow": "tests/fixtures/overflow_acetate_vs_growth"})
    vjson, html = step.build(ctx)
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["reference_model"].startswith("experimental literature")
    assert vjson["model_ref"].startswith("v2ecoli baseline FBA")
    assert "acetate-carbon yield" in html
    # Committed output must be deterministic — no timestamp baked in.
    assert not vjson.get("generated")


def test_verdict_matches_pre_migration_golden(tmp_path, core):
    """The Gen-1 -> Gen-2 migration bar: the baked 2026-07-22 arm still grades
    `mismatch` (no overflow). Renaming curve_response -> threshold_linear changed
    the meter wording, not the verdict."""
    step = AcetateOverflowCard({}, core=core)
    ctx = _ctx(tmp_path, {"acetate_overflow": "tests/fixtures/overflow_acetate_vs_growth"})
    vjson, _ = step.build(ctx)
    assert vjson["overall"] == _GOLDEN_OVERALL
    axes = {ax["id"]: ax["verdict"] for grp in vjson.get("groups", {}).values()
            for ax in grp.get("axes", [])}
    assert axes == {_GOLDEN_AXIS: "mismatch"}


def test_verdict_group_is_reachable_by_report_card_axis(tmp_path):
    """The card -> acceptance-spine join (pbg_v2ecoli/evaluators.py). The Gen-2
    flush writes `<card>.verdict.json`, so the evaluator must resolve that name,
    not only the Gen-1 `report_card_verdict.json`."""
    from pbg_v2ecoli.evaluators import evaluate_report_card_group

    card_dir = tmp_path / "viz" / "report_card"
    card_dir.mkdir(parents=True)
    (card_dir / "acetate_overflow.verdict.json").write_text(
        '{"overall": "mismatch", "groups": {"overflow": {"axes": '
        '[{"id": "metabolism.acetate_overflow", "verdict": "mismatch"}]}}}')
    test = {"measure": {"kind": "report_card_axis", "card": "viz/report_card",
                        "card_name": "acetate_overflow", "group": "overflow"}}
    out = evaluate_report_card_group(test, None, tmp_path)
    assert out["result"] == "FAIL"          # a mismatch axis fails the group
    assert out["provenance"]["overall"] == "mismatch"
