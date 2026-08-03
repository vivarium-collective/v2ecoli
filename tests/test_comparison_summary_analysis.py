"""Task 4 (comparison convergence Phase 1): comparison_summary Analysis
renders the `summary` report card from two workbench runs — a candidate and a
reference. Hermetic — no engine run, no S3; loads via monkeypatched
``load_run_observables`` (synthetic) and the real committed
``tests/fixtures/redux_cards`` zarr pair (Task 3's fixtures).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
from v2ecoli.workflow.analyses.comparison_summary import (
    ComparisonSummary, comparison_summary)

FIXTURES = Path(__file__).parent / "fixtures" / "redux_cards"
CANDIDATE_ZARR = FIXTURES / "v2ecoli_seed00.zarr"
REFERENCE_ZARR = FIXTURES / "vecoli_seed00.zarr"


# --------------------------------------------------------------------------- #
# registration — same mechanism as the example Analysis (growth_overlay)
# --------------------------------------------------------------------------- #

def test_comparison_summary_is_registered_like_the_example_analysis():
    from v2ecoli.workflow.analyses.growth_overlay import GrowthOverlay

    assert issubclass(ComparisonSummary, Analysis)
    assert issubclass(GrowthOverlay, Analysis)
    assert ANALYSIS_REGISTRY["comparison_summary"] is ComparisonSummary
    assert ANALYSIS_REGISTRY["growth_overlay"] is GrowthOverlay


def test_comparison_summary_discoverable_via_analyses_package_import():
    """Importing the analyses package (the __init__.py registration path
    every other ported analysis uses) registers comparison_summary too."""
    import importlib

    import v2ecoli.workflow.analyses as analyses_pkg
    importlib.reload(analyses_pkg)
    assert "comparison_summary" in ANALYSIS_REGISTRY


# --------------------------------------------------------------------------- #
# comparison_summary() — synthetic known-difference pairing
# --------------------------------------------------------------------------- #

def test_comparison_summary_grades_known_differences(monkeypatch):
    """One observable differs by +2% (within the 5% tolerance band), the other
    by +15% (beyond the 10% drift band -> mismatch). The Analysis must grade
    each independently and roll the group/overall verdict up to the worst."""
    times = np.array([0.0, 10.0, 20.0, 30.0])
    ref_a = np.array([100.0, 101.0, 99.0, 100.0])
    ref_b = np.array([50.0, 51.0, 49.0, 50.0])

    ref_obs = {"a": (times, ref_a), "b": (times, ref_b)}
    cand_obs = {"a": (times, ref_a * 1.02), "b": (times, ref_b * 1.15)}

    def fake_load(run_ref, observables=None, *, study_dir=None, runs_db=None):
        return dict(cand_obs) if run_ref == "candidate" else dict(ref_obs)

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_summary.load_run_observables",
        fake_load)

    out = comparison_summary("candidate", "reference",
                             observables=["a", "b"], seeds=3)

    assert set(out) == {"card_html", "verdict"}
    verdict = out["verdict"]
    axes = {ax["label"]: ax for ax in verdict["groups"]["observables"]["axes"]}

    assert axes["a"]["verdict"] == "within_tol"
    assert axes["a"]["detail"]["median_rel"] == pytest.approx(0.02, abs=1e-9)
    assert axes["b"]["verdict"] == "mismatch"
    assert axes["b"]["detail"]["median_rel"] == pytest.approx(0.15, abs=1e-9)
    # worst-of rollup
    assert verdict["groups"]["observables"]["verdict"] == "mismatch"
    assert verdict["overall"] == "mismatch"

    html = out["card_html"]
    # per-observable |Δ| surfaced as a percent
    assert "2.0%" in html
    assert "15.0%" in html
    # status glyphs (never color-alone — see scripts/_compare/theme.py STATUS)
    assert "✓" in html  # within_tol
    assert "✗" in html  # mismatch
    assert "3 seeds" in html


def test_comparison_summary_analysis_step_wraps_the_function(monkeypatch):
    """The registered Step reads candidate_run/reference_run/seeds off its own
    config (Step config, not sim-history state) and delegates to the pure
    function."""
    times = np.array([0.0, 10.0])
    ref_obs = {"x": (times, np.array([10.0, 10.0]))}
    cand_obs = {"x": (times, np.array([10.0, 10.0]))}

    def fake_load(run_ref, observables=None, *, study_dir=None, runs_db=None):
        return dict(cand_obs) if run_ref == "cand-run" else dict(ref_obs)

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_summary.load_run_observables",
        fake_load)

    from v2ecoli.core import build_core

    step = ComparisonSummary(config={
        "candidate_run": "cand-run", "reference_run": "ref-run",
        "seeds": 2, "observables": ["x"],
    }, core=build_core())
    out = step.update()
    assert set(out) == {"card_html", "verdict"}
    axes = out["verdict"]["groups"]["observables"]["axes"]
    assert axes[0]["label"] == "x"
    assert axes[0]["verdict"] == "within_tol"


# --------------------------------------------------------------------------- #
# real fixtures (Task 3's committed redux_cards zarr pair) — no monkeypatch
# --------------------------------------------------------------------------- #

def test_comparison_summary_renders_from_real_fixtures():
    out = comparison_summary(str(CANDIDATE_ZARR), str(REFERENCE_ZARR), seeds=1)

    verdict = out["verdict"]
    axes = verdict["groups"]["observables"]["axes"]
    labels = {ax["label"] for ax in axes}
    assert {"cell_mass", "dry_mass"} <= labels
    # a matched v2ecoli<->vEcoli pair from the same fixture family: every
    # graded axis should have SOME magnitude recorded (not silently ungraded).
    graded = [ax for ax in axes if ax["verdict"] != "ungraded"]
    assert graded, "expected at least one graded observable from the fixture pair"
    for ax in graded:
        assert ax["detail"]["median_rel"] is not None

    html = out["card_html"]
    assert len(html.encode("utf-8")) > 1000
    assert "cell_mass" in html
    assert "1 seeds" in html
