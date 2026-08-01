"""Comparison convergence Phase 2, Task 2: comparison_cards Analysis renders
the FULL wired report-card set from two workbench runs — a candidate and a
reference. Hermetic — no engine run, no S3.

Two data sources, matching comparison_summary's test split:
  - a synthetic monkeypatched pairing with a KNOWN relative delta, to check
    the standard/parca grading + the assembled verdict rollup precisely
    (uses real observable names -- cell_mass/dry_mass -- since standard/parca
    hardcode comparison_report_card.OBSERVABLES/MASS_OBS, not the caller's
    `observables` list);
  - the real committed `tests/fixtures/redux_cards` zarr pair (Task 3's
    fixtures, also used by comparison_summary's test), which is the only way
    to hermetically exercise trajectory/distribution/metabolism/composition
    (they read real zarr stores off disk, not the loaded observables dict).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
from v2ecoli.workflow.analyses.comparison_cards import (
    ComparisonCards, DEFAULT_CARDS, WIRED_CARDS, comparison_cards)

FIXTURES = Path(__file__).parent / "fixtures" / "redux_cards"
CANDIDATE_ZARR = FIXTURES / "v2ecoli_seed00.zarr"
REFERENCE_ZARR = FIXTURES / "vecoli_seed00.zarr"

ZARR_CARDS = {"trajectory", "distribution", "metabolism", "composition"}
PER_OBS_CARDS = {"standard", "parca"}


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

def test_comparison_cards_is_registered_like_the_example_analysis():
    from v2ecoli.workflow.analyses.growth_overlay import GrowthOverlay

    assert issubclass(ComparisonCards, Analysis)
    assert issubclass(GrowthOverlay, Analysis)
    assert ANALYSIS_REGISTRY["comparison_cards"] is ComparisonCards


def test_comparison_cards_discoverable_via_analyses_package_import():
    import importlib

    import v2ecoli.workflow.analyses as analyses_pkg
    importlib.reload(analyses_pkg)
    assert "comparison_cards" in ANALYSIS_REGISTRY


# --------------------------------------------------------------------------- #
# wired-card catalogue sanity
# --------------------------------------------------------------------------- #

def test_wired_cards_excludes_statistical():
    """The plan requires statistical to be DEFERRED (needs a multi-seed
    ensemble) -- assert it never silently lands in the wired set."""
    assert "statistical" not in WIRED_CARDS
    assert set(WIRED_CARDS) == {"summary", "standard", "parca", "trajectory",
                                "distribution", "metabolism", "composition"}
    assert set(DEFAULT_CARDS) == set(WIRED_CARDS)


# --------------------------------------------------------------------------- #
# synthetic known-difference pairing -- standard/parca grading + verdict rollup
# --------------------------------------------------------------------------- #

def test_comparison_cards_grades_known_differences(monkeypatch):
    """cell_mass differs by +2% (within 5% tolerance), dry_mass by +15%
    (beyond the 10% drift band -> mismatch). standard + parca must grade
    each independently; the assembled verdict rolls up to the worst; the
    summary card must be built from THOSE groups (never faking a
    `statistical` group that wasn't requested/rendered)."""
    times = np.array([0.0, 10.0, 20.0, 30.0])
    ref_cell = np.array([100.0, 101.0, 99.0, 100.0])
    ref_dry = np.array([50.0, 51.0, 49.0, 50.0])

    ref_obs = {"cell_mass": (times, ref_cell), "dry_mass": (times, ref_dry)}
    cand_obs = {"cell_mass": (times, ref_cell * 1.02),
               "dry_mass": (times, ref_dry * 1.15)}

    def fake_load(run_ref, observables=None, *, study_dir=None, runs_db=None):
        return dict(cand_obs) if run_ref == "candidate" else dict(ref_obs)

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.load_run_observables",
        fake_load)

    out = comparison_cards("candidate", "reference",
                           observables=["cell_mass", "dry_mass"], seeds=3,
                           cards=["summary", "standard", "parca"])

    assert set(out) == {"cards", "verdict", "deferred"}
    assert out["deferred"] == {}
    assert set(out["cards"]) == {"summary", "standard", "parca"}

    verdict = out["verdict"]
    assert verdict["schema"] == "report_card_verdict/v1"
    assert verdict["model_ref"] == "candidate"
    assert verdict["reference_model"] == "reference"
    assert set(verdict["groups"]) == {"standard", "parca"}
    # both groups saw the same +15% dry_mass divergence -> both worst-of mismatch
    assert verdict["groups"]["standard"]["verdict"] == "mismatch"
    assert verdict["groups"]["parca"]["verdict"] == "mismatch"
    assert verdict["overall"] == "mismatch"
    # `statistical` was never requested/rendered -> never appears as a group
    assert "statistical" not in verdict["groups"]

    standard_axes = {ax["label"]: ax for ax in verdict["groups"]["standard"]["axes"]}
    assert standard_axes["cell mass (fg)"]["verdict"] == "within_tol"
    assert standard_axes["dry mass (fg)"]["verdict"] == "mismatch"

    # summary card built from standard+parca groups (both mismatch) -> gate
    # is ungraded (GRADED={statistical,parca} and parca IS present+mismatch,
    # so gate reflects parca).
    summary_html = out["cards"]["summary"]
    assert "✓" in summary_html or "✗" in summary_html
    assert "3 seeds" in summary_html

    standard_html = out["cards"]["standard"]
    assert "dry mass" in standard_html.lower()


def test_comparison_cards_step_wraps_the_function(monkeypatch):
    """The registered Step reads its config (not sim-history state) and
    delegates to the pure function, same wrapping pattern as ComparisonSummary."""
    times = np.array([0.0, 10.0])
    ref_obs = {"cell_mass": (times, np.array([10.0, 10.0]))}
    cand_obs = {"cell_mass": (times, np.array([10.0, 10.0]))}

    def fake_load(run_ref, observables=None, *, study_dir=None, runs_db=None):
        return dict(cand_obs) if run_ref == "cand-run" else dict(ref_obs)

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.load_run_observables",
        fake_load)

    from v2ecoli.core import build_core

    step = ComparisonCards(config={
        "candidate_run": "cand-run", "reference_run": "ref-run",
        "seeds": 2, "observables": ["cell_mass"], "cards": ["standard"],
    }, core=build_core())
    out = step.update()
    assert set(out) == {"cards", "verdict", "deferred"}
    assert out["verdict"]["groups"]["standard"]["verdict"] == "within_tol"


# --------------------------------------------------------------------------- #
# deferred statistical -- honest, never faked
# --------------------------------------------------------------------------- #

def test_statistical_card_is_deferred_not_faked(monkeypatch):
    times = np.array([0.0, 10.0])
    obs = {"cell_mass": (times, np.array([10.0, 10.0]))}

    def fake_load(run_ref, observables=None, *, study_dir=None, runs_db=None):
        return dict(obs)

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.load_run_observables",
        fake_load)

    out = comparison_cards("candidate", "reference",
                           observables=["cell_mass"],
                           cards=["standard", "statistical", "made_up_card"])

    # requested-but-not-wired cards never render card_html...
    assert "statistical" not in out["cards"]
    assert "made_up_card" not in out["cards"]
    # ...and never appear as a verdict group (no fake grading).
    assert "statistical" not in out["verdict"]["groups"]
    assert "made_up_card" not in out["verdict"]["groups"]
    # ...but ARE reported, honestly, with a reason.
    assert "multi-seed" in out["deferred"]["statistical"].lower()
    assert "made_up_card" in out["deferred"]["made_up_card"]
    # the card that WAS wired still rendered normally.
    assert "standard" in out["cards"]


# --------------------------------------------------------------------------- #
# real fixtures (Task 3's committed redux_cards zarr pair) — full card set
# --------------------------------------------------------------------------- #

def test_comparison_cards_renders_full_wired_set_from_real_fixtures():
    out = comparison_cards(str(CANDIDATE_ZARR), str(REFERENCE_ZARR), seeds=1)

    assert out["deferred"] == {}
    assert set(out["cards"]) == set(WIRED_CARDS)

    verdict = out["verdict"]
    assert verdict["schema"] == "report_card_verdict/v1"
    # every graded card (all but trajectory, which is purely descriptive)
    # shows up as a verdict group.
    assert set(verdict["groups"]) == set(WIRED_CARDS) - {"summary"}
    assert verdict["groups"]["trajectory"]["verdict"] == "ungraded"
    assert verdict["groups"]["trajectory"]["axes"] == []

    # standard/parca: real |Δ| percents, not "--".
    standard_axes = verdict["groups"]["standard"]["axes"]
    graded = [ax for ax in standard_axes if ax["verdict"] != "ungraded"]
    assert graded, "expected at least one graded standard axis from the fixture pair"
    for ax in graded:
        assert ax["detail"]["median_rel"] is not None

    # composition grades via rel_tol (works even for a single-run pairing).
    comp_axes = verdict["groups"]["composition"]["axes"]
    assert comp_axes and any(ax["verdict"] != "ungraded" for ax in comp_axes)

    for card in WIRED_CARDS:
        html = out["cards"][card]
        assert isinstance(html, str) and html, f"{card} card rendered empty html"

    # summary card is self-contained HTML built from the other groups.
    summary_html = out["cards"]["summary"]
    assert len(summary_html.encode("utf-8")) > 200
    assert "1 seeds" in summary_html

    standard_html = out["cards"]["standard"]
    assert "cell mass" in standard_html.lower()

    trajectory_html = out["cards"]["trajectory"]
    assert len(trajectory_html.encode("utf-8")) > 500  # real plotly fragment(s)


def test_comparison_cards_honors_a_requested_subset_from_real_fixtures():
    out = comparison_cards(str(CANDIDATE_ZARR), str(REFERENCE_ZARR), seeds=1,
                           cards=["distribution", "metabolism"])

    assert out["deferred"] == {}
    assert set(out["cards"]) == {"distribution", "metabolism"}
    assert set(out["verdict"]["groups"]) == {"distribution", "metabolism"}
    # distribution: n=1 per side -> grade_axis's ttest branch honestly
    # reports ungraded (never a fabricated significance test).
    assert all(ax["verdict"] == "ungraded"
              for ax in out["verdict"]["groups"]["distribution"]["axes"])
