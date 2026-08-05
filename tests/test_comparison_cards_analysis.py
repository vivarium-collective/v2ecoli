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

import json
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


def test_step_forwards_study_dir_and_runs_db_from_config(monkeypatch):
    """The env worker threads ``study_dir``/``runs_db`` into the analysis config
    when this runs as a per-study ``analyses:`` entry; the Step MUST forward
    them to ``comparison_cards`` so runs resolve by sim_name off the study's
    ``runs.db`` and the verdict persists (else comparison_matrix rolls up
    nothing). Regression for the real-worker verdict path."""
    captured = {}

    def fake_comparison_cards(candidate_run, reference_run, **kwargs):
        captured["candidate_run"] = candidate_run
        captured["reference_run"] = reference_run
        captured.update(kwargs)
        return {"cards": {}, "verdict": {"overall": "within_tol"}, "deferred": []}

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.comparison_cards",
        fake_comparison_cards)

    from v2ecoli.core import build_core

    step = ComparisonCards(config={
        "candidate_run": "basal", "reference_run": "reference",
        "study_dir": "/ws/studies/basal", "runs_db": "/ws/studies/basal/runs.db",
    }, core=build_core())
    step.update()

    assert captured["study_dir"] == "/ws/studies/basal"
    assert captured["runs_db"] == "/ws/studies/basal/runs.db"


def test_step_leaves_run_store_context_none_for_direct_callers():
    """A direct/unit caller that omits study_dir/runs_db gets None (the
    self-contained path), not a crash."""
    from v2ecoli.core import build_core
    step = ComparisonCards(config={"candidate_run": None, "reference_run": None},
                           core=build_core())
    assert step.config.get("study_dir") is None
    assert step.config.get("runs_db") is None


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


# --------------------------------------------------------------------------- #
# Phase B Task A: verdict persistence to the canonical per-study path, so
# comparison_matrix (Task 4) can read it from disk.
# --------------------------------------------------------------------------- #

def _fake_load_factory(cand_obs, ref_obs):
    def fake_load(run_ref, observables=None, *, study_dir=None, runs_db=None):
        return dict(cand_obs) if run_ref == "candidate" else dict(ref_obs)
    return fake_load


def test_comparison_cards_persists_verdict_when_study_dir_given(monkeypatch, tmp_path):
    """When called as a per-study analysis (study_dir given), the verdict
    must land at the exact path comparison_matrix's disk loader reads:
    ``<study_dir>/report_card_verdict.json`` (directly under study_dir, not
    nested under a per-condition subdirectory)."""
    times = np.array([0.0, 10.0, 20.0, 30.0])
    ref_cell = np.array([100.0, 101.0, 99.0, 100.0])
    cand_obs = {"cell_mass": (times, ref_cell)}
    ref_obs = {"cell_mass": (times, ref_cell)}

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.load_run_observables",
        _fake_load_factory(cand_obs, ref_obs))

    study_dir = tmp_path / "studies" / "basal"
    out = comparison_cards("candidate", "reference",
                           observables=["cell_mass"], seeds=1,
                           cards=["standard"], study_dir=study_dir)

    verdict_path = study_dir / "report_card_verdict.json"
    assert verdict_path.is_file()
    persisted = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert persisted == out["verdict"]
    assert persisted["groups"]["standard"]["verdict"] == "within_tol"


def test_comparison_cards_skips_persistence_when_study_dir_is_none(monkeypatch, tmp_path,
                                                                    capfd):
    """Direct/unit callers (study_dir=None, the default) get the verdict back
    but nothing is written to disk -- no accidental cwd pollution."""
    times = np.array([0.0, 10.0])
    obs = {"cell_mass": (times, np.array([10.0, 10.0]))}

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.load_run_observables",
        _fake_load_factory(obs, obs))
    monkeypatch.chdir(tmp_path)

    out = comparison_cards("candidate", "reference",
                           observables=["cell_mass"], cards=["standard"])

    assert out["verdict"]["groups"]["standard"]["verdict"] == "within_tol"
    assert not (tmp_path / "report_card_verdict.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_comparison_cards_verdict_round_trips_through_comparison_matrix(
        monkeypatch, tmp_path):
    """The write path (comparison_cards -> write_study_verdict) and the read
    path (comparison_matrix -> _load_study_verdict) must agree: a verdict
    written by comparison_cards for study "basal" must be the SAME verdict
    comparison_matrix reads back for config_studies=["basal"], not a
    placeholder."""
    from v2ecoli.workflow.analyses.comparison_matrix import comparison_matrix

    times = np.array([0.0, 10.0, 20.0, 30.0])
    ref_cell = np.array([100.0, 101.0, 99.0, 100.0])
    ref_dry = np.array([50.0, 51.0, 49.0, 50.0])
    ref_obs = {"cell_mass": (times, ref_cell), "dry_mass": (times, ref_dry)}
    # dry_mass +15% -> beyond drift band -> mismatch, so the round-tripped
    # matrix HTML must show a real mismatch glyph, not an ungraded placeholder.
    cand_obs = {"cell_mass": (times, ref_cell), "dry_mass": (times, ref_dry * 1.15)}

    monkeypatch.setattr(
        "v2ecoli.workflow.analyses.comparison_cards.load_run_observables",
        _fake_load_factory(cand_obs, ref_obs))

    workspace = tmp_path
    study_dir = workspace / "studies" / "basal"
    written = comparison_cards("candidate", "reference",
                               observables=["cell_mass", "dry_mass"], seeds=1,
                               cards=["standard"], study_dir=study_dir)["verdict"]

    out = comparison_matrix(config_studies=["basal"], workspace=workspace)
    html = out["matrix_html"]

    # Not the ungraded placeholder: the matrix actually rendered the
    # standard group's real axes, matching what comparison_cards computed.
    assert "basal" in html
    assert "✗" in html and "Mismatch" in html
    assert written["groups"]["standard"]["verdict"] == "mismatch"
