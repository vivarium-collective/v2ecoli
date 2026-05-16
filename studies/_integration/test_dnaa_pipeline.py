"""Cross-study integration tests for the dnaA-replication investigation.

These tests consume the most-recent baseline run from EACH dnaa-* study
and check that the chained predictions hold:

  dnaa-01's steady DnaA pool → consistent across dnaa-02..06
  dnaa-02's DnaA-ATP fraction → matches what dnaa-04's mechanism expects
  dnaa-03's chromosomal occupancy timing → consistent with dnaa-04 initiation timing
  dnaa-05's CV narrowing claim → really narrows vs dnaa-02 baseline

Today everything is xfail because none of the upstream simulations have
been run yet. As each upstream study completes (status: ran, tests
passing), the corresponding integration test flips from xfail to a
real assertion.

Run via:
  pytest studies/_integration/ -v

This is the v0 scaffold; the assertion bodies are placeholders. The
shape of each test is what matters — clear about which two studies'
data it joins and what the joining predicate is.
"""
from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

import pytest
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT   = WORKSPACE_ROOT / "studies"


def _latest_baseline_history(study_slug: str) -> list[dict] | None:
    """Load the most-recent completed baseline run for a study.

    Returns None if the study has no runs.db or no completed runs.
    """
    db = STUDIES_ROOT / study_slug / "runs.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        try:
            row = conn.execute(
                "SELECT run_id FROM runs_meta WHERE status='completed' "
                "ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        run_id = row[0]
        import json
        rows = conn.execute(
            "SELECT step, global_time, state FROM history "
            "WHERE simulation_id=? ORDER BY step ASC",
            (run_id,),
        ).fetchall()
        return [{"step": s, "time": t, "state": json.loads(st)} for s, t, st in rows]
    finally:
        conn.close()


def _require_run(slug: str):
    """Skip the test if the named study has no completed run yet."""
    h = _latest_baseline_history(slug)
    if h is None:
        pytest.skip(f"no completed run for {slug} — run it first")
    return h


# ── Integration tests ──────────────────────────────────────────────────────

@pytest.mark.xfail(reason="dnaa-02 ATP/ADP/apo split not implemented yet (gap-1)")
def test_dnaa_atp_fraction_matches_across_dnaa02_and_dnaa05():
    """dnaa-02 reports DnaA-ATP fraction X; dnaa-05's full-cycle run
    should oscillate AROUND X (peak ≈ 1.5×X, trough ≈ 0.5×X)."""
    h2 = _require_run("dnaa-02-atp-hydrolysis")
    h5 = _require_run("dnaa-05-rida-ddah-dars")
    # Placeholder: compute medians and oscillation amplitude.
    raise NotImplementedError("upstream code (gap-1 dnaa-02) not landed")


@pytest.mark.xfail(reason="dnaa-03 box-binding model not implemented yet")
def test_dnaa03_oric_timing_matches_dnaa04_initiation():
    """dnaa-03's oriC-occupancy-reaches-50% timestamp should be within
    ±2 min of dnaa-04's first initiation event."""
    h3 = _require_run("dnaa-03-box-binding")
    h4 = _require_run("dnaa-04-initiation-mechanism")
    raise NotImplementedError("upstream code (gap-1 dnaa-03) not landed")


@pytest.mark.xfail(reason="dnaa-04 mechanism + dnaa-02 intrinsic-only haven't both run")
def test_dnaa05_cv_narrowing_is_real():
    """Acceptance criterion in investigation.yaml. The inter-initiation
    timing CV from a dnaa-05 baseline run must be ≤ 70% of the CV from
    a dnaa-02-only baseline run."""
    h2 = _require_run("dnaa-02-atp-hydrolysis")
    h5 = _require_run("dnaa-05-rida-ddah-dars")
    raise NotImplementedError("waiting on dnaa-02 + dnaa-05 baselines")


@pytest.mark.xfail(reason="dnaa-06 SeqA process not implemented")
def test_dnaa06_seqa_prevents_reinitiation_in_dnaa04_run():
    """With SeqA on (dnaa-06 wildtype) the dnaa-04 baseline-mechanism
    run should have exactly one initiation event per generation
    across all seeds."""
    h6 = _require_run("dnaa-06-seqa-sequestration")
    raise NotImplementedError("waiting on dnaa-06 SeqA implementation")


def test_investigation_yaml_acceptance_criteria_resolve():
    """Smoke test: every (study, behavior) in investigation.yaml's
    acceptance_criteria must resolve to a real entry. This runs today
    against the YAML and protects against drift as studies evolve."""
    inv = yaml.safe_load(
        (WORKSPACE_ROOT / "investigations" / "dnaa-replication" / "investigation.yaml")
        .read_text()
    )
    failures = []
    for crit in inv.get("acceptance_criteria") or []:
        slug = crit.get("study")
        behavior_name = crit.get("behavior")
        study_path = STUDIES_ROOT / slug / "study.yaml"
        if not study_path.exists():
            failures.append(f"acceptance_criterion: study {slug!r} not found")
            continue
        study_spec = yaml.safe_load(study_path.read_text())
        names = {b.get("name") for b in (study_spec.get("expected_behavior") or [])}
        if behavior_name not in names:
            failures.append(
                f"acceptance_criterion: behavior {behavior_name!r} not in "
                f"study {slug!r}'s expected_behavior list. Available: "
                + ", ".join(sorted(names))
            )
    assert not failures, "Acceptance-criteria drift:\n  " + "\n  ".join(failures)
