"""Phase 5 tests: validation sweep orchestrator internals.

Focuses on the script's PURE LOGIC (run-spec construction, pre-flight
checks, metric extraction, acceptance evaluation, report generation,
CLI arg parsing) — does NOT run actual multi-hour sims. The sweep itself
is a separate user-driven action.

Test tiers:

§A — RunSpec + orchestration (no cache):
- build_run_specs Cartesian product
- needs_run skip-if-exists / force semantics
- output-path conventions

§B — Pre-flight (no cache, partial cache):
- check_cache surfaces missing / stale cache cleanly
- check_media_available rejects bogus media ids
- preflight composition

§C — Metrics + report (synthetic data):
- compute_metrics on a synthesized run.db
- _exponential_fit_per_hour numerics
- check_acceptance against fixture metrics
- generate_report round-trips

§D — CLI (smoke):
- _parse_args defaults
- --preflight-only short-circuits
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


# ============================================================
# §A — RunSpec + orchestration
# ============================================================

def test_build_run_specs_cartesian_product() -> None:
    from consensus_validation_sweep import build_run_specs, RunSpec

    specs = build_run_specs(
        media=["minimal", "minimal_acetate"],
        seeds=[0, 1, 2],
        max_generations=5,
        max_steps=20000,
    )
    assert len(specs) == 6  # 2 * 3
    assert all(isinstance(s, RunSpec) for s in specs)
    assert {(s.media, s.seed) for s in specs} == {
        ("minimal", 0), ("minimal", 1), ("minimal", 2),
        ("minimal_acetate", 0), ("minimal_acetate", 1), ("minimal_acetate", 2),
    }
    for s in specs:
        assert s.max_generations == 5
        assert s.max_steps == 20000


def test_run_spec_paths_are_well_structured(tmp_path: Path) -> None:
    from consensus_validation_sweep import RunSpec

    spec = RunSpec(media="minimal_acetate", seed=2, max_generations=5, max_steps=1)
    out = spec.out_subdir(tmp_path)
    db = spec.db_path(tmp_path)
    assert out == tmp_path / "minimal_acetate" / "seed-2"
    assert db == tmp_path / "minimal_acetate" / "seed-2" / "run.db"
    assert spec.run_id() == "consensus-minimal_acetate-seed2"


def test_needs_run_skips_when_db_exists(tmp_path: Path) -> None:
    from consensus_validation_sweep import RunSpec, needs_run

    spec = RunSpec(media="minimal", seed=0, max_generations=1, max_steps=1)
    # No db yet — must run.
    assert needs_run(spec, tmp_path, force=False) is True
    # Create the db file (touch).
    db = spec.db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_text("")
    # Now skip unless force.
    assert needs_run(spec, tmp_path, force=False) is False
    assert needs_run(spec, tmp_path, force=True) is True


# ============================================================
# §B — Pre-flight
# ============================================================

def test_check_cache_reports_missing_dir(tmp_path: Path) -> None:
    from consensus_validation_sweep import check_cache

    missing = tmp_path / "does_not_exist"
    problems = check_cache(missing)
    assert problems
    assert any("does not exist" in p for p in problems)


def test_preflight_reports_problems_when_cache_missing(tmp_path: Path) -> None:
    from consensus_validation_sweep import preflight

    missing = tmp_path / "no_cache"
    problems = preflight(missing, ["minimal"])
    assert problems
    # First problem must be the cache check — subsequent checks gated on it.
    assert any("does not exist" in p for p in problems)


CACHE = "out/cache"
_needs_real_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


@_needs_real_cache
def test_check_media_available_accepts_canonical_media() -> None:
    from consensus_validation_sweep import check_media_available, DEFAULT_MEDIA

    problems = check_media_available(CACHE, DEFAULT_MEDIA)
    assert not problems, (
        f"DEFAULT_MEDIA must be valid against the canonical cache: {problems}"
    )


@_needs_real_cache
def test_check_media_available_rejects_bogus_media() -> None:
    from consensus_validation_sweep import check_media_available

    problems = check_media_available(CACHE, ["definitely_not_a_real_media"])
    assert problems
    assert any("definitely_not_a_real_media" in p for p in problems)


# ============================================================
# §C — Metrics + report
# ============================================================

def _synth_history_row(t: float, cell_mass: float, fc: list[float] | None = None,
                       elong_rate: float | None = None) -> tuple:
    state = {"listeners": {"mass": {"cell_mass": cell_mass}}}
    if fc is not None:
        state["listeners"].setdefault("growth_limits", {})["fraction_trna_charged"] = fc
    if elong_rate is not None:
        state["listeners"].setdefault("ribosome_data", {})[
            "effective_elongation_rate"
        ] = elong_rate
    return (t, json.dumps(state))


def _make_synth_db(db_path: Path, sim_id: str, rows: list[tuple]) -> None:
    """Build a minimal sqlite db with a history table matching the columns
    compute_metrics reads. Mirrors the SQLiteEmitter's schema enough for
    the metric extraction logic without dragging in the real emitter.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE history ("
            "simulation_id TEXT, time REAL, state TEXT, generation INTEGER"
            ")"
        )
        for t, state_json in rows:
            conn.execute(
                "INSERT INTO history (simulation_id, time, state, generation) "
                "VALUES (?, ?, ?, ?)",
                (sim_id, t, state_json, 1),
            )
        conn.commit()
    finally:
        conn.close()


def test_compute_metrics_returns_nan_for_missing_db(tmp_path: Path) -> None:
    from consensus_validation_sweep import compute_metrics

    m = compute_metrics(tmp_path / "missing.db", media="minimal", seed=0)
    assert math.isnan(m.growth_rate_per_hour)
    assert math.isnan(m.mean_charging_fraction)
    assert m.n_steps == 0
    assert any("missing" in n for n in m.notes)


def test_compute_metrics_extracts_growth_and_charging(tmp_path: Path) -> None:
    from consensus_validation_sweep import compute_metrics

    db = tmp_path / "run.db"
    sim_id = "consensus-minimal-seed0"
    # 60 minutes of synthetic exponential growth: m(t) = 1.0 * exp(0.5 * t / 3600)
    # so growth rate should be ~0.5/h.
    rows = []
    for i in range(0, 3600, 60):
        t = float(i)
        mass = 1.0 * math.exp(0.5 * t / 3600.0)
        rows.append(_synth_history_row(
            t, mass,
            fc=[0.9, 0.85, 0.88],  # mean 0.876
            elong_rate=18.0,
        ))
    _make_synth_db(db, sim_id, rows)

    m = compute_metrics(db, media="minimal", seed=0)
    assert m.n_steps == 60
    assert abs(m.growth_rate_per_hour - 0.5) < 0.01, (
        f"growth rate fit drift: got {m.growth_rate_per_hour}, expected ~0.5"
    )
    assert abs(m.mean_charging_fraction - (0.9 + 0.85 + 0.88) / 3) < 1e-9
    assert abs(m.mean_elongation_rate_aa_per_s - 18.0) < 1e-9


def test_exponential_fit_handles_degenerate_input() -> None:
    from consensus_validation_sweep import _exponential_fit_per_hour

    assert math.isnan(_exponential_fit_per_hour([], []))
    assert math.isnan(_exponential_fit_per_hour([1.0], [1.0]))
    # All zero or negative values → no positive values → NaN.
    assert math.isnan(_exponential_fit_per_hour([0.0, 1.0], [-1.0, 0.0]))


def test_check_acceptance_passes_when_metrics_meet_targets() -> None:
    from consensus_validation_sweep import Metrics, check_acceptance

    metrics = [
        Metrics(media="minimal", seed=0, n_generations=5, n_steps=4000,
                growth_rate_per_hour=0.7,
                mean_charging_fraction=0.92,
                ppgpp_starvation_rise=float("nan")),
        Metrics(media="minimal_acetate", seed=0, n_generations=5, n_steps=4000,
                growth_rate_per_hour=0.4,
                mean_charging_fraction=0.88,
                ppgpp_starvation_rise=3.0),
    ]
    checks = check_acceptance(metrics)
    by_name = {c.name: c for c in checks}
    assert by_name["trna_charging_fraction"].passed is True
    assert by_name["ppgpp_starvation_rise"].passed is True
    assert by_name["growth_rate_plausibility"].passed is True


def test_check_acceptance_fails_when_charging_below_target() -> None:
    from consensus_validation_sweep import Metrics, check_acceptance

    metrics = [
        Metrics(media="minimal", seed=0, n_generations=5, n_steps=4000,
                growth_rate_per_hour=0.7,
                mean_charging_fraction=0.50),  # well below 0.85
    ]
    checks = check_acceptance(metrics)
    fc_check = next(c for c in checks if c.name == "trna_charging_fraction")
    assert fc_check.passed is False


def test_generate_report_writes_json(tmp_path: Path) -> None:
    from consensus_validation_sweep import (
        AcceptanceCheck, Metrics, generate_report,
    )

    metrics = [
        Metrics(media="minimal", seed=0, n_generations=5, n_steps=4000,
                growth_rate_per_hour=0.7, mean_charging_fraction=0.9),
    ]
    checks = [
        AcceptanceCheck(name="dummy", passed=True, actual=1, expected=1),
    ]
    out = tmp_path / "report.json"
    report = generate_report(metrics, checks, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    # Structural checks (full-dict equality fails on NaN values in metrics).
    assert loaded["summary"] == report["summary"]
    assert loaded["summary"]["passed_all"] is True
    assert loaded["summary"]["checks_passed"] == 1
    assert loaded["summary"]["total_runs"] == 1
    assert len(loaded["metrics"]) == 1
    assert loaded["metrics"][0]["media"] == "minimal"
    assert loaded["metrics"][0]["growth_rate_per_hour"] == 0.7
    assert len(loaded["acceptance"]) == 1
    assert loaded["acceptance"][0]["name"] == "dummy"


# ============================================================
# §D — CLI
# ============================================================

def test_parse_args_defaults() -> None:
    from consensus_validation_sweep import _parse_args, DEFAULT_MEDIA, DEFAULT_SEEDS

    args = _parse_args([])
    assert args.cache_dir == "out/cache"
    assert args.media == DEFAULT_MEDIA
    assert args.seeds == DEFAULT_SEEDS
    assert args.max_generations == 5
    assert args.preflight_only is False
    assert args.force is False


def test_parse_args_overrides() -> None:
    from consensus_validation_sweep import _parse_args

    args = _parse_args([
        "--cache-dir", "out/alt",
        "--media", "minimal", "minimal_acetate",
        "--seeds", "7",
        "--max-generations", "2",
        "--preflight-only",
        "--force",
    ])
    assert args.cache_dir == "out/alt"
    assert args.media == ["minimal", "minimal_acetate"]
    assert args.seeds == [7]
    assert args.max_generations == 2
    assert args.preflight_only is True
    assert args.force is True


def test_main_preflight_only_short_circuits_with_missing_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from consensus_validation_sweep import main

    missing = tmp_path / "no_cache"
    rc = main([
        "--cache-dir", str(missing),
        "--preflight-only",
    ])
    assert rc == 1  # pre-flight failure exit code
    out = capsys.readouterr().out
    assert "PRE-FLIGHT FAILED" in out
