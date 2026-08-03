"""Task 3 (comparison convergence Phase 1): the run-store adapter translates
a `vivarium-workbench` run reference into the observables shape the existing
report cards consume. Hermetic — no engine run, no S3.

Fixtures used:
  - ``tests/fixtures/redux_cards/v2ecoli_seed00.zarr`` — a REAL, committed
    "v2ecoli-format" zarr store (see tests/fixtures/redux_cards/README.md),
    used unmodified (read-only) as a stand-in for a workbench run's stored
    xarray-emitter output — confirmed structurally identical (lineage_seed
    group with direct observable-leaf children, ``generation=N`` data vars)
    to what the workbench's XArrayEmitter writes (see run_store_adapter.py's
    module docstring for the citations).
  - a synthetic ``runs.db`` (the vendored ``runs_meta`` DDL) built here to
    exercise the run_id -> emitter_path -> zarr resolution path without any
    real workbench install.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts._compare.run_store_adapter import (
    RunStoreError, load_run_observables, resolve_run_store)

FIXTURE_ZARR = (Path(__file__).parent.parent / "fixtures" / "redux_cards"
                / "v2ecoli_seed00.zarr")

RUNS_META_DDL = """
CREATE TABLE IF NOT EXISTS runs_meta (
    run_id        TEXT PRIMARY KEY,
    spec_id       TEXT NOT NULL,
    label         TEXT,
    params_json   TEXT,
    started_at    REAL NOT NULL,
    completed_at  REAL,
    n_steps       INTEGER,
    status        TEXT NOT NULL,
    sim_name      TEXT,
    generation_id TEXT,
    emitter_path  TEXT
);
"""


def _make_runs_db(runs_db: Path, run_id: str, emitter_path: str) -> None:
    conn = sqlite3.connect(runs_db)
    try:
        conn.executescript(RUNS_META_DDL)
        conn.execute(
            "INSERT INTO runs_meta(run_id, spec_id, started_at, status, emitter_path)"
            " VALUES (?, 'vecoli', 0.0, 'complete', ?)",
            (run_id, emitter_path),
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# resolve_run_store — direct path forms
# --------------------------------------------------------------------------- #

def test_resolve_direct_zarr_path_returns_as_is():
    assert resolve_run_store(str(FIXTURE_ZARR)) == FIXTURE_ZARR


def test_resolve_run_dir_with_bare_zarr_child(tmp_path):
    run_dir = tmp_path / "out" / "run_abc123"
    zarr_dir = run_dir / "v2ecoli_seed00.zarr"
    zarr_dir.mkdir(parents=True)
    assert resolve_run_store(run_dir) == zarr_dir


def test_resolve_run_dir_with_store_zarr_nested(tmp_path):
    # vwb XArrayEmitter convention: <run_dir>/store.zarr
    run_dir = tmp_path / "out" / "run_xyz"
    store = run_dir / "store.zarr"
    store.mkdir(parents=True)
    assert resolve_run_store(run_dir) == store


def test_resolve_missing_path_raises():
    with pytest.raises(RunStoreError):
        resolve_run_store("/no/such/path/anywhere.zarr")


# --------------------------------------------------------------------------- #
# resolve_run_store — run_id via runs.db
# --------------------------------------------------------------------------- #

def test_resolve_via_run_id_and_runs_db(tmp_path):
    study_dir = tmp_path / "study"
    run_dir = study_dir / "out" / "r1"
    zarr_dir = run_dir / "vecoli_seed00.zarr"
    zarr_dir.mkdir(parents=True)
    _make_runs_db(study_dir / "runs.db", "r1", "out/r1/vecoli_seed00.zarr")

    resolved = resolve_run_store("r1", study_dir=study_dir)
    assert resolved == zarr_dir


def test_resolve_via_run_id_mapping_form(tmp_path):
    study_dir = tmp_path / "study2"
    run_dir = study_dir / "out" / "r2"
    zarr_dir = run_dir / "v2ecoli_seed00.zarr"
    zarr_dir.mkdir(parents=True)
    _make_runs_db(study_dir / "runs.db", "r2", "out/r2/v2ecoli_seed00.zarr")

    resolved = resolve_run_store({"run_id": "r2", "study_dir": study_dir})
    assert resolved == zarr_dir


def test_resolve_via_run_id_unknown_raises(tmp_path):
    study_dir = tmp_path / "study3"
    study_dir.mkdir(parents=True)
    _make_runs_db(study_dir / "runs.db", "known", "out/known/x.zarr")
    with pytest.raises(RunStoreError):
        resolve_run_store("unknown-run", study_dir=study_dir)


def test_resolve_run_id_without_study_dir_or_runs_db_raises():
    with pytest.raises(RunStoreError):
        resolve_run_store("bare-run-id")


# --------------------------------------------------------------------------- #
# load_run_observables — the card-facing contract
# --------------------------------------------------------------------------- #

def test_load_run_observables_matches_read_pbg_local_directly():
    """The adapter must return exactly what the trajectory/distribution/
    metabolism/composition cards already get from ``read_pbg_local`` — same
    keys, same (times, values) arrays — for the SAME committed fixture."""
    from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local

    expected = read_pbg_local(str(FIXTURE_ZARR), OBSERVABLES)
    got = load_run_observables(str(FIXTURE_ZARR))

    assert set(got) == set(expected)
    assert "cell_mass" in got
    assert "_generation" in got
    for obs in expected:
        exp_t, exp_v = expected[obs]
        got_t, got_v = got[obs]
        assert list(got_t) == list(exp_t)
        assert list(got_v) == list(exp_v)


def test_load_run_observables_shape_and_known_values():
    """Pin the concrete shape a card would consume: {obs: (times, values)}
    with the fixture's known first sample (verified directly against the
    fixture zarr; see this file's setup)."""
    obs = load_run_observables(str(FIXTURE_ZARR), observables=["cell_mass"])

    assert set(obs) >= {"cell_mass", "_generation"}
    times, values = obs["cell_mass"]
    assert len(times) == len(values) == 5
    assert times[0] == pytest.approx(60.2)
    assert values[0] == pytest.approx(1280.973673270876)


def test_load_run_observables_via_run_dir(tmp_path):
    """End-to-end: run_id -> runs.db -> emitter_path -> zarr -> observables,
    using a symlink to the real fixture zarr as the resolved run store (so
    the adapter's resolution AND read path are both exercised together)."""
    study_dir = tmp_path / "study"
    run_dir = study_dir / "out" / "cand1"
    run_dir.mkdir(parents=True)
    linked = run_dir / "v2ecoli_seed00.zarr"
    linked.symlink_to(FIXTURE_ZARR, target_is_directory=True)
    _make_runs_db(study_dir / "runs.db", "cand1", "out/cand1/v2ecoli_seed00.zarr")

    obs = load_run_observables("cand1", study_dir=study_dir,
                               observables=["dry_mass"])
    assert "dry_mass" in obs
    times, values = obs["dry_mass"]
    assert len(times) == 5
    assert values[0] == pytest.approx(384.4834763764661)
