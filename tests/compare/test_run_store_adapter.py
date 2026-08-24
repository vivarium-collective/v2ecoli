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


def _make_runs_db_rows(runs_db: Path, rows: "list[dict]") -> None:
    """Like ``_make_runs_db`` but inserts an arbitrary set of rows — used by
    the sim_name-resolution tests below, which need distinct ``sim_name``
    columns (and, for the ambiguity-ordering test, multiple rows sharing one
    sim_name at different ``started_at`` times)."""
    conn = sqlite3.connect(runs_db)
    try:
        conn.executescript(RUNS_META_DDL)
        for row in rows:
            conn.execute(
                "INSERT INTO runs_meta(run_id, spec_id, started_at, status,"
                " sim_name, emitter_path) VALUES (?, 'vecoli', ?, 'complete', ?, ?)",
                (row["run_id"], row.get("started_at", 0.0), row.get("sim_name"),
                 row["emitter_path"]),
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


# --------------------------------------------------------------------------- #
# resolve_run_store — sim_name resolution (Phase B native comparison_cards
# passes candidate_run/reference_run as sim_names, not run_ids — the
# baseline/candidate's is <config> (e.g. "basal"), the variant's is
# "reference" — see run_store_adapter.py's _lookup_emitter_path docstring).
# --------------------------------------------------------------------------- #

def _sim_name_runs_db(tmp_path) -> Path:
    runs_db = tmp_path / "runs.db"
    _make_runs_db_rows(runs_db, [
        {"run_id": "rid-cand", "sim_name": "basal",
         "emitter_path": str(tmp_path / "cand" / "store.zarr")},
        {"run_id": "rid-ref", "sim_name": "reference",
         "emitter_path": str(tmp_path / "ref" / "store.zarr")},
    ])
    (tmp_path / "cand" / "store.zarr").mkdir(parents=True)
    (tmp_path / "ref" / "store.zarr").mkdir(parents=True)
    return runs_db


def test_resolve_by_sim_name_candidate(tmp_path):
    runs_db = _sim_name_runs_db(tmp_path)
    resolved = resolve_run_store("basal", runs_db=runs_db)
    assert resolved == tmp_path / "cand" / "store.zarr"


def test_resolve_by_sim_name_reference(tmp_path):
    runs_db = _sim_name_runs_db(tmp_path)
    resolved = resolve_run_store("reference", runs_db=runs_db)
    assert resolved == tmp_path / "ref" / "store.zarr"


def test_resolve_by_run_id_still_tried_first(tmp_path):
    runs_db = _sim_name_runs_db(tmp_path)
    resolved = resolve_run_store("rid-cand", runs_db=runs_db)
    assert resolved == tmp_path / "cand" / "store.zarr"


def test_resolve_unknown_ref_raises_clear_error(tmp_path):
    runs_db = _sim_name_runs_db(tmp_path)
    with pytest.raises(RunStoreError):
        resolve_run_store("totally-unknown", runs_db=runs_db)


def test_resolve_mapping_form_run_id_still_works(tmp_path):
    runs_db = _sim_name_runs_db(tmp_path)
    resolved = resolve_run_store({"run_id": "rid-ref"}, runs_db=runs_db)
    assert resolved == tmp_path / "ref" / "store.zarr"


def test_resolve_by_sim_name_picks_most_recent_on_ambiguity(tmp_path):
    """Two rows share sim_name="basal" (e.g. a re-run) — resolution must
    pick the most recently started one, not an arbitrary row."""
    runs_db = tmp_path / "runs.db"
    older = tmp_path / "older" / "store.zarr"
    newer = tmp_path / "newer" / "store.zarr"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    _make_runs_db_rows(runs_db, [
        {"run_id": "rid-old", "sim_name": "basal", "started_at": 1.0,
         "emitter_path": str(older)},
        {"run_id": "rid-new", "sim_name": "basal", "started_at": 2.0,
         "emitter_path": str(newer)},
    ])
    resolved = resolve_run_store("basal", runs_db=runs_db)
    assert resolved == newer


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
