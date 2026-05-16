"""Test fixtures for dnaa-01-expression-dynamics behavioral tests.

Loads simulation history from this study's runs.db (the SQLite DB that
process_bigraph.emitter.SQLiteEmitter writes via the dashboard's run
machinery).

Skips cleanly when no runs exist yet, so the test suite is meaningful
even before the first baseline / variant has been executed.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


STUDY_DIR = Path(__file__).resolve().parents[1]
RUNS_DB = STUDY_DIR / "runs.db"


def _open_runs_db() -> sqlite3.Connection | None:
    if not RUNS_DB.exists():
        return None
    return sqlite3.connect(str(RUNS_DB))


def _latest_run_id(conn: sqlite3.Connection, name: str | None) -> str | None:
    """Return the most recently started simulation_id, optionally filtered
    by run name (matches a `simulation_set:` entry in study.yaml)."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='simulations'"
    )
    if cur.fetchone() is None:
        return None
    if name is None:
        row = conn.execute(
            "SELECT simulation_id FROM simulations "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT simulation_id FROM simulations WHERE name=? "
            "ORDER BY started_at DESC LIMIT 1",
            (name,),
        ).fetchone()
    return row[0] if row else None


def _load_history(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    """Return a list of {step, time, state} dicts, ordered by step.

    Each row's ``state`` column is a JSON blob of the per-step subtree the
    composite's emit_schema declared. The fixture's tests treat the result
    as a list of ``{step, time, state}`` records.
    """
    rows = conn.execute(
        "SELECT step, global_time, state FROM history WHERE simulation_id=? "
        "ORDER BY step ASC",
        (run_id,),
    ).fetchall()
    return [
        {"step": step, "time": time, "state": json.loads(state)}
        for step, time, state in rows
    ]


@pytest.fixture(scope="session")
def baseline_history():
    """Latest `baseline-steady-state` run, or skip if none."""
    conn = _open_runs_db()
    if conn is None:
        pytest.skip(f"no runs.db at {RUNS_DB}; run the baseline first "
                    f"(python studies/dnaa-01-expression-dynamics/sims/run_baseline.py)")
    try:
        run_id = (_latest_run_id(conn, "baseline-steady-state")
                  or _latest_run_id(conn, None))
        if run_id is None:
            pytest.skip("no runs in runs.db")
        return _load_history(conn, run_id)
    finally:
        conn.close()


@pytest.fixture(scope="session")
def stop_synthesis_history():
    """Latest `stop-dnaA-synthesis` variant run, or skip."""
    conn = _open_runs_db()
    if conn is None:
        pytest.skip(f"no runs.db at {RUNS_DB}; run the variant first")
    try:
        run_id = _latest_run_id(conn, "stop-dnaA-synthesis")
        if run_id is None:
            pytest.skip("no stop-dnaA-synthesis run in runs.db")
        return _load_history(conn, run_id)
    finally:
        conn.close()


# ─── Identifier helpers ─────────────────────────────────────────────────────

DNAA_MONOMER_ID = "MONOMER0-160[c]"
DNAA_MRNA_ID = "EG10235_RNA"
DNAA_TF_ID = "MONOMER0-160"


def bulk_count(state: dict, molecule_id: str) -> int | None:
    """Return the count of a bulk molecule by ID from one history snapshot.

    The bulk store is a `bulk_array`: a structured array with `id` and
    `count` columns, serialized to JSON as two parallel lists.
    """
    # Try canonical agent path: agents.<id>.bulk
    agents = state.get("agents") or {}
    if not agents:
        return None
    first_agent = next(iter(agents.values()))
    bulk = first_agent.get("bulk")
    if bulk is None:
        return None
    if isinstance(bulk, dict) and "id" in bulk and "count" in bulk:
        ids = bulk["id"]
        counts = bulk["count"]
    elif isinstance(bulk, list) and bulk and isinstance(bulk[0], (list, tuple)):
        # Tuples of (id, count) — alternate serialization.
        ids = [row[0] for row in bulk]
        counts = [row[1] for row in bulk]
    else:
        return None
    try:
        idx = ids.index(molecule_id)
    except ValueError:
        return None
    return counts[idx]


def listener_value(state: dict, path: str):
    """Walk a dotted path inside the first agent's listener tree.

    e.g. `listener_value(state, 'listeners.rnap_data.rna_init_event')`.
    Returns None if any segment is missing.
    """
    agents = state.get("agents") or {}
    if not agents:
        return None
    cursor = next(iter(agents.values()))
    for seg in path.split("."):
        if not isinstance(cursor, dict) or seg not in cursor:
            return None
        cursor = cursor[seg]
    return cursor
