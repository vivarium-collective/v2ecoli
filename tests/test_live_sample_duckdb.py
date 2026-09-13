"""``_live_sample`` against a REAL DuckDB connection, while a query is running.

The unit tests for ``sampled_span`` use a fake sampler, which proves the span and
heartbeat plumbing but not the thing most likely to be wrong in the gather: a
second thread touching DuckDB while the first thread is mid-query.
``_run_duckdb_name``'s own docstring is explicit that a cursor is NOT safe to use
from two threads at once -- only separate cursors on one connection are -- so the
sampler gets its own cursor, and that is the claim these tests check.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

pytestmark = pytest.mark.fast

pbg_events = pytest.importorskip(
    "process_bigraph.events", reason="process-bigraph >= 1.9 (feat/events) required"
)
duckdb = pytest.importorskip("duckdb")

from v2ecoli.workflow import events as revents  # noqa: E402
from v2ecoli.workflow.analysis_runner import _live_sample  # noqa: E402


def _events_from(capsys) -> list[dict]:
    out = capsys.readouterr().out
    parsed = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return parsed


def test_live_sample_reports_real_duckdb_numbers():
    conn = duckdb.connect()
    sample = _live_sample(conn.cursor())
    assert "duckdb_memory_mb" in sample, sample
    assert "duckdb_temp_mb" in sample, sample
    assert "rss_mb" in sample, sample
    assert all(isinstance(v, (int, float)) for v in sample.values()), sample


def test_live_sample_never_raises_on_a_dead_cursor():
    """A metric must never fail the analysis -- including after the conn closed."""
    conn = duckdb.connect()
    cur = conn.cursor()
    conn.close()
    sample = _live_sample(cur)  # must not raise
    assert "duckdb_memory_mb" not in sample  # degraded, not fatal
    assert "rss_mb" in sample  # the part that still works still works


def test_sampling_a_separate_cursor_is_safe_while_a_query_runs(capsys):
    """The gather's actual shape: probe thread + query thread, one connection.

    Uses a deliberately slow query so the sampler is guaranteed to run while the
    main thread is blocked inside DuckDB, which is exactly the situation that
    would surface a cursor-sharing bug.

    The query is sized so the test cannot be a wall-clock race: a 40M-row ordered
    window measures ~0.95 s locally against a 20 ms sampling interval, i.e. ~45
    expected samples. My first draft used a 10M-row aggregate, which DuckDB
    finishes in 9 ms -- the test failed, and it was the TEST that was wrong, not
    the code. Do not shrink this query to make the suite faster; a margin this
    wide is the only thing keeping the assertion non-flaky.
    """
    pbg_events.configure("stdout")
    conn = duckdb.connect()
    query_cursor = conn.cursor()
    probe_cursor = conn.cursor()          # the fix: NOT query_cursor

    with revents.sampled_span(
        "analysis.group", lambda c=probe_cursor: _live_sample(c),
        interval_s=0.02, name="synthetic", scale="multigeneration",
    ):
        # ~0.95 s of real DuckDB work: sampled ~45 times mid-flight
        got = query_cursor.execute(
            "SELECT count(*) FROM (SELECT i, sum(i) OVER (ORDER BY i) "
            "FROM range(40000000) t(i))"
        ).fetchone()

    assert got[0] == 40_000_000

    events = _events_from(capsys)
    samples = [e for e in events if e.get("event") == "analysis.sample"]
    assert samples, "sampler never ran while the query was blocked"
    assert any("duckdb_temp_mb" in (t.get("payload") or {}) for t in samples), \
        f"no DuckDB reading reached an event: {[t.get('payload') for t in samples[:2]]}"

    ends = [e for e in events if e.get("event") == "span.end"]
    assert ends and ends[-1]["payload"]["status"] == "ok"
    assert ends[-1]["payload"]["attrs"]["scale"] == "multigeneration"


def test_no_sampler_thread_survives_the_query():
    pbg_events.configure("stdout")
    conn = duckdb.connect()
    before = threading.active_count()
    for _ in range(3):
        with revents.sampled_span("analysis.group",
                                  lambda c=conn.cursor(): _live_sample(c),
                                  interval_s=0.01, name="x"):
            conn.cursor().execute("SELECT count(*) FROM range(200000)").fetchone()
    time.sleep(0.1)
    assert threading.active_count() <= before
