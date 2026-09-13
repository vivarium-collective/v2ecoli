"""``sampled_span`` -- a span that heartbeats while its body is blocked.

The gather's failure mode is one ``step.update(...)`` that blocks for hours and
then dies (sms-ecoli#166, Run 1). A plain span records that the work started and
never finished; every number that would explain WHY -- DuckDB's temp-directory
usage, RSS -- lives inside the blocked call and is currently sampled once, after
it returns, which on that path never happens. These tests pin the behaviour that
closes the gap, without needing DuckDB or a cluster.
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

from v2ecoli.workflow import events as revents  # noqa: E402


def _events_from(capsys) -> list[dict]:
    out = capsys.readouterr().out
    parsed = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return parsed


def test_a_blocked_body_still_reports_progress(capsys):
    """The point of the whole exercise: samples arrive DURING the body."""
    pbg_events.configure("stdout")
    readings = iter([{"duckdb_temp_mb": 100.0}, {"duckdb_temp_mb": 200.0},
                     {"duckdb_temp_mb": 300.0}])

    def sampler():
        return next(readings, {"duckdb_temp_mb": 999.0})

    with revents.sampled_span("analysis.group", sampler, interval_s=0.02,
                              name="ptools_rna_multigeneration"):
        time.sleep(0.25)

    events = _events_from(capsys)
    samples = [e for e in events if e.get("event") == "analysis.sample"]
    assert samples, "no sample arrived while the body was blocked"
    seen = [t.get("payload", {}).get("duckdb_temp_mb") for t in samples]
    assert any(v is not None for v in seen), f"sampler output never reached an event: {samples[:2]}"

    names = [e.get("event") for e in events]
    assert "span.start" in names and "span.end" in names


def test_a_raising_body_ends_the_span_as_an_error_and_reraises(capsys):
    """An OOM inside the gather must surface as an error span, not a lost span."""
    pbg_events.configure("stdout")
    with pytest.raises(MemoryError):
        with revents.sampled_span("analysis.group", lambda: {"duckdb_temp_mb": 1.0},
                                  interval_s=0.01, name="x"):
            raise MemoryError("Out of Memory Error: failed to offload data block")

    ends = [e for e in _events_from(capsys) if e.get("event") == "span.end"]
    assert ends, "no span.end on the failing path"
    assert ends[-1]["payload"]["status"] == "error"
    assert "MemoryError" in (ends[-1]["payload"].get("error") or "")


def test_a_raising_sampler_never_breaks_the_body(capsys):
    """Observability must never take down the run it is observing."""
    pbg_events.configure("stdout")

    def bad_sampler():
        raise RuntimeError("duckdb_memory() unavailable")

    with revents.sampled_span("analysis.group", bad_sampler, interval_s=0.01, name="x"):
        time.sleep(0.08)
    # reaching here at all is the assertion; confirm the span still closed ok
    ends = [e for e in _events_from(capsys) if e.get("event") == "span.end"]
    assert ends and ends[-1]["payload"]["status"] == "ok"


def test_the_sampler_thread_does_not_outlive_the_span():
    """A daemon thread per analysis group would leak across a long sweep."""
    pbg_events.configure("stdout")
    before = threading.active_count()
    for _ in range(5):
        with revents.sampled_span("analysis.group", lambda: {"a": 1},
                                  interval_s=0.01, name="x"):
            time.sleep(0.03)
    time.sleep(0.1)
    assert threading.active_count() <= before, "sampler threads leaked"


def test_it_is_inert_when_no_sink_is_configured(capsys):
    """Off a dispatched run there must be no threads and no output."""
    pbg_events.configure("none")
    before = threading.active_count()
    with revents.sampled_span("analysis.group", lambda: {"a": 1}, interval_s=0.01, name="x"):
        time.sleep(0.05)
    assert threading.active_count() <= before
    assert not _events_from(capsys)
