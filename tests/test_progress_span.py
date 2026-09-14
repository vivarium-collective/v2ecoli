"""``progress_span`` -- a span plus a throttled report for a caller-driven loop.

The companion to ``sampled_span``. That one samples a blocking call from a
thread because the caller never gets control back; this one is for a loop whose
body IS the natural callback, where the need is rate limiting rather than a
thread. ParCa's per-condition fit is the motivating case: up to 200 iterations,
~6 min per condition on the cluster, and until now completely silent between
"starting" and "converged".
"""

from __future__ import annotations

import json
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
        if line.startswith("{"):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return parsed


def test_the_first_report_is_never_throttled(capsys):
    """"It started and is progressing" must not wait a whole interval."""
    pbg_events.configure("stdout")
    with revents.progress_span("parca.fit_condition", interval_s=3600,
                               condition="basal") as report:
        assert report(iteration=0, degree_of_fit=1.0) is True

    progress = [e for e in _events_from(capsys)
                if e.get("event") == "parca.fit_condition.progress"]
    assert len(progress) == 1
    assert progress[0]["payload"]["iteration"] == 0
    # the span's identity must ride on the progress event too, or the stream is
    # unreadable without joining on span ids
    assert progress[0]["payload"]["condition"] == "basal"


def test_later_reports_are_throttled(capsys):
    """200 iterations must not become 200 events."""
    pbg_events.configure("stdout")
    with revents.progress_span("parca.fit_condition", interval_s=3600,
                               condition="basal") as report:
        emitted = [report(iteration=i, degree_of_fit=1.0 / (i + 1)) for i in range(200)]

    assert emitted[0] is True
    assert not any(emitted[1:]), "throttle let a second report through"
    progress = [e for e in _events_from(capsys)
                if e.get("event") == "parca.fit_condition.progress"]
    assert len(progress) == 1


def test_the_interval_actually_releases(capsys):
    pbg_events.configure("stdout")
    with revents.progress_span("parca.fit_condition", interval_s=0.05,
                               condition="basal") as report:
        assert report(iteration=0, degree_of_fit=1.0) is True
        assert report(iteration=1, degree_of_fit=0.5) is False
        time.sleep(0.08)
        assert report(iteration=2, degree_of_fit=0.25) is True


def test_a_raising_body_ends_the_span_as_an_error_and_reraises(capsys):
    """"Fitting did not converge" must surface as a failed span, not a lost one."""
    pbg_events.configure("stdout")
    with pytest.raises(Exception, match="did not converge"):
        with revents.progress_span("parca.fit_condition", condition="basal") as report:
            report(iteration=0, degree_of_fit=1.0)
            raise Exception("Fitting did not converge")

    ends = [e for e in _events_from(capsys) if e.get("event") == "span.end"]
    assert ends and ends[-1]["payload"]["status"] == "error"
    assert "did not converge" in (ends[-1]["payload"].get("error") or "")
    assert ends[-1]["payload"]["attrs"]["condition"] == "basal"


def test_a_failing_sink_disables_reporting_instead_of_raising(capsys):
    """Observability must never take down the fit it is observing."""
    pbg_events.configure("stdout")
    emitter = pbg_events.get_emitter()
    original = emitter.event

    def boom(*a, **k):
        raise RuntimeError("sink exploded")

    with revents.progress_span("parca.fit_condition", interval_s=0,
                               condition="basal") as report:
        emitter.event = boom
        try:
            assert report(iteration=0, degree_of_fit=1.0) is False  # swallowed
            assert report(iteration=1, degree_of_fit=0.5) is False  # stays dead
        finally:
            emitter.event = original


def test_it_is_inert_when_no_sink_is_configured(capsys):
    pbg_events.configure("none")
    with revents.progress_span("parca.fit_condition", interval_s=0,
                               condition="basal") as report:
        report(iteration=0, degree_of_fit=1.0)
    assert not _events_from(capsys)
