"""EmitterHistorySummary — a generic post-sim visualization for a run whose
captured output is a plain in-memory emitter's persisted trajectory
(``emitter_history.json``, written by ``viva_api.compose.run_pbg``'s
``_persist_emitter_history`` fallback — backlog item 88) and/or just the
always-present ``final_state.json`` snapshot.

Unlike every class under ``v2ecoli/visualizations/`` (``ColonyVisualization``/
``ColonyGrowthGif``/etc — all ``viva_superpowers.visualization.Visualization``,
the process-bigraph-native accumulate/render or update() Step family, wired
live INTO a composite document and invoked during/immediately after a run in
the SAME process), this is a ``v2ecoli.workflow.post_sim.Visualization``
(= ``VisualizationStep``, the POST_SIM_REGISTRY family ``run_flush()``
discovers via ``iter_post_sim("visualization")``) — a genuinely SEPARATE
mechanism for a post-hoc analysis reading a FINISHED run's on-disk output,
possibly from an entirely different process (e.g. a standalone AWS Batch
"analysis flush" job with no access to the original composite/Step
instances). This is the first concrete subclass of that family in v2ecoli;
every other registered "visualization" kind slot has been empty until now.

Deliberately lives here, not under ``v2ecoli/visualizations/`` (whose own
``__init__.py`` — and a real regression test, ``test_visualizations_
discovery.py::test_visualizations_are_visualization_subclasses`` — assert
every entry ``bigraph_schema.allocate_core()`` auto-registers under that
module prefix belongs to the OTHER, live-Step family). Mirrors the existing
``v2ecoli/workflow/analyses/`` and ``v2ecoli/workflow/report_cards/``
convention instead: a caller that wants this registered must explicitly
``import v2ecoli.workflow.post_sim_visualizations`` first (``scripts/
run_multi_node_analysis.py`` does this before calling ``run_flush``) — plain
``import v2ecoli`` alone does not cascade into it, exactly as it does not for
``workflow.analyses``/``workflow.report_cards`` either (see ``v2ecoli/
__init__.py``, which only imports ``composites``/``visualizations``).

Reads ``out_dir`` directly (``VisualizationStep``'s own documented "override
inputs() to consume the run extraction instead" escape hatch) rather than the
DuckDB/hive-parquet machinery every other v2ecoli analysis needs
(``v2ecoli.workflow.analysis_runner.history_files`` only globs ``*.pq`` —
there is none to find for a plain in-memory emitter). This is what makes it
render something real for ANY composite dispatched through the generic
``run_pbg.py`` runner with a non-file-backed emitter, not colony-specific —
deliberately no per-composite hardcoding (no assumption of "cells"/"agents"/
"mass"/etc. — that is composite-specific interpretation, left to a future,
composite-aware step, exactly as ``v2ecoli/composites/ecoli_colony.py``'s own
comment anticipates: "richer colony-specific views ... are being added as
post-hoc analyses that read the emitter output in the flush").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v2ecoli.workflow.post_sim import Visualization

_MAX_PREVIEW_KEYS = 12


def _load_json(path: Path) -> Any:
    """Best-effort JSON load: a missing or unparseable file is a legitimate
    "nothing here" (never this function's error to raise) -- run_pbg.py's own
    _persist_emitter_history is itself a best-effort fallback that may simply
    not have run for a given composite."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _summarize_entries(entries: Any) -> dict[str, Any]:
    """A best-effort, shape-agnostic summary of one emitter's gathered rows —
    a list of ``(time, state)`` tuples/lists or flat dicts carrying their own
    ``time`` key (the same dual shape run_pbg.py's own JSON round-trip
    produces: a Python tuple serializes as a 2-element JSON list). Counts
    records and derives the observed time span; never assumes any
    domain-specific state keys."""
    if not isinstance(entries, list):
        return {"n_records": 0, "t_start": None, "t_end": None}
    times: list[float] = []
    for entry in entries:
        t: Any = None
        if isinstance(entry, list | tuple) and len(entry) == 2:
            t = entry[0]
        elif isinstance(entry, dict):
            t = entry.get("time")
        if isinstance(t, int | float):
            times.append(float(t))
    return {
        "n_records": len(entries),
        "t_start": min(times) if times else None,
        "t_end": max(times) if times else None,
    }


def _render_html(
    history: dict[str, Any] | None, final_state: dict[str, Any] | None, summaries: dict[str, Any]
) -> str:
    from html import escape

    if summaries:
        rows = "".join(
            f"<tr><td>{escape(str(path))}</td><td>{s['n_records']}</td>"
            f"<td>{s['t_start']}</td><td>{s['t_end']}</td></tr>"
            for path, s in summaries.items()
        )
        history_html = (
            "<table border='1' cellpadding='4' cellspacing='0'>"
            "<tr><th>emitter path</th><th>records</th><th>t start</th><th>t end</th></tr>"
            f"{rows}</table>"
        )
    else:
        history_html = "<p>No in-memory emitter history was captured for this run.</p>"

    if isinstance(final_state, dict):
        keys = sorted(final_state.keys())
        shown = keys[:_MAX_PREVIEW_KEYS]
        more = f" (+{len(keys) - _MAX_PREVIEW_KEYS} more)" if len(keys) > _MAX_PREVIEW_KEYS else ""
        final_state_html = f"<p>{escape(', '.join(shown)) or '(none)'}{more}</p>"
    else:
        final_state_html = "<p>final_state.json not available for this run.</p>"

    return (
        "<h2>Emitter history summary</h2>"
        f"{history_html}"
        "<h3>Final state top-level keys</h3>"
        f"{final_state_html}"
    )


class EmitterHistorySummary(Visualization):
    """Renders a basic, honest summary of a run's captured emitter history
    (record counts + observed time span per emitter path) plus the top-level
    keys present in ``final_state.json``. Degrades to a clear "nothing
    captured" notice — never raises — when neither file is present under
    ``out_dir``: a run whose every emitter was file-backed (so nothing had to
    fall back to ``emitter_history.json``) is this step's expected, honest
    no-op, not an error.
    """

    name = "emitter_history_summary"
    config_schema: dict = {}

    def inputs(self) -> dict[str, Any]:
        return {"out_dir": "string"}

    def render(self, out_dir: str) -> "tuple[str, dict] | None":
        base = Path(out_dir)
        history = _load_json(base / "emitter_history.json")
        final_state = _load_json(base / "final_state.json")
        if history is None and final_state is None:
            return None

        summaries: dict[str, Any] = {}
        if isinstance(history, dict):
            summaries = {path: _summarize_entries(entries) for path, entries in history.items()}

        html = _render_html(history, final_state, summaries)
        data = {
            "emitters": summaries,
            "final_state_keys": sorted(final_state.keys()) if isinstance(final_state, dict) else [],
        }
        return html, data

    def update(self, state: dict[str, Any], interval: Any = None) -> dict[str, Any]:
        # Base VisualizationStep.update() hardcodes state["study"] -> render(study);
        # this step consumes out_dir instead (see this module's own docstring and
        # VisualizationStep.render's own "override inputs() to consume the run
        # extraction instead" contract), so update() must be overridden too.
        out_dir = state.get("out_dir") or ""
        res = self.render(out_dir)
        if not res:
            return {"view": "", "data": {}}
        view, data = res
        return {"view": view, "data": data}
