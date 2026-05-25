"""DnaA-investigation Visualization Steps.

Two classes referenced by ``studies/dnaa-0{2,3,4}/study.yaml``:

  - ``DnaAStateVisualization``: nucleotide-state pool composition + DnaA-ATP
    fraction over time, with the Boesen 2024 physiological band overlaid.
    Used by dnaa-02..04 (any study that emits the dnaA_cycle listener).

  - ``DnaABoxOccupancyVisualization``: DnaA-box occupancy trajectories
    (chromosomal vs oriC vs dnaAp; high- vs low-affinity oriC). Used by
    dnaa-03 + dnaa-04.

Inputs are bound from the run's SQLiteEmitter history via the study yaml's
``visualizations[].config.inputs_map`` block (port → observable path). The
classes are tolerant of partial data: missing ports render as empty traces
so a study still shows *something* even when only a subset of observables
exist.

Rendering uses Plotly via CDN — produces self-contained interactive HTML
that drops into the dashboard's Visualizations tab and also into the
downloadable investigation report.
"""

from __future__ import annotations

import json
import math
from html import escape
from typing import Any, Sequence

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_document


_PLOTLY_CDN = (
    '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8">'
    '</script>'
)


def _to_runs(value: Any) -> list[list[float]]:
    """Normalize a port value into a list of run-trajectories.

    Single-run input arrives as ``list[float]``; multi-run as
    ``list[list[float]]``. Returns the multi-run form, treating scalars /
    None as empty.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return []
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return [list(v or []) for v in value]
    return [list(value)]


def _coerce_labels(value: Any, n: int) -> list[str]:
    if isinstance(value, (list, tuple)) and value:
        labels = [str(v) for v in value]
        if len(labels) < n:
            labels.extend(f"run {i}" for i in range(len(labels), n))
        return labels[:n]
    return [f"run {i}" for i in range(n)]


def _trace(name: str, x: Sequence[float], y: Sequence[float],
           color: str | None = None, dash: str | None = None) -> dict:
    line: dict[str, Any] = {"width": 2}
    if color:
        line["color"] = color
    if dash:
        line["dash"] = dash
    def _num(v):
        # Tolerate non-numeric points (None, NaN, or a stray dict/list from an
        # empty-container '{}' capture) — render them as a gap rather than
        # crashing the whole report on float({}).
        if v is None or isinstance(v, (dict, list)):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    return {
        "type": "scatter",
        "mode": "lines",
        "name": name,
        "x": list(x),
        "y": [_num(v) for v in y],
        "line": line,
    }


#: Cell-cycle event visual styles. Distinct color + dash patterns so
#: stacked overlays read at a glance: initiation = solid green
#: (positive event, ATP-driven), division = dashed red (terminal event,
#: state-reshape). Completion deferred until v2ecoli exposes a clean
#: completion-event observable (current replication_data only carries
#: in-progress fork state).
_EVENT_STYLES: dict[str, dict] = {
    "initiation": {"color": "#16a34a", "dash": "solid", "label": "init"},
    "division":   {"color": "#dc2626", "dash": "dash",  "label": "div"},
    # "completion": deferred — needs a dedicated listener
}


def _event_shapes(events: dict[str, Sequence[float]] | None) -> list[dict]:
    """Build Plotly `shape` dicts (vertical lines) marking cell-cycle events.

    Args:
      events: ``{"initiation": [t0, t1, ...], "division": [...]}``.
        Keys not in ``_EVENT_STYLES`` are skipped. None / empty values
        yield no shapes.

    Returns: list of Plotly shape dicts ready to merge into
    ``layout["shapes"]``. Lines span the full y-axis via paper-y refs
    so they sit visually above every trace.
    """
    if not events:
        return []
    out: list[dict] = []
    for ev_type, times in events.items():
        style = _EVENT_STYLES.get(ev_type)
        if not style or not times:
            continue
        for t in times:
            try:
                t = float(t)
            except (TypeError, ValueError):
                continue
            out.append({
                "type": "line",
                "xref": "x", "yref": "paper",
                "x0": t, "x1": t, "y0": 0, "y1": 1,
                "line": {"color": style["color"], "width": 1.5,
                         "dash": style["dash"]},
                "opacity": 0.6,
                "layer": "above",
            })
    return out


def _event_annotations(events: dict[str, Sequence[float]] | None) -> list[dict]:
    """Build short text labels for each event line (top of plot)."""
    if not events:
        return []
    out: list[dict] = []
    for ev_type, times in events.items():
        style = _EVENT_STYLES.get(ev_type)
        if not style or not times:
            continue
        for t in times:
            try:
                t = float(t)
            except (TypeError, ValueError):
                continue
            out.append({
                "x": t, "y": 1.02, "xref": "x", "yref": "paper",
                "text": style["label"], "showarrow": False,
                "font": {"size": 9, "color": style["color"]},
                "xanchor": "center", "yanchor": "bottom",
            })
    return out


def _plotly_div(div_id: str, traces: list[dict], layout: dict,
                events: dict[str, Sequence[float]] | None = None) -> str:
    """Emit a Plotly chart inline as a div + script.

    Chart height auto-scales with trace count so multi-run overlays
    (10+ traces) don't shrink the chart to fit a wrapped horizontal
    legend. Formula: 280 px chart area + 28 px per legend row, where a
    row wraps every ~4 traces. Caps the lower bound at 340 px so
    single-trace charts don't shrink. Layout `height` is set in
    parallel so Plotly's internal layout engine sizes things right
    (legend orientation: "h", y: -0.25 puts the legend BELOW the
    chart, where it needs the room).

    ``events``: optional ``{"initiation": [t0, t1, ...], "division": [...]}``
    — overlays vertical lines + top labels at each event time, color+
    dash-coded per ``_EVENT_STYLES``. Caller-supplied ``layout.shapes``
    / ``layout.annotations`` are preserved (event ones appended).
    """
    n_traces = max(1, len(traces) if traces else 1)
    legend_rows = max(1, math.ceil(n_traces / 4))
    # Empirically Plotly's horizontal legend renders at ~36 px per row
    # (font + symbol + padding); 28 px undershot for ≥40-trace cases
    # (legend overlapped the x-axis label — Haochen 2026-05-25 screenshot).
    # Also force an explicit bottom margin sized for the legend rather
    # than relying on Plotly's auto-margin (which under-allocates when
    # `legend.y` is set to a negative offset below the plot area).
    chart_h = max(340, 320 + 36 * legend_rows)
    layout_with_h = {**layout}
    layout_with_h.setdefault("height", chart_h)
    layout_with_h.setdefault("autosize", True)
    layout_with_h.setdefault("margin", {})
    if isinstance(layout_with_h.get("margin"), dict):
        layout_with_h["margin"].setdefault("b", 60 + 36 * legend_rows)
    if events:
        shapes = list(layout_with_h.get("shapes") or [])
        shapes.extend(_event_shapes(events))
        layout_with_h["shapes"] = shapes
        anns = list(layout_with_h.get("annotations") or [])
        anns.extend(_event_annotations(events))
        layout_with_h["annotations"] = anns
    data_json = json.dumps(traces)
    layout_json = json.dumps(layout_with_h)
    return (
        f'<div id="{escape(div_id)}" style="width:100%;height:{chart_h}px"></div>\n'
        f'<script>Plotly.newPlot("{escape(div_id)}", {data_json}, '
        f'{layout_json}, {{responsive: true}});</script>\n'
    )


def _empty_note(msg: str) -> str:
    return (
        f'<div style="padding:16px;border:1px dashed #cbd5e1;color:#64748b;'
        f'background:#f8fafc;border-radius:6px;font-size:0.9em">{escape(msg)}'
        f'</div>'
    )


def _detect_events_from_inputs(state: dict[str, Any]) -> dict[str, list[float]]:
    """Detect initiation + division event timepoints from common inputs.

    Uses ``state.get('number_of_oric')`` as the event signal:
      * step-up (2→4, 4→8, etc.) ⇒ ``initiation`` event at that tick
      * step-down (4→2, 2→1, etc.) ⇒ ``division`` event at that tick

    For multi-run inputs, uses the FIRST run's trace as the canonical
    event timing (overlaying events from N runs would clutter the
    chart; the first run is treated as representative). Pair the
    chart's caption with a "events shown for run 0" note when N>1.

    Requires the caller to wire ``number_of_oric`` + ``time`` into the
    viz's ``inputs_map`` (both as ``list[float]`` ports). Returns
    ``{}`` if either is missing — overlays are best-effort, never
    load-bearing.
    """
    noric_runs = _to_runs(state.get("number_of_oric"))
    time_runs = _to_runs(state.get("time"))
    if not noric_runs or not time_runs:
        return {}
    noric = noric_runs[0]
    times = time_runs[0]
    if len(noric) < 2 or len(times) != len(noric):
        return {}
    inits: list[float] = []
    divs: list[float] = []
    for i in range(1, len(noric)):
        prev, curr = noric[i - 1], noric[i]
        if prev is None or curr is None:
            continue
        try:
            prev_n = int(round(float(prev)))
            curr_n = int(round(float(curr)))
        except (TypeError, ValueError):
            continue
        if curr_n > prev_n:
            inits.append(float(times[i]))
        elif curr_n < prev_n:
            divs.append(float(times[i]))
    return {"initiation": inits, "division": divs}


# ---------------------------------------------------------------------------
# DnaA nucleotide-state visualization
# ---------------------------------------------------------------------------

# Default-palette so single- and multi-run runs render with stable colors.
_PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b", "#0891b2"]


class DnaAStateVisualization(Visualization):
    """DnaA nucleotide-state pool composition + ATP-fraction trajectory.

    Wire via ``inputs_map`` in the study yaml so the dashboard pulls the
    right observables out of the SQLiteEmitter history. Example::

        visualizations:
          - name: dnaa_state
            address: local:DnaAStateVisualization
            config:
              title: "DnaA nucleotide-state cycle"
              inputs_map:
                apo_count:      listeners.dnaA_cycle.apo_count
                atp_count:      listeners.dnaA_cycle.atp_count
                adp_count:      listeners.dnaA_cycle.adp_count
                atp_fraction:   listeners.dnaA_cycle.atp_fraction
                time:           global_time

    All inputs are optional; a chart panel is rendered for whatever subset
    is provided.
    """

    config_schema = {
        **Visualization.config_schema,
        # Boesen 2024 physiological band for DnaA-ATP / total DnaA.
        "atp_band_low":  {"_type": "float", "_default": 0.20},
        "atp_band_high": {"_type": "float", "_default": 0.50},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "apo_count":     "list[float]",
            "atp_count":     "list[float]",
            "adp_count":     "list[float]",
            "atp_fraction":  "list[float]",
            "adp_fraction":  "list[float]",
            "apo_fraction":  "list[float]",
            "time":          "list[float]",
            # Optional: drives the initiation/division vertical-line
            # overlay. Detected from number_of_oric step-ups / step-
            # downs via _detect_events_from_inputs. Wire it via
            # inputs_map: `number_of_oric: listeners.replication_data.number_of_oric`
            "number_of_oric": "list[float]",
            "_run_labels":   "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "DnaA nucleotide-state cycle"
        band_low = float(self.config.get("atp_band_low", 0.20))
        band_high = float(self.config.get("atp_band_high", 0.50))
        events = _detect_events_from_inputs(state)

        apo_runs = _to_runs(state.get("apo_count"))
        atp_runs = _to_runs(state.get("atp_count"))
        adp_runs = _to_runs(state.get("adp_count"))
        atpf_runs = _to_runs(state.get("atp_fraction"))
        time_runs = _to_runs(state.get("time"))

        n_runs = max(len(apo_runs), len(atp_runs), len(adp_runs),
                     len(atpf_runs), 0)
        if n_runs == 0:
            body = _empty_note(
                "No dnaA_cycle observables in this run's history yet. "
                "Re-run the study with the SQLiteEmitter wired through "
                "pbg_runner to populate this chart.")
            return {"html": render_document(title=title, body_html=body)}

        run_labels = _coerce_labels(state.get("_run_labels"), n_runs)

        def _times_for(traces: list[list[float]], i: int) -> list[float]:
            if i < len(time_runs) and time_runs[i]:
                return time_runs[i]
            if i < len(traces) and traces[i]:
                return list(range(len(traces[i])))
            return []

        # ─── Pool composition (counts) ──────────────────────────────────
        count_traces: list[dict] = []
        for i in range(n_runs):
            label = run_labels[i]
            color_atp = _PALETTE[i % len(_PALETTE)]
            if i < len(atp_runs) and atp_runs[i]:
                count_traces.append(_trace(
                    f"DnaA-ATP · {label}", _times_for(atp_runs, i),
                    atp_runs[i], color=color_atp))
            if i < len(adp_runs) and adp_runs[i]:
                count_traces.append(_trace(
                    f"DnaA-ADP · {label}", _times_for(adp_runs, i),
                    adp_runs[i], color=color_atp, dash="dash"))
            if i < len(apo_runs) and apo_runs[i]:
                count_traces.append(_trace(
                    f"apo-DnaA · {label}", _times_for(apo_runs, i),
                    apo_runs[i], color=color_atp, dash="dot"))

        count_chart = _plotly_div(
            "dnaa-state-counts",
            count_traces,
            {
                "title": {"text": "Pool composition (counts)"},
                "xaxis": {"title": {"text": "time (s)"}},
                "yaxis": {"title": {"text": "molecules / cell"}},
                "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
                "legend": {"orientation": "h", "y": -0.25},
            },
            events=events,
        ) if count_traces else _empty_note(
            "No count observables in history; check inputs_map for "
            "apo_count / atp_count / adp_count.")

        # ─── ATP-fraction with Boesen band ──────────────────────────────
        frac_traces: list[dict] = []
        # Band drawn as a filled invisible-edge area between the two y-values.
        # Use a synthetic x-range spanning all runs.
        all_t: list[float] = []
        for tr in time_runs:
            all_t.extend(tr or [])
        for fr in atpf_runs:
            if not (any((time_runs[i] if i < len(time_runs) else None)
                        for i in range(n_runs))):
                all_t.extend(range(len(fr or [])))
        if all_t:
            tmin, tmax = min(all_t), max(all_t)
            # Lower band edge (invisible line)
            frac_traces.append({
                "type": "scatter", "mode": "lines",
                "name": "Boesen band lo", "showlegend": False,
                "x": [tmin, tmax], "y": [band_low, band_low],
                "line": {"width": 0},
                "hoverinfo": "skip",
            })
            # Upper band edge (fill to lower)
            frac_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"Boesen 2024 band [{band_low:.2f},"
                        f" {band_high:.2f}]",
                "x": [tmin, tmax], "y": [band_high, band_high],
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": "rgba(16,185,129,0.15)",
                "hoverinfo": "skip",
            })
        for i in range(n_runs):
            if i < len(atpf_runs) and atpf_runs[i]:
                frac_traces.append(_trace(
                    f"DnaA-ATP fraction · {run_labels[i]}",
                    _times_for(atpf_runs, i), atpf_runs[i],
                    color=_PALETTE[i % len(_PALETTE)]))

        frac_chart = _plotly_div(
            "dnaa-state-fraction",
            frac_traces,
            {
                "title": {"text": "DnaA-ATP fraction (Boesen 2024 band "
                          f"[{band_low:.2f}, {band_high:.2f}] shaded)"},
                "xaxis": {"title": {"text": "time (s)"}},
                "yaxis": {"title": {"text": "DnaA-ATP / total DnaA"},
                          "range": [0, 1]},
                "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
                "legend": {"orientation": "h", "y": -0.25},
            },
            events=events,
        ) if any(i < len(atpf_runs) and atpf_runs[i] for i in range(n_runs)
                 ) else _empty_note(
            "No atp_fraction observable; either the dnaA_cycle listener "
            "is not wired into the composite or inputs_map.atp_fraction "
            "is missing.")

        body = (
            f"{_PLOTLY_CDN}\n"
            f'<div style="padding:0 12px 24px 12px">\n'
            f"{count_chart}\n{frac_chart}\n"
            f"</div>"
        )
        return {"html": render_document(title=title, body_html=body)}


# ---------------------------------------------------------------------------
# DnaA-box occupancy visualization (dnaa-03 / dnaa-04)
# ---------------------------------------------------------------------------


class DnaABoxOccupancyVisualization(Visualization):
    """DnaA-box occupancy trajectories: chromosomal buffer + oriC + dnaAp.

    Wire via ``inputs_map`` (all optional, missing → blank panel):

        visualizations:
          - name: dnaa_box_occupancy
            address: local:DnaABoxOccupancyVisualization
            config:
              inputs_map:
                chromosome_fraction:  listeners.dnaA_binding.chromosome.occupied_fraction
                oric_high:            listeners.dnaA_binding.oric.high_affinity_occupied
                oric_low:             listeners.dnaA_binding.oric.low_affinity_occupied
                dnaap_fraction:       listeners.dnaA_binding.dnaap.occupied
                free_atp:             listeners.dnaA_binding.free_atp
                free_adp:             listeners.dnaA_binding.free_adp
                bound_total:          listeners.dnaA_binding.bound_total
                time:                 global_time
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "chromosome_fraction": "list[float]",
            "oric_high":           "list[float]",
            "oric_low":            "list[float]",
            "dnaap_fraction":      "list[float]",
            "free_atp":            "list[float]",
            "free_adp":            "list[float]",
            "bound_total":         "list[float]",
            "time":                "list[float]",
            "number_of_oric":      "list[float]",   # event overlay (optional)
            "_run_labels":         "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "DnaA-box occupancy"
        events = _detect_events_from_inputs(state)

        ch = _to_runs(state.get("chromosome_fraction"))
        oh = _to_runs(state.get("oric_high"))
        ol = _to_runs(state.get("oric_low"))
        dp = _to_runs(state.get("dnaap_fraction"))
        fa = _to_runs(state.get("free_atp"))
        fd = _to_runs(state.get("free_adp"))
        bt = _to_runs(state.get("bound_total"))
        time_runs = _to_runs(state.get("time"))

        n_runs = max(len(ch), len(oh), len(ol), len(dp),
                     len(fa), len(fd), len(bt), 0)
        if n_runs == 0:
            body = _empty_note(
                "No dnaA_binding observables in this run's history yet. "
                "Re-run with the SQLiteEmitter wired up (pbg_runner) and "
                "with dnaa_box_binding active in the composite.")
            return {"html": render_document(title=title, body_html=body)}

        run_labels = _coerce_labels(state.get("_run_labels"), n_runs)

        def _times_for(traces: list[list[float]], i: int) -> list[float]:
            if i < len(time_runs) and time_runs[i]:
                return time_runs[i]
            if i < len(traces) and traces[i]:
                return list(range(len(traces[i])))
            return []

        # ─── Occupancy fractions over time ──────────────────────────────
        occ_traces: list[dict] = []
        for i in range(n_runs):
            label = run_labels[i]
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(ch) and ch[i]:
                occ_traces.append(_trace(
                    f"chromosomal · {label}", _times_for(ch, i),
                    ch[i], color=color))
            if i < len(oh) and oh[i]:
                occ_traces.append(_trace(
                    f"oriC high · {label}", _times_for(oh, i),
                    oh[i], color=color, dash="dash"))
            if i < len(ol) and ol[i]:
                occ_traces.append(_trace(
                    f"oriC low · {label}", _times_for(ol, i),
                    ol[i], color=color, dash="dot"))
            if i < len(dp) and dp[i]:
                occ_traces.append(_trace(
                    f"dnaAp · {label}", _times_for(dp, i),
                    dp[i], color=color, dash="longdash"))

        occ_chart = _plotly_div(
            "dnaa-box-occ",
            occ_traces,
            {
                "title": {"text": "Box occupancy fractions"},
                "xaxis": {"title": {"text": "time (s)"}},
                "yaxis": {"title": {"text": "fraction occupied"},
                          "range": [0, 1.05]},
                "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
                "legend": {"orientation": "h", "y": -0.25},
            },
            events=events,
        ) if occ_traces else _empty_note(
            "No occupancy observables; check inputs_map for "
            "chromosome_fraction / oric_high / oric_low / dnaap_fraction.")

        # ─── Free vs bound DnaA pools ───────────────────────────────────
        pool_traces: list[dict] = []
        for i in range(n_runs):
            label = run_labels[i]
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(fa) and fa[i]:
                pool_traces.append(_trace(
                    f"free DnaA-ATP · {label}", _times_for(fa, i),
                    fa[i], color=color))
            if i < len(fd) and fd[i]:
                pool_traces.append(_trace(
                    f"free DnaA-ADP · {label}", _times_for(fd, i),
                    fd[i], color=color, dash="dash"))
            if i < len(bt) and bt[i]:
                pool_traces.append(_trace(
                    f"bound total · {label}", _times_for(bt, i),
                    bt[i], color=color, dash="dot"))

        pool_chart = _plotly_div(
            "dnaa-box-pools",
            pool_traces,
            {
                "title": {"text": "Free vs bound DnaA pools"},
                "xaxis": {"title": {"text": "time (s)"}},
                "yaxis": {"title": {"text": "molecules / cell"}},
                "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
                "legend": {"orientation": "h", "y": -0.25},
            },
            events=events,
        ) if pool_traces else _empty_note(
            "No pool observables; check inputs_map for free_atp / "
            "free_adp / bound_total.")

        body = (
            f"{_PLOTLY_CDN}\n"
            f'<div style="padding:0 12px 24px 12px">\n'
            f"{occ_chart}\n{pool_chart}\n"
            f"</div>"
        )
        return {"html": render_document(title=title, body_html=body)}
