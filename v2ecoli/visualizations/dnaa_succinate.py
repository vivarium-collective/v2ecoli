"""DnaA / succinate-steady-state Visualization Steps.

Two classes for the fresh-start dnaa-replication investigation
(see investigations/dnaa-replication/feedback-2026-05-29-fresh-start/
feedback.pdf):

  - DnaaSteadyStateVisualization (dnaa-0): three-panel timeseries
    (oriC count, cell mass, DnaA monomer count) for confirming the
    succinate 10-generation single-daughters lineage reaches periodic
    steady state — Rashmi's acceptance criterion (oriC 1↔2 never 4,
    cell mass periodic from gen-3 onwards).

  - DnaaExpressionVisualization (dnaa-1): four-panel timeseries
    (DnaA monomer count, DnaA concentration = count / cell_mass,
    dnaA mRNA count, dnaA mRNA initiation events) for confirming
    Mechanism A's runtime perturbation lands the DnaA pool in the
    Schmidt 2016 / Mori 2021 [300, 800] band stably across generations.

Both classes read observables from study-yaml inputs_map (port →
observable path), use generation-boundary markers (vertical dashed
lines) so the periodic structure is visually obvious, and render to
self-contained interactive HTML via Plotly CDN.

NO RIDA. NO intrinsic-hydrolysis. NO box binding. These visualizations
are scoped strictly to the two foundational studies. Additional
mechanism-specific viz come in future rounds, after dnaa-0 and dnaa-1
pass.
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
    """Normalize a port value to multi-run shape (list[list[float]])."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) == 0:
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


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _trace(name: str, x: Sequence[float], y: Sequence[float],
           color: str | None = None, dash: str | None = None,
           mode: str = "lines") -> dict:
    line: dict[str, Any] = {"width": 2}
    if color:
        line["color"] = color
    if dash:
        line["dash"] = dash
    return {
        "type": "scatter",
        "mode": mode,
        "name": name,
        "x": list(x),
        "y": [_num(v) for v in y],
        "line": line,
    }


_SEC_PER_MIN = 60.0


def _to_minutes(runs: list[list[float]]) -> list[list[float]]:
    """Convert per-run time axes from seconds to minutes (Rashmi 2026-05-30:
    'mark the x axis time in minutes')."""
    out: list[list[float]] = []
    for run in runs:
        out.append([(_num(t) or 0.0) / _SEC_PER_MIN for t in (run or [])])
    return out


def _generation_shapes(division_times: Sequence[float] | None) -> list[dict]:
    """Vertical dashed lines at each division time (generation boundary)."""
    if not division_times:
        return []
    out: list[dict] = []
    for t in division_times:
        try:
            ts = float(t)
        except (TypeError, ValueError):
            continue
        out.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": ts, "x1": ts, "y0": 0, "y1": 1,
            "line": {"color": "#94a3b8", "width": 1, "dash": "dash"},
            "opacity": 0.6, "layer": "below",
        })
    return out


def _generation_annotations(division_times: Sequence[float] | None,
                            x_max: float | None) -> list[dict]:
    """'gen 1', 'gen 2', ... labels centered in each inter-division span
    (Rashmi 2026-05-30: 'label the number of generations'). division_times
    are the generation boundaries; the span [0, first], [first, second], …,
    [last, x_max] is one generation each."""
    if not division_times or x_max is None:
        return []
    edges: list[float] = [0.0]
    for t in division_times:
        try:
            edges.append(float(t))
        except (TypeError, ValueError):
            continue
    edges.append(float(x_max))
    anns: list[dict] = []
    for k in range(len(edges) - 1):
        mid = (edges[k] + edges[k + 1]) / 2.0
        anns.append({
            "xref": "x", "yref": "paper", "x": mid, "y": 1.0,
            "yanchor": "bottom", "showarrow": False,
            "text": f"gen {k + 1}",
            "font": {"size": 10, "color": "#64748b"},
        })
    return anns


def _plotly_div(div_id: str, traces: list[dict], layout: dict) -> str:
    data_json = json.dumps(traces)
    layout_json = json.dumps(layout)
    return (
        f'<div id="{escape(div_id)}" style="width:100%;height:260px"></div>\n'
        f'<script>Plotly.newPlot("{escape(div_id)}", {data_json}, '
        f'{layout_json}, {{responsive: true}});</script>\n'
    )


def _empty_note(msg: str) -> str:
    return (
        f'<div style="padding:16px;border:1px dashed #cbd5e1;color:#64748b;'
        f'background:#f8fafc;border-radius:6px;font-size:0.9em">{escape(msg)}'
        f'</div>'
    )


def _times_for(values: list[list[float]] | None,
               times: list[list[float]] | None, i: int) -> list[float]:
    """Pick a time axis: shared `times` if present, else integer ticks."""
    if times and i < len(times) and times[i]:
        return times[i]
    if values and i < len(values) and values[i]:
        return list(range(len(values[i])))
    return []


_PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b", "#0891b2"]


class DnaaSteadyStateVisualization(Visualization):
    """dnaa-0 acceptance viz — three panels: oriC, cell mass, DnaA monomer total.

    Wire via inputs_map in the study yaml so the dashboard / parquet_viz
    renderer pulls the right observables. Example::

        visualizations:
          - name: dnaa_steady_state
            address: local:DnaaSteadyStateVisualization
            config:
              title: "dnaa-0 — succinate steady state (oriC, mass, DnaA)"
              inputs_map:
                oric_count:        listeners.replication_data.number_of_oric
                cell_mass:         listeners.mass.cell_mass
                dnaa_monomer_total: listeners.monomer_counts  # aggregated downstream
                time:              global_time
                division_times:    listeners.mass.division_times  # optional

    All inputs are optional — a panel renders for whatever subset is
    provided. The division_times input (when set) draws vertical dashed
    lines so the periodic structure is visually obvious.
    """

    config_schema = {
        **Visualization.config_schema,
        "dnaa_band_low": {"_type": "float", "_default": 300.0},
        "dnaa_band_high": {"_type": "float", "_default": 800.0},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "oric_count":         "list[float]",
            "cell_mass":          "list[float]",
            "dnaa_monomer_total": "list[float]",
            "time":               "list[float]",
            "division_times":     "list[float]",
            "_run_labels":        "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "dnaa-0 — succinate steady state"
        band_low = float(self.config.get("dnaa_band_low", 300.0))
        band_high = float(self.config.get("dnaa_band_high", 800.0))

        oric_runs = _to_runs(state.get("oric_count"))
        mass_runs = _to_runs(state.get("cell_mass"))
        dnaa_runs = _to_runs(state.get("dnaa_monomer_total"))
        time_runs = _to_minutes(_to_runs(state.get("time")))  # x axis in minutes
        div_times = state.get("division_times") or []
        # division_times may be a flat list (single run) or per-run; flatten the
        # single-run case for the shapes overlay
        if div_times and isinstance(div_times[0], (list, tuple)):
            div_times = div_times[0] if div_times else []
        # boundaries arrive in seconds; render in minutes to match the x axis
        div_times = [(_num(t) or 0.0) / 60.0 for t in div_times]
        x_max = max((max(r) for r in time_runs if r), default=None)

        n_runs = max(len(oric_runs), len(mass_runs), len(dnaa_runs), 0)
        if n_runs == 0:
            return {"html": render_document(
                title=title,
                body_html=_empty_note(
                    "No succinate observables yet. Re-run the study so the "
                    "emitter populates listeners.replication_data.number_of_oric, "
                    "listeners.mass.cell_mass, and listeners.monomer_counts "
                    "(at the dnaA index).")
            )}

        run_labels = _coerce_labels(state.get("_run_labels"), n_runs)
        shapes = _generation_shapes(div_times)
        gen_anns = _generation_annotations(div_times, x_max)

        # --- Panel 1: oriC count -------------------------------------------
        oric_traces: list[dict] = []
        for i in range(n_runs):
            ys = oric_runs[i] if i < len(oric_runs) else []
            xs = _times_for(oric_runs, time_runs, i)
            color = _PALETTE[i % len(_PALETTE)]
            oric_traces.append(_trace(run_labels[i], xs, ys, color=color))

        oric_layout = {
            "title": {"text": "(1) oriC count — periodic 1 ↔ 2 at steady state "
                              "(transient may reach 4 before gen 3)"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "oriC count (copies)", "rangemode": "tozero",
                      "dtick": 1},
            "shapes": shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        # --- Panel 2: cell mass -------------------------------------------
        mass_traces: list[dict] = []
        for i in range(n_runs):
            ys = mass_runs[i] if i < len(mass_runs) else []
            xs = _times_for(mass_runs, time_runs, i)
            color = _PALETTE[i % len(_PALETTE)]
            mass_traces.append(_trace(run_labels[i], xs, ys, color=color))

        mass_layout = {
            "title": {"text": "(2) cell mass — sawtooth periodic from gen-3 onwards"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "cell mass (fg)"},
            "shapes": shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        # --- Panel 3: DnaA monomer total ----------------------------------
        dnaa_traces: list[dict] = []
        for i in range(n_runs):
            ys = dnaa_runs[i] if i < len(dnaa_runs) else []
            xs = _times_for(dnaa_runs, time_runs, i)
            color = _PALETTE[i % len(_PALETTE)]
            dnaa_traces.append(_trace(run_labels[i], xs, ys, color=color))

        dnaa_shapes = list(shapes)
        dnaa_shapes.extend([
            {"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
             "y0": band_low, "y1": band_high,
             "fillcolor": "#22c55e", "opacity": 0.08,
             "line": {"width": 0}, "layer": "below"},
        ])
        dnaa_layout = {
            "title": {"text": f"(3) DnaA monomer count — band [{int(band_low)}, "
                              f"{int(band_high)}] shaded"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "DnaA monomer (count)"},
            "shapes": dnaa_shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        body = (
            _PLOTLY_CDN
            + _plotly_div("dnaa0-oric", oric_traces, oric_layout)
            + _plotly_div("dnaa0-mass", mass_traces, mass_layout)
            + _plotly_div("dnaa0-monomer", dnaa_traces, dnaa_layout)
        )
        return {"html": render_document(title=title, body_html=body)}


class DnaaExpressionVisualization(Visualization):
    """dnaa-1 acceptance viz — four panels per Rashmi's PDF ask.

    Panels: (1) DnaA monomer count (all forms via monomer_counts listener),
    (2) DnaA concentration (count / cell_mass), (3) dnaA mRNA count,
    (4) dnaA mRNA initiation events.

    Example study-yaml wiring::

        visualizations:
          - name: dnaa_expression
            address: local:DnaaExpressionVisualization
            config:
              title: "dnaa-1 — Mechanism A V=2e-3 (succinate)"
              inputs_map:
                dnaa_monomer_total: listeners.monomer_counts  # at dnaA index
                cell_mass:          listeners.mass.cell_mass
                dnaa_mrna_count:    listeners.rna_counts.mRNA_counts  # at dnaA mRNA index
                dnaa_init_events:   listeners.rnap_data.rna_init_event_per_cistron  # at dnaA cistron
                time:               global_time
                division_times:     listeners.mass.division_times
    """

    config_schema = {
        **Visualization.config_schema,
        "dnaa_band_low":  {"_type": "float", "_default": 300.0},
        "dnaa_band_high": {"_type": "float", "_default": 800.0},
        "target_init_rate_per_min": {"_type": "float", "_default": 1.5},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "dnaa_monomer_total": "list[float]",
            "cell_mass":          "list[float]",
            "dnaa_mrna_count":    "list[float]",
            "dnaa_init_events":   "list[float]",
            "time":               "list[float]",
            "division_times":     "list[float]",
            "_run_labels":        "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "dnaa-1 — Mechanism A on succinate"
        band_low = float(self.config.get("dnaa_band_low", 300.0))
        band_high = float(self.config.get("dnaa_band_high", 800.0))

        band_target_rate = float(self.config.get("target_init_rate_per_min", 1.5))

        mono_runs = _to_runs(state.get("dnaa_monomer_total"))
        mass_runs = _to_runs(state.get("cell_mass"))
        mrna_runs = _to_runs(state.get("dnaa_mrna_count"))
        init_runs = _to_runs(state.get("dnaa_init_events"))
        time_runs = _to_minutes(_to_runs(state.get("time")))  # x axis in minutes
        div_times = state.get("division_times") or []
        if div_times and isinstance(div_times[0], (list, tuple)):
            div_times = div_times[0] if div_times else []
        div_times = [(_num(t) or 0.0) / 60.0 for t in div_times]
        x_max = max((max(r) for r in time_runs if r), default=None)

        n_runs = max(len(mono_runs), len(mass_runs), len(mrna_runs),
                     len(init_runs), 0)
        if n_runs == 0:
            return {"html": render_document(
                title=title,
                body_html=_empty_note(
                    "No DnaA expression observables yet. Wire the emitter to "
                    "capture listeners.monomer_counts (at dnaA index), "
                    "listeners.mass.cell_mass, listeners.rna_counts.mRNA_counts "
                    "(at dnaA mRNA TU index), and "
                    "listeners.rnap_data.rna_init_event_per_cistron (at dnaA "
                    "cistron index).")
            )}

        run_labels = _coerce_labels(state.get("_run_labels"), n_runs)
        shapes = _generation_shapes(div_times)
        gen_anns = _generation_annotations(div_times, x_max)

        # --- Panel 1: DnaA monomer count (band-shaded) --------------------
        mono_traces = []
        for i in range(n_runs):
            ys = mono_runs[i] if i < len(mono_runs) else []
            xs = _times_for(mono_runs, time_runs, i)
            mono_traces.append(_trace(run_labels[i], xs, ys,
                                       color=_PALETTE[i % len(_PALETTE)]))
        mono_shapes = list(shapes) + [
            {"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
             "y0": band_low, "y1": band_high,
             "fillcolor": "#22c55e", "opacity": 0.08,
             "line": {"width": 0}, "layer": "below"},
        ]
        mono_layout = {
            "title": {"text": f"(1) DnaA monomer count — band [{int(band_low)}, "
                              f"{int(band_high)}] shaded"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "DnaA monomer (count)"},
            "shapes": mono_shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        # --- Panel 2: DnaA concentration (count / cell_mass) --------------
        conc_traces = []
        for i in range(n_runs):
            mono = mono_runs[i] if i < len(mono_runs) else []
            mass = mass_runs[i] if i < len(mass_runs) else []
            n = min(len(mono), len(mass))
            xs = _times_for(mono_runs, time_runs, i)[:n]
            ys = []
            for j in range(n):
                m, cm = _num(mono[j]), _num(mass[j])
                ys.append(m / cm if (m is not None and cm and cm > 0) else None)
            conc_traces.append(_trace(run_labels[i], xs, ys,
                                       color=_PALETTE[i % len(_PALETTE)]))
        conc_layout = {
            "title": {"text": "(2) DnaA concentration (count / cell mass)"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "DnaA / cell mass (count · fg⁻¹)"},
            "shapes": shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        # --- Panel 3: dnaA mRNA count ------------------------------------
        mrna_traces = []
        for i in range(n_runs):
            ys = mrna_runs[i] if i < len(mrna_runs) else []
            xs = _times_for(mrna_runs, time_runs, i)
            mrna_traces.append(_trace(run_labels[i], xs, ys,
                                       color=_PALETTE[i % len(_PALETTE)],
                                       mode="lines+markers"))
        mrna_layout = {
            "title": {"text": "(3) dnaA mRNA count"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "dnaA mRNA (count)", "rangemode": "tozero"},
            "shapes": shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        # --- Panel 4: dnaA mRNA initiation events ------------------------
        init_traces = []
        for i in range(n_runs):
            ys = init_runs[i] if i < len(init_runs) else []
            xs = _times_for(init_runs, time_runs, i)
            init_traces.append(_trace(run_labels[i], xs, ys,
                                       color=_PALETTE[i % len(_PALETTE)]))
        # Rashmi 2026-05-30: the PDF target is the biologically-expected dnaA
        # RNA-synthesis RATE; Mechanism A V=2e-3 landed ~1.01 events/min, close
        # to expectation, which in turn put the DnaA pool in band. Draw the
        # target rate as a reference line so the chart reads as a rate check.
        init_shapes = list(shapes) + [
            {"type": "line", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
             "y0": band_target_rate, "y1": band_target_rate,
             "line": {"color": "#dc2626", "width": 1, "dash": "dot"},
             "opacity": 0.7, "layer": "below"},
        ]
        init_layout = {
            "title": {"text": f"(4) dnaA mRNA initiation events — target "
                              f"≈ {band_target_rate:g} event/min (biological rate)"},
            "xaxis": {"title": "time (min)"},
            "yaxis": {"title": "dnaA init events (per emit window)",
                      "rangemode": "tozero"},
            "shapes": init_shapes,
            "annotations": gen_anns,
            "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        }

        body = (
            _PLOTLY_CDN
            + _plotly_div("dnaa1-monomer", mono_traces, mono_layout)
            + _plotly_div("dnaa1-conc", conc_traces, conc_layout)
            + _plotly_div("dnaa1-mrna", mrna_traces, mrna_layout)
            + _plotly_div("dnaa1-init", init_traces, init_layout)
        )
        return {"html": render_document(title=title, body_html=body)}
