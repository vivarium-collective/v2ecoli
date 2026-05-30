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
    """dnaa-1 acceptance viz — six panels reproducing Rashmi's PDF figure
    ("V=2e-3 seed=1 — 7-gen lineage on succinate"):

      (1) oriC number          (2) cell mass
      (3) DnaA monomer count (all forms; [300,800] band shaded)
      (4) DnaA concentration (count / cell_mass)
      (5) dnaA mRNA count      (6) dnaA mRNA initiation events (per-tick
                                    barcode + biological-rate reference)

    Example study-yaml wiring::

        visualizations:
          - name: dnaa_expression
            address: local:DnaaExpressionVisualization
            config:
              title: "dnaa-1 — Mechanism A V=2e-3 (succinate)"
              inputs_map:
                oric_count:         listeners.replication_data.number_of_oric
                cell_mass:          listeners.mass.cell_mass
                dnaa_monomer_total: listeners.monomer_counts  # at dnaA index
                dnaa_mrna_count:    listeners.rna_counts.mRNA_counts  # at dnaA mRNA index
                dnaa_init_events:   listeners.rnap_data.rna_init_event_per_cistron  # at dnaA cistron
                time:               global_time
                division_times:     listeners.mass.division_times
    """

    config_schema = {
        **Visualization.config_schema,
        "dnaa_band_low":  {"_type": "float", "_default": 300.0},
        "dnaa_band_high": {"_type": "float", "_default": 800.0},
        "target_init_rate_per_min": {"_type": "float", "_default": 1.0},
        "subtitle": {"_type": "string", "_default": ""},
        "init_caption": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "oric_count":         "list[float]",
            "cell_mass":          "list[float]",
            "dnaa_monomer_total": "list[float]",
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

        band_target_rate = float(self.config.get("target_init_rate_per_min", 1.0))
        subtitle = self.config.get("subtitle") or ""

        oric_runs = _to_runs(state.get("oric_count"))
        mass_runs = _to_runs(state.get("cell_mass"))
        mono_runs = _to_runs(state.get("dnaa_monomer_total"))
        mrna_runs = _to_runs(state.get("dnaa_mrna_count"))
        init_runs = _to_runs(state.get("dnaa_init_events"))
        time_runs = _to_minutes(_to_runs(state.get("time")))  # x axis in minutes
        div_times = state.get("division_times") or []
        if div_times and isinstance(div_times[0], (list, tuple)):
            div_times = div_times[0] if div_times else []
        div_times = [(_num(t) or 0.0) / 60.0 for t in div_times]
        x_max = max((max(r) for r in time_runs if r), default=None)

        n_runs = max(len(oric_runs), len(mass_runs), len(mono_runs),
                     len(mrna_runs), len(init_runs), 0)
        if n_runs == 0:
            return {"html": render_document(
                title=title,
                body_html=_empty_note(
                    "No DnaA expression observables yet. Wire the emitter to "
                    "capture listeners.replication_data.number_of_oric, "
                    "listeners.mass.cell_mass, listeners.monomer_counts (at "
                    "dnaA index), listeners.rna_counts.mRNA_counts (at dnaA "
                    "mRNA TU index), and "
                    "listeners.rnap_data.rna_init_event_per_cistron (at dnaA "
                    "cistron index).")
            )}

        run_labels = _coerce_labels(state.get("_run_labels"), n_runs)
        shapes = _generation_shapes(div_times)
        gen_anns = _generation_annotations(div_times, x_max)

        def _panel(runs, mode="lines"):
            tr = []
            for i in range(n_runs):
                ys = runs[i] if i < len(runs) else []
                xs = _times_for(runs, time_runs, i)
                tr.append(_trace(run_labels[i], xs, ys,
                                 color=_PALETTE[i % len(_PALETTE)], mode=mode))
            return tr

        def _layout(title_text, yaxis, extra_shapes=None, **yaxis_extra):
            yax = {"title": yaxis}
            yax.update(yaxis_extra)
            return {
                "title": {"text": title_text},
                "xaxis": {"title": "time (min)"},
                "yaxis": yax,
                "shapes": list(shapes) + list(extra_shapes or []),
                "annotations": gen_anns,
                "margin": {"l": 60, "r": 20, "t": 44, "b": 38},
            }

        # --- Panel 1: oriC number -----------------------------------------
        oric_layout = _layout(
            "(1) oriC number — periodic 1 ↔ 2 (never 4 at steady state)",
            "oriC count (copies)", rangemode="tozero", dtick=1)

        # --- Panel 2: cell mass -------------------------------------------
        mass_layout = _layout("(2) cell mass", "cell mass (fg)")

        # --- Panel 3: DnaA monomer count (band-shaded) --------------------
        band = [{"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
                 "y0": band_low, "y1": band_high, "fillcolor": "#22c55e",
                 "opacity": 0.08, "line": {"width": 0}, "layer": "below"}]
        mono_layout = _layout(
            f"(3) DnaA monomer count — band [{int(band_low)}, {int(band_high)}] shaded",
            "DnaA monomer (count)", extra_shapes=band)

        # --- Panel 4: DnaA concentration (count / cell_mass) --------------
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
        conc_layout = _layout("(4) DnaA concentration (count / cell mass)",
                              "DnaA / cell mass (count · fg⁻¹)")

        # --- Panel 5: dnaA mRNA count ------------------------------------
        mrna_layout = _layout("(5) dnaA mRNA count", "dnaA mRNA (count)",
                              rangemode="tozero")

        # --- Panel 6: dnaA mRNA initiation events (per-tick barcode) ------
        # Rashmi's figure shows each transcription-initiation event as a
        # vertical line, with a caption "total N events over T min; mean
        # rate = R/min". Render as a bar barcode + biological-rate ref line.
        init_traces = []
        for i in range(n_runs):
            ys = [(_num(v) or 0.0) for v in (init_runs[i] if i < len(init_runs) else [])]
            xs = _times_for(init_runs, time_runs, i)
            init_traces.append({
                "type": "bar", "name": run_labels[i],
                "x": list(xs), "y": ys,
                "marker": {"color": _PALETTE[i % len(_PALETTE)],
                           "line": {"width": 0}},
            })
        # caption from the followed run (run 0)
        primary = [(_num(v) or 0.0) for v in (init_runs[0] if init_runs else [])]
        total_ev = sum(primary)
        mean_rate = (total_ev / x_max) if (x_max and x_max > 0) else 0.0
        init_shapes = [
            {"type": "line", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
             "y0": band_target_rate, "y1": band_target_rate,
             "line": {"color": "#dc2626", "width": 1, "dash": "dot"},
             "opacity": 0.7, "layer": "below"}]
        init_caption = self.config.get("init_caption") or (
            f"{int(total_ev)} events over {int(x_max or 0)} min; "
            f"mean {mean_rate:.2f}/min")
        init_layout = _layout(
            f"(6) dnaA mRNA initiation events — {init_caption} "
            f"(target ≈ {band_target_rate:g}/min)",
            "init events (per tick)", extra_shapes=init_shapes,
            rangemode="tozero")
        init_layout["bargap"] = 0

        subtitle_html = (
            f'<div style="text-align:center;color:#dc2626;font-size:0.85em;'
            f'margin:2px 0 8px">{escape(subtitle)}</div>' if subtitle else "")

        body = (
            _PLOTLY_CDN
            + subtitle_html
            + _plotly_div("dnaa1-oric", _panel(oric_runs), oric_layout)
            + _plotly_div("dnaa1-mass", _panel(mass_runs), mass_layout)
            + _plotly_div("dnaa1-monomer", _panel(mono_runs), mono_layout)
            + _plotly_div("dnaa1-conc", conc_traces, conc_layout)
            + _plotly_div("dnaa1-mrna", _panel(mrna_runs, mode="lines+markers"), mrna_layout)
            + _plotly_div("dnaa1-init", init_traces, init_layout)
        )
        return {"html": render_document(title=title, body_html=body)}


class DnaaChromosomeVisualization(Visualization):
    """Chromosome-state viz — replication-cycle structure for the succinate
    DnaA investigation. Three panels:

      (1) Cell-cycle counts — oriC, full chromosomes, active replisomes over
          time. The signature cycle: 1 chromosome + 1 oriC → initiation
          (oriC → 2, a replisome pair appears) → replication → termination
          (chromosome → 2) → division back to 1.
      (2) Replication fork map — each active fork's genomic position (signed,
          as a fraction of the half-genome; 0 = oriC, ±1 = terminus) plotted
          vs time. Fork pairs spring from oriC at initiation and travel
          outward to ter.
      (3) DnaA-box occupancy — bound fraction (total − free)/total, with the
          free and total box counts.

    Built from observables every succinate run already emits:
    listeners.replication_data.{number_of_oric, fork_coordinates,
    free_DnaA_boxes, total_DnaA_boxes} and
    listeners.unique_molecule_counts.{full_chromosome, active_replisome}.
    """

    config_schema = {**Visualization.config_schema}

    def inputs(self) -> dict[str, Any]:
        return {
            "oric_count":        "list[float]",
            "chromosome_count":  "list[float]",
            "replisome_count":   "list[float]",
            "fork_times":        "list[float]",   # flat: one entry per fork-tick
            "fork_positions":    "list[float]",   # flat: matching genomic coords
            "free_dnaa_boxes":   "list[float]",
            "total_dnaa_boxes":  "list[float]",
            "time":              "list[float]",
            "division_times":    "list[float]",
            "_run_labels":       "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "chromosome state (succinate)"

        oric_runs = _to_runs(state.get("oric_count"))
        chrom_runs = _to_runs(state.get("chromosome_count"))
        repl_runs = _to_runs(state.get("replisome_count"))
        free_runs = _to_runs(state.get("free_dnaa_boxes"))
        total_runs = _to_runs(state.get("total_dnaa_boxes"))
        time_runs = _to_minutes(_to_runs(state.get("time")))
        div_times = state.get("division_times") or []
        if div_times and isinstance(div_times[0], (list, tuple)):
            div_times = div_times[0] if div_times else []
        div_times = [(_num(t) or 0.0) / 60.0 for t in div_times]
        x_max = max((max(r) for r in time_runs if r), default=None)

        n_runs = max(len(oric_runs), len(chrom_runs), len(repl_runs), 0)
        if n_runs == 0:
            return {"html": render_document(
                title=title,
                body_html=_empty_note(
                    "No chromosome-state observables yet. Wire the emitter to "
                    "listeners.replication_data.{number_of_oric, "
                    "fork_coordinates, free_DnaA_boxes, total_DnaA_boxes} and "
                    "listeners.unique_molecule_counts.{full_chromosome, "
                    "active_replisome}.")
            )}

        run_labels = _coerce_labels(state.get("_run_labels"), n_runs)
        shapes = _generation_shapes(div_times)
        gen_anns = _generation_annotations(div_times, x_max)

        def _layout(title_text, yaxis, extra_shapes=None, **yax_extra):
            yax = {"title": yaxis}
            yax.update(yax_extra)
            return {
                "title": {"text": title_text},
                "xaxis": {"title": "time (min)"},
                "yaxis": yax,
                "shapes": list(shapes) + list(extra_shapes or []),
                "annotations": gen_anns,
                "margin": {"l": 60, "r": 20, "t": 44, "b": 38},
            }

        # --- Panel 1: cell-cycle counts -----------------------------------
        count_traces = []
        series = [("oriC", oric_runs, "#2563eb"),
                  ("chromosomes", chrom_runs, "#16a34a"),
                  ("active replisomes", repl_runs, "#dc2626")]
        for name, runs, color in series:
            if not runs:
                continue
            xs = _times_for(runs, time_runs, 0)
            count_traces.append(_trace(name, xs, runs[0], color=color,
                                       mode="lines"))
        count_layout = _layout(
            "(1) cell-cycle counts — oriC · chromosomes · replisomes",
            "count", rangemode="tozero", dtick=1)

        # --- Panel 2: replication fork map --------------------------------
        ft = [(_num(t) or 0.0) / 60.0 for t in (state.get("fork_times") or [])]
        fp = [_num(p) for p in (state.get("fork_positions") or [])]
        # normalise positions to ±1 (fraction of half-genome) by the max |pos|
        scale = max((abs(p) for p in fp if p is not None), default=1.0) or 1.0
        fp_norm = [(p / scale) if p is not None else None for p in fp]
        fork_traces = [{
            "type": "scatter", "mode": "markers", "name": "fork position",
            "x": ft, "y": fp_norm,
            "marker": {"color": "#9333ea", "size": 3, "opacity": 0.5},
        }]
        fork_layout = _layout(
            "(2) replication fork map — 0 = oriC, ±1 = terminus",
            "fork position (fraction of half-genome)",
            extra_shapes=[{"type": "line", "xref": "paper", "yref": "y",
                           "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                           "line": {"color": "#94a3b8", "width": 1},
                           "layer": "below"}])
        fork_layout["yaxis"]["range"] = [-1.1, 1.1]

        # --- Panel 3: DnaA-box occupancy ----------------------------------
        occ_traces = []
        if free_runs and total_runs:
            free = free_runs[0]
            total = total_runs[0]
            n = min(len(free), len(total))
            xs = _times_for(total_runs, time_runs, 0)[:n]
            bound_frac = []
            for j in range(n):
                f, t = _num(free[j]), _num(total[j])
                bound_frac.append(((t - f) / t) if (t and t > 0 and f is not None) else None)
            occ_traces.append(_trace("bound fraction", xs, bound_frac,
                                     color="#0891b2", mode="lines"))
        occ_layout = _layout(
            "(3) DnaA-box occupancy — bound fraction (total − free)/total",
            "bound fraction", rangemode="tozero")
        occ_layout["yaxis"]["range"] = [0, 1.05]

        body = (
            _PLOTLY_CDN
            + _plotly_div("chrom-counts", count_traces, count_layout)
            + _plotly_div("chrom-forks", fork_traces, fork_layout)
            + _plotly_div("chrom-boxocc", occ_traces, occ_layout)
        )
        return {"html": render_document(title=title, body_html=body)}


class DnaaChromosomeMapVisualization(Visualization):
    """Circular chromosome map — multi-timepoint snapshots of the 4.6 Mb
    circular chromosome showing oriC (top), Ter (bottom), replication-fork
    replisomes, and RNAP positions, laid out as a row of circles (the v1
    dnaa 'chromosome timeline' diagram).

    Each genomic coordinate (signed nt, 0 = oriC) maps to an angle around the
    circle: oriC at 12 o'clock, both arms running down to Ter at 6 o'clock.
    Replisome forks (orange ▲) spring from oriC at initiation and travel out
    to Ter; RNAP (blue ·) dot the backbone by transcription density.

    State (built by the renderer):
      snapshots: list of dicts, each
        {t_min: float, n_chr: int, forks: [coord,...], rnaps: [coord,...]}
      half_genome: float (nt; default 2.32e6 → 4.64 Mb genome)
    """

    config_schema = {**Visualization.config_schema}

    def inputs(self) -> dict[str, Any]:
        return {"snapshots": "list", "half_genome": "float", "_caption": "string"}

    @staticmethod
    def _xy(coord: float, half: float, r: float = 1.0):
        # f in [-1,1]; angle 90° (top, oriC) → -90°/270° (bottom, Ter)
        f = max(-1.0, min(1.0, (coord or 0.0) / half))
        ang = math.radians(90.0 - f * 180.0)
        return r * math.cos(ang), r * math.sin(ang)

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "chromosome map (succinate)"
        snaps = state.get("snapshots") or []
        half = float(state.get("half_genome") or 2.32e6)
        caption = state.get("_caption") or ""
        if not snaps:
            return {"html": render_document(title=title, body_html=_empty_note(
                "No chromosome snapshots. Wire fork_coordinates + "
                "active_rnap_coordinates from the run."))}

        def _one_circle_traces(forks, rnaps):
            """Trace list for ONE chromosome circle at origin (radius 1).
            Backbone + replication bubbles (one green arc per fork pair,
            from a fork through oriC to its partner, inset + nested for
            multifork, with radial tethers) + RNAP/replisome/oriC/Ter."""
            ring_x, ring_y = [], []
            for k in range(0, 361, 5):
                a = math.radians(k)
                ring_x.append(math.cos(a)); ring_y.append(math.sin(a))
            tr = [{"type": "scatter", "mode": "lines", "x": ring_x, "y": ring_y,
                   "line": {"color": "#cbd5e1", "width": 1.5},
                   "hoverinfo": "skip", "showlegend": False}]
            sf = sorted(forks); npr = len(sf) // 2
            prs = [(sf[i], sf[-1 - i]) for i in range(npr)]
            prs.sort(key=lambda p: -(abs(p[0]) + abs(p[1])) / 2.0)
            ins = ([0.80] * npr if npr <= 1
                   else [0.82 - 0.34 * j / (npr - 1) for j in range(npr)])
            for (f_lo, f_hi), r_b in zip(prs, ins):
                a_hi = math.radians(90.0 - (f_hi / half) * 180.0)
                a_lo = math.radians(90.0 - (f_lo / half) * 180.0)
                bx, by = [], []
                for k in range(81):
                    a = a_hi + (a_lo - a_hi) * k / 80.0
                    bx.append(r_b * math.cos(a)); by.append(r_b * math.sin(a))
                tr.append({"type": "scatter", "mode": "lines", "x": bx, "y": by,
                           "line": {"color": "#10b981", "width": 4}, "opacity": 0.65,
                           "hoverinfo": "skip", "showlegend": False})
                for f in (f_lo, f_hi):
                    a = math.radians(90.0 - (f / half) * 180.0)
                    tr.append({"type": "scatter", "mode": "lines",
                               "x": [r_b * math.cos(a), math.cos(a)],
                               "y": [r_b * math.sin(a), math.sin(a)],
                               "line": {"color": "#10b981", "width": 1}, "opacity": 0.4,
                               "hoverinfo": "skip", "showlegend": False})
            rxs = [self._xy(c, half)[0] for c in rnaps]
            rys = [self._xy(c, half)[1] for c in rnaps]
            fxs = [self._xy(c, half)[0] for c in forks]
            fys = [self._xy(c, half)[1] for c in forks]
            ox, oy = self._xy(0.0, half); tx, ty = self._xy(half, half)
            tr += [
                {"type": "scatter", "mode": "markers", "x": rxs, "y": rys,
                 "marker": {"color": "#2563eb", "size": 4, "opacity": 0.55},
                 "hoverinfo": "skip", "showlegend": False},
                {"type": "scatter", "mode": "markers", "x": fxs, "y": fys,
                 "marker": {"color": "#f59e0b", "size": 11, "symbol": "triangle-up",
                            "line": {"color": "#b45309", "width": 1}},
                 "hoverinfo": "skip", "showlegend": False},
                {"type": "scatter", "mode": "markers", "x": [ox], "y": [oy],
                 "marker": {"color": "#16a34a", "size": 14},
                 "hoverinfo": "skip", "showlegend": False},
                {"type": "scatter", "mode": "markers", "x": [tx], "y": [ty],
                 "marker": {"color": "#dc2626", "size": 11, "symbol": "square"},
                 "hoverinfo": "skip", "showlegend": False},
            ]
            return tr

        divs = []
        for s in snaps:
            forks = s.get("forks") or []
            rnaps = s.get("rnaps") or []
            n_chr = max(1, int(s.get("n_chr", 1) or 1))
            npr = len(sorted(forks)) // 2
            # One full_chromosome → one circle. Distribute each fork pair (and
            # its two forks) and the RNAPs across the chromosomes, so after
            # termination the cell shows TWO chromosomes — each able to carry
            # its own replication bubble if it has re-initiated.
            sf = sorted(forks)
            fork_pairs = [(sf[i], sf[-1 - i]) for i in range(npr)]
            per_forks = [[] for _ in range(n_chr)]
            for pi, (lo, hi) in enumerate(fork_pairs):
                per_forks[pi % n_chr].extend([lo, hi])
            per_rnaps = [rnaps[i::n_chr] for i in range(n_chr)]

            size = 230 if n_chr == 1 else max(120, int(230 / n_chr))
            circle_divs = []
            for i in range(n_chr):
                traces = _one_circle_traces(per_forks[i], per_rnaps[i])
                layout = {
                    "xaxis": {"visible": False, "range": [-1.18, 1.18],
                              "scaleanchor": "y", "scaleratio": 1},
                    "yaxis": {"visible": False, "range": [-1.18, 1.18]},
                    "margin": {"l": 2, "r": 2, "t": 2, "b": 2},
                    "showlegend": False, "plot_bgcolor": "#ffffff",
                }
                cid = f"chrmap-{s.get('t_min', 0):.0f}-{i}"
                circle_divs.append(
                    f'<div id="{escape(cid)}" style="display:inline-block;'
                    f'width:{size}px;height:{size}px"></div>'
                    f'<script>Plotly.newPlot("{escape(cid)}", '
                    f'{json.dumps(traces)}, {json.dumps(layout)}, '
                    f'{{responsive:true, displayModeBar:false}});</script>')
            title = (f"t = {s.get('t_min', 0):.0f} min · {n_chr} chromosome"
                     f"{'s' if n_chr != 1 else ''} · "
                     f"{npr} fork pair{'s' if npr != 1 else ''}")
            divs.append(
                f'<div style="display:inline-block;vertical-align:top;'
                f'text-align:center;margin:0 6px">'
                f'<div style="font-weight:600;font-size:12px;color:#334155;'
                f'margin-bottom:2px">{escape(title)}</div>'
                f'<div style="white-space:nowrap">{"".join(circle_divs)}</div>'
                f'</div>')

        legend = (
            '<div style="text-align:center;font-size:0.85em;color:#475569;'
            'margin:6px 0">'
            '<span style="color:#16a34a">● oriC</span> &nbsp; '
            '<span style="color:#dc2626">■ Ter</span> &nbsp; '
            '<span style="color:#f59e0b">▲ replisome fork</span> &nbsp; '
            '<span style="color:#10b981">▬ replication bubble</span> &nbsp; '
            '<span style="color:#2563eb">· RNAP</span></div>')
        cap = (f'<div style="text-align:center;font-size:0.8em;color:#64748b;'
               f'margin-bottom:6px">{escape(caption)}</div>' if caption else "")
        body = (_PLOTLY_CDN + cap
                + '<div style="white-space:nowrap;overflow-x:auto;text-align:center">'
                + "".join(divs) + '</div>' + legend)
        return {"html": render_document(title=title, body_html=body)}


class DnaaFormsVisualization(Visualization):
    """dnaa-2 viz — DnaA nucleotide-state partitioning across the lineage.

    Three panels (per-generation, from the lineage's cell states):
      (1) DnaA forms — apo, DnaA-ATP, DnaA-ADP, and total, with the
          [300, 800] total-DnaA band shaded.
      (2) DnaA-ATP fraction — DnaA-ATP / (apo+ATP+ADP), with the Boesen 2024
          [0.2, 0.5] biological band shaded.
      (3) cellular ATP / ADP bulk pools (and their ratio).

    State (built by the renderer):
      gens, apo, atp, adp (DnaA-form counts per gen),
      atp_pool, adp_pool (bulk ATP[c]/ADP[c] per gen).
    """

    config_schema = {
        **Visualization.config_schema,
        "atp_band_low":  {"_type": "float", "_default": 0.2},
        "atp_band_high": {"_type": "float", "_default": 0.5},
        "dnaa_band_low":  {"_type": "float", "_default": 300.0},
        "dnaa_band_high": {"_type": "float", "_default": 800.0},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "gens": "list[float]", "apo": "list[float]", "atp": "list[float]",
            "adp": "list[float]", "atp_pool": "list[float]",
            "adp_pool": "list[float]", "_caption": "string",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "DnaA-form distribution"
        lo = float(self.config.get("atp_band_low", 0.2))
        hi = float(self.config.get("atp_band_high", 0.5))
        dlo = float(self.config.get("dnaa_band_low", 300.0))
        dhi = float(self.config.get("dnaa_band_high", 800.0))
        gens = [_num(g) for g in (state.get("gens") or [])]
        apo = [_num(v) or 0.0 for v in (state.get("apo") or [])]
        atp = [_num(v) or 0.0 for v in (state.get("atp") or [])]
        adp = [_num(v) or 0.0 for v in (state.get("adp") or [])]
        atp_pool = [_num(v) for v in (state.get("atp_pool") or [])]
        adp_pool = [_num(v) for v in (state.get("adp_pool") or [])]
        caption = state.get("_caption") or ""
        if not gens:
            return {"html": render_document(title=title, body_html=_empty_note(
                "No DnaA-form data. Provide per-generation apo / DnaA-ATP / "
                "DnaA-ADP counts and ATP[c]/ADP[c] pools."))}

        total = [apo[i] + atp[i] + adp[i] for i in range(len(gens))]
        frac = [(atp[i] / total[i] if total[i] else None) for i in range(len(gens))]

        def _xax():
            return {"title": "generation", "dtick": 1}

        forms_band = [{"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
                       "y0": dlo, "y1": dhi, "fillcolor": "#22c55e", "opacity": 0.08,
                       "line": {"width": 0}, "layer": "below"}]
        forms_traces = [
            _trace("total DnaA", gens, total, color="#111827", mode="lines+markers"),
            _trace("DnaA-ATP", gens, atp, color="#2563eb", mode="lines+markers"),
            _trace("DnaA-ADP", gens, adp, color="#dc2626", mode="lines+markers"),
            _trace("apo-DnaA", gens, apo, color="#9333ea", mode="lines+markers"),
        ]
        forms_layout = {
            "title": {"text": f"(1) DnaA forms — total band [{int(dlo)}, {int(dhi)}] shaded"},
            "xaxis": _xax(), "yaxis": {"title": "count", "rangemode": "tozero"},
            "shapes": forms_band, "margin": {"l": 60, "r": 20, "t": 40, "b": 40}}

        atp_band = [{"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1,
                     "y0": lo, "y1": hi, "fillcolor": "#22c55e", "opacity": 0.12,
                     "line": {"width": 0}, "layer": "below"}]
        valid_fr = [f for f in frac if f is not None]
        mean_fr = sum(valid_fr) / len(valid_fr) if valid_fr else 0.0
        frac_layout = {
            "title": {"text": f"(2) DnaA-ATP fraction — Boesen band [{lo}, {hi}] "
                              f"shaded (lineage mean ≈ {mean_fr:.3f})"},
            "xaxis": _xax(),
            "yaxis": {"title": "DnaA-ATP / total", "range": [0, 1.05]},
            "shapes": atp_band, "margin": {"l": 60, "r": 20, "t": 40, "b": 40}}
        frac_traces = [_trace("DnaA-ATP fraction", gens, frac, color="#2563eb",
                              mode="lines+markers")]

        pool_traces = []
        if any(v is not None for v in atp_pool):
            pool_traces.append(_trace("ATP[c]", gens, atp_pool, color="#0891b2",
                                      mode="lines+markers"))
        if any(v is not None for v in adp_pool):
            pool_traces.append(_trace("ADP[c]", gens, adp_pool, color="#f59e0b",
                                      mode="lines+markers"))
        pool_layout = {
            "title": {"text": "(3) cellular nucleotide pools — ATP[c], ADP[c]"},
            "xaxis": _xax(),
            "yaxis": {"title": "molecules / cell", "rangemode": "tozero"},
            "margin": {"l": 70, "r": 20, "t": 40, "b": 40}}

        cap = (f'<div style="text-align:center;font-size:0.82em;color:#64748b;'
               f'margin:4px 0 10px">{escape(caption)}</div>' if caption else "")
        body = (_PLOTLY_CDN + cap
                + _plotly_div("dnaa2-forms", forms_traces, forms_layout)
                + _plotly_div("dnaa2-frac", frac_traces, frac_layout)
                + _plotly_div("dnaa2-pools", pool_traces, pool_layout))
        return {"html": render_document(title=title, body_html=body)}
