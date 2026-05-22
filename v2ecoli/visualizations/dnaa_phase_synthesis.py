"""DnaA phase-synthesis Visualization Steps inspired by the
'v2ecoli — replication initiation report' PDF (2026-05-17).

Three additional viz classes, each contributing a distinct panel-type
not covered by the existing dnaa.py module:

  - DnaAInitiationGate
      Gate-trigger signal + SeqA-v0 refractory window over time. The
      "DnaA-ATP-per-oriC gate signal" + "WITH SeqA: refractory window
      after each initiation" panels from the PDF, generalised.

  - DnaACycleSummary
      Multi-panel synthesis: DnaA pool composition, DnaA-box occupancy,
      active-replisome count, mass-at-oriC over time. Mirrors the PDF's
      "Trajectory under the new architecture" + "Phase X: chromosome
      dynamics" composite cards.

  - DnaAFluxContributions
      Stacked bars of per-tick hydrolysis events: intrinsic + RIDA
      (+ future DDAH). Mirrors the PDF's "DnaA-ATP hydrolysis flux
      contributors (cumulative-difference per ~50 s window)" panels.

Same Plotly-via-CDN rendering convention as dnaa.py. All inputs are
optional; missing observables render as empty panels with a clear
note so the chart still tells a partial story.
"""

from __future__ import annotations

import json
import math
from html import escape
from typing import Any, Sequence

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_document
from v2ecoli.visualizations.dnaa import (
    _PLOTLY_CDN,
    _to_runs,
    _coerce_labels,
    _trace,
    _plotly_div,
    _empty_note,
    _PALETTE,
)


def _numz(v):
    """Coerce to float, mapping None/NaN/non-numeric (e.g. a stray empty-
    container '{}' from an unresolved observable) to 0 — so a missing/dict
    point renders flat instead of crashing report generation on float({})."""
    if v is None or isinstance(v, (dict, list)):
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f


# ---------------------------------------------------------------------------
# DnaAInitiationGate — fires + refractory + gate signal
# ---------------------------------------------------------------------------


class DnaAInitiationGate(Visualization):
    """DnaA-driven initiation gate signal + SeqA-v0 refractory window.

    Reads outputs of DnaaInitiationMechanism (the shadow observer):

      listeners.dnaA_initiation.would_fire             (0/1 per tick)
      listeners.dnaA_initiation.in_refractory          (0/1 per tick)
      listeners.dnaA_initiation.cumulative_fires       (int, monotone)
      listeners.dnaA_initiation.t_since_last_fire_s    (float)
      listeners.dnaA_initiation.oric_high_obs          (float)
      listeners.dnaA_initiation.atp_fraction_obs       (float)

    Renders three stacked panels:

      1. Cumulative would_fire count + binary tick markers.
      2. Refractory window indicator + t_since_last_fire trajectory.
      3. Trigger inputs (oriC high occupancy + atp_fraction) with
         threshold reference lines so the would_fire events line up
         visually with the underlying gate logic.
    """

    config_schema = {
        **Visualization.config_schema,
        "oric_high_threshold":    {"_type": "float", "_default": 0.8},
        "atp_fraction_threshold": {"_type": "float", "_default": 0.3},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "would_fire":             "list[float]",
            "in_refractory":          "list[float]",
            "cumulative_fires":       "list[float]",
            "t_since_last_fire_s":    "list[float]",
            "oric_high_obs":          "list[float]",
            "atp_fraction_obs":       "list[float]",
            "time":                   "list[float]",
            "_run_labels":            "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "DnaA initiation gate"
        oric_t = float(self.config.get("oric_high_threshold", 0.8))
        atp_t  = float(self.config.get("atp_fraction_threshold", 0.3))

        wf  = _to_runs(state.get("would_fire"))
        ref = _to_runs(state.get("in_refractory"))
        cum = _to_runs(state.get("cumulative_fires"))
        tsl = _to_runs(state.get("t_since_last_fire_s"))
        oh  = _to_runs(state.get("oric_high_obs"))
        af  = _to_runs(state.get("atp_fraction_obs"))
        time_runs = _to_runs(state.get("time"))

        n = max(len(wf), len(ref), len(cum), len(tsl), len(oh), len(af), 0)
        if n == 0:
            body = _empty_note(
                "No dnaA_initiation observables yet. Run the dnaa-04 "
                "composite (dnaa_04_with_dnaa_initiation_trigger) and "
                "wire inputs_map at listeners.dnaA_initiation.*")
            return {"html": render_document(title=title, body_html=body)}

        labels = _coerce_labels(state.get("_run_labels"), n)

        def _t(traces, i):
            if i < len(time_runs) and time_runs[i]:
                return time_runs[i]
            if i < len(traces) and traces[i]:
                return list(range(len(traces[i])))
            return []

        # Panel 1: cumulative fires + binary would_fire markers
        p1_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(cum) and cum[i]:
                p1_traces.append(_trace(
                    f"cumulative fires · {labels[i]}",
                    _t(cum, i), cum[i], color=color))
            if i < len(wf) and wf[i]:
                # Render as scatter markers so single-tick events stay visible.
                times_i = _t(wf, i)
                fire_x = [t for t, v in zip(times_i, wf[i]) if v]
                fire_y = [(cum[i][j] if i < len(cum) and j < len(cum[i]) else 0)
                          for j, v in enumerate(wf[i]) if v]
                if fire_x:
                    p1_traces.append({
                        "type": "scatter", "mode": "markers",
                        "name": f"would_fire events · {labels[i]}",
                        "x": fire_x, "y": fire_y,
                        "marker": {"color": color, "size": 10, "symbol": "triangle-up",
                                   "line": {"width": 1, "color": "#0f172a"}},
                    })
        p1 = _plotly_div("dnaa-init-fires", p1_traces, {
            "title": {"text": "Cumulative would-fire events (▲ = each fire)"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "fire count"}},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p1_traces else _empty_note(
            "No would_fire / cumulative_fires data; missing inputs_map?")

        # Panel 2: refractory window + t_since_last_fire
        p2_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(ref) and ref[i]:
                p2_traces.append(_trace(
                    f"in_refractory · {labels[i]}",
                    _t(ref, i), ref[i], color=color, dash="dash"))
            if i < len(tsl) and tsl[i]:
                # Clip the t_since_last_fire to a sane upper bound for plotting
                clipped = [min(_numz(v), 1800) for v in tsl[i]]
                p2_traces.append(_trace(
                    f"t since last fire (s, clip 1800) · {labels[i]}",
                    _t(tsl, i), clipped, color=color))
        p2 = _plotly_div("dnaa-init-refractory", p2_traces, {
            "title": {"text": "SeqA-v0 refractory window + time-since-last-fire"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "s (or 0/1 indicator)"}},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p2_traces else _empty_note(
            "No refractory / time-since-fire data.")

        # Panel 3: trigger inputs with thresholds
        p3_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(oh) and oh[i]:
                p3_traces.append(_trace(
                    f"oriC high occupancy · {labels[i]}",
                    _t(oh, i), oh[i], color=color))
            if i < len(af) and af[i]:
                p3_traces.append(_trace(
                    f"DnaA-ATP fraction · {labels[i]}",
                    _t(af, i), af[i], color=color, dash="dash"))
        # Threshold reference lines
        all_t: list[float] = []
        for tr in time_runs:
            all_t.extend(tr or [])
        if all_t:
            tmin, tmax = min(all_t), max(all_t)
            p3_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"oriC threshold = {oric_t:.2f}",
                "x": [tmin, tmax], "y": [oric_t, oric_t],
                "line": {"width": 1, "color": "#16a34a", "dash": "dot"},
            })
            p3_traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"ATP-frac threshold = {atp_t:.2f}",
                "x": [tmin, tmax], "y": [atp_t, atp_t],
                "line": {"width": 1, "color": "#dc2626", "dash": "dot"},
            })
        p3 = _plotly_div("dnaa-init-inputs", p3_traces, {
            "title": {"text": "Trigger inputs vs thresholds"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "fraction (0–1)"}, "range": [0, 1.05]},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p3_traces else _empty_note(
            "No oric_high / atp_fraction observables.")

        body = (
            f"{_PLOTLY_CDN}\n"
            f'<div style="padding:0 12px 24px 12px">\n'
            f"{p1}\n{p2}\n{p3}\n"
            f"</div>"
        )
        return {"html": render_document(title=title, body_html=body)}


# ---------------------------------------------------------------------------
# DnaACycleSummary — multi-panel "trajectory under the new architecture"
# ---------------------------------------------------------------------------


class DnaACycleSummary(Visualization):
    """Multi-panel synthesis of the DnaA cycle + replication trajectory.

    Mirrors the PDF report's "Trajectory under the new architecture"
    summary card: a single rendered HTML page with four stacked panels.

      1. DnaA pool composition (ATP, ADP, apo counts)
      2. DnaA-box occupancy (chromosome, oriC high, dnaAp)
      3. Active-replisome count + chromosome count
      4. Mass-at-oriC + cell mass

    Designed to be the headline "did the cycle work?" visualization the
    biologist sees first.
    """

    config_schema = {**Visualization.config_schema}

    def inputs(self) -> dict[str, Any]:
        return {
            # Pool composition (from dnaa-02 listener)
            "apo_count":      "list[float]",
            "atp_count":      "list[float]",
            "adp_count":      "list[float]",
            # Box occupancy (from dnaa-03 listener)
            "chromosome_fraction":  "list[float]",
            "oric_high_occupied":   "list[float]",
            "dnaap_fraction":       "list[float]",
            # Replication state (from v2ecoli's existing chromosome listener)
            "n_replisomes":   "list[float]",
            "n_chromosomes":  "list[float]",
            # Mass (from listeners.mass)
            "cell_mass_fg":   "list[float]",
            "mass_at_oric":   "list[float]",
            "time":           "list[float]",
            "_run_labels":    "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "DnaA cycle — trajectory summary"

        apo = _to_runs(state.get("apo_count"))
        atp = _to_runs(state.get("atp_count"))
        adp = _to_runs(state.get("adp_count"))
        ch  = _to_runs(state.get("chromosome_fraction"))
        oh  = _to_runs(state.get("oric_high_occupied"))
        dp  = _to_runs(state.get("dnaap_fraction"))
        nrep = _to_runs(state.get("n_replisomes"))
        nchr = _to_runs(state.get("n_chromosomes"))
        mass = _to_runs(state.get("cell_mass_fg"))
        moric = _to_runs(state.get("mass_at_oric"))
        time_runs = _to_runs(state.get("time"))

        n = max(len(apo), len(atp), len(adp), len(ch), len(oh), len(dp),
                len(nrep), len(nchr), len(mass), len(moric), 0)
        if n == 0:
            body = _empty_note(
                "No observables yet. Wire inputs_map to dnaa_cycle.* / "
                "dnaA_binding.* / mass.* / unique listener paths and re-run.")
            return {"html": render_document(title=title, body_html=body)}

        labels = _coerce_labels(state.get("_run_labels"), n)

        def _t(traces, i):
            if i < len(time_runs) and time_runs[i]:
                return time_runs[i]
            if i < len(traces) and traces[i]:
                return list(range(len(traces[i])))
            return []

        # Panel 1: pool composition
        p1_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            for series, label, dash in [
                (atp, "DnaA-ATP", None),
                (adp, "DnaA-ADP", "dash"),
                (apo, "apo-DnaA", "dot"),
            ]:
                if i < len(series) and series[i]:
                    p1_traces.append(_trace(
                        f"{label} · {labels[i]}",
                        _t(series, i), series[i],
                        color=color, dash=dash))
        p1 = _plotly_div("dnaa-summary-pool", p1_traces, {
            "title": {"text": "1. Pool composition (counts)"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "molecules / cell"}},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p1_traces else _empty_note("No pool data.")

        # Panel 2: box occupancy
        p2_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            for series, label, dash in [
                (ch, "chromosomal", None),
                (oh, "oriC high",   "dash"),
                (dp, "dnaAp",       "dot"),
            ]:
                if i < len(series) and series[i]:
                    p2_traces.append(_trace(
                        f"{label} · {labels[i]}",
                        _t(series, i), series[i],
                        color=color, dash=dash))
        p2 = _plotly_div("dnaa-summary-boxes", p2_traces, {
            "title": {"text": "2. DnaA-box occupancy"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "fraction"}, "range": [0, 1.05]},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p2_traces else _empty_note(
            "No DnaA-box occupancy data (dnaa-03 not in this run?).")

        # Panel 3: replication state
        p3_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(nrep) and nrep[i]:
                p3_traces.append(_trace(
                    f"active replisomes · {labels[i]}",
                    _t(nrep, i), nrep[i], color=color))
            if i < len(nchr) and nchr[i]:
                p3_traces.append(_trace(
                    f"chromosome count · {labels[i]}",
                    _t(nchr, i), nchr[i], color=color, dash="dash"))
        p3 = _plotly_div("dnaa-summary-reps", p3_traces, {
            "title": {"text": "3. Active replisomes + chromosome count"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "count"}},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p3_traces else _empty_note(
            "No replication-state data (unique.active_replisome not wired?).")

        # Panel 4: mass
        p4_traces: list[dict] = []
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            if i < len(mass) and mass[i]:
                p4_traces.append(_trace(
                    f"cell mass · {labels[i]}",
                    _t(mass, i), mass[i], color=color))
            if i < len(moric) and moric[i]:
                p4_traces.append(_trace(
                    f"mass at oriC · {labels[i]}",
                    _t(moric, i), moric[i], color=color, dash="dash"))
        p4 = _plotly_div("dnaa-summary-mass", p4_traces, {
            "title": {"text": "4. Cell mass + mass-at-oriC"},
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "mass (fg)"}},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if p4_traces else _empty_note(
            "No mass data (listeners.mass.* not wired?).")

        body = (
            f"{_PLOTLY_CDN}\n"
            f'<div style="padding:0 12px 24px 12px">\n'
            f"{p1}\n{p2}\n{p3}\n{p4}\n"
            f"</div>"
        )
        return {"html": render_document(title=title, body_html=body)}


# ---------------------------------------------------------------------------
# DnaAFluxContributions — stacked bars of per-tick hydrolysis events
# ---------------------------------------------------------------------------


class DnaAFluxContributions(Visualization):
    """Stacked bars of per-tick DnaA-ATP → DnaA-ADP hydrolysis events.

    Inspired by the PDF's "DnaA-ATP hydrolysis flux contributors
    (cumulative-difference per ~50 s window)" panels. Two contributors
    shipped today (intrinsic + RIDA-v0); the third slot is reserved for
    DDAH when dnaa-05 lands.

    Inputs are the per-tick listener emits:
      listeners.dnaA_cycle.intrinsic_hydrolysis_events  (per tick)
      listeners.dnaA_cycle.rida_events                   (per tick)

    Renders a single grouped-bar chart showing events per tick by source.
    """

    config_schema = {**Visualization.config_schema}

    def inputs(self) -> dict[str, Any]:
        return {
            "intrinsic_events": "list[float]",
            "rida_events":      "list[float]",
            "time":             "list[float]",
            "_run_labels":      "list[string]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        title = self.config.get("title") or "DnaA hydrolysis flux contributors"

        intr = _to_runs(state.get("intrinsic_events"))
        rida = _to_runs(state.get("rida_events"))
        time_runs = _to_runs(state.get("time"))

        n = max(len(intr), len(rida), 0)
        if n == 0:
            body = _empty_note(
                "No flux observables. Wire inputs_map to "
                "listeners.dnaA_cycle.intrinsic_hydrolysis_events and "
                "listeners.dnaA_cycle.rida_events.")
            return {"html": render_document(title=title, body_html=body)}

        labels = _coerce_labels(state.get("_run_labels"), n)

        def _t(traces, i):
            if i < len(time_runs) and time_runs[i]:
                return time_runs[i]
            if i < len(traces) and traces[i]:
                return list(range(len(traces[i])))
            return []

        traces: list[dict] = []
        for i in range(n):
            label = labels[i]
            xs = (_t(intr, i) if i < len(intr) and intr[i]
                  else _t(rida, i) if i < len(rida) and rida[i]
                  else [])
            if i < len(intr) and intr[i]:
                traces.append({
                    "type": "bar",
                    "name": f"intrinsic · {label}",
                    "x": xs,
                    "y": [_numz(v) for v in intr[i]],
                    "marker": {"color": "#6366f1"},
                })
            if i < len(rida) and rida[i]:
                traces.append({
                    "type": "bar",
                    "name": f"RIDA · {label}",
                    "x": xs,
                    "y": [_numz(v) for v in rida[i]],
                    "marker": {"color": "#f59e0b"},
                })

        chart = _plotly_div("dnaa-flux", traces, {
            "title": {"text": "Hydrolysis events per tick by pathway"},
            "barmode": "stack",
            "xaxis": {"title": {"text": "time (s)"}},
            "yaxis": {"title": {"text": "events / tick"}},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "h", "y": -0.25},
        }) if traces else _empty_note("No flux series.")

        body = (
            f"{_PLOTLY_CDN}\n"
            f'<div style="padding:0 12px 24px 12px">\n'
            f"{chart}\n"
            f"</div>"
        )
        return {"html": render_document(title=title, body_html=body)}
