"""Interactive Plotly fragments shared by the trajectory/distribution report
cards. Each `*_html` function concatenates one `fig.to_html(full_html=False)`
fragment per axis/observable, loading plotly.js via CDN once (the first
figure) and inlining nothing thereafter (`include_plotlyjs=False`) so a card
with several figures doesn't re-download the library per figure.

Color convention (matches the rest of the comparison report): colors come
from the shared theme (`scripts/_compare/theme.py`) so charts and the HTML
report never drift apart — vEcoli is the ``reference`` engine color,
v2ecoli is the ``candidate`` engine color.
"""
from __future__ import annotations

from scripts._compare import theme

VE_COLOR = theme.ENGINE["reference"]   # vEcoli
V2_COLOR = theme.ENGINE["candidate"]   # v2ecoli


def overlay_html(per_obs: dict, title: str = "") -> str:
    """One value-vs-time overlay figure per observable.

    ``per_obs`` maps observable -> {"v2": [(times, values), ...],
    "ve": [(times, values), ...], "gen_bounds": [t, ...]}; the list holds one
    ``(times, values)`` trace per seed (usually just one). Generation-boundary
    times, if given, are drawn as vertical dashed lines.
    """
    import plotly.graph_objects as go

    parts = []
    first = True
    for obs, d in per_obs.items():
        ve_traces = d.get("ve") or []
        v2_traces = d.get("v2") or []
        if not ve_traces and not v2_traces:
            continue
        fig = go.Figure()
        for i, (t, v) in enumerate(ve_traces):
            fig.add_scatter(
                x=list(t), y=list(v), mode="lines",
                name="vEcoli" if len(ve_traces) == 1 else f"vEcoli seed{i}",
                legendgroup="ve", showlegend=(i == 0),
                line=dict(color=VE_COLOR))
        for i, (t, v) in enumerate(v2_traces):
            fig.add_scatter(
                x=list(t), y=list(v), mode="lines",
                name="v2ecoli" if len(v2_traces) == 1 else f"v2ecoli seed{i}",
                legendgroup="v2", showlegend=(i == 0),
                line=dict(color=V2_COLOR))
        for gb in d.get("gen_bounds") or []:
            fig.add_vline(x=float(gb), line=dict(color="#9ca3af", dash="dot", width=1))
        fig.update_layout(
            title=f"{title} — {obs}".strip(" —"), height=280,
            margin=dict(l=40, r=10, t=30, b=30),
            hovermode="x unified", template="simple_white")
        parts.append(fig.to_html(include_plotlyjs=("cdn" if first else False),
                                 full_html=False))
        first = False
    return "".join(parts)


def _mean_std_by_timepoint(traces: list):
    """Reduce a list of ``(times, values)`` seed traces to per-timepoint
    mean/σ, aligning on the SHORTEST trace among ``traces`` (seeds are not
    guaranteed to share a length, e.g. divergent generation counts). Returns
    ``(times, means, stds)`` truncated to that shortest length, or
    ``(None, None, None)`` if ``traces`` is empty.
    """
    import statistics

    if not traces:
        return None, None, None
    n = min(len(v) for _, v in traces)
    if n == 0:
        return None, None, None
    times = list(traces[0][0])[:n]
    means = []
    stds = []
    for i in range(n):
        vals = [v[i] for _, v in traces]
        means.append(statistics.fmean(vals))
        stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return times, means, stds


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def overlay_band_html(per_obs: dict, title: str = "") -> str:
    """One value-vs-time overlay figure per observable, cross-seed band form.

    Same ``per_obs`` shape as :func:`overlay_html`, but instead of drawing
    one raw line per seed, reduces each engine's seed traces to a
    per-timepoint mean +/- 1 sigma (see ``_mean_std_by_timepoint``) and draws
    a 2px mean line with a low-opacity shaded band in the engine's identity
    color. Single y-axis, unified crosshair hover (dataviz method: never
    dual-axis).
    """
    import plotly.graph_objects as go

    parts = []
    first = True
    for obs, d in per_obs.items():
        ve_traces = d.get("ve") or []
        v2_traces = d.get("v2") or []
        if not ve_traces and not v2_traces:
            continue
        fig = go.Figure()
        for label, traces, color, legendgroup in (
            ("v2ecoli", v2_traces, V2_COLOR, "v2"),
            ("vEcoli", ve_traces, VE_COLOR, "ve"),
        ):
            times, means, stds = _mean_std_by_timepoint(traces)
            if times is None:
                continue
            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]
            fig.add_scatter(
                x=times, y=upper, mode="lines",
                line=dict(width=0, color=color),
                legendgroup=legendgroup, showlegend=False, hoverinfo="skip")
            fig.add_scatter(
                x=times, y=lower, mode="lines",
                line=dict(width=0, color=color),
                fill="tonexty", fillcolor=_hex_to_rgba(color, 0.15),
                legendgroup=legendgroup, showlegend=False, hoverinfo="skip",
                name=f"{label} ±1σ")
            fig.add_scatter(
                x=times, y=means, mode="lines", name=label,
                legendgroup=legendgroup, showlegend=True,
                line=dict(color=color, width=2))
        for gb in d.get("gen_bounds") or []:
            fig.add_vline(x=float(gb), line=dict(color="#9ca3af", dash="dot", width=1))
        fig.update_layout(
            title=f"{title} — {obs}".strip(" —"), height=280,
            margin=dict(l=40, r=10, t=30, b=30),
            hovermode="x unified", template="simple_white")
        parts.append(fig.to_html(include_plotlyjs=("cdn" if first else False),
                                 full_html=False))
        first = False
    return "".join(parts)


def _median_relative_deltas(ve_traces: list, v2_traces: list):
    """Per-timepoint median relative delta ``(v2 - ve) / ve``, pairing v2/ve
    seed traces by index (seed N of v2 against seed N of ve — the harness's
    matched-timepoint convention), truncated to the shortest paired trace
    length. Returns ``(times, deltas)``, or ``(None, None)`` if there are no
    v2/ve pairs.

    A timepoint where EVERY pair has ``ve == 0`` gets ``math.nan`` rather
    than ``0.0``: the relative delta is undefined there (division by the
    reference value), and rendering it as ``0.0`` would draw a false
    "agrees / within tolerance" point at a spot of undefined-or-unbounded
    divergence. Plotly renders a NaN y-value as a gap in the line, not a
    point, so the gap is visible instead of a misleading zero.
    """
    import math
    import statistics

    n_pairs = min(len(ve_traces), len(v2_traces))
    if n_pairs == 0:
        return None, None
    n_t = min(min(len(v) for _, v in ve_traces[:n_pairs]),
              min(len(v) for _, v in v2_traces[:n_pairs]))
    if n_t == 0:
        return None, None
    times = list(ve_traces[0][0])[:n_t]
    deltas = []
    for i in range(n_t):
        rel = []
        for p in range(n_pairs):
            ve_v = ve_traces[p][1][i]
            v2_v = v2_traces[p][1][i]
            if ve_v == 0:
                continue
            rel.append((v2_v - ve_v) / ve_v)
        deltas.append(statistics.median(rel) if rel else math.nan)
    return times, deltas


def delta_panel_html(per_obs: dict, tol: float, stat: dict | None = None) -> str:
    """One relative-delta-vs-time figure per observable.

    For each observable, pairs v2/ve seed traces by index (seed N of v2
    against seed N of ve — the harness's matched-timepoint convention) and
    plots the MEDIAN relative delta ``(v2 - ve) / ve`` across those pairs at
    each timepoint (see ``_median_relative_deltas``; a timepoint where every
    pair's reference value is 0 renders as a gap, never a fake 0.0). Shades
    the tolerance band ``[-tol, +tol]`` in a neutral (non-engine, non-status)
    gray. If ``stat`` (e.g. ``{"kind": "Welch-t", "p": 0.4}``) is given, adds
    an inline annotation naming the stat and its p-value.
    """
    import plotly.graph_objects as go

    parts = []
    first = True
    for obs, d in per_obs.items():
        ve_traces = d.get("ve") or []
        v2_traces = d.get("v2") or []
        times, median_delta = _median_relative_deltas(ve_traces, v2_traces)
        if times is None:
            continue

        fig = go.Figure()
        fig.add_hrect(y0=-tol, y1=tol, fillcolor="rgba(156, 163, 175, 0.18)",
                     line_width=0, layer="below")
        fig.add_hline(y=0, line=dict(color="#9ca3af", width=1, dash="dot"))
        fig.add_scatter(
            x=times, y=median_delta, mode="lines+markers",
            name="median relative Δ (v2 vs ve)",
            connectgaps=False,
            line=dict(color=V2_COLOR, width=2))
        if stat:
            kind = stat.get("kind", "")
            p_val = stat.get("p")
            text = f"{kind}: p={p_val}" if p_val is not None else str(kind)
            fig.add_annotation(
                xref="paper", yref="paper", x=0.98, y=0.95,
                showarrow=False, align="right", text=text,
                bgcolor="rgba(255,255,255,0.7)")
        fig.update_layout(
            title=f"Δ {obs}", height=240,
            margin=dict(l=40, r=10, t=30, b=30),
            hovermode="x unified", template="simple_white",
            yaxis_title="relative Δ")
        parts.append(fig.to_html(include_plotlyjs=("cdn" if first else False),
                                 full_html=False))
        first = False
    return "".join(parts)


def grouped_bar_html(categories: list, ve_values: list, v2_values: list,
                     title: str = "", yaxis_title: str = "", first: bool = False) -> str:
    """One grouped-bar figure: vEcoli vs v2ecoli value per category.

    ``categories`` labels the x-axis (e.g. mass-fraction names); ``ve_values``/
    ``v2_values`` are the same length, one bar height per category per engine.
    A missing value (``None``) is skipped rather than rendered as a zero bar.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_bar(x=categories, y=[v if v is not None else None for v in ve_values],
               name="vEcoli", marker_color=VE_COLOR)
    fig.add_bar(x=categories, y=[v if v is not None else None for v in v2_values],
               name="v2ecoli", marker_color=V2_COLOR)
    fig.update_layout(
        title=title, barmode="group", height=300,
        margin=dict(l=40, r=10, t=30, b=30), yaxis_title=yaxis_title,
        template="simple_white")
    return fig.to_html(include_plotlyjs=("cdn" if first else False), full_html=False)


def violin_html(axis_records: list, title: str = "") -> str:
    """One violin+strip figure per axis record.

    Each record: {"label": str, "v2_values": [float,...],
    "ve_values": [float,...], "meter": str}. Points are overlaid
    (``points="all"``) so the (often n=1) raw observations stay visible.
    """
    import plotly.graph_objects as go

    parts = []
    first = True
    for rec in axis_records:
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=rec.get("ve_values") or [], name="vEcoli", line_color=VE_COLOR,
            points="all", box_visible=True, meanline_visible=True))
        fig.add_trace(go.Violin(
            y=rec.get("v2_values") or [], name="v2ecoli", line_color=V2_COLOR,
            points="all", box_visible=True, meanline_visible=True))
        label = rec.get("label", "")
        meter = rec.get("meter", "")
        subtitle = f"{title} — {label}".strip(" —")
        if meter:
            subtitle = f"{subtitle} ({meter})"
        fig.update_layout(
            title=subtitle, height=320,
            margin=dict(l=40, r=10, t=40, b=30), template="simple_white")
        parts.append(fig.to_html(include_plotlyjs=("cdn" if first else False),
                                 full_html=False))
        first = False
    return "".join(parts)
