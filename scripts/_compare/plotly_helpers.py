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
