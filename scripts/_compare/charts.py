"""
Shared inline-SVG chart helpers.

Single source of truth for ``sparkline()`` and ``multiline_svg()``, used by
both ``reports/composite_comparison.py`` and the new harness report module.
``reports/composite_comparison.py`` imports these as ``_sparkline`` and
``_multiline_svg`` so all existing call-sites remain unchanged.
"""
from __future__ import annotations

# Per-engine palette — shared with composite_comparison.py and the harness
# report.  Keep in sync: this is the authoritative definition.
PALETTE = ["#3730a3", "#b45309", "#15803d", "#9d174d", "#0e7490", "#6d28d9"]


def sparkline(snaps, key, w=260, h=44, color="#3730a3"):
    """Inline SVG sparkline of snapshot[key] over time."""
    pts = [(float(s.get("time", 0)), float(s.get(key, 0) or 0)) for s in snaps]
    pts = [(t, v) for t, v in pts if v == v]  # drop NaN
    if len(pts) < 2:
        return "<span class='ref'>—</span>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0; dy = (y1 - y0) or 1.0
    coords = " ".join(
        f"{(t-x0)/dx*(w-4)+2:.1f},{h-2-((v-y0)/dy*(h-6)):.1f}" for t, v in pts)
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='1.5' "
            f"points='{coords}'/></svg>")


def _fmt_axis(v: float) -> str:
    """Compact numeric tick label."""
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1e4 or av < 1e-2:
        return f"{v:.1e}"
    if av >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def multiline_svg(series, w=300, h=120, baseline_zero=True, axes=False,
                  x_label="time (s)", colors=None, vlines=None):
    """Shared-axis multi-line SVG. ``series`` is a list aligned to engine index;
    each element is a list of (x, y) points (or None/empty to skip). All lines
    share one auto-scaled x/y range so absolute divergence is directly visible.

    ``axes=False`` (default) renders the compact, full-width sparkline used by
    ``composite_comparison.py``. ``axes=True`` reserves margins and draws
    numbered x (time) and y (value) tick labels with undistorted text.

    ``colors`` optionally overrides the per-series stroke (list aligned to
    ``series``); when ``None`` the cyclic ``PALETTE`` is used. ``vlines`` is an
    optional list of x positions drawn as light dashed verticals (axes mode) —
    used to mark generation boundaries.

    Returns ``(svg_str, (y0, y1))`` — the SVG string and the y-axis extent.
    """
    def _color(i):
        if colors and i < len(colors) and colors[i]:
            return colors[i]
        return PALETTE[i % len(PALETTE)]
    gxs, gys = [], []
    for pts in series:
        for t, v in (pts or []):
            if v == v:
                gxs.append(t); gys.append(v)
    if len(gxs) < 2:
        return "<span class='ref'>—</span>", (0.0, 0.0)
    x0, x1 = min(gxs), max(gxs)
    y0, y1 = min(gys), max(gys)
    if baseline_zero:
        y0 = min(y0, 0.0)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0

    if not axes:
        pad = 7
        lines = []
        for i, pts in enumerate(series):
            pts = [(t, v) for t, v in (pts or []) if v == v]
            if len(pts) < 2:
                continue
            coords = " ".join(
                f"{pad+(t-x0)/dx*(w-2*pad):.1f},{h-pad-((v-y0)/dy*(h-2*pad)):.1f}"
                for t, v in pts)
            lines.append(
                f"<polyline fill='none' stroke='{_color(i)}' "
                f"stroke-width='1.5' points='{coords}'/>")
        svg = (f"<svg width='100%' height='{h}' viewBox='0 0 {w} {h}' "
               f"preserveAspectRatio='none'>"
               f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' "
               f"stroke='#e2e8f0'/>{''.join(lines)}</svg>")
        return svg, (y0, y1)

    # Axes mode: fixed (undistorted) SVG with margins for numbered ticks.
    mL, mR, mT, mB = 48, 8, 10, 28
    pw, ph = w - mL - mR, h - mT - mB

    def X(t):
        return mL + (t - x0) / dx * pw

    def Y(v):
        return mT + (1 - (v - y0) / dy) * ph

    lines = []
    for i, pts in enumerate(series):
        pts = [(t, v) for t, v in (pts or []) if v == v]
        if len(pts) < 2:
            continue
        coords = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
        lines.append(
            f"<polyline fill='none' stroke='{_color(i)}' "
            f"stroke-width='1.4' points='{coords}'/>")

    axis = (f"<line x1='{mL}' y1='{mT}' x2='{mL}' y2='{mT+ph}' stroke='#cbd5e1'/>"
            f"<line x1='{mL}' y1='{mT+ph}' x2='{mL+pw}' y2='{mT+ph}' "
            f"stroke='#cbd5e1'/>")
    for vx in (vlines or []):
        if x0 <= vx <= x1:
            axis += (f"<line x1='{X(vx):.1f}' y1='{mT}' x2='{X(vx):.1f}' "
                     f"y2='{mT+ph}' stroke='#cbd5e1' stroke-width='1' "
                     f"stroke-dasharray='3,3'/>")

    def _t(x, y, s, anchor="middle"):
        return (f"<text x='{x:.0f}' y='{y:.0f}' font-size='8' fill='#64748b' "
                f"text-anchor='{anchor}'>{s}</text>")

    ymid = (y0 + y1) / 2
    yticks = (_t(mL - 5, Y(y1) + 3, _fmt_axis(y1), "end")
              + _t(mL - 5, Y(ymid) + 3, _fmt_axis(ymid), "end")
              + _t(mL - 5, Y(y0) + 3, _fmt_axis(y0), "end"))
    xmid = (x0 + x1) / 2
    yb = mT + ph + 11
    xticks = (_t(mL, yb, _fmt_axis(x0), "start")
              + _t(X(xmid), yb, _fmt_axis(xmid), "middle")
              + _t(mL + pw, yb, _fmt_axis(x1), "end"))
    xcap = _t(mL + pw / 2, mT + ph + 23, x_label, "middle")
    svg = (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
           f"{axis}{''.join(lines)}{yticks}{xticks}{xcap}</svg>")
    return svg, (y0, y1)
