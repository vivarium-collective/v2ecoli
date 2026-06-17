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


def multiline_svg(series, w=300, h=120, baseline_zero=True):
    """Shared-axis multi-line SVG. ``series`` is a list aligned to engine index;
    each element is a list of (x, y) points (or None/empty to skip). All lines
    share one auto-scaled x/y range so absolute divergence is directly visible.

    Returns ``(svg_str, (y0, y1))`` — the SVG string and the y-axis extent.
    """
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
    pad = 7
    lines = []
    for i, pts in enumerate(series):
        pts = [(t, v) for t, v in (pts or []) if v == v]
        if len(pts) < 2:
            continue
        coords = " ".join(
            f"{pad+(t-x0)/dx*(w-2*pad):.1f},{h-pad-((v-y0)/dy*(h-2*pad)):.1f}"
            for t, v in pts)
        lines.append(f"<polyline fill='none' stroke='{PALETTE[i % len(PALETTE)]}' "
                     f"stroke-width='1.5' points='{coords}'/>")
    svg = (f"<svg width='100%' height='{h}' viewBox='0 0 {w} {h}' "
           f"preserveAspectRatio='none'>"
           f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' "
           f"stroke='#e2e8f0'/>{''.join(lines)}</svg>")
    return svg, (y0, y1)
