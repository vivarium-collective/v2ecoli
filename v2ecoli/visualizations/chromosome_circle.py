"""Circular chromosome diagram (SVG) for v2ecoli figures.

Renders E. coli circular chromosomes with feature markers (oriC, ter, RNAPs,
replisomes, DnaA boxes, regulatory regions) at their actual genomic
coordinates. Multiple panels can show different timepoints side-by-side;
each panel can itself contain multiple chromosomes (post-replication
2-chromosome states).

Moved here from pbg_superpowers/study_charts.py (where it was misplaced —
pbg_superpowers is the generic framework and shouldn't know about E. coli
genome biology). The handful of SVG-primitive helpers it uses are
vendored inline so this module is self-contained and doesn't depend on
private pbg_superpowers internals.
"""
from __future__ import annotations

import math
from typing import Sequence

# ─── Vendored SVG primitives ────────────────────────────────────────────
# Small enough to inline; keeps this module self-contained.

_PALETTE = {
    "ink":     "#0f172a",
    "muted":   "#64748b",
    "axis":    "#334155",
}


def _esc(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_open(width: int, height: int, title: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="-apple-system,sans-serif" font-size="13">'
        + (f'<title>{title}</title>' if title else '')
        + f'<rect width="{width}" height="{height}" fill="white"/>'
    )


def _svg_close() -> str:
    return "</svg>"


def _title_bar(width: int, title: str, subtitle: str = "") -> str:
    out = [
        f'<rect x="0" y="0" width="{width}" height="44" fill="{_PALETTE["ink"]}"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" fill="white" '
        f'font-weight="600" font-size="15">{_esc(title)}</text>',
    ]
    if subtitle:
        out.append(
            f'<text x="{width / 2}" y="60" text-anchor="middle" '
            f'fill="{_PALETTE["muted"]}" font-size="11">{_esc(subtitle)}</text>'
        )
    return "".join(out)


# ─── Public function ────────────────────────────────────────────────────


def chromosome_circle_svg(
    title: str,
    *,
    panels: Sequence[dict],
    genome_len: int,
    subtitle: str = "",
    width: int = 1400,
    panel_radius: int = 130,
    panel_spacing: int = 60,
) -> str:
    """Render circular chromosome diagram(s) with feature markers.

    Each panel dict::

        label: str                 — title above the panel
        chromosomes: list[dict]    — one or more circles, each with:
            features: dict[str, list[dict]]
                where each feature dict has:
                  coord:    int     — genomic position in bp (0 = oriC)
                  marker:   str     — 'circle' | 'square' | 'triangle_up' | 'tick'
                  color:    str     — hex color
                  size:     int     — marker radius in px
                  category: str     — used to deduplicate legend entries

    Example::

        chromosome_circle_svg(
            "DnaA-box landscape", panels=[{
                "label": "t=0",
                "chromosomes": [{
                    "features": {
                        "oriC":       [{"coord": 0,         "marker": "circle",       "color": "#16a34a", "size": 8, "category": "OriC"}],
                        "ter":        [{"coord": 2_320_000, "marker": "square",       "color": "#dc2626", "size": 7, "category": "Ter"}],
                        "boxes":      [{"coord": c,         "marker": "circle",       "color": "#94a3b8", "size": 2, "category": "DnaA-box"} for c in box_coords],
                        "replisomes": [{"coord": 1_331_293, "marker": "triangle_up",  "color": "#f59e0b", "size": 7, "category": "Replisome"}],
                    },
                }],
            }],
            genome_len=4_641_652,
        )
    """
    PT = 60 if subtitle else 50
    PB = 80

    n_panels = len(panels)
    if n_panels == 0:
        return _svg_open(width, 80, title) + _title_bar(width, title, subtitle) + _svg_close()

    cell_w = (width - panel_spacing * (n_panels + 1)) / n_panels
    if cell_w < panel_radius * 2 + 20:
        panel_radius = max(60, int((cell_w - 20) / 2))

    max_chr = max(len(p.get("chromosomes", [{}])) for p in panels)
    chr_spacing = 20
    chr_unit_h = panel_radius * 2 + chr_spacing + 30
    panel_total_h = chr_unit_h * max_chr + 40
    height = PT + panel_total_h + PB

    parts = [_svg_open(width, height, title), _title_bar(width, title, subtitle)]

    legend_entries: dict[str, dict] = {}

    def marker_path(cx, cy, mk, size, color, angle=0.0):
        if mk == "circle":
            return f'<circle cx="{cx}" cy="{cy}" r="{size}" fill="{color}"/>'
        if mk == "square":
            return f'<rect x="{cx - size}" y="{cy - size}" width="{size * 2}" height="{size * 2}" fill="{color}"/>'
        if mk == "triangle_up":
            return (
                f'<polygon points="{cx},{cy - size} '
                f'{cx - size},{cy + size * 0.7} {cx + size},{cy + size * 0.7}" fill="{color}"/>'
            )
        if mk == "tick":
            dx = math.sin(angle) * size
            dy = -math.cos(angle) * size
            return (
                f'<line x1="{cx - dx}" y1="{cy - dy}" x2="{cx + dx}" y2="{cy + dy}" '
                f'stroke="{color}" stroke-width="1.2"/>'
            )
        return f'<circle cx="{cx}" cy="{cy}" r="{size}" fill="{color}"/>'

    for pi, panel in enumerate(panels):
        panel_cx = panel_spacing + cell_w / 2 + (cell_w + panel_spacing) * pi

        chromosomes = panel.get("chromosomes", [])
        parts.append(
            f'<text x="{panel_cx}" y="{PT + 10}" text-anchor="middle" '
            f'fill="{_PALETTE["ink"]}" font-weight="600" font-size="13">'
            f'{_esc(panel.get("label", ""))}</text>'
        )

        for ci, chrom in enumerate(chromosomes):
            cy_center = PT + 30 + chr_unit_h * ci + panel_radius
            parts.append(
                f'<circle cx="{panel_cx}" cy="{cy_center}" r="{panel_radius}" '
                f'fill="none" stroke="#cbd5e1" stroke-width="2"/>'
            )

            features = chrom.get("features", {})
            order_by_size = []
            for cat_key, items in features.items():
                for item in items:
                    order_by_size.append((item.get("size", 3), cat_key, item))
            order_by_size.sort(key=lambda t: t[0])
            for _, cat_key, item in order_by_size:
                coord = item.get("coord", 0)
                angle = (coord / genome_len) * 2 * math.pi
                size = item.get("size", 3)
                offset = 0 if size > 4 else -size
                r = panel_radius + offset if item.get("outside") else panel_radius
                cx = panel_cx + r * math.sin(angle)
                cy = cy_center - r * math.cos(angle)
                color = item.get("color", "#64748b")
                marker = item.get("marker", "circle")
                parts.append(marker_path(cx, cy, marker, size, color, angle))
                cat_name = item.get("category", cat_key)
                if cat_name not in legend_entries:
                    legend_entries[cat_name] = {
                        "marker": marker, "color": color, "size": min(size, 6),
                    }

    ly = height - 30
    parts.append(
        f'<text x="{panel_spacing}" y="{ly}" fill="{_PALETTE["axis"]}" '
        f'font-weight="600">Legend:</text>'
    )
    lx = panel_spacing + 80
    for cat, meta in legend_entries.items():
        parts.append(marker_path(lx + 6, ly - 4, meta["marker"], meta["size"], meta["color"]))
        parts.append(
            f'<text x="{lx + 18}" y="{ly}" fill="{_PALETTE["axis"]}">{_esc(cat)}</text>'
        )
        lx += 8 + 18 + 8 + 9 * len(cat)

    parts.append(_svg_close())
    return "\n".join(parts)
