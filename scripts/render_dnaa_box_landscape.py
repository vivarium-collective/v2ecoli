"""Render the DnaA-box occupancy LANDSCAPE figure for dnaa-03.

Adapted from PR #28's `_chromosome_diagram_static` / `_box_occupancy_view`.
Renders all 325 active DnaA-binding sites at their actual chromosomal
coordinates around a single circular E. coli chromosome, color-coded by
region (oriC, dnaA promoter, chromosomal consensus) and affinity tier.

Uses v2ecoli.visualizations.chromosome_circle.chromosome_circle_svg (the
generic E. coli circular-chromosome SVG renderer) — no composite run
needed; the catalog (v2ecoli.data.dnaa_box_catalog) is the source.

Usage:
  .venv/bin/python scripts/render_dnaa_box_landscape.py [--study dnaa-03-box-binding]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
from v2ecoli.data.dnaa_box_catalog import active_boxes, CHROMOSOME_LENGTH_BP
from v2ecoli.visualizations.chromosome_circle import chromosome_circle_svg


# Color palette by region/tier.
_COLORS = {
    "ORIC_high":                "#16a34a",  # green
    "ORIC_low":                 "#f59e0b",  # amber
    "DNAAP_high":               "#7c3aed",  # purple
    "DNAAP_low":                "#a78bfa",  # light purple
    "CHROMOSOMAL_TITRATION":    "#94a3b8",  # grey
    "DATA":                     "#0ea5e9",  # blue
    "DARS1":                    "#0d9488",  # teal
    "DARS2":                    "#06b6d4",  # cyan
}


def _affinity_value(b) -> str:
    """Normalize affinity_class which may be an enum or string."""
    v = b.affinity_class
    return v.value if hasattr(v, "value") else str(v)


def _region_value(b) -> str:
    """Normalize region_type which may be an enum or string."""
    v = b.region_type
    return v.value if hasattr(v, "value") else str(v)


def _categorize(b) -> tuple[str, str, int]:
    """Return (color, category-name, marker-size) for a box."""
    region = _region_value(b)
    aff = _affinity_value(b)
    key = f"{region}_{aff}" if region in ("ORIC", "DNAAP") else region
    color = _COLORS.get(key, "#64748b")
    if region == "ORIC":
        cat = f"oriC {aff}-affinity ({_count_by(region, aff)} boxes)"
        size = 7
    elif region == "DNAAP":
        cat = f"dnaA promoter {aff}-affinity ({_count_by(region, aff)} boxes)"
        size = 5
    elif region == "CHROMOSOMAL_TITRATION":
        cat = "chromosomal consensus (307 boxes)"
        size = 2
    else:
        cat = f"{region} ({_count_by(region)} boxes)"
        size = 4
    return color, cat, size


_CACHE: dict = {}


def _count_by(region: str, aff: str | None = None) -> int:
    """Count active boxes for a region [+ affinity]."""
    if (region, aff) not in _CACHE:
        n = 0
        for b in active_boxes():
            if _region_value(b) == region and (aff is None or _affinity_value(b) == aff):
                n += 1
        _CACHE[(region, aff)] = n
    return _CACHE[(region, aff)]


def build_panel() -> dict:
    """One panel showing every active box."""
    boxes_by_cat: dict[str, list[dict]] = {}
    for b in active_boxes():
        color, cat, size = _categorize(b)
        marker = "circle"
        boxes_by_cat.setdefault(cat, []).append({
            "coord": int(b.position),
            "marker": marker,
            "color": color,
            "size": size,
            "category": cat,
        })
    # Always add oriC (green) + Ter (red) anchors so the legend has them.
    features = {
        "_oriC_anchor": [{"coord": 0, "marker": "circle", "color": "#16a34a",
                          "size": 9, "category": "OriC"}],
        "_ter_anchor": [{"coord": CHROMOSOME_LENGTH_BP // 2, "marker": "square",
                          "color": "#dc2626", "size": 8, "category": "Ter"}],
    }
    features.update(boxes_by_cat)
    return {"label": "DnaA-binding sites on the E. coli chromosome",
            "chromosomes": [{"features": features}]}


def render_html(svg: str, title: str) -> str:
    body_h = 800
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"<style>html,body{{height:{body_h}px;overflow:hidden}}"
        "body{margin:0;padding:0;background:#fff;font-family:-apple-system,system-ui,sans-serif}"
        ".wrap{padding:16px 22px;height:100%;box-sizing:border-box;display:flex;flex-direction:column}"
        "h1{font-size:1.05em;margin:0 0 4px 0;color:#0f172a}"
        ".sub{color:#6b7280;font-size:0.85em;margin-bottom:12px}"
        ".svg-host{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden}"
        ".svg-host svg{max-width:100%;max-height:100%;height:auto;width:auto;"
        "border:1px solid #e5e7eb;border-radius:6px}"
        "</style></head><body><div class='wrap'>"
        f"<h1>{title}</h1>"
        "<div class='sub'>Every active DnaA-binding site at its real chromosomal "
        "coordinate (oriC at top, Ter at bottom). Green = high-affinity oriC boxes "
        "(R1/R2/R4); amber = low-affinity oriC boxes (R5M/τ2/I1-3/C1-3); "
        "purple = dnaA-promoter boxes; grey dots = 307 chromosomal consensus boxes "
        "that sequester DnaA away from oriC.</div>"
        f"<div class='svg-host'>{svg}</div>"
        "</div></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="dnaa-03-box-binding")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    panel = build_panel()
    title = args.title or "dnaa-03 — DnaA-binding sites across the chromosome"
    svg = chromosome_circle_svg(
        title=title,
        subtitle="325 active sites (11 oriC + 7 dnaA-promoter + 307 chromosomal consensus)",
        panels=[panel],
        genome_len=CHROMOSOME_LENGTH_BP,
        width=900,
        panel_radius=280,
    )

    html = render_html(svg, title)
    out = Path("studies") / args.study / "viz" / "dnaa_box_landscape.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[box_landscape] wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"[box_landscape] {len(panel['chromosomes'][0]['features'])} feature categories")
    n_boxes = sum(len(v) for v in panel['chromosomes'][0]['features'].values())
    print(f"[box_landscape] total marks rendered: {n_boxes}")


if __name__ == "__main__":
    main()
