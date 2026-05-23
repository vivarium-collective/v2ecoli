"""Render the DnaA-box occupancy LANDSCAPE figure for dnaa-03.

Renders DnaA-binding sites at their REAL chromosomal coordinates from
v2ecoli sim_data (`sim_data.process.replication.motif_coordinates
["DnaA_box"]`, exposed live as `agents/0/unique/DnaA_box.coordinates`).
The sequence-motif scan against the genome yields ~456 putative DnaA-box
sites; v1 of dnaa-03's box-binding step constrains itself to the ~307
high-confidence consensus subset, but the FULL motif landscape is what
this figure surfaces. oriC (green) and Ter (red square) are anchored at
their canonical positions for orientation.

Uses v2ecoli.visualizations.chromosome_circle.chromosome_circle_svg
(circular SVG renderer, moved out of pbg_superpowers).

Usage:
  .venv/bin/python scripts/render_dnaa_box_landscape.py \\
      --study dnaa-03-box-binding \\
      --spec v2ecoli.composites.baseline_recipes.dnaa_03_with_box_binding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from bigraph_schema import allocate_core
from process_bigraph import Composite
from pbg_superpowers.composite_generator import _REGISTRY, build_generator
import v2ecoli.composites  # noqa: F401
from v2ecoli.visualizations.chromosome_circle import chromosome_circle_svg

# E. coli K-12 MG1655 chromosome length (bp).
CHROMOSOME_LENGTH_BP = 4_641_652
ORIC_ABS_BP = 3_925_859  # canonical oriC absolute position; relative coords are oriC = 0


def _signed_to_absolute(signed_bp: int, genome_len: int = CHROMOSOME_LENGTH_BP) -> int:
    """Convert oriC-relative signed coord → wrapped absolute (0 = oriC for the SVG)."""
    return signed_bp % genome_len


def build_panel(comp: Composite) -> dict:
    """Build one panel from the live composite's DnaA_box unique molecules."""
    boxes = comp.state["agents"]["0"]["unique"]["DnaA_box"]
    active_mask = boxes["_entryState"].view(np.bool_)
    active = boxes[active_mask]

    coords = active["coordinates"]  # signed; oriC = 0
    bound = active["DnaA_bound"] if "DnaA_bound" in active.dtype.names else None

    bound_pts: list[dict] = []
    free_pts: list[dict] = []
    for i, c in enumerate(coords):
        rec = {
            "coord": _signed_to_absolute(int(c)),
            "marker": "circle",
            "size": 2,
        }
        if bound is not None and bool(bound[i]):
            rec.update(color="#16a34a", category=f"bound DnaA-box")
            bound_pts.append(rec)
        else:
            rec.update(color="#94a3b8", category=f"unbound DnaA-box")
            free_pts.append(rec)

    features = {
        "_oriC_anchor": [{"coord": 0, "marker": "circle", "color": "#16a34a",
                          "size": 10, "category": "OriC"}],
        "_ter_anchor": [{"coord": CHROMOSOME_LENGTH_BP // 2, "marker": "square",
                          "color": "#dc2626", "size": 9, "category": "Ter"}],
        "free": free_pts,
        "bound": bound_pts,
    }
    # Rename categories with live counts.
    if free_pts:
        for r in free_pts:
            r["category"] = f"unbound DnaA-box ({len(free_pts)})"
    if bound_pts:
        for r in bound_pts:
            r["category"] = f"bound DnaA-box ({len(bound_pts)})"
    return {"label": f"DnaA-box landscape — {len(coords)} sites "
                     f"({len(bound_pts)} bound / {len(free_pts)} unbound at t = {comp.state.get('global_time', 0):.0f}s)",
            "chromosomes": [{"features": features}]}


def render_html(svg: str, title: str, n_total: int, n_bound: int) -> str:
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
        f"<div class='sub'>{n_total} DnaA-box motif positions from v2ecoli sim_data "
        f"(<code>sim_data.process.replication.motif_coordinates[\"DnaA_box\"]</code>), "
        f"placed at their real chromosomal coordinates. {n_bound} bound (green) / "
        f"{n_total - n_bound} unbound (grey). Anchors: oriC (green dot, top), Ter "
        "(red square, bottom). The dnaa-03 box-binding step acts on the ~307 "
        "high-confidence consensus subset; the full ~456 motif landscape is what "
        "the genome-wide sequence scan finds.</div>"
        f"<div class='svg-host'>{svg}</div>"
        "</div></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="dnaa-03-box-binding")
    ap.add_argument(
        "--spec",
        default="v2ecoli.composites.baseline_recipes.dnaa_03_with_box_binding",
        help="composite spec id",
    )
    ap.add_argument("--steps", type=int, default=600,
                    help="how many ticks to run before snapshotting (longer = more boxes bound)")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    core = allocate_core()
    entry = _REGISTRY[args.spec]
    doc = build_generator(entry, overrides={"seed": args.seed, "cache_dir": args.cache_dir})
    comp = Composite({"state": doc.get("state", doc)}, core=core)

    # Optionally run a bit so DnaA boxes are bound (snapshot the resulting state).
    done = 0
    while done < args.steps:
        n = min(args.chunk, args.steps - done)
        try:
            comp.run(n)
        except Exception as e:
            print(f"[box_landscape] composite stopped at {done}s: {str(e)[:80]}")
            break
        done += n
        if "0" not in (comp.state.get("agents") or {}):
            break
    print(f"[box_landscape] snapshot at t={done}s")

    panel = build_panel(comp)
    title = args.title or "dnaa-03 — DnaA-box landscape on the E. coli chromosome"

    boxes = comp.state["agents"]["0"]["unique"]["DnaA_box"]
    active = boxes[boxes["_entryState"].view(np.bool_)]
    n_total = len(active)
    n_bound = int(active["DnaA_bound"].sum()) if "DnaA_bound" in active.dtype.names else 0

    svg = chromosome_circle_svg(
        title=title,
        subtitle=f"{n_total} motif positions ({n_bound} bound) at t = {done}s",
        panels=[panel],
        genome_len=CHROMOSOME_LENGTH_BP,
        width=900,
        panel_radius=280,
    )

    html = render_html(svg, title, n_total, n_bound)
    out = Path("studies") / args.study / "viz" / "dnaa_box_landscape.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[box_landscape] wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"[box_landscape] {n_total} total boxes ({n_bound} bound) at t={done}s")


if __name__ == "__main__":
    main()
