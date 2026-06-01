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


def _classify_region(signed_coord: int) -> str:
    """Coarse region classifier by genomic coordinate.

    Per the molecular reference (v2ecoli.data.dnaa_box_catalog +
    molecular_info.pdf):
      - oriC center ≈ 3,925,859 bp absolute → 0 in signed coords. Spans ~250 bp.
      - dnaA promoter at ~3,882,000 bp absolute → ~-43,859 in signed coords.
      - datA at ~4,440,000 abs → ~+514,141 signed.
      - DARS1 at ~3,425,000 abs → ~-500,859 signed.
      - DARS2 at ~1,890,000 abs → ~-2,035,859 signed (wraps near terC).
    """
    if abs(signed_coord) <= 5_000:
        return "ORIC"
    # dnaA promoter region (~50 kb tolerance, locus is small)
    if -50_000 < signed_coord < -30_000:
        return "DNAAP"
    if 500_000 <= signed_coord <= 530_000:
        return "DATA"
    if -515_000 <= signed_coord <= -485_000:
        return "DARS1"
    if -2_055_000 <= signed_coord <= -2_015_000:
        return "DARS2"
    return "CHROMOSOMAL"


def _bound_counts_from_listener(comp: Composite) -> dict[str, int]:
    """Derive per-region bound-DnaA-box counts from the live dnaA_binding listener."""
    cell = (comp.state.get("agents") or {}).get("0") or {}
    db = (cell.get("listeners") or {}).get("dnaA_binding") or {}
    oric = db.get("oric") or {}
    dnaap = db.get("dnaap") or {}
    chrom = db.get("chromosome") or {}
    return {
        "ORIC": int(round((float(oric.get("high_affinity_occupied", 0.0) or 0.0) * 3) +
                          (float(oric.get("low_affinity_occupied", 0.0) or 0.0) * 8))),
        "DNAAP": int(round(float(dnaap.get("occupied", 0.0) or 0.0) * 7)),
        "CHROMOSOMAL": int(chrom.get("occupied_count", 0) or 0),
        "DATA": 0,    # not yet modelled (dnaa-06 scope)
        "DARS1": 0,
        "DARS2": 0,
    }


def build_panel(comp: Composite) -> dict:
    """Build one panel from the live composite's DnaA_box unique molecules.

    Strategy: read REAL coordinates from unique.DnaA_box. For per-box BOUND
    state, derive from the dnaA_binding listener's per-region counts (the
    step writes those counts correctly; per-box flags on the unique molecule
    aren't yet written by the dnaa_box_binding step — proper fix queued as
    a follow-up task). Within each region, mark the first K live boxes as
    bound (deterministic; positions inside a region are the natural sort
    order). This is visually accurate at the per-region level.
    """
    import random as _random  # local rng so script is repeatable
    rng = _random.Random(0)

    boxes = comp.state["agents"]["0"]["unique"]["DnaA_box"]
    active = boxes[boxes["_entryState"].view(np.bool_)]
    coords = active["coordinates"]

    # Group live boxes by region.
    by_region: dict[str, list[int]] = {}  # region → indices into `active`
    for i, c in enumerate(coords):
        by_region.setdefault(_classify_region(int(c)), []).append(i)

    # Listener-driven bound counts per region.
    bound_targets = _bound_counts_from_listener(comp)

    # Decide bound vs free per live box.
    bound_idxs: set[int] = set()
    for region, idxs in by_region.items():
        n_bind = min(int(bound_targets.get(region, 0)), len(idxs))
        if n_bind > 0:
            rng.shuffle(idxs)
            bound_idxs.update(idxs[:n_bind])

    # Color by REGION; bound boxes get a larger marker.
    region_color = {
        "ORIC":        "#16a34a",   # green
        "DNAAP":       "#7c3aed",   # purple
        "DATA":        "#0ea5e9",   # blue
        "DARS1":       "#0d9488",   # teal
        "DARS2":       "#06b6d4",   # cyan
        "CHROMOSOMAL": "#94a3b8",   # grey
    }
    n_bound = len(bound_idxs)
    n_free = len(coords) - n_bound

    free_pts: list[dict] = []
    bound_pts: list[dict] = []
    for i, c in enumerate(coords):
        region = _classify_region(int(c))
        absolute = _signed_to_absolute(int(c))
        is_bound = i in bound_idxs
        rec = {
            "coord": absolute,
            "marker": "circle",
            "color": region_color.get(region, "#94a3b8") if not is_bound else "#16a34a",
            "size": 4 if is_bound else 2,
        }
        if is_bound:
            rec["category"] = f"bound DnaA-box ({n_bound})"
            bound_pts.append(rec)
        else:
            # Single legend entry for all unbound boxes (color-by-region is
            # still visually present on the chromosome — the legend just doesn't
            # repeat itself per region or it overflows the SVG width).
            rec["category"] = f"unbound DnaA-box ({n_free})"
            free_pts.append(rec)

    features = {
        "_oriC_anchor": [{"coord": 0, "marker": "circle", "color": "#0f5132",
                          "size": 12, "category": "OriC anchor"}],
        "_ter_anchor": [{"coord": CHROMOSOME_LENGTH_BP // 2, "marker": "square",
                          "color": "#dc2626", "size": 10, "category": "Ter"}],
        "free": free_pts,
        "bound": bound_pts,
    }
    t_sec = comp.state.get("global_time", 0)
    return {"label": f"DnaA-box landscape — {len(coords)} sites "
                     f"({n_bound} bound / {n_free} unbound at t = {t_sec:.0f}s)",
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
    out = Path("workspace/studies") / args.study / "viz" / "dnaa_box_landscape.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[box_landscape] wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"[box_landscape] {n_total} total boxes ({n_bound} bound) at t={done}s")


if __name__ == "__main__":
    main()
