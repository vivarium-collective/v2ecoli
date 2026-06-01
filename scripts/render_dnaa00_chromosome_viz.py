"""Render the dnaa-00 chromosome-state visualization for the investigation report.

Builds the dnaa_00_baseline composite, runs it past the first replication
initiation (~3000 sim seconds), captures live snapshots from `composite.state`
at intervals (avoiding the SQLite-emit-loses-numpy-dtype problem), and calls
`v2ecoli.visualizations.workflow._plot_chromosome_state` to render the
multi-panel chromosome figure (circular maps at 5 timepoints + replication-fork
progress timeseries + mass growth).

Output: a self-contained HTML at
`workspace/studies/dnaa-00-parameter-foundation/viz/chromosome_state.html` that wraps the
matplotlib PNG (base64) in a small page suitable for the report's iframe embed.

Usage:
  .venv/bin/python scripts/render_dnaa00_chromosome_viz.py [--steps 3600]
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
import v2ecoli.composites  # noqa: F401  - registers composites
from v2ecoli.visualizations.workflow import _plot_chromosome_state


def extract_snapshot(state: dict, t: float) -> dict | None:
    """Pull the chromosome-state fields out of the live agent state.

    Unique molecules live under ``cell['unique'][<type>]`` as numpy structured
    arrays (with `_entryState` boolean field marking active entries). Listeners
    live under ``cell['listeners']`` as plain dicts.
    """
    cell = (state.get("agents") or {}).get("0")
    if not cell:
        return None
    mass = cell.get("listeners", {}).get("mass", {}) if isinstance(cell.get("listeners"), dict) else {}
    unique = cell.get("unique") or {}

    fc = unique.get("full_chromosome")
    n_chrom = 0
    if fc is not None and hasattr(fc, "dtype") and "_entryState" in fc.dtype.names:
        n_chrom = int(fc["_entryState"].view(np.bool_).sum())

    rep = unique.get("active_replisome")
    fork_coords: list[int] = []
    fork_domains: list[int] = []
    if rep is not None and hasattr(rep, "dtype") and "_entryState" in rep.dtype.names:
        active_rep = rep[rep["_entryState"].view(np.bool_)]
        if len(active_rep) > 0 and "coordinates" in rep.dtype.names:
            fork_coords = [int(c) for c in active_rep["coordinates"]]
            if "domain_index" in active_rep.dtype.names:
                fork_domains = [int(d) for d in active_rep["domain_index"]]

    domains = unique.get("chromosome_domain")
    n_domains = 0
    domain_children: dict[int, list[int]] = {}
    if domains is not None and hasattr(domains, "dtype") and "_entryState" in domains.dtype.names:
        active_dom = domains[domains["_entryState"].view(np.bool_)]
        n_domains = int(len(active_dom))
        if {"domain_index", "child_domains"}.issubset(set(active_dom.dtype.names)):
            for entry in active_dom:
                kids = entry["child_domains"]
                kids = [int(k) for k in kids if int(k) >= 0]
                domain_children[int(entry["domain_index"])] = kids

    rnap = unique.get("active_RNAP")
    rnap_coords: list[int] = []
    rnap_domains: list[int] = []
    n_rnap = 0
    if rnap is not None and hasattr(rnap, "dtype") and "_entryState" in rnap.dtype.names:
        active_rnap = rnap[rnap["_entryState"].view(np.bool_)]
        n_rnap = len(active_rnap)
        if n_rnap > 0 and "coordinates" in rnap.dtype.names:
            rnap_coords = [int(c) for c in active_rnap["coordinates"]]
            if "domain_index" in active_rnap.dtype.names:
                rnap_domains = [int(d) for d in active_rnap["domain_index"]]

    # dnaA-binding (dnaa-03 / load-and-trigger). Fractions of occupied boxes
    # at oriC for the high-affinity tier (3 boxes: R1/R2/R4) and low-affinity
    # tier (8 boxes: R5M/τ2/I1-3/C1-3). Multiply by tier sizes to get counts.
    dnaa_binding = cell.get("listeners", {}).get("dnaA_binding", {}) or {}
    oric_binding = dnaa_binding.get("oric", {}) or {}
    high_frac = float(oric_binding.get("high_affinity_occupied", 0.0) or 0.0)
    low_frac = float(oric_binding.get("low_affinity_occupied", 0.0) or 0.0)

    return {
        "time": float(t),
        "n_chromosomes": n_chrom,
        "n_domains": n_domains,
        "fork_coords": fork_coords,
        "fork_domains": fork_domains,
        "rnap_coords": rnap_coords,
        "rnap_domains": rnap_domains,
        "domain_children": domain_children,
        "n_rnap": n_rnap,
        "dna_mass": float(mass.get("dna_mass", 0)),
        "dry_mass": float(mass.get("dry_mass", 0)),
        "protein_mass": float(mass.get("protein_mass", 0)),
        # dnaa-03 box-binding fields (may be 0 in other studies' composites).
        "oric_high_frac": high_frac,
        "oric_low_frac": low_frac,
        "oric_high_count": high_frac * 3,
        "oric_low_count": low_frac * 8,
    }


def render_html(b64_png: str, title: str) -> str:
    """Wrap a base64 PNG in an iframe-embed-friendly self-contained HTML page.

    Uses the same height-clamp pattern as comparative_viz (``html,body
    {height:Npx;overflow:hidden}``) so the dashboard's iframe insertion
    code can pin the iframe to a known height — otherwise scrollHeight
    measurements on a ``width:100%;height:auto`` image fail before
    layout completes, leaving the iframe at its 200px min-height.

    The image gets ``max-height: 920px`` so it scales DOWN to fit but
    keeps its aspect ratio; centered horizontally so narrower iframes
    don't stretch it wide-and-short.
    """
    body_h = 1000  # 920 image + chrome (title + subtitle + padding)
    img_max_h = 920
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"<style>html,body{{height:{body_h}px;overflow:hidden}}"
        "body{margin:0;padding:0;background:#fff;font-family:-apple-system,system-ui,sans-serif}"
        ".wrap{padding:16px 22px;height:100%;box-sizing:border-box;display:flex;flex-direction:column}"
        "h1{font-size:1.05em;margin:0 0 4px 0;color:#0f172a}"
        ".sub{color:#6b7280;font-size:0.85em;margin-bottom:12px}"
        f"img{{display:block;margin:0 auto;max-width:100%;max-height:{img_max_h}px;width:auto;height:auto;"
        "border:1px solid #e5e7eb;border-radius:6px;object-fit:contain}"
        "</style></head><body><div class='wrap'>"
        f"<h1>{title}</h1>"
        "<div class='sub'>Circular chromosome maps at 5 timepoints across the cell cycle, plus "
        "replication-fork progress and mass-growth timeseries. Gold triangles = replisomes; "
        "blue dots = RNAPs; green dot = oriC; red square = Ter. Stacked circles = replicating chromosomes.</div>"
        f'<img alt="chromosome state" src="data:image/png;base64,{b64_png}"/>'
        "</div></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000,
                    help="how many sim ticks to run (default 5000 = one emergent "
                         "cycle on the Stage-1 glycerol cache; bumped from 600 "
                         "per Haochen 2026-05-25 ask to cover a full cell cycle)")
    ap.add_argument("--chunk", type=int, default=60, help="snapshot interval (default 60s)")
    ap.add_argument("--cache-dir", default="out/cache-stage1-glycerol",
                    help="composite cache dir (default: Stage-1 glycerol)")
    ap.add_argument(
        "--out",
        default="workspace/studies/dnaa-00-parameter-foundation/viz/chromosome_state.html",
        help="output HTML path (relative to workspace root)",
    )
    ap.add_argument(
        "--spec",
        default="v2ecoli.composites.baseline_recipes.dnaa_00_baseline_with_dnaa_readout",
        help="composite spec id",
    )
    args = ap.parse_args()

    core = allocate_core()
    entry = _REGISTRY[args.spec]
    doc = build_generator(entry, overrides={"seed": 0, "cache_dir": args.cache_dir})
    comp = Composite({"state": doc.get("state", doc)}, core=core)

    snapshots: list[dict] = []
    snap0 = extract_snapshot(comp.state, 0.0)
    if snap0:
        snapshots.append(snap0)

    done = 0
    while done < args.steps:
        n = min(args.chunk, args.steps - done)
        try:
            comp.run(n)
        except Exception as e:
            print(f"[chromosome_viz] composite stopped at {done}s: {str(e)[:80]}")
            break
        done += n
        snap = extract_snapshot(comp.state, float(done))
        if snap:
            snapshots.append(snap)
        # Stop on division (agents/0 removed).
        if "0" not in (comp.state.get("agents") or {}):
            print(f"[chromosome_viz] division at {done}s; stopping")
            break

    print(f"[chromosome_viz] captured {len(snapshots)} snapshots over {done} sim s")
    if snapshots:
        forks_at_end = snapshots[-1].get("fork_coords", [])
        rnaps_at_end = snapshots[-1].get("rnap_coords", [])
        chroms_at_end = snapshots[-1].get("n_chromosomes", 0)
        print(f"[chromosome_viz] final state: {chroms_at_end} chromosome(s), "
              f"{len(forks_at_end)} replisomes, {len(rnaps_at_end)} RNAPs")

    title = "dnaa-00 — chromosome state across the cell cycle"
    b64 = _plot_chromosome_state(snapshots, title=title)
    html = render_html(b64, title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"[chromosome_viz] wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
