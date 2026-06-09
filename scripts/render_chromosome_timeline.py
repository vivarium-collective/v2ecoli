"""Render the per-study chromosome-replication timeline visualization.

The figure (adapted from PR #28's `_chromosome_timeline_plot`) carries the
"replisomes drive RIDA flux" story: 5 chromosome diagrams across the cell
cycle (each disc with replication bubble, RNAPs, and replisome triangles)
+ a bottom step-plot of n_chromosomes / active-replisomes over time with
initiation and chromosome-doubling events marked.

Used by dnaa-02 (RIDA-only) and dnaa-06 (full reset network). Parameterised
so other RIDA-relevant studies can call it too.

Usage:
  .venv/bin/python scripts/render_chromosome_timeline.py \\
      --study dnaa-02-atp-hydrolysis \\
      --spec v2ecoli.composites.baseline_recipes.dnaa_02_with_intrinsic_hydrolysis \\
      --steps 3600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
from bigraph_schema import allocate_core
from process_bigraph import Composite
from pbg_superpowers.composite_generator import _REGISTRY, build_generator
import v2ecoli.composites  # noqa: F401
from v2ecoli.visualizations.workflow import _plot_chromosome_timeline

# Re-use the snapshot extractor from the dnaa-00 script (same logic — it
# reads cell['unique'][active_replisome|active_RNAP|chromosome_domain|...]).
from scripts.render_dnaa00_chromosome_viz import extract_snapshot  # noqa: E402


def render_html(b64_png: str, title: str) -> str:
    body_h = 1000
    img_max_h = 920
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"<style>html,body{{height:{body_h}px;overflow:hidden}}"
        "body{margin:0;padding:0;background:#fff;font-family:-apple-system,system-ui,sans-serif}"
        ".wrap{padding:16px 22px;height:100%;box-sizing:border-box;display:flex;flex-direction:column}"
        "h1{font-size:1.05em;margin:0 0 4px 0;color:#0f172a}"
        ".sub{color:#6b7280;font-size:0.85em;margin-bottom:12px}"
        f"img{{display:block;margin:0 auto;max-width:100%;max-height:{img_max_h}px;"
        "width:auto;height:auto;border:1px solid #e5e7eb;border-radius:6px;object-fit:contain}"
        "</style></head><body><div class='wrap'>"
        f"<h1>{title}</h1>"
        "<div class='sub'>Top row: chromosome diagrams at 5 timepoints. Gold triangles = "
        "active replisomes; green arcs = newly-synthesized daughter strand; blue dots = RNAPs. "
        "Bottom: replisome and chromosome counts over time. Red dashed = initiation event "
        "(new forks fire). Blue dotted = chromosome replication completed. "
        "RIDA flux is gated on the active-replisome count.</div>"
        f'<img alt="chromosome timeline" src="data:image/png;base64,{b64_png}"/>'
        "</div></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True,
                    help="study slug (e.g. dnaa-02-atp-hydrolysis)")
    ap.add_argument("--spec", required=True,
                    help="composite spec id (e.g. v2ecoli.composites.baseline_recipes.dnaa_02_with_intrinsic_hydrolysis)")
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--title", default=None,
                    help="figure title (default derived from --study)")
    args = ap.parse_args()

    core = allocate_core()
    entry = _REGISTRY[args.spec]
    doc = build_generator(entry, overrides={"seed": args.seed, "cache_dir": args.cache_dir})
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
            print(f"[chromosome_timeline] composite stopped at {done}s: {str(e)[:80]}")
            break
        done += n
        snap = extract_snapshot(comp.state, float(done))
        if snap:
            snapshots.append(snap)
        if "0" not in (comp.state.get("agents") or {}):
            print(f"[chromosome_timeline] division at {done}s; stopping")
            break

    print(f"[chromosome_timeline] captured {len(snapshots)} snapshots over {done} sim s")

    title = args.title or f"{args.study} — replication timeline (replisomes drive RIDA flux)"
    b64 = _plot_chromosome_timeline(snapshots, title=title)
    html = render_html(b64, title)

    out = Path("workspace/studies") / args.study / "viz" / "chromosome_timeline.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[chromosome_timeline] wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
