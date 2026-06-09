"""Render the oriC tier-split (load-and-trigger) figure for dnaa-03.

Adapted from PR #28 `_oric_tier_split_plot`. Shows the count of occupied
high-affinity boxes (3 — R1/R2/R4) vs low-affinity (8 — R5M/τ2/I1-3/C1-3)
over time, with tier-ceiling lines.

Usage:
  .venv/bin/python scripts/render_oric_tier_split.py \\
      --study dnaa-03-box-binding \\
      --spec v2ecoli.composites.baseline_recipes.dnaa_03_with_box_binding \\
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
from v2ecoli.visualizations.workflow import _plot_oric_tier_split
from scripts.render_dnaa00_chromosome_viz import extract_snapshot  # noqa: E402


def render_html(b64_png: str, title: str) -> str:
    body_h = 540
    img_max_h = 460
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
        "<div class='sub'>The 3 high-affinity boxes (R1/R2/R4, Kd ≈ 1 nM) saturate quickly. "
        "The 8 low-affinity boxes (R5M/τ2/I1-3/C1-3, Kd > 100 nM, ATP-only, cooperative) "
        "fill more slowly as DnaA-ATP pool grows. The low-tier fill rate (not the high-tier "
        "occupancy) is what gates initiation in the load-and-trigger model.</div>"
        f'<img alt="oriC tier split" src="data:image/png;base64,{b64_png}"/>'
        "</div></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--title", default=None)
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
            print(f"[oric_tier_split] composite stopped at {done}s: {str(e)[:80]}")
            break
        done += n
        snap = extract_snapshot(comp.state, float(done))
        if snap:
            snapshots.append(snap)
        if "0" not in (comp.state.get("agents") or {}):
            print(f"[oric_tier_split] division at {done}s; stopping")
            break

    print(f"[oric_tier_split] captured {len(snapshots)} snapshots over {done} sim s")
    if snapshots:
        s = snapshots[-1]
        print(f"  final oric_high={s['oric_high_count']:.2f}/3 oric_low={s['oric_low_count']:.2f}/8")

    title = args.title or f"{args.study} — load-and-trigger at oriC"
    b64 = _plot_oric_tier_split(snapshots, title=title)
    html = render_html(b64, title)

    out = Path("workspace/studies") / args.study / "viz" / "oric_tier_split.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[oric_tier_split] wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
