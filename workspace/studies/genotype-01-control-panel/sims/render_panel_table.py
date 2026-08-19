#!/usr/bin/env python
"""Render the per-gene outcomes table for genotype-01-control-panel as SVG.

Reads data/panel_summary.json (the committed run artifact) and writes a
self-contained SVG table: design (gene, class, wiring) x structural card x
ParCa outcome (completed / failed at step N + error summary) x expression
readout. SVG because the workbench study page's Visualizations tab discovers
charts/*.svg (+ raster) only — HTML files are invisible to it
(vivarium_workbench.lib.study_charts.discover_static_study_charts).

Run from the study directory (the visualizations[].render contract):
    python sims/render_panel_table.py --out charts/panel_outcomes.svg
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

STUDY_DIR = Path(__file__).resolve().parents[1]

# text inks + reserved status colors (icon + word carry state; color reinforces)
INK, MUTED, GOOD, BAD, LINE = "#1a1a2e", "#6b7280", "#1a7f37", "#b42318", "#e5e7eb"

CLASS_LABEL = {"A": "A · unwired, dispensable",
               "B": "B · wired, non-essential",
               "C": "C · essential machinery"}


def outcome_lines(rec: dict) -> tuple[bool, str, str]:
    p = rec.get("parca", {})
    if p.get("exit") == 0 and p.get("state_written"):
        fit = rec.get("mechanistic_fit_status") or {}
        ok = all(v == "ok" for v in fit.values()) if fit else False
        return True, "✓ completed — 9/9 steps", (
            "fits ok ×3" if ok else "fit status mixed — see summary")
    tail = "\n".join(p.get("failure_tail") or [])
    step = re.search(r"step_(\d+)", tail)
    err = re.search(r"KeyError: np\.str_\('([^']+)'\)", tail)
    line1 = f"✗ failed at step {int(step.group(1))}" if step else "✗ failed"
    line2 = (f"KeyError('{err.group(1)}')" if err
             else "see failure_tail")
    return False, line1, line2


def delta_lines(rec: dict) -> tuple[str, str]:
    d = rec.get("expression_delta_vs_wt") or {}
    if "median_rel" not in d:
        return "—", "no built state"
    return (f"median {d['median_rel']:.2%}",
            f"{d['frac_gt_5pct']:.2%} of cistrons >5%")


def _wrap(s: str, width: int = 36) -> tuple[str, str]:
    """Word-aware two-line split."""
    if len(s) <= width:
        return s, ""
    cut = s.rfind(" ", 0, width + 1)
    cut = cut if cut > 0 else width
    return s[:cut], s[cut + 1:cut + 1 + width]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="charts/panel_outcomes.svg")
    args = ap.parse_args()
    data = json.loads((STUDY_DIR / "data" / "panel_summary.json").read_text())
    panel = data["panel"]

    W = 1080
    X = [16, 120, 320, 560, 800, 950]   # column x anchors
    HEADERS = ["Gene", "Class", "In-model wiring (verified)",
               "ParCa outcome", "Structural card", "Expr Δ vs WT"]
    top, row_h = 118, 44
    H = top + row_h * len(panel) + 58
    n_ok = sum(1 for r in panel if r.get("parca", {}).get("exit") == 0)

    def txt(x, y, s, fill=INK, size=12.5, weight="normal", family="system-ui, sans-serif"):
        return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
                f'font-weight="{weight}" font-family="{family}">{escape(s)}</text>')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Per-gene knockout outcomes table">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        txt(16, 28, "Control panel — per-gene knockout outcomes",
            size=17, weight="600"),
        txt(16, 50, f"{len(panel)} ParCa-level knockouts, fast regime · "
                    f"{n_ok}/{len(panel)} builds completed · every failure is "
                    f"the identical step-3 set_ppgpp_expression crash (F-03; "
                    f"393 ppGpp-regulated genes share this exposure)",
            fill=MUTED, size=12),
        f'<line x1="16" y1="{top - 26}" x2="{W - 16}" y2="{top - 26}" '
        f'stroke="#d1d5db" stroke-width="1.5"/>',
    ]
    for x, h in zip(X, HEADERS):
        parts.append(txt(x, top - 34, h, fill=MUTED, size=11.5, weight="600"))

    for i, rec in enumerate(panel):
        y = top + i * row_h
        ok, l1, l2 = outcome_lines(rec)
        d1, d2 = delta_lines(rec)
        card_ok = rec.get("card", {}).get("overall_structural_ok")
        parts += [
            txt(X[0], y, rec["gene"], weight="600"),
            txt(X[0], y + 15, rec["gene_id"], fill=MUTED, size=10.5,
                family="ui-monospace, monospace"),
            txt(X[1], y, CLASS_LABEL.get(rec["class"], rec["class"]), size=12),
            txt(X[2], y, _wrap(rec.get("wiring", ""))[0], size=11.5),
            txt(X[2], y + 15, _wrap(rec.get("wiring", ""))[1], fill=MUTED, size=11.5),
            txt(X[3], y, l1, fill=(GOOD if ok else BAD), size=12, weight="600"),
            txt(X[3], y + 15, l2, fill=MUTED, size=10.5,
                family="ui-monospace, monospace"),
            txt(X[4], y, "✓ within_tol" if card_ok else "✗ failure",
                fill=(GOOD if card_ok else BAD), size=12),
            txt(X[4], y + 15, "6 fit-free axes", fill=MUTED, size=10.5),
            txt(X[5], y, d1, size=12),
            txt(X[5], y + 15, d2, fill=MUTED, size=10.5),
            f'<line x1="16" y1="{y + 26}" x2="{W - 16}" y2="{y + 26}" '
            f'stroke="{LINE}" stroke-width="1"/>',
        ]

    parts += [
        txt(16, H - 22, "Δ excludes the deleted gene; the >5% tail is the "
            "target's operon neighbors (TU-share reattribution) plus "
            "numerically-zero cistrons — F-04. Source: data/panel_summary.json",
            fill=MUTED, size=11),
        "</svg>",
    ]

    out = STUDY_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
