"""Wrap the colony figures + GIF as self-contained HTML embeds for the read-only
dashboard / investigation report.

The read-only dashboard does NOT serve a study's raw `charts/*.svg` or
`colony.gif`; figures appear publicly only as self-contained HTML under
`reports/figures/<study>/` (auto-discovered + published to
gh-pages:dashboard/reports/figures/). This wraps the existing SVGs + the GIF
into such embeds so they render in the investigation results online.

    python .../colonies-02-parallel-multigen-perf/sims/make_report_embeds.py
"""
from __future__ import annotations

import base64
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
# STUDY_DIR = <root>/workspace/investigations/colonies/studies/<study>
# parents: [0]studies [1]colonies [2]investigations [3]workspace [4]repo-root
WS_ROOT = STUDY_DIR.parents[4]   # repo root (reports/ lives here, not under workspace/)
STUDY = STUDY_DIR.name
REPORTS = WS_ROOT / "reports" / "figures"

PAGE = ("<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        "<style>body{{margin:0;background:#fff;font-family:system-ui,sans-serif}}"
        ".cap{{padding:8px 12px;color:#444;font-size:13px}}"
        "svg,img{{max-width:100%;height:auto;display:block;margin:auto}}</style></head>"
        "<body><div class='wrap'>{body}</div><div class='cap'>{cap}</div></body></html>")


def write(study: str, name: str, body: str, cap: str, title: str):
    out = REPORTS / study
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.html").write_text(PAGE.format(title=title, body=body, cap=cap))
    return out / f"{name}.html"


def svg_embed(study, name, svg_path: Path, cap, title):
    return write(study, name, svg_path.read_text(), cap, title)


def gif_embed(study, name, gif_path: Path, cap, title):
    b64 = base64.b64encode(gif_path.read_bytes()).decode()
    return write(study, name, f"<img src='data:image/gif;base64,{b64}'>", cap, title)


charts = STUDY_DIR / "charts"
made = []
made.append(gif_embed(STUDY, "1-colony-growth-animation", STUDY_DIR / "colony.gif",
    "Colony growing by natural division (1->2->4->6); fixed physics. Capsules are cells; "
    "daughters are hue-shifted from their mother.", "Colony growth"))
made.append(svg_embed(STUDY, "2-rss-vs-tick-by-ncells", charts / "01_rss_vs_tick_by_ncells.svg",
    "RSS vs tick (staged N=1->2->4). Step-ups = per-cell footprint; within-plateau slope = the leak.",
    "RSS vs tick by cell count"))
made.append(svg_embed(STUDY, "3-footprint-and-leak-by-ncells", charts / "02_footprint_and_leak_by_ncells.svg",
    "Per-cell footprint (each cell = a full baseline WCM + own sim_data) and per-tick leak rate by cell count.",
    "Footprint + leak by N"))
made.append(svg_embed(STUDY, "4-memory-decomposition", charts / "03_memory_decomposition.svg",
    "RSS split: numpy-object bytes flat, native growing -> the leak is native (macOS leaks: ~13MB LLVM/Numba; #253).",
    "Memory decomposition"))
# Also surface the decomposition under the leak study (its F-05 evidence).
made.append(svg_embed("colonies-03-wcm-rss-leak", "1-memory-decomposition",
    charts / "03_memory_decomposition.svg",
    "F-05: the dominant leak is native (numpy flat, growth native); macOS leaks localized ~13MB to the LLVM/Numba JIT. #253.",
    "Native leak decomposition"))

for p in made:
    print("wrote", p.relative_to(WS_ROOT))
