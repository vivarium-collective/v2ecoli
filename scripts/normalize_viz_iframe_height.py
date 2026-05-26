"""Normalize every reports/figures/**/*.html so the dashboard iframe auto-sizer
expands to show the full figure without scrollbars.

The dashboard's _fitEmbed (walkthrough.js) detects a CSS height clamp via the
regex /html,body\\{height:(\\d+)px/ in the embedded HTML, and pins the iframe
to that height. If absent, it falls back to scrollHeight measurements which
were unreliable on matplotlib-PNG HTMLs from earlier sessions — leading to
visible scrollbars across the report.

This script injects (or upgrades) an `html,body{height:Hpx;overflow:hidden}`
rule at the start of every .html file under reports/figures/, picking H by
detecting whether the file is matplotlib-PNG, plotly, or a planning summary.

Idempotent: a file already containing `html,body{height:Hpx` is left untouched.

Run from worktree root:
    python scripts/normalize_viz_iframe_height.py
"""
from __future__ import annotations
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


# Pick a pinned-height per detected content type. Slightly generous so 6-panel
# figures, multi-row layouts and long captions all fit.
HEIGHT_MATPLOTLIB = 760     # data:image/png base64 - one panel + caption
HEIGHT_MATPLOTLIB_TALL = 900  # 6-panel / tall figures
HEIGHT_PLOTLY = 820          # plotly.js charts
HEIGHT_STUDY_PLAN = 1200     # study_plan_summary cards (long table layout)
HEIGHT_DEFAULT = 780


CLAMP_RE = re.compile(r"html\s*,\s*body\s*\{\s*height\s*:\s*(\d+)px", re.I)


def _detect_height(html: str, filename: str) -> int:
    """Heuristic content-type detector → pinned iframe height."""
    name = filename.lower()
    if "study_plan_summary" in name or "study-plan-summary" in name:
        return HEIGHT_STUDY_PLAN
    if "plotly" in html.lower() or "plot.ly" in html.lower():
        return HEIGHT_PLOTLY
    if "v2.html" in name or "6-panel" in html or "6 panel" in html or "_v2." in name:
        return HEIGHT_MATPLOTLIB_TALL
    if "data:image/png;base64" in html or "<img " in html.lower():
        return HEIGHT_MATPLOTLIB
    return HEIGHT_DEFAULT


def _inject_clamp(html: str, height: int) -> str:
    """Inject a body-height clamp as the first rule in the document.

    Strategy: insert a fresh <style> tag right after the <head> opening tag.
    Don't try to merge with existing <style> blocks — the regex auto-sizer
    just needs to FIND the clamp early in the source; CSS cascade handles
    the rest.
    """
    clamp = (
        f"<style>html,body{{height:{height}px;overflow:hidden;"
        "margin:0;padding:0}}</style>"
    ).replace("{{", "{").replace("}}", "}")  # ensure literal braces

    if "<head>" in html:
        return html.replace("<head>", "<head>" + clamp, 1)
    # Fallback: inject after <html>
    if "<html>" in html:
        return html.replace("<html>", "<html><head>" + clamp + "</head>", 1)
    # Last resort: prepend
    return clamp + html


def main():
    root = Path("reports/figures")
    if not root.is_dir():
        print(f"no figures dir at {root}"); return
    files = sorted(root.rglob("*.html"))
    print(f"scanning {len(files)} html files under {root}/")
    n_already, n_updated = 0, 0
    for path in files:
        try:
            html = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  SKIP {path}: read error {e}")
            continue
        m = CLAMP_RE.search(html)
        if m:
            n_already += 1
            continue
        height = _detect_height(html, path.name)
        new_html = _inject_clamp(html, height)
        path.write_text(new_html, encoding="utf-8")
        n_updated += 1
        print(f"  + {path}  → {height}px")
    print(f"\ndone: {n_updated} updated, {n_already} already had clamp")


if __name__ == "__main__":
    main()
