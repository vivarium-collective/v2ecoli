"""Render the basal-condition phenotype report card from a workflow run.

Reads a ``basal_phenotype_card`` analysis.json (produced by
``v2ecoli-workflow --config configs/basal_phenotype_card.json``) and the pinned
reference, grades the measured ensemble, and writes the card as Markdown + HTML.

    python reports/basal_phenotype_card_report.py \\
        --analysis out/basal_phenotype_card/analysis.json

Default output: out/basal_phenotype_card/report_card.{md,html}

See docs/meta_report_cards.md for the card design. The grade logic lives in
v2ecoli.library.report_card (shared with the grade test).
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess

from v2ecoli.library.report_card import (
    card_from_analysis, load_json, render_html, render_markdown,
)


_DEFAULT_REFERENCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "basal_phenotype_reference.json")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", required=True,
                   help="Path to the run's analysis.json")
    p.add_argument("--reference", default=_DEFAULT_REFERENCE,
                   help="Path to the pinned reference fixture")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: alongside the analysis.json)")
    p.add_argument("--model-ref", default=None,
                   help="Model identifier for the card header (default: git sha)")
    args = p.parse_args()

    analysis = load_json(args.analysis)
    reference = load_json(args.reference)
    card = card_from_analysis(analysis)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.analysis))
    os.makedirs(out_dir, exist_ok=True)
    model_ref = args.model_ref or _git_sha()
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    md = render_markdown(card, reference, model_ref=model_ref, generated=generated)
    html = render_html(card, reference, model_ref=model_ref, generated=generated)

    md_path = os.path.join(out_dir, "report_card.md")
    html_path = os.path.join(out_dir, "report_card.html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(md)
    print(f"\nWrote {md_path}\n      {html_path}")


if __name__ == "__main__":
    main()
