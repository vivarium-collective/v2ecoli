"""CLI wrapper for the aerobic acetate-overflow report card.

The card itself is a Gen-2 ``ReportCardStep``
(``v2ecoli/workflow/report_cards/acetate_overflow_card.py``) and normally renders
via the flush into ``workspace/studies/overflow-acetate-vs-growth/viz/report_card/``.
This script exists for the one thing that can't live in an import-safe library:
rendering a standalone copy into the (gitignored) ``docs/`` tree for inspection,
with a human-readable generation timestamp.

Prefer the Gen-2 path:
    python scripts/study_report_cards.py --study overflow-acetate-vs-growth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from v2ecoli.library.report_card import grade_card, render_html, verdict_json  # noqa: E402
from v2ecoli.library.overflow import (  # noqa: E402,F401 — re-exported for importers
    _BASAN, _MODEL_JSON, FIXTURES, basan_curve, build, build_overflow,
    cnode_to_card, mg1655_anchor, model_curve, vemuri_curve)

OUT = REPO / "docs/report_cards/metabolism_overflow/overflow_acetate_vs_growth"


def main(model_json: str | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    card, reference, model = build(Path(model_json) if model_json else _MODEL_JSON)
    generated = time.strftime("%Y-%m-%d %H:%M")
    model_ref = reference["stimulus"]["measured_model"]
    report = grade_card(card, reference)
    (OUT / "report_card.html").write_text(
        render_html(card, reference, model_ref=model_ref, generated=generated))
    with open(OUT / "overflow_reference.json", "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)
    with open(OUT / "report_card_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_json(report, model_ref=model_ref,
                               reference_model=reference["stimulus"]["reference_model"],
                               generated=generated), f, indent=2, ensure_ascii=False)
    print(f"overall: {report['overall']}")
    for path, a in report["axes"].items():
        print(f"  {path:30} {a['verdict']:11} {a.get('meter','')}")
    print(f"\nwrote {OUT}/")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-json", default=None, help="baked GUR-titration arm JSON")
    args = ap.parse_args()
    main(args.model_json)
