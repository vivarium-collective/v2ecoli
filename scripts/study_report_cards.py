# scripts/study_report_cards.py
"""Generate per-study report cards by running the report-card Steps.

Each registered ``ReportCardStep`` (tests, vs_vecoli, ...) emits a ``view`` (HTML)
+ ``data`` (verdict map); this runner writes them to
``workspace/studies/<name>/viz/report_card/<name>.{html,verdict.json}``, which the
dashboard auto-discovers (no dashboard changes). The ``tests`` card is universal
and run-free; ``vs_vecoli`` stages a pre-generated v2ecoli<->vEcoli comparison
verdict (declared per study via ``report_card_refs.vs_vecoli``).

Usage:
  python scripts/study_report_cards.py --study all [--card all] [--prune]
  python scripts/study_report_cards.py --study showcase-2-baseline-figures --card vs_vecoli
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bigraph_schema import allocate_core  # noqa: E402

from v2ecoli.workflow.report_cards import (  # noqa: E402
    applicable, prune, write_card)


def _all_studies(ws_root: Path) -> list[str]:
    sdir = ws_root / "workspace" / "studies"
    if not sdir.is_dir():
        return []
    return sorted(p.name for p in sdir.iterdir() if (p / "study.yaml").is_file())


def generate_study(ws_root: Path, name: str, core, only: "str | None",
                   do_prune: bool) -> dict:
    from v2ecoli.workflow.report_cards import StudyContext
    ctx = StudyContext.load(ws_root, name)
    written: list[str] = []
    for step in applicable(ctx, core, only=only):
        try:
            res = step.build(ctx)
        except Exception as e:  # noqa: BLE001 — one card never aborts the run
            print(f"  ! {name}/{step.name}: skip ({e})")
            continue
        if not res:
            continue
        vjson, html = res
        write_card(ctx, step.name, vjson, html)
        written.append(step.name)
        print(f"  ✓ {name}/{step.name} [{vjson.get('overall', '?')}]")
    if do_prune:
        for s in prune(ctx, keep=set(written)):
            print(f"  - {name}/{s}: pruned")
    return {"study": name, "written": written}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="all", help="study name or 'all'")
    ap.add_argument("--card", default="all", help="card name or 'all'")
    ap.add_argument("--prune", action="store_true",
                    help="delete report_card/* not produced this run")
    args = ap.parse_args(argv)
    studies = _all_studies(REPO_ROOT) if args.study == "all" else [args.study]
    only = None if args.card == "all" else args.card
    core = allocate_core()
    total = 0
    for s in studies:
        total += len(generate_study(REPO_ROOT, s, core, only, args.prune)["written"])
    print(f"done — {total} cards across {len(studies)} studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
