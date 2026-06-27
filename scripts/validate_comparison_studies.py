#!/usr/bin/env python3
"""Validate that comparison studies match their manifest (drift guard).

For the v2ecoli-vecoli-comparison investigation, assert per study: its
`condition` exists in the manifest; its report_card_axis behavior_test groups
exactly equal the manifest's graded cards for that condition; each test's `card`
path is the canonical <CARD_ROOT>/<condition>. Exits non-zero on any drift so
this can run in CI / pre-merge. This is how "studies reference the manifest"
stays honest without auto-generating them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from scripts.scaffold_comparison_studies import (
    REPO, INVEST, CARD_ROOT, GRADED, condition_name)


def validate(manifest_path: str, ws_root: str) -> list[str]:
    spec = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    default_cards = (spec.get("defaults", {}) or {}).get("cards") or ["standard"]
    manifest_conds = {}
    for entry in spec.get("configs", []):
        cond = condition_name(entry)
        cards = entry.get("cards") or default_cards
        manifest_conds[cond] = sorted(c for c in cards if c in GRADED)

    problems = []
    studies_dir = Path(ws_root) / "workspace/investigations" / INVEST / "studies"
    if not studies_dir.is_dir():
        return [f"no studies dir: {studies_dir}"]
    for sdir in sorted(studies_dir.glob("*")):
        spath = sdir / "study.yaml"
        if not spath.exists():
            continue
        study = yaml.safe_load(spath.read_text(encoding="utf-8")) or {}
        cond = study.get("condition") or sdir.name
        if cond not in manifest_conds:
            problems.append(f"{sdir.name}: condition {cond!r} not in manifest")
            continue
        axis_tests = [t for t in study.get("behavior_tests", [])
                      if (t.get("measure") or {}).get("kind") == "report_card_axis"]
        groups = sorted(t["measure"].get("group", "") for t in axis_tests)
        if groups != manifest_conds[cond]:
            problems.append(
                f"{cond}: behavior_test groups {groups} != manifest graded "
                f"cards {manifest_conds[cond]}")
        expected_card = f"{CARD_ROOT}/{cond}"
        for t in axis_tests:
            if t["measure"].get("card") != expected_card:
                problems.append(
                    f"{cond}: test {t.get('name')!r} card "
                    f"{t['measure'].get('card')!r} != {expected_card!r}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="comparison manifest JSON")
    ap.add_argument("--ws-root", default=str(REPO), help="repo/workspace root")
    args = ap.parse_args(argv)
    problems = validate(args.manifest, args.ws_root)
    for p in problems:
        print(f"DRIFT: {p}", file=sys.stderr)
    if problems:
        return 1
    print("comparison studies OK (match manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
