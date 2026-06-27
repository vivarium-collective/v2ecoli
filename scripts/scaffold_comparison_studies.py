#!/usr/bin/env python3
"""One-time scaffold: manifest -> investigation.yaml + per-condition study.yaml.

Idempotent: never overwrites an existing study unless --force. After scaffolding
the files are hand-owned (spec Decision 4); this is NOT part of the run/render
loop. Studies REFERENCE the manifest (comparison_manifest + condition); the
validator (scripts/validate_comparison_studies.py) guards drift.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
INVEST = "v2ecoli-vecoli-comparison"
CARD_ROOT = f"docs/report_cards/{INVEST}"
GRADED = {"standard", "statistical"}   # cards that produce a gating test


def condition_name(entry: dict) -> str:
    """A manifest entry's condition key: its explicit `name`, else the config
    filename stem with a leading 'cond_' and trailing scale suffix (_NxN) stripped."""
    if entry.get("name"):
        return entry["name"]
    stem = os.path.splitext(os.path.basename(entry["config"]))[0]
    if stem.startswith("cond_"):
        stem = stem[len("cond_"):]
    return re.sub(r"_\d+x\d+$", "", stem)


def build_study(cond: str, cards: list, manifest_rel: str) -> dict:
    graded = [c for c in cards if c in GRADED]
    return {
        "schema_version": 4,
        "name": cond,
        "investigation": INVEST,
        "title": f"v2ecoli reproduces vEcoli on {cond}",
        "status": "evaluated",
        "comparison_manifest": manifest_rel,
        "condition": cond,
        "question": f"Does v2ecoli reproduce vEcoli on the {cond} condition?",
        "report_cards": [f"{CARD_ROOT}/{cond}/index.html"],
        "behavior_tests": [
            {"name": f"{c}-vs-vecoli",
             "classification": "primary",
             "question": f"Does v2ecoli reproduce vEcoli on {cond} ({c} card)?",
             "measure": {"kind": "report_card_axis",
                         "card": f"{CARD_ROOT}/{cond}", "group": c}}
            for c in graded],
        "runs": [
            {"name": f"{cond}-comparison", "kind": "analysis", "canonical": True,
             "description": f"v2e-compare study {cond}"}],
        "pipeline_gate": {"prerequisites": [], "enables": []},
    }


def build_investigation(conds: list) -> dict:
    return {
        "schema_version": 4,
        "name": INVEST,
        "title": "v2ecoli ↔ vEcoli comparison",
        "question": "Does v2ecoli reproduce vEcoli across nutrient conditions?",
        "studies": sorted(conds),
    }


def scaffold(manifest_path: str, ws_root: str, force: bool = False) -> list:
    spec = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    default_cards = (spec.get("defaults", {}) or {}).get("cards") or ["standard"]
    inv_dir = Path(ws_root) / "workspace/investigations" / INVEST
    studies_dir = inv_dir / "studies"
    try:
        manifest_rel = os.path.relpath(manifest_path, REPO)
    except ValueError:
        manifest_rel = manifest_path
    written = []
    conds = []
    for entry in spec.get("configs", []):
        cond = condition_name(entry)
        conds.append(cond)
        cards = entry.get("cards") or default_cards
        spath = studies_dir / cond / "study.yaml"
        if spath.exists() and not force:
            continue
        spath.parent.mkdir(parents=True, exist_ok=True)
        spath.write_text(
            yaml.safe_dump(build_study(cond, cards, manifest_rel), sort_keys=False,
                           allow_unicode=True), encoding="utf-8")
        written.append(spath)
    inv_dir.mkdir(parents=True, exist_ok=True)
    ipath = inv_dir / "investigation.yaml"
    if force or not ipath.exists():
        ipath.write_text(
            yaml.safe_dump(build_investigation(conds), sort_keys=False,
                           allow_unicode=True), encoding="utf-8")
        written.append(ipath)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="comparison manifest JSON")
    ap.add_argument("--ws-root", default=str(REPO), help="repo/workspace root")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing studies/investigation")
    args = ap.parse_args(argv)
    written = scaffold(args.manifest, args.ws_root, force=args.force)
    for p in written:
        print(f"wrote {p}")
    if not written:
        print("nothing to write (all studies exist; use --force to overwrite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
