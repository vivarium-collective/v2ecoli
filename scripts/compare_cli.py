#!/usr/bin/env python3
"""v2e-compare — one front door for the comparison-harness investigation.

  v2e-compare run <manifest> [--ray] [--out DIR] [--render-only]
  v2e-compare study <name|path> [--ray] [--manifest M] [--out DIR] [--render-only]

`run` drives the whole investigation: scaffold studies if missing -> run both
engines per condition -> emit per-condition verdicts (via the renderer) ->
validate studies-vs-manifest -> report dashboard-ready. `study` runs ONE
condition, resolving its manifest from the study's own `comparison_manifest`
back-link (spec Decision 3). Sims run serial+local by default; --ray (or
V2E_MODE=ray) fans conditions out in parallel for the mini.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import run_comparison  # noqa: E402
from scripts import scaffold_comparison_studies as scaffold_mod  # noqa: E402
from scripts import validate_comparison_studies as validate_mod  # noqa: E402

INVEST = scaffold_mod.INVEST
CARD_ROOT = scaffold_mod.CARD_ROOT
STUDIES = REPO / "workspace/investigations" / INVEST / "studies"


def _abs_manifest(manifest: str) -> str:
    return manifest if os.path.isabs(manifest) else str(REPO / manifest)


def _run_investigation(manifest, out, ray, render_only) -> int:
    manifest = _abs_manifest(manifest)
    if not render_only:                                   # 1. scaffold if missing
        scaffold_mod.scaffold(manifest, str(REPO), force=False)
    mode = "ray" if ray else "serial"
    argv = [manifest, "--out", out, "--mode", mode]       # 2-3. run + verdict (render writes verdict)
    if render_only:
        argv.append("--render-only")
    rc = run_comparison.main(argv)
    if rc:
        return rc
    problems = validate_mod.validate(manifest, str(REPO))  # 4. validate
    for p in problems:
        print(f"DRIFT: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"investigation ready: workspace/investigations/{INVEST}")  # 5.
    return 0


def _resolve_study(name_or_path):
    p = Path(name_or_path)
    if p.name == "study.yaml":
        spath = p
    elif p.is_dir():
        spath = p / "study.yaml"
    else:
        spath = STUDIES / name_or_path / "study.yaml"
    if not spath.exists():
        sys.exit(f"study not found: {spath}")
    return yaml.safe_load(spath.read_text(encoding="utf-8")) or {}, spath


def _run_study(name_or_path, manifest_override, out, ray, render_only) -> int:
    study, spath = _resolve_study(name_or_path)
    manifest = manifest_override or study.get("comparison_manifest")
    cond = study.get("condition") or study.get("name")
    if not manifest:
        sys.exit(f"{spath}: no comparison_manifest (pass --manifest)")
    if not cond:
        sys.exit(f"{spath}: no condition/name")
    mode = "ray" if ray else "serial"
    argv = [_abs_manifest(manifest), "--out", out, "--mode", mode,
            "--condition", cond]
    if render_only:
        argv.append("--render-only")
    rc = run_comparison.main(argv)
    if rc == 0:
        print(f"study '{cond}' done; verdict: {CARD_ROOT}/{cond}/"
              f"report_card_verdict.json")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="v2e-compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the whole comparison investigation")
    pr.add_argument("manifest")
    pr.add_argument("--out", default="out/report")
    pr.add_argument("--ray", action="store_true")
    pr.add_argument("--render-only", action="store_true")

    ps = sub.add_parser("study", help="run a single study/condition")
    ps.add_argument("name", help="study name or path")
    ps.add_argument("--manifest", default=None, help="override the study's back-link")
    ps.add_argument("--out", default="out/report")
    ps.add_argument("--ray", action="store_true")
    ps.add_argument("--render-only", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return _run_investigation(args.manifest, args.out, args.ray, args.render_only)
    return _run_study(args.name, args.manifest, args.out, args.ray, args.render_only)


if __name__ == "__main__":
    raise SystemExit(main())
