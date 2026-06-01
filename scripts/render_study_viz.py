"""Render all visualizations for a study from its parquet-runs output.

Usage:
    python scripts/render_study_viz.py <study_slug> [<study_slug> ...]
    python scripts/render_study_viz.py --all-dnaa

Reads the latest parquet run under ``workspace/studies/<slug>/parquet-runs/``, resolves
each viz registered in ``study.yaml`` against the parquet columns, and writes
HTML files to ``workspace/studies/<slug>/viz/``.
"""
from __future__ import annotations

import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study_slug", nargs="*",
                   help="study slug(s) to render; e.g. dnaa-02-atp-hydrolysis")
    p.add_argument("--all-dnaa", action="store_true",
                   help="render all dnaa-* studies that have visualizations")
    args = p.parse_args()

    from v2ecoli.library.parquet_viz import render_study_visualizations

    targets: list[str] = list(args.study_slug)
    if args.all_dnaa:
        import glob
        for syaml in sorted(glob.glob("workspace/studies/dnaa-*/study.yaml")):
            slug = syaml.split("/")[2]
            targets.append(slug)
    targets = list(dict.fromkeys(targets))  # de-dupe, preserve order
    if not targets:
        p.print_help()
        return 2

    rc = 0
    for slug in targets:
        print(f"\n== {slug} ==")
        try:
            written = render_study_visualizations(slug)
            if not written:
                print("  (no visualizations registered or no parquet run)")
        except FileNotFoundError as e:
            print(f"  skip: {e}")
        except Exception as e:
            print(f"  error: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
