"""Regenerate the HTML golden fixtures by running each report on current main.

USAGE: cd v2ecoli && uv run python scripts/regenerate_viz_goldens.py [--only NAME ...]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "visualizations"
OUT_DIR = REPO_ROOT / "out" / "reports"

# (script_path, extra_args, fixed_output_path_or_None)
# fixed_output_path: relative to REPO_ROOT, where the script writes its HTML.
#   If None, the script accepts --output <path> so we pass OUT_DIR/<name>.html.
REPORTS = {
    "network": (
        "reports/network_report.py",
        ["--no-open"],
        None,  # accepts --output
    ),
    "compare": (
        "reports/compare_report.py",
        [],
        None,  # accepts --output
    ),
    "workflow": (
        "reports/workflow_report.py",
        [],
        "out/workflow/workflow_report.html",  # fixed output location
    ),
    "multigeneration": (
        "reports/multigeneration_report.py",
        [],
        "out/multigeneration/multigeneration_report.html",  # fixed output location
    ),
    "colony": (
        "reports/colony_report.py",
        [],
        "out/colony/colony_report.html",  # fixed output location
    ),
    "benchmark": (
        "reports/benchmark_report.py",
        [],
        None,  # no --output; globs for output
    ),
    "v1_v2": (
        "reports/v1_v2_report.py",
        [],
        "out/comparison/comparison.html",  # fixed output location
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Regenerate only these names (default: all)")
    args = parser.parse_args()
    targets = args.only or list(REPORTS)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for name in targets:
        if name not in REPORTS:
            print(f"skip: unknown report '{name}'", file=sys.stderr)
            continue
        script, extra_args, fixed_output = REPORTS[name]
        dest_html = OUT_DIR / f"{name}.html"
        cmd = ["uv", "run", "python", script]

        if fixed_output is None:
            # Script accepts --output <path>; direct output into OUT_DIR.
            try:
                subprocess.run(
                    cmd + ["--output", str(dest_html), *extra_args],
                    check=True, cwd=str(REPO_ROOT),
                )
            except subprocess.CalledProcessError:
                # Retry with --out (some scripts use --out instead of --output)
                try:
                    subprocess.run(
                        cmd + ["--out", str(dest_html), *extra_args],
                        check=True, cwd=str(REPO_ROOT),
                    )
                except subprocess.CalledProcessError as e:
                    # Last resort: run without --output and glob.
                    try:
                        subprocess.run(cmd + extra_args,
                                       check=True, cwd=str(REPO_ROOT))
                    except subprocess.CalledProcessError as e2:
                        failures.append((name, str(e2)))
                        continue
                    # Glob OUT_DIR or repo root for any matching html
                    candidates = (
                        sorted(OUT_DIR.glob(f"*{name}*.html"))
                        or sorted(REPO_ROOT.glob(f"out/**/*{name}*.html"))
                    )
                    if candidates:
                        dest_html = candidates[-1]
                    else:
                        failures.append((name, f"no output found for {name}"))
                        continue
        else:
            # Script writes to a fixed path; run it, then copy.
            try:
                subprocess.run(cmd + extra_args, check=True, cwd=str(REPO_ROOT))
            except subprocess.CalledProcessError as e:
                failures.append((name, str(e)))
                continue
            fixed_path = REPO_ROOT / fixed_output
            if not fixed_path.exists():
                # Try to find via glob
                stem = fixed_path.stem
                candidates = sorted(REPO_ROOT.glob(f"out/**/*{stem}*.html"))
                if candidates:
                    fixed_path = candidates[-1]
                else:
                    failures.append((name, f"expected output not found: {fixed_path}"))
                    continue
            shutil.copy(fixed_path, dest_html)

        if not dest_html.exists():
            failures.append((name, f"output html not found after run: {dest_html}"))
            continue

        golden = FIXTURES_DIR / f"{name}.golden.html"
        shutil.copy(dest_html, golden)
        print(f"[{name}] wrote {golden}")

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for n, err in failures:
            print(f"  {n}: {err}", file=sys.stderr)
        return 1
    print(f"\n{len(targets)} golden(s) regenerated under {FIXTURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
