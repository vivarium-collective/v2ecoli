"""Render the single-cell mechanics card (pytest-as-evidence).

The single-cell mechanical invariants (cell grows, replication completes, division
conserves, daughters viable) are single-cell + procedural, so — unlike the
population cards — their *measurement* stays in pytest. This CLI runs the
`behavior`-marked checks (or reads a junit XML from a prior run), maps each
test's pass / fail / skip onto a boolean axis declared in the reference, and
renders the card through the *shared* grader + renderer.

    python reports/single_cell_mechanics_report.py
    python reports/single_cell_mechanics_report.py --junitxml out/behavior.xml

A skipped check (missing checkpoint / trajectory) renders ``ungraded`` — honest,
not a pass. Default output: docs/report_cards/single_cell_mechanics/report_card.{md,html}.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from v2ecoli.library.report_card import (
    _set_path, load_json, render_html, render_markdown,
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_REFERENCE = os.path.join(_HERE, "tests", "fixtures",
                                  "single_cell_mechanics_reference.json")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _run_pytest_junit(tests: str, junit_path: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", tests, "-m", "behavior", "-q",
         "-p", "no:cacheprovider", f"--junitxml={junit_path}"],
        check=False)


def _parse_junit(path: str) -> dict[str, str]:
    """Map test-function name -> 'pass' | 'fail' | 'skip' from a junit XML."""
    out: dict[str, str] = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        func = (tc.get("name") or "").split("[")[0]  # strip parametrize ids
        if tc.find("failure") is not None or tc.find("error") is not None:
            out[func] = "fail"
        elif tc.find("skipped") is not None:
            out[func] = "skip"
        else:
            out[func] = "pass"
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", default="tests/test_model_behavior.py")
    p.add_argument("--junitxml", default=None,
                   help="Use an existing junit XML instead of running pytest")
    p.add_argument("--reference", default=_DEFAULT_REFERENCE)
    p.add_argument("--out-dir",
                   default=os.path.join("docs", "report_cards", "single_cell_mechanics"))
    args = p.parse_args()

    reference = load_json(args.reference)

    junit = args.junitxml
    if not junit:
        fd, junit = tempfile.mkstemp(suffix=".xml", prefix="behavior_")
        os.close(fd)
        print(f"[pytest] running `pytest -m behavior {args.tests}` …")
        _run_pytest_junit(args.tests, junit)
    results = _parse_junit(junit)

    # Map each axis's source test onto a boolean card node (None == skipped).
    card: dict = {}
    for path, spec in (reference.get("axes") or {}).items():
        status = results.get(spec.get("test"))
        _set_path(card, path,
                  True if status == "pass" else False if status == "fail" else None)

    os.makedirs(args.out_dir, exist_ok=True)
    model_ref = _git_sha()
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = render_markdown(card, reference, model_ref=model_ref, generated=generated)
    html = render_html(card, reference, model_ref=model_ref, generated=generated)

    md_path = os.path.join(args.out_dir, "report_card.md")
    html_path = os.path.join(args.out_dir, "report_card.html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(md)
    print(f"\nWrote {md_path}\n      {html_path}")


if __name__ == "__main__":
    main()
