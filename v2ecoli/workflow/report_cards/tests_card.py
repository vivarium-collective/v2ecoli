# v2ecoli/workflow/report_cards/tests_card.py
from __future__ import annotations

import re
from typing import Any

from v2ecoli.library.report_card import grade_card, render_verdict_html, verdict_json
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext

_STATUS_TO_VERDICT = {
    "passed": "within_tol", "pass": "within_tol", "within_tol": "within_tol",
    "failed": "mismatch", "fail": "mismatch", "mismatch": "mismatch",
    "drift": "drift",
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_") or "test"


def _criterion_str(pass_if: dict) -> str:
    op = pass_if.get("op")
    if op == "in_range":
        return f"in [{pass_if.get('low')}, {pass_if.get('high')}]"
    if op in ("gt", "ge", "lt", "le", "eq"):
        sym = {"gt": ">", "ge": "≥", "lt": "<", "le": "≤", "eq": "="}[op]
        return f"{sym} {pass_if.get('value', pass_if.get('threshold', ''))}"
    return op or ""


def _axes_from_list(tests: list) -> tuple[dict, dict]:
    """Build (reference_axes, card_tests) from a list of test dicts."""
    reference_axes: dict[str, Any] = {}
    card_tests: dict[str, Any] = {}
    for t in tests:
        tname = t.get("name", "test")
        slug = _slug(tname)
        path = f"tests.{slug}"
        status = str(t.get("status", "")).lower()
        verdict = _STATUS_TO_VERDICT.get(status, "ungraded")
        group = (t.get("classification") or "tests").capitalize()
        crit_str = _criterion_str(t.get("pass_if") or {})
        measure = t.get("measure") or {}
        value = measure.get("value")
        detail = t.get("question") or measure.get("detail") or ""
        reference_axes[path] = {
            "label": tname, "group": group,
            "criterion": {"type": "status", "criterion_str": crit_str},
        }
        card_tests[slug] = {
            "verdict": verdict, "value": value,
            "meter": crit_str, "detail": {"text": detail},
        }
    return reference_axes, card_tests


def _axes_from_pytest_dict(tests: dict) -> tuple[dict, dict]:
    """Build (reference_axes, card_tests) from a pytest auto-discover dict."""
    reference_axes: dict[str, Any] = {}
    card_tests: dict[str, Any] = {}

    last_results = tests.get("last_results")
    if last_results:
        for i, r in enumerate(last_results):
            if not isinstance(r, dict):
                continue
            label = r.get("name") or r.get("nodeid") or f"test_{i}"
            slug = _slug(label)
            path = f"tests.{slug}"
            outcome_raw = str(r.get("outcome") or r.get("status") or "").lower()
            verdict = _STATUS_TO_VERDICT.get(outcome_raw, "ungraded")
            reference_axes[path] = {
                "label": label, "group": "Pytest",
                "criterion": {"type": "status", "criterion_str": outcome_raw},
            }
            card_tests[slug] = {
                "verdict": verdict, "value": None,
                "meter": outcome_raw, "detail": {"text": ""},
            }

    if not reference_axes:
        # No usable results yet — emit a single placeholder axis
        n_targets = len(tests.get("pytest_args") or [])
        meter = f"{n_targets} pytest target(s); no results recorded yet"
        slug = "pytest_auto_discover"
        path = f"tests.{slug}"
        reference_axes[path] = {
            "label": "pytest auto-discover", "group": "Pytest",
            "criterion": {"type": "status", "criterion_str": "no results recorded yet"},
        }
        card_tests[slug] = {
            "verdict": "ungraded", "value": None,
            "meter": meter, "detail": {"text": ""},
        }

    return reference_axes, card_tests


class TestsCard(ReportCardStep):
    name = "tests"

    def applies(self, study: StudyContext) -> bool:
        tests = study.spec.get("tests")
        return bool(tests)  # None / [] / {} → False; non-empty list or dict → True

    def build(self, study: StudyContext):
        tests = study.spec.get("tests")
        if not tests:
            return None

        if isinstance(tests, list):
            reference_axes, card_tests = _axes_from_list(tests)
        else:
            # pytest auto-discover dict schema
            reference_axes, card_tests = _axes_from_pytest_dict(tests)

        if not reference_axes:
            return None

        card: dict[str, Any] = {"tests": card_tests}
        reference = {
            "title": f"{study.spec.get('name', study.study_name)} — default tests",
            "stimulus": {"reference_model": "behavioral spec",
                         "measured_model": "v2ecoli"},
            "axes": reference_axes,
        }
        report = grade_card(card, reference)
        vjson = verdict_json(report, model_ref="v2ecoli",
                             reference_model="behavioral spec")
        vjson["title"] = reference["title"]
        html = render_verdict_html(vjson, title=reference["title"])
        return vjson, html
