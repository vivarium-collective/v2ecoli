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


class TestsCard(ReportCardStep):
    name = "tests"

    def applies(self, study: StudyContext) -> bool:
        return bool(study.spec.get("tests"))

    def build(self, study: StudyContext):
        tests = study.spec.get("tests") or []
        if not tests:
            return None
        reference_axes: dict[str, Any] = {}
        card: dict[str, Any] = {"tests": {}}
        for t in tests:
            tname = t.get("name", "test")
            slug = _slug(tname)
            path = f"tests.{slug}"
            status = str(t.get("status", "")).lower()
            verdict = _STATUS_TO_VERDICT.get(status, "ungraded")
            group = (t.get("classification") or "tests").capitalize()
            crit_str = _criterion_str(t.get("pass_if") or {})
            measure = t.get("measure") or {}
            value = measure.get("value")  # present only if the study recorded one
            detail = t.get("question") or measure.get("detail") or ""
            reference_axes[path] = {
                "label": tname, "group": group,
                "criterion": {"type": "status", "criterion_str": crit_str},
            }
            card["tests"][slug] = {
                "verdict": verdict, "value": value,
                "meter": crit_str, "detail": {"text": detail},
            }
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
