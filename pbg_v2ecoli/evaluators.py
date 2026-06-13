"""Workspace-local study evaluators registered into the pbg-superpowers seam.

register_evaluators(registry) is discovered + called by
pbg_superpowers.study_evaluator.load_workspace_evaluators (mirrors build_core()).
The framework stays report-card-agnostic; all report-card logic lives here.
"""
import json
from pathlib import Path

# A group's outcome = the worst (most severe) axis verdict in that group.
_SEVERITY = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def register_evaluators(registry: dict) -> None:
    registry["report_card_axis"] = evaluate_report_card_group


def evaluate_report_card_group(test: dict, reader, ws_root) -> dict:
    """Grade one study test against one group of a report card's verdict JSON.

    measure: {kind: report_card_axis, card: <dir relative to ws_root>, group: <name>}
    Aggregation: any mismatch -> FAIL; any drift (no mismatch) -> PASS + caveat;
    only within_tol -> PASS; all ungraded / missing -> result 'ungraded' (skip).
    """
    measure = test.get("measure") or {}
    card_dir = measure.get("card", "")
    group = measure.get("group", "")
    vpath = Path(ws_root) / card_dir / "report_card_verdict.json"

    if not vpath.is_file():
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"verdict json not found: {vpath}"}
    try:
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"unreadable verdict json: {exc}"}

    g = (verdict.get("groups") or {}).get(group)
    if g is None:
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"group {group!r} absent in card {card_dir!r}"}

    axes = g.get("axes") or []
    verdicts = [a.get("verdict", "ungraded") for a in axes]
    worst = max(verdicts, key=lambda v: _SEVERITY.get(v, 0)) if verdicts else "ungraded"

    provenance = {"card": card_dir, "group": group,
                  "overall": verdict.get("overall"),
                  "axis_verdicts": [{"id": a.get("id"), "verdict": a.get("verdict")}
                                    for a in axes]}

    if worst == "mismatch":
        return {"result": "FAIL", "evaluated_by": "report_card",
                "detail": f"group {group}: mismatch axis present", "provenance": provenance}
    if worst == "drift":
        return {"result": "PASS", "caveat": "drift", "evaluated_by": "report_card",
                "detail": f"group {group}: within tolerance with drift", "provenance": provenance}
    if worst == "within_tol":
        return {"result": "PASS", "evaluated_by": "report_card",
                "detail": f"group {group}: all axes within tolerance", "provenance": provenance}
    return {"result": "ungraded", "evaluated_by": "report_card",
            "detail": f"group {group}: all axes ungraded", "provenance": provenance}
