"""Workspace-local study evaluators registered into the pbg-superpowers seam.

register_evaluators(registry) is discovered + called by
viva_superpowers.study_evaluator.load_workspace_evaluators (mirrors build_core()).
The framework stays report-card-agnostic; all report-card logic lives here.
"""
import json
from pathlib import Path

# A group's outcome = the worst (most severe) axis verdict in that group.
_SEVERITY = {"mismatch": 3, "drift": 2, "within_tol": 1, "ungraded": 0}


def register_evaluators(registry: dict) -> None:
    registry["report_card_axis"] = evaluate_report_card_group


def _find_verdict(card_dir: Path, card_name: "str | None" = None) -> "Path | None":
    """Locate a card's verdict JSON, spanning both naming conventions.

    Gen-1 render CLIs write ``<card_dir>/report_card_verdict.json``; the Gen-2 flush
    writes ``<study>/viz/report_card/<card>.verdict.json``
    (``v2ecoli/workflow/report_cards/__init__.py::write_card``). Reading only the
    Gen-1 name made a flush-produced verdict unreachable from a `report_card_axis`
    study test — the join between a report card and the acceptance spine — so a
    Gen-2 study could not gate on its own card.

    Resolution order: the Gen-1 filename, then ``<card_name>.verdict.json`` when the
    test names one, then a unique ``*.verdict.json`` in the directory. The last step
    stays deliberately strict: with several cards flushed to one directory the
    correct one is ambiguous, so the test must name it rather than have us guess.
    """
    gen1 = card_dir / "report_card_verdict.json"
    if gen1.is_file():
        return gen1
    if card_name:
        named = card_dir / f"{card_name}.verdict.json"
        return named if named.is_file() else None
    if card_dir.is_dir():
        found = sorted(card_dir.glob("*.verdict.json"))
        if len(found) == 1:
            return found[0]
    return None


def evaluate_report_card_group(test: dict, reader, ws_root) -> dict:
    """Grade one study test against one group of a report card's verdict JSON.

    measure: {kind: report_card_axis, card: <dir relative to ws_root>, group: <name>,
              card_name: <optional Gen-2 card name>}
    Aggregation: any mismatch -> FAIL; any drift (no mismatch) -> PASS + caveat;
    only within_tol -> PASS; all ungraded / missing -> result 'ungraded' (skip).
    """
    measure = test.get("measure") or {}
    card_dir = measure.get("card", "")
    group = measure.get("group", "")
    vpath = _find_verdict(Path(ws_root) / card_dir, measure.get("card_name"))

    if vpath is None:
        return {"result": "ungraded", "evaluated_by": "report_card",
                "detail": f"verdict json not found under: {Path(ws_root) / card_dir}"}
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
