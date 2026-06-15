import json
from pathlib import Path

from pbg_v2ecoli.evaluators import evaluate_report_card_group, register_evaluators


def _ws_with_verdict(tmp_path: Path, groups: dict) -> Path:
    card = tmp_path / "docs" / "card"
    card.mkdir(parents=True)
    (card / "report_card_verdict.json").write_text(json.dumps({
        "schema": "report_card_verdict/v1", "overall": "mismatch", "groups": groups,
    }), encoding="utf-8")
    return tmp_path


def _test(group):
    return {"name": f"{group}-test",
            "measure": {"kind": "report_card_axis", "card": "docs/card", "group": group}}


def test_register_exposes_report_card_axis():
    reg = {}
    register_evaluators(reg)
    assert "report_card_axis" in reg


def test_mismatch_group_fails(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"flux": {"verdict": "mismatch",
        "axes": [{"id": "o2", "verdict": "mismatch"}]}})
    out = evaluate_report_card_group(_test("flux"), None, ws)
    assert out["result"] == "FAIL"
    assert out["evaluated_by"] == "report_card"


def test_drift_group_passes_with_caveat(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"ribo": {"verdict": "drift",
        "axes": [{"id": "total", "verdict": "drift"}, {"id": "ef", "verdict": "within_tol"}]}})
    out = evaluate_report_card_group(_test("ribo"), None, ws)
    assert out["result"] == "PASS"
    assert out["caveat"] == "drift"


def test_within_tol_group_passes(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"comp": {"verdict": "within_tol",
        "axes": [{"id": "protein", "verdict": "within_tol"}]}})
    out = evaluate_report_card_group(_test("comp"), None, ws)
    assert out["result"] == "PASS"
    assert "caveat" not in out


def test_all_ungraded_group_skips(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"x": {"verdict": "ungraded",
        "axes": [{"id": "a", "verdict": "ungraded"}]}})
    out = evaluate_report_card_group(_test("x"), None, ws)
    assert out["result"] == "ungraded"


def test_missing_file_or_group_skips(tmp_path):
    ws = _ws_with_verdict(tmp_path, {"present": {"verdict": "within_tol", "axes": []}})
    out = evaluate_report_card_group(_test("absent"), None, ws)
    assert out["result"] == "ungraded"
    assert "absent" in out.get("detail", "")
