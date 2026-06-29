import json
from scripts._compare.verdict import (
    worst, build_condition_verdict, write_condition_verdict)


def test_worst_orders_by_severity():
    assert worst(["within_tol", "mismatch", "drift"]) == "mismatch"
    assert worst(["within_tol", "drift"]) == "drift"
    assert worst([]) == "ungraded"
    assert worst(["bogus"]) == "ungraded"


def test_build_groups_per_card_and_overall_is_worst():
    cards = {
        "standard": {"verdict": "drift", "axes": [
            {"id": "standard.rna_mass", "label": "RNA mass", "verdict": "drift"}]},
        "config": {"verdict": "ungraded", "axes": []},
    }
    doc = build_condition_verdict("basal", cards)
    assert doc["schema"] == "report_card_verdict/v1"
    assert set(doc["groups"]) == {"standard", "config"}
    assert doc["groups"]["config"]["verdict"] == "ungraded"
    assert doc["overall"] == "drift"
    assert doc["model_ref"] == "v2ecoli @ basal"
    assert doc["reference_model"] == "vEcoli @ basal"


def test_group_verdict_falls_back_to_worst_axis_when_absent():
    cards = {"standard": {"axes": [
        {"id": "a", "verdict": "within_tol"}, {"id": "b", "verdict": "mismatch"}]}}
    doc = build_condition_verdict("with_aa", cards)
    assert doc["groups"]["standard"]["verdict"] == "mismatch"


def test_write_creates_per_condition_file(tmp_path):
    p = write_condition_verdict(tmp_path, "basal", {
        "standard": {"verdict": "within_tol",
                     "axes": [{"id": "x", "verdict": "within_tol"}]}})
    assert p == tmp_path / "basal" / "report_card_verdict.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["overall"] == "within_tol"


def test_verdict_feeds_report_card_axis_evaluator(tmp_path):
    # The core proof that gating needs no new code: the evaluator reads our file.
    from pbg_v2ecoli.evaluators import evaluate_report_card_group
    write_condition_verdict(tmp_path, "basal", {
        "standard": {"verdict": "mismatch",
                     "axes": [{"id": "standard.rna", "verdict": "mismatch"}]}})
    test = {"measure": {"kind": "report_card_axis",
                        "card": "basal", "group": "standard"}}
    res = evaluate_report_card_group(test, None, str(tmp_path))
    assert res["result"] == "FAIL"
