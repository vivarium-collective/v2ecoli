# tests/test_scaffold_comparison_studies.py
import json
import yaml
from scripts.scaffold_comparison_studies import (
    condition_name, build_study, scaffold, INVEST, CARD_ROOT)


def test_condition_name_uses_explicit_name_then_strips_stem():
    assert condition_name({"config": "configs/cond_basal_1x4.json"}) == "basal"
    assert condition_name({"config": "x/cond_with_aa.json"}) == "with_aa"
    assert condition_name({"config": "c/cond_basal_4x4.json",
                           "name": "basal_4x4"}) == "basal_4x4"


def test_build_study_one_test_per_graded_card():
    s = build_study("basal", ["config", "parca", "standard"], "comparison_spec.json")
    assert s["condition"] == "basal"
    assert s["comparison_manifest"] == "comparison_spec.json"
    assert s["pipeline_gate"] == {"prerequisites": [], "enables": []}
    groups = [t["measure"]["group"] for t in s["behavior_tests"]]
    assert groups == ["standard"]            # config/parca are not graded
    t = s["behavior_tests"][0]
    assert t["measure"]["kind"] == "report_card_axis"
    assert t["measure"]["card"] == f"{CARD_ROOT}/basal"


def _manifest(tmp_path):
    m = tmp_path / "spec.json"
    m.write_text(json.dumps({
        "defaults": {"cards": ["config", "parca", "standard"]},
        "configs": [
            {"config": "configs/cond_basal_1x4.json"},
            {"config": "configs/cond_basal_4x4.json", "name": "basal_4x4",
             "cards": ["config", "parca", "statistical"]}]}), encoding="utf-8")
    return m


def test_scaffold_writes_investigation_and_studies(tmp_path):
    written = scaffold(str(_manifest(tmp_path)), str(tmp_path))
    base = tmp_path / "workspace/investigations" / INVEST
    assert (base / "investigation.yaml").exists()
    assert (base / "studies/basal/study.yaml").exists()
    assert (base / "studies/basal_4x4/study.yaml").exists()
    inv = yaml.safe_load((base / "investigation.yaml").read_text(encoding="utf-8"))
    assert sorted(inv["studies"]) == ["basal", "basal_4x4"]
    s44 = yaml.safe_load(
        (base / "studies/basal_4x4/study.yaml").read_text(encoding="utf-8"))
    assert [t["measure"]["group"] for t in s44["behavior_tests"]] == ["statistical"]


def test_scaffold_is_idempotent_without_force(tmp_path):
    m = _manifest(tmp_path)
    scaffold(str(m), str(tmp_path))
    spath = tmp_path / "workspace/investigations" / INVEST / "studies/basal/study.yaml"
    spath.write_text("name: basal\nEDITED: true\n", encoding="utf-8")
    written = scaffold(str(m), str(tmp_path))            # no force
    assert spath not in written
    assert "EDITED" in spath.read_text(encoding="utf-8")  # not clobbered
    written2 = scaffold(str(m), str(tmp_path), force=True)
    assert spath in written2
    assert "EDITED" not in spath.read_text(encoding="utf-8")
