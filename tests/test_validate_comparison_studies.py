import json
import yaml
from scripts.scaffold_comparison_studies import scaffold, INVEST, CARD_ROOT
from scripts.validate_comparison_studies import validate


def _setup(tmp_path):
    m = tmp_path / "spec.json"
    m.write_text(json.dumps({
        "defaults": {"cards": ["config", "parca", "standard"]},
        "configs": [{"config": "configs/cond_basal_1x4.json"}]}), encoding="utf-8")
    scaffold(str(m), str(tmp_path))
    return m, (tmp_path / "workspace/investigations" / INVEST
               / "studies/basal/study.yaml")


def test_validate_passes_on_scaffolded(tmp_path):
    m, _ = _setup(tmp_path)
    assert validate(str(m), str(tmp_path)) == []


def test_validate_flags_unknown_condition(tmp_path):
    m, spath = _setup(tmp_path)
    s = yaml.safe_load(spath.read_text(encoding="utf-8"))
    s["condition"] = "no_such_condition"
    spath.write_text(yaml.safe_dump(s), encoding="utf-8")
    problems = validate(str(m), str(tmp_path))
    assert any("not in manifest" in p for p in problems)


def test_validate_flags_group_mismatch(tmp_path):
    m, spath = _setup(tmp_path)
    s = yaml.safe_load(spath.read_text(encoding="utf-8"))
    s["behavior_tests"][0]["measure"]["group"] = "statistical"  # manifest says standard
    spath.write_text(yaml.safe_dump(s), encoding="utf-8")
    problems = validate(str(m), str(tmp_path))
    assert any("graded cards" in p for p in problems)


def test_validate_flags_bad_card_path(tmp_path):
    m, spath = _setup(tmp_path)
    s = yaml.safe_load(spath.read_text(encoding="utf-8"))
    s["behavior_tests"][0]["measure"]["card"] = "docs/report_cards/wrong/basal"
    spath.write_text(yaml.safe_dump(s), encoding="utf-8")
    problems = validate(str(m), str(tmp_path))
    assert any("card" in p for p in problems)
