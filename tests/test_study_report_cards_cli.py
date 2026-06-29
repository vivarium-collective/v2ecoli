# tests/test_study_report_cards_cli.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.study_report_cards as cli


def _study(tmp_path, name, spec):
    sd = tmp_path / "workspace" / "studies" / name
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(spec))
    return sd


def test_generate_study_emits_tests_card(core, tmp_path):
    sd = _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "classification": "primary", "status": "passed",
         "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    r = cli.generate_study(tmp_path, "demo", core, only=None, do_prune=True)
    assert "tests" in r["written"]
    rc = sd / "viz" / "report_card"
    assert (rc / "tests.html").is_file()
    assert json.loads((rc / "tests.verdict.json").read_text())["overall"] == "within_tol"


def test_only_filters_to_one_card(core, tmp_path):
    _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    r = cli.generate_study(tmp_path, "demo", core, only="vs_vecoli", do_prune=False)
    assert r["written"] == []   # tests excluded by --card vs_vecoli; no ref -> none


def test_prune_drops_stale_card(core, tmp_path):
    sd = _study(tmp_path, "demo", {"name": "Demo", "tests": [
        {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]})
    rc = sd / "viz" / "report_card"
    rc.mkdir(parents=True)
    (rc / "old.html").write_text("<i></i>")
    cli.generate_study(tmp_path, "demo", core, only=None, do_prune=True)
    assert not (rc / "old.html").is_file()    # stale pruned
    assert (rc / "tests.html").is_file()
