# tests/test_tests_card.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.tests_card import TestsCard


def _ctx(tmp_path, tests):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": "Demo", "tests": tests}))
    return StudyContext.load(tmp_path, "demo")


def test_one_axis_per_test_overall_is_worst(core, tmp_path):
    ctx = _ctx(tmp_path, [
        {"name": "doubling-time-in-band", "classification": "primary",
         "status": "passed", "pass_if": {"op": "in_range", "low": 35, "high": 55}},
        {"name": "mass-fraction", "classification": "primary",
         "status": "failed", "pass_if": {"op": "in_range", "low": 0.40, "high": 0.55}},
    ])
    m = TestsCard({}, core=core)
    assert m.applies(ctx) is True
    vjson, html = m.build(ctx)
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["overall"] == "mismatch"               # worst of pass + fail
    assert "doubling-time-in-band" in html and "mass-fraction" in html
    assert "in [35, 55]" in html                        # criterion string surfaced


def test_absent_when_no_tests(core, tmp_path):
    assert TestsCard({}, core=core).applies(_ctx(tmp_path, [])) is False
