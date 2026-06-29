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


def test_pytest_dict_schema_empty_results_gets_placeholder_card(core, tmp_path):
    """Dict schema with no last_results → applies True, one placeholder axis, ungraded."""
    ctx = _ctx(tmp_path, {"auto_discover": True, "pytest_args": ["t.py"], "last_results": None})
    m = TestsCard({}, core=core)
    assert m.applies(ctx) is True
    result = m.build(ctx)
    assert result is not None, "build() must not return None for a pytest-dict study"
    vjson, html = result
    assert vjson["overall"] == "ungraded"
    total_axes = sum(len(g["axes"]) for g in vjson["groups"].values())
    assert total_axes == 1
    assert "pytest" in html.lower()


def test_pytest_dict_schema_with_results_renders_outcomes(core, tmp_path):
    """Dict schema with last_results list → two axes, overall mismatch (pass+fail)."""
    ctx = _ctx(tmp_path, {
        "last_results": [
            {"name": "test_a", "outcome": "passed"},
            {"name": "test_b", "outcome": "failed"},
        ]
    })
    m = TestsCard({}, core=core)
    vjson, html = m.build(ctx)
    total_axes = sum(len(g["axes"]) for g in vjson["groups"].values())
    assert total_axes == 2
    assert vjson["overall"] == "mismatch"


def test_tests_null_does_not_apply(core, tmp_path):
    """tests: null → applies() False."""
    ctx = _ctx(tmp_path, None)
    assert TestsCard({}, core=core).applies(ctx) is False
