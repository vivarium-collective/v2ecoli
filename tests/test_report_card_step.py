# tests/test_report_card_step.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.steps.base import V2Step
from v2ecoli.workflow.report_cards import (
    REPORT_CARD_REGISTRY, ReportCardStep, StudyContext, applicable, prune, write_card)


class _DemoCard(ReportCardStep):
    name = "demo_card"

    def applies(self, study):
        return bool(study.spec.get("demo"))

    def build(self, study):
        return ({"schema": "report_card_verdict/v1", "overall": "drift"},
                "<div>demo</div>")


def _ctx(tmp_path, spec=None):
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(spec or {"name": "demo"}))
    return StudyContext.load(tmp_path, "demo")


def test_reportcardstep_is_v2step_with_view_data_ports(core):
    step = _DemoCard({}, core=core)
    assert isinstance(step, V2Step)
    assert step.outputs() == {"view": "string", "data": "map"}
    assert step.inputs() == {"study": "any"}


def test_subclass_auto_registers():
    assert REPORT_CARD_REGISTRY.get("demo_card") is _DemoCard


def test_update_returns_view_and_data(core, tmp_path):
    ctx = _ctx(tmp_path, {"name": "demo", "demo": True})
    out = _DemoCard({}, core=core).update({"study": ctx})
    assert out["view"] == "<div>demo</div>"
    assert out["data"]["overall"] == "drift"


def test_studycontext_loads_spec_and_paths(tmp_path):
    ctx = _ctx(tmp_path, {"name": "Demo", "tests": [{"name": "t"}]})
    assert ctx.study_name == "demo"
    assert ctx.spec["name"] == "Demo"
    assert ctx.card_dir.name == "report_card"
    assert ctx.run_zarr_paths() == []


def test_write_card_writes_both_files_and_sanitizes(tmp_path):
    ctx = _ctx(tmp_path)
    p = write_card(ctx, "tests", {"overall": "drift", "x": float("inf")}, "<i>hi</i>")
    assert p.name == "tests.html"
    assert p.read_text() == "<i>hi</i>"
    vj = json.loads((ctx.card_dir / "tests.verdict.json").read_text())
    assert vj["overall"] == "drift"
    assert vj["x"] is None  # inf -> null (bundle-safe)


def test_prune_removes_stale_only(tmp_path):
    ctx = _ctx(tmp_path)
    write_card(ctx, "keep", {"overall": "within_tol"}, "<i></i>")
    write_card(ctx, "stale", {"overall": "within_tol"}, "<i></i>")
    assert prune(ctx, keep={"keep"}) == ["stale"]
    assert (ctx.card_dir / "keep.html").is_file()
    assert not (ctx.card_dir / "stale.html").is_file()
    assert not (ctx.card_dir / "stale.verdict.json").is_file()


def test_applicable_selects_by_applies_and_allowlist(core, tmp_path):
    on = _ctx(tmp_path, {"name": "demo", "demo": True})
    # only='demo_card' isolates from other registered cards; applies() True here
    assert [s.name for s in applicable(on, core, only="demo_card")] == ["demo_card"]
    off = _ctx(tmp_path, {"name": "demo"})  # no 'demo' key -> applies() False
    assert applicable(off, core, only="demo_card") == []
    # explicit report_cards allowlist excluding demo_card -> not emitted
    excl = _ctx(tmp_path, {"name": "demo", "demo": True, "report_cards": ["tests"]})
    assert applicable(excl, core, only="demo_card") == []
