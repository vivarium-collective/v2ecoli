# tests/test_vs_vecoli_card.py
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.vs_vecoli_card import VsVecoliCard


def _ctx(tmp_path, refs=None):
    (tmp_path / "workspace.yaml").write_text(
        yaml.safe_dump({"name": "test-ws", "layout": {"studies": "workspace/studies"}}))
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    spec = {"name": "Demo"}
    if refs:
        spec["report_card_refs"] = refs
    (sd / "study.yaml").write_text(yaml.safe_dump(spec))
    return StudyContext.load(tmp_path, "demo")


def _write_verdict(tmp_path):
    p = tmp_path / "docs" / "rc" / "basal" / "report_card_verdict.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "schema": "report_card_verdict/v1", "overall": "drift",
        "reference_model": "vEcoli @ basal", "model_ref": "v2ecoli @ basal",
        "groups": {"standard": {"verdict": "drift", "axes": [
            {"id": "physiology.cell_mass", "label": "Cell mass",
             "verdict": "within_tol", "value": 1.2, "meter": ""}]}}}))
    return "docs/rc/basal/report_card_verdict.json"


def test_absent_without_ref(core, tmp_path):
    assert VsVecoliCard({}, core=core).applies(_ctx(tmp_path)) is False


def test_absent_when_ref_missing_file(core, tmp_path):
    ctx = _ctx(tmp_path, refs={"vs_vecoli": "docs/rc/nope.json"})
    assert VsVecoliCard({}, core=core).applies(ctx) is False


def test_renders_from_declared_verdict(core, tmp_path):
    rel = _write_verdict(tmp_path)
    ctx = _ctx(tmp_path, refs={"vs_vecoli": rel})
    m = VsVecoliCard({}, core=core)
    assert m.applies(ctx) is True
    vjson, html = m.build(ctx)
    assert vjson["overall"] == "drift"
    assert "Cell mass" in html


def test_builtin_cards_registered():
    from v2ecoli.workflow.report_cards import REPORT_CARD_REGISTRY
    assert "tests" in REPORT_CARD_REGISTRY
    assert "vs_vecoli" in REPORT_CARD_REGISTRY
