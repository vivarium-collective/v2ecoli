# tests/test_run_flush.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import run_flush


def _study(tmp_path, slug="demo", tests=None):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(
        {"name": slug, "tests": tests if tests is not None else [
            {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]}))
    return sd


def test_run_flush_places_report_card(core, tmp_path):
    sd = _study(tmp_path, "demo")
    res = run_flush("out/x", {"study": "demo"}, tmp_path, core=core)
    assert res["study"] == "demo"
    placed = {(p["kind"], p["name"]) for p in res["placed"]}
    assert ("report_card", "tests") in placed
    assert (sd / "viz" / "report_card" / "tests.html").is_file()


def test_run_flush_skips_card_that_raises(core, tmp_path, monkeypatch):
    _study(tmp_path, "demo")
    # Force one card's build to raise; the flush must skip it and still return.
    import v2ecoli.workflow.report_cards.tests_card as tc
    monkeypatch.setattr(tc.TestsCard, "build",
                        lambda self, study: (_ for _ in ()).throw(RuntimeError("boom")))
    res = run_flush("out/x", {"study": "demo"}, tmp_path, core=core)
    assert any(s["name"] == "tests" for s in res["skipped"])
