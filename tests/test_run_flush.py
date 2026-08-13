# tests/test_run_flush.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.flush import run_flush


def _mk_workspace(tmp_path):
    # Declare the nested studies layout the shared viva_workspace resolver reads
    # (matches v2ecoli's real workspace.yaml).
    (tmp_path / "workspace.yaml").write_text(
        yaml.safe_dump({"name": "test-ws", "layout": {"studies": "workspace/studies"}})
    )


def _study(tmp_path, slug="demo", tests=None):
    _mk_workspace(tmp_path)
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


def test_run_flush_skips_step_when_placement_raises(core, tmp_path, monkeypatch):
    import yaml
    _mk_workspace(tmp_path)
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(
        {"name": "demo", "tests": [
            {"name": "t1", "status": "passed", "pass_if": {"op": "in_range", "low": 1, "high": 2}}]}))
    import v2ecoli.workflow.flush as flush_mod
    monkeypatch.setattr(flush_mod, "place_output",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    res = flush_mod.run_flush("out/x", {"study": "demo"}, tmp_path, core=core)
    assert any(s["name"] == "t1" or s["name"] == "tests" for s in res["skipped"]) or res["placed"] == []
    # the key invariant: run_flush returned normally (did not propagate the OSError)
    assert "placed" in res and "skipped" in res
