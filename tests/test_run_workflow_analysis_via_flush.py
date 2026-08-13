import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_run_workflow_does_not_call_run_analyses_directly(monkeypatch, tmp_path):
    """The standalone run_analyses call is gone; analyses go through the flush."""
    import v2ecoli.workflow.run as run_mod
    import v2ecoli.workflow.analysis_runner as ar

    called = {"direct": 0}
    monkeypatch.setattr(ar, "run_analyses",
                        lambda *a, **k: called.__setitem__("direct", called["direct"] + 1) or {},
                        raising=False)
    # the flush is where analyses should now be driven — stub it so we can assert
    # run_workflow no longer invokes run_analyses on its own.
    import v2ecoli.workflow.flush as flush_mod
    monkeypatch.setattr(flush_mod, "run_flush",
                        lambda *a, **k: {"placed": [], "skipped": [], "study": "demo"},
                        raising=False)

    import yaml
    (tmp_path / "workspace.yaml").write_text(
        yaml.safe_dump({"name": "test-ws", "layout": {"studies": "workspace/studies"}}))
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": "demo"}))

    cfg = {"study": "demo", "out_dir": "out/x", "ws_root": str(tmp_path),
           "analysis_options": {"single": {"x": {}}}}
    res = run_mod._maybe_flush(cfg, "out/x", {"complete": True})
    # _maybe_flush drove the flush (stubbed), and did NOT call run_analyses directly
    assert res.get("flush", {}).get("study") == "demo"
    assert called["direct"] == 0
