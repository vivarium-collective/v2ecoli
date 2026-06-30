# tests/test_run_workflow_flush_hook.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _study(tmp_path, slug="demo"):
    import yaml
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
    return sd


def test_run_workflow_calls_flush_when_study_resolvable(monkeypatch, tmp_path):
    import v2ecoli.workflow.run as run_mod
    _study(tmp_path, "demo")  # resolve_owning_study runs for real — needs the study

    calls = {}
    def _fake_flush(out_dir, config, ws_root, **kw):
        calls["study"] = config.get("study")
        return {"placed": [{"kind": "report_card", "name": "tests", "path": "p"}],
                "skipped": [], "study": config.get("study")}
    # _maybe_flush does `from v2ecoli.workflow.flush import run_flush` internally,
    # so patch run_flush on the flush module (where the local import resolves it).
    import v2ecoli.workflow.flush as flush_mod
    monkeypatch.setattr(flush_mod, "run_flush", _fake_flush, raising=False)

    cfg = {"study": "demo", "out_dir": "out/x", "ws_root": str(tmp_path)}
    res = run_mod._maybe_flush(cfg, "out/x", {"complete": True})
    assert res["flush"]["study"] == "demo"
    assert calls["study"] == "demo"


def test_maybe_flush_noop_without_study(tmp_path):
    import v2ecoli.workflow.run as run_mod
    res = run_mod._maybe_flush({"out_dir": "out/workflow", "ws_root": str(tmp_path)},
                               "out/workflow", {"complete": True})
    assert "flush" not in res


def test_maybe_flush_swallows_resolve_error(monkeypatch, tmp_path):
    import v2ecoli.workflow.run as run_mod
    import v2ecoli.workflow.flush as flush_mod
    def _boom(*a, **k):
        raise PermissionError("nope")
    monkeypatch.setattr(flush_mod, "resolve_owning_study", _boom, raising=False)
    res = run_mod._maybe_flush({"study": "demo", "ws_root": str(tmp_path)},
                               "out/x", {"complete": True})
    assert res["complete"] is True          # run result preserved, no exception
    assert "error" in res.get("flush", {})  # error captured into result['flush']
