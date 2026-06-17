# tests/test_compare_orchestrator.py
import scripts._compare.orchestrator as orch


def test_vecoli_sim_uses_passed_repo(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd=None):
        captured["cmd"], captured["cwd"] = cmd, cwd

    monkeypatch.setattr(orch, "_run", fake_run)
    monkeypatch.setattr(orch, "is_stale", lambda *a, **k: True)
    monkeypatch.setattr(orch, "mark_done", lambda *a, **k: None)
    orch.run_vecoli_sim(config_path="c.json", out_dir="out/v",
                       token="t", vecoli_repo="/tmp/fork")
    assert captured["cwd"] == "/tmp/fork"
    assert captured["cmd"][0] == "/tmp/fork/.venv/bin/python"
