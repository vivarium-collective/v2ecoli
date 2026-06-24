# tests/test_compare_orchestrator.py
import scripts._compare.orchestrator as orch


def test_vecoli_sim_uses_passed_repo(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd=None, env=None, retries=0):
        captured["cmd"], captured["cwd"], captured["env"] = cmd, cwd, env

    monkeypatch.setattr(orch, "_run", fake_run)
    monkeypatch.setattr(orch, "is_stale", lambda *a, **k: True)
    monkeypatch.setattr(orch, "mark_done", lambda *a, **k: None)
    orch.run_vecoli_sim(config_path="c.json", out_dir="out/v",
                       token="t", vecoli_repo="/tmp/fork")
    assert captured["cwd"] == "/tmp/fork"
    assert captured["cmd"][0] == "/tmp/fork/.venv/bin/python"
    # vEcoli's venv must be first on PATH so Nextflow tasks use its python.
    assert captured["env"]["PATH"].startswith("/tmp/fork/.venv/bin:")
