from pathlib import Path

from scripts._compare import orchestrator


def test_run_v2_parca_skips_when_fresh(tmp_path, monkeypatch):
    out = tmp_path / "v2parca"
    out.mkdir()
    (out / ".done").write_text("ok")
    called = {"n": 0}
    monkeypatch.setattr(orchestrator.subprocess, "run",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    result = orchestrator.run_v2_parca(out_dir=out, cache_dir=tmp_path / "c",
                                       mode="full")
    assert called["n"] == 0          # cache hit → no subprocess
    assert result == out


def test_run_v2_parca_invokes_cli_when_stale(tmp_path, monkeypatch):
    out = tmp_path / "v2parca"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        (out).mkdir(parents=True, exist_ok=True)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator.run_v2_parca(out_dir=out, cache_dir=tmp_path / "c",
                              mode="full")
    assert "v2ecoli-parca" in captured["cmd"]
    assert "--mode" in captured["cmd"] and "full" in captured["cmd"]


def test_run_vecoli_parca_uses_vecoli_python_and_save_intermediates(
        tmp_path, monkeypatch):
    out = tmp_path / "vparca"
    captured = {}

    def fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        out.mkdir(parents=True, exist_ok=True)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator.run_vecoli_parca(config_path="/x/cfg.json", out_dir=out)
    assert captured["cmd"][0].endswith("/vEcoli/.venv/bin/python")
    assert "--save-intermediates" in captured["cmd"]
    assert captured["cwd"].endswith("/vEcoli")


def test_run_vecoli_sim_drives_nextflow_workflow(tmp_path, monkeypatch):
    out = tmp_path / "vsim"
    captured = {}

    def fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        out.mkdir(parents=True, exist_ok=True)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator.run_vecoli_sim(config_path="/x/vsim_cfg.json", out_dir=out)
    assert captured["cmd"][0].endswith("/vEcoli/.venv/bin/python")
    assert captured["cmd"][1:4] == ["-m", "runscripts.workflow", "--config"]
    assert captured["cwd"].endswith("/vEcoli")


def test_run_vecoli_sim_skips_when_fresh(tmp_path, monkeypatch):
    out = tmp_path / "vsim"
    out.mkdir()
    (out / ".done").write_text("ok")
    called = {"n": 0}
    monkeypatch.setattr(orchestrator.subprocess, "run",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    assert orchestrator.run_vecoli_sim(config_path="/x.json", out_dir=out) == out
    assert called["n"] == 0
