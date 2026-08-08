from pathlib import Path

from scripts._compare import orchestrator
from scripts._compare.reference import ReferenceEngine


def _ref(repo="/Users/eranagmon/code/vEcoli"):
    return ReferenceEngine.from_spec({"repo": repo, "kind": "vecoli"})


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
    # Invoked by absolute path inside the v2 venv (no reliance on PATH).
    assert captured["cmd"][0].endswith("/v2ecoli-parca")
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
    orchestrator.run_vecoli_parca(reference=_ref(), config_path="/x/cfg.json",
                                  out_dir=out)
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
    orchestrator.run_vecoli_sim(reference=_ref(), config_path="/x/vsim_cfg.json",
                                out_dir=out)
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
    assert orchestrator.run_vecoli_sim(
        reference=_ref(), config_path="/x.json", out_dir=out) == out
    assert called["n"] == 0


def test_run_retries_then_succeeds(monkeypatch):
    """A transient non-zero exit is retried; a later success is accepted so a
    single seed's flaky Nextflow launch doesn't drop it from a batch."""
    rcs = iter([1, 1, 0])  # fail, fail, succeed
    seen = {"n": 0}

    def fake_run(cmd, **kwargs):
        seen["n"] += 1
        class R:
            returncode = next(rcs)
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator._run(["x"], retries=2)        # must not raise
    assert seen["n"] == 3


def test_run_raises_after_exhausting_retries(monkeypatch):
    seen = {"n": 0}

    def fake_run(cmd, **kwargs):
        seen["n"] += 1
        class R:
            returncode = 1
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    try:
        orchestrator._run(["x"], retries=2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError after exhausting retries")
    assert seen["n"] == 3                       # original + 2 retries


def test_vecoli_sim_retries_transient_failure(tmp_path, monkeypatch):
    out = tmp_path / "vsim"
    rcs = iter([1, 0])                          # one transient failure, then ok
    monkeypatch.setattr(orchestrator.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": next(rcs)})())
    orchestrator.run_vecoli_sim(reference=_ref(str(tmp_path)), config_path="c.json",
                                out_dir=out, token="t")
    assert (out / ".done").exists()             # recovered → marked done


def test_vecoli_parca_uses_reference_commands(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(orchestrator, "_run", lambda cmd, cwd=None, env=None, retries=0: captured.update(cmd=cmd, cwd=cwd, env=env))
    monkeypatch.setattr(orchestrator, "is_stale", lambda *a, **k: True)
    monkeypatch.setattr(orchestrator, "mark_done", lambda *a, **k: None)
    ref = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    orchestrator.run_vecoli_parca(reference=ref, config_path="/c.json", out_dir=tmp_path)
    assert captured["cmd"][0] == "/abs/vEcoli/.venv/bin/python"
    assert "runscripts/parca.py" in captured["cmd"]
    assert captured["cwd"] == "/abs/vEcoli"
    assert captured["env"]["PATH"].startswith("/abs/vEcoli/.venv/bin:")
