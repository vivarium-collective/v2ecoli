"""Tests that compare_harness.py writes a report even when stages fail."""
import scripts.compare_harness as h


def test_harness_writes_report_when_sim_stage_fails(tmp_path, monkeypatch):
    # Config stage: avoid the real vEcoli subprocess.
    monkeypatch.setattr(h, "resolve_vecoli_config",
                        lambda p, vecoli_repo=None: {"experiment_id": "x",
                                                     "generations": 2})
    # Sim stage blows up; ParCa diff fails too (cached files absent in tmp).
    def boom(**kwargs):
        raise RuntimeError("sim exploded")
    monkeypatch.setattr(h.orchestrator, "run_vecoli_sim", boom)
    monkeypatch.setattr(h.orchestrator, "run_v2_sim", boom)
    out = tmp_path / "report.html"
    h.main(["--config", "ignored.json", "-o", str(out),
            "--workdir", str(tmp_path / "work"),
            "--vecoli-simdata", str(tmp_path / "missing.cPickle"),
            "--v2-cache", str(tmp_path / "missing_cache")])
    html = out.read_text(encoding="utf-8")
    # report still written; both engine stages degrade to error sections.
    assert "ParCa / sim_data (cached)" in html
    assert "2-generation sim dynamics" in html
    assert "sim exploded" in html
    # loaded-config panel is always present
    assert "Loaded config" in html
