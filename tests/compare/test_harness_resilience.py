"""Tests that compare_harness.py writes a report even when stages fail."""
import scripts.compare_harness as h


def test_harness_writes_report_when_parca_stage_fails(tmp_path, monkeypatch):
    # Config stage: avoid the real vEcoli subprocess.
    monkeypatch.setattr(h, "resolve_vecoli_config",
                        lambda p, vecoli_repo=None: {"experiment_id": "x",
                                                     "generations": 2})
    # ParCa stage blows up.
    def boom(**kwargs):
        raise RuntimeError("parca exploded")
    monkeypatch.setattr(h.orchestrator, "run_vecoli_parca", boom)
    monkeypatch.setattr(h.orchestrator, "run_v2_parca", boom)
    # Sim stage also can't run without ParCa; make it raise too if reached.
    monkeypatch.setattr(h.orchestrator, "run_vecoli_sim", boom)
    monkeypatch.setattr(h.orchestrator, "run_v2_sim", boom)
    out = tmp_path / "report.html"
    h.main(["--config", "ignored.json", "-o", str(out),
            "--workdir", str(tmp_path / "work")])
    html = out.read_text(encoding="utf-8")
    # report still written, both downstream sections present as errors
    assert "ParCa / sim_data" in html
    assert "2-generation sim dynamics" in html
    assert "parca exploded" in html
    assert "Config &amp; schema diff" in html or "Config & schema diff" in html
