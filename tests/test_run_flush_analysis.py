# tests/test_run_flush_analysis.py
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow import flush as flush_mod
from v2ecoli.workflow.flush import run_flush


def _study(tmp_path, slug="demo"):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({"name": slug}))
    return sd


def test_run_flush_runs_and_places_analyses(core, tmp_path, monkeypatch):
    sd = _study(tmp_path, "demo")
    out = tmp_path / "out" / "run1"

    def _fake_run_analyses(sweep_dir, analysis_options, *a, **k):
        viz = Path(sweep_dir) / "viz"
        viz.mkdir(parents=True, exist_ok=True)
        (viz / "mass_fraction__seed_0.html").write_text("<div>mf</div>")
        return {}
    # run_flush imports run_analyses lazily from analysis_runner; patch it there.
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "run_analyses", _fake_run_analyses, raising=False)

    cfg = {"study": "demo", "analysis_options": {"single": {"mass_fraction": {}}}}
    res = run_flush(str(out), cfg, tmp_path, core=core, kinds=("analysis",))
    assert any(p["kind"] == "analysis" and p["name"] == "mass_fraction__seed_0"
               for p in res["placed"])
    assert (sd / "viz" / "mass_fraction__seed_0.html").is_file()


def test_run_flush_analysis_skips_on_error(core, tmp_path, monkeypatch):
    _study(tmp_path, "demo")
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "run_analyses",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
                        raising=False)
    cfg = {"study": "demo", "analysis_options": {"single": {"x": {}}}}
    res = run_flush(str(tmp_path / "out"), cfg, tmp_path, core=core, kinds=("analysis",))
    assert any(s["name"] == "analyses" for s in res["skipped"])


def test_run_flush_no_analysis_options_noop(core, tmp_path):
    _study(tmp_path, "demo")
    res = run_flush(str(tmp_path / "out"), {"study": "demo"}, tmp_path,
                    core=core, kinds=("analysis",))
    assert res["placed"] == [] and res["skipped"] == []


def test_default_kinds_include_analysis():
    import inspect
    sig = inspect.signature(run_flush)
    assert sig.parameters["kinds"].default == ("analysis", "report_card", "visualization")
