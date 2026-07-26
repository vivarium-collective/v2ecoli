import importlib.util
import json
import pathlib

STUDIES = pathlib.Path("workspace/studies")  # registry: studies live top-level (Spec 1 migration)


def _load(run_path):
    spec = importlib.util.spec_from_file_location(run_path.stem + "_mod", run_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mother_machine_study_writes_artifacts(tmp_path):
    mod = _load(STUDIES / "colonies-05-mother-machine" / "sims" / "run.py")
    out = mod.main(n_steps=3, out_dir=tmp_path, n_channels=3)
    assert out["n_final"] >= 1
    assert (tmp_path / "phenotypes.json").exists()
    charts = tmp_path / "charts"
    assert (charts / "colony.gif").exists()
    for fig in ("size_at_division.png", "interdivision_time.png", "added_size.png"):
        assert (charts / fig).exists()
    json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_daughter_machine_study_writes_artifacts(tmp_path):
    mod = _load(STUDIES / "colonies-06-daughter-machine" / "sims" / "run.py")
    out = mod.main(n_steps=3, out_dir=tmp_path, env_size=30.0)
    assert out["n_final"] >= 1
    assert (tmp_path / "charts" / "colony.gif").exists()
