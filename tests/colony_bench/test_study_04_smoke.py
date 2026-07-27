import json, importlib.util, pathlib

STUDY = pathlib.Path("workspace/studies/colonies-04-device-phenotype-harness")  # registry: top-level (Spec 1)

def test_study_run_writes_phenotypes(tmp_path):
    spec = importlib.util.spec_from_file_location("c04run", STUDY / "sims" / "run.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = mod.main(geometry="free_colony", tier="simple", n_ticks=5, out_dir=tmp_path)
    pheno = json.loads((tmp_path / "phenotypes.json").read_text(encoding="utf-8"))
    assert "n_division_events" in pheno
    assert out["n_final"] >= 2
