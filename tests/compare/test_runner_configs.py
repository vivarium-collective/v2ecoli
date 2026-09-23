import inspect
from scripts._compare import runner
from scripts._compare.study_spec import specs_from_configs
from scripts._compare.reference import ReferenceEngine


def _spec(config):
    ctx = {"invest_name": "whole-cell-model-comparison",
           "reference": ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"}),
           "configs": [{"name": "s", "config": config, "condition": "basal"}],
           "v2_cache": "vc", "ve_cache": "vec",
           "defaults": {"seeds": 4, "gens": 1, "cards": ["parca"]}, "inv_dir": None}
    return specs_from_configs(ctx)[0]


def test_condition_config_uses_condition_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv) or type("P", (), {"returncode": 0})())
    runner._run_engines(_spec("basal"), out="out/x", mode="serial")
    v2, ve = calls
    assert "--condition" in v2 and "basal" in v2
    assert "--from-vecoli-config" not in v2      # bare condition → no swap flag


def test_path_config_uses_swap_flag_on_both(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv) or type("P", (), {"returncode": 0})())
    runner._run_engines(_spec("configs/redux.json"), out="out/x", mode="serial")
    v2, ve = calls
    assert "--from-vecoli-config" in v2 and "configs/redux.json" in v2
    assert "--from-vecoli-config" in ve


def test_append_store_spares_the_runner_s_OWN_rmtree(tmp_path, monkeypatch):
    """⛔⛔ ONE FLAG, TWO DELETION SITES.

    `--append-store` reaches `run_multigen_xarray(overwrite=…)`. But the runner
    has its OWN `shutil.rmtree` of the per-seed store, upstream of that — so a
    resumed stage deleted the generations it was about to resume from, then
    failed looking for the parent it had just removed (`KeyError:
    emitstep_gen=1`), which reads as an emitter bug.

    Measured before the fix: stage 1 closed with 10 data files on disk; stage 2
    started and the store was empty.
    """
    import scripts.run_comparison_ensemble as rce

    src = inspect.getsource(rce.make_run_one)
    # The rmtree must be guarded by append_store, not unconditional.
    assert "not append_store" in src, (
        "the runner's own rmtree ignores --append-store; a chain's later stage "
        "will delete its predecessor's generations")
    i_guard = src.index("not append_store")
    i_rmtree = src.index("shutil.rmtree(store_path)")
    assert i_guard < i_rmtree, (
        "the append_store guard must precede the rmtree it protects")
