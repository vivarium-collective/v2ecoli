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
