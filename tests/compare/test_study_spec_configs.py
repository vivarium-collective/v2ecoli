from scripts._compare.study_spec import specs_from_configs
from scripts._compare.reference import ReferenceEngine


def _ctx(configs):
    return {
        "invest_name": "whole-cell-model-comparison",
        "reference": ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"}),
        "configs": configs,
        "v2_cache": "out/cache_full",
        "ve_cache": "out/compare_harness/vecoli_parca",
        "defaults": {"seeds": 4, "gens": 1, "cards": ["parca", "statistical"]},
        "inv_dir": None,
    }


def test_one_spec_per_config_with_defaults():
    specs = specs_from_configs(_ctx([{"name": "basal", "config": "basal"}]))
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "basal" and s.config == "basal" and s.condition == "basal"
    assert s.seeds == 4 and s.gens == 1 and s.cards == ["parca", "statistical"]


def test_path_config_carries_swap_and_condition_override():
    specs = specs_from_configs(_ctx([
        {"name": "redux_basal", "config": "configs/redux.json", "condition": "basal", "seeds": 6},
    ]))
    s = specs[0]
    assert s.config == "configs/redux.json"    # a swap is just a config path
    assert s.condition == "basal"              # explicit override
    assert s.seeds == 6                          # per-entry override wins over defaults


def test_condition_defaults_to_name_when_absent():
    specs = specs_from_configs(_ctx([{"name": "acetate", "config": "acetate"}]))
    assert specs[0].condition == "acetate"
