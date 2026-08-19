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


# --- companion fork processes declared by the study -------------------------

def _study_ctx():
    c = _ctx([])
    c["default_cards"] = ["parca"]      # _spec_from_study reads this, not `defaults`
    return c


def _study_yaml(tmp_path, body):
    import yaml
    d = tmp_path / "a-study"
    d.mkdir()
    (d / "study.yaml").write_text(yaml.safe_dump(body))
    return d / "study.yaml"


def test_study_declares_companion_processes_at_top_level(tmp_path):
    from scripts._compare.study_spec import _spec_from_study
    path = _study_yaml(tmp_path, {
        "name": "s", "condition": "basal",
        "from_vecoli_config": "configs/redux.json",
        "inject_processes": ["companion-listener"],
    })
    spec = _spec_from_study(path, _study_ctx())
    assert spec.inject_processes == ["companion-listener"]


def test_study_declares_companion_processes_under_comparison(tmp_path):
    from scripts._compare.study_spec import _spec_from_study
    path = _study_yaml(tmp_path, {
        "name": "s", "condition": "basal",
        "comparison": {"config": "configs/redux.json",
                       "inject_processes": ["companion-listener"]},
    })
    spec = _spec_from_study(path, _study_ctx())
    assert spec.inject_processes == ["companion-listener"]


def test_a_study_that_declares_none_gets_an_empty_list_not_none(tmp_path):
    # runner.py iterates this; None would need a guard at every call site.
    from scripts._compare.study_spec import _spec_from_study
    path = _study_yaml(tmp_path, {
        "name": "s", "condition": "basal", "from_vecoli_config": "configs/redux.json"})
    spec = _spec_from_study(path, _study_ctx())
    assert spec.inject_processes == []


def test_top_level_companions_win_over_the_comparison_block(tmp_path):
    from scripts._compare.study_spec import _spec_from_study
    path = _study_yaml(tmp_path, {
        "name": "s", "condition": "basal", "from_vecoli_config": "configs/redux.json",
        "inject_processes": ["top-level"],
        "comparison": {"inject_processes": ["nested"]}})
    assert _spec_from_study(path, _study_ctx()).inject_processes == ["top-level"]


def test_an_empty_top_level_list_falls_through_rather_than_clearing(tmp_path):
    # Precedence is by truthiness, not presence — so top-level cannot be used to
    # switch off a companion declared under `comparison:`. Pinning the documented
    # behaviour so a future reader does not assume the override works.
    from scripts._compare.study_spec import _spec_from_study
    path = _study_yaml(tmp_path, {
        "name": "s", "condition": "basal",
        "inject_processes": [],
        "comparison": {"config": "configs/redux.json", "inject_processes": ["nested"]}})
    assert _spec_from_study(path, _study_ctx()).inject_processes == ["nested"]


def test_a_companion_on_a_bare_condition_config_is_rejected_not_dropped(tmp_path):
    # Injection only happens on the --from-vecoli-config path, so this
    # declaration would reach the runner and be discarded with no error.
    import pytest
    from scripts._compare.study_spec import _spec_from_study
    path = _study_yaml(tmp_path, {
        "name": "basal", "condition": "basal",
        "inject_processes": ["companion-listener"]})
    with pytest.raises(ValueError, match="silently discarded"):
        _spec_from_study(path, _study_ctx())
