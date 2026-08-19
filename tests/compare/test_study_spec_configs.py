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


# --- the investigation route must carry companions too ----------------------
# run_investigation builds specs ONLY through specs_from_configs, from
# investigation.yaml's configs[] entries — which do NOT carry this key. The
# declaration lives in the study.yaml, which that route names but never read.
# So it evaporated on one of two first-class routes while still sitting in the
# file, looking correct.
#
# These drive a REAL study.yaml on disk through the investigation route. The
# earlier version of these tests declared the key on the investigation entry
# instead, which fenced a surface that should not have existed and let the
# documented one stay broken.

def _study_on_disk(tmp_path, monkeypatch, name, body):
    import scripts._compare.study_spec as ss
    d = tmp_path / "workspace" / "studies" / name
    d.mkdir(parents=True)
    import yaml as _y
    (d / "study.yaml").write_text(_y.safe_dump(body))
    monkeypatch.setattr(ss, "REPO", tmp_path)
    return d / "study.yaml"


def test_investigation_route_reads_companions_from_the_study_yaml(tmp_path, monkeypatch):
    _study_on_disk(tmp_path, monkeypatch, "s", {
        "name": "s", "condition": "basal", "from_vecoli_config": "configs/redux.json",
        "inject_processes": ["companion-listener"]})
    specs = specs_from_configs(_ctx([{"name": "s", "config": "configs/redux.json"}]))
    assert specs[0].inject_processes == ["companion-listener"]


def test_investigation_route_also_reads_the_comparison_block(tmp_path, monkeypatch):
    _study_on_disk(tmp_path, monkeypatch, "s", {
        "name": "s", "condition": "basal",
        "comparison": {"inject_processes": ["companion-listener"]}})
    specs = specs_from_configs(_ctx([{"name": "s", "config": "configs/redux.json"}]))
    assert specs[0].inject_processes == ["companion-listener"]


def test_investigation_route_defaults_to_an_empty_list(tmp_path, monkeypatch):
    _study_on_disk(tmp_path, monkeypatch, "s", {"name": "s", "condition": "basal"})
    specs = specs_from_configs(_ctx([{"name": "s", "config": "configs/redux.json"}]))
    assert specs[0].inject_processes == []


def test_investigation_route_rejects_companions_it_cannot_inject(tmp_path, monkeypatch):
    import pytest
    _study_on_disk(tmp_path, monkeypatch, "basal", {
        "name": "basal", "condition": "basal",
        "inject_processes": ["companion-listener"]})
    with pytest.raises(ValueError, match="silently discarded"):
        specs_from_configs(_ctx([{"name": "basal", "config": "basal"}]))


def test_a_missing_study_yaml_is_not_an_error(tmp_path, monkeypatch):
    # Not every configs[] entry has a study dir on disk.
    import scripts._compare.study_spec as ss
    monkeypatch.setattr(ss, "REPO", tmp_path)
    specs = specs_from_configs(_ctx([{"name": "absent", "config": "configs/redux.json"}]))
    assert specs[0].inject_processes == []


def test_companions_resolve_within_an_alternate_workspace(tmp_path):
    # `inv_dir` is <workspace>/investigations/<name>, so studies are its sibling.
    # Deriving from it — as the legacy members path already did — means an
    # investigation rooted outside REPO still finds its studies. Hardcoding REPO
    # read no companions and did not fire the guard, silently.
    import yaml as _y
    ws = tmp_path / "alt_workspace"
    inv_dir = ws / "investigations" / "inv"
    inv_dir.mkdir(parents=True)
    sdir = ws / "studies" / "s"
    sdir.mkdir(parents=True)
    (sdir / "study.yaml").write_text(_y.safe_dump({
        "name": "s", "condition": "basal", "from_vecoli_config": "configs/redux.json",
        "inject_processes": ["companion-listener"]}))
    ctx = _ctx([{"name": "s", "config": "configs/redux.json"}])
    ctx["inv_dir"] = inv_dir
    specs = specs_from_configs(ctx)
    assert specs[0].inject_processes == ["companion-listener"]
    assert specs[0].study_path == str(sdir / "study.yaml")
