import json
import os
import pytest
from v2ecoli.workflow.config import load_config_with_inheritance, _merge_configs
from v2ecoli.workflow.variants import expand_branches


def test_merge_overlay_wins(tmp_path):
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    overlay = {"a": 2, "nested": {"y": 3}}
    _merge_configs(base, overlay)
    assert base["a"] == 2
    assert base["nested"] == {"x": 1, "y": 3}


def test_inheritance_priority(tmp_path):
    (tmp_path / "C.json").write_text(json.dumps({"v": "C", "only_c": 1}))
    (tmp_path / "B.json").write_text(json.dumps({"inherit_from": ["C.json"], "v": "B"}))
    (tmp_path / "D.json").write_text(json.dumps({"v": "D", "only_d": 1}))
    (tmp_path / "A.json").write_text(
        json.dumps({"inherit_from": ["B.json", "D.json"], "v": "A"}))
    cfg = load_config_with_inheritance(str(tmp_path / "A.json"), config_dir=str(tmp_path))
    # Priority A > B > C > D
    assert cfg["v"] == "A"
    assert cfg["only_c"] == 1
    assert cfg["only_d"] == 1


def test_list_keys_merge_and_dedup(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"add_processes": ["z_proc", "m_proc"]}))
    (tmp_path / "top.json").write_text(
        json.dumps({"inherit_from": ["base.json"], "add_processes": ["m_proc", "a_proc"]}))
    cfg = load_config_with_inheritance(str(tmp_path / "top.json"), config_dir=str(tmp_path))
    # Contract: concatenate across the chain, dedup, sorted order.
    assert cfg["add_processes"] == ["a_proc", "m_proc", "z_proc"]


def test_circular_inheritance_raises(tmp_path):
    (tmp_path / "A.json").write_text(json.dumps({"inherit_from": ["B.json"]}))
    (tmp_path / "B.json").write_text(json.dumps({"inherit_from": ["A.json"]}))
    with pytest.raises(ValueError, match="circular"):
        load_config_with_inheritance(str(tmp_path / "A.json"), config_dir=str(tmp_path))


def _vecoli_reference_merge(base_config: dict, overlay_config: dict) -> None:
    """Reference merger transcribed verbatim from vEcoli
    ``runscripts/workflow.py:_merge_configs`` (lines 626-644). Kept inline so the
    parity assertion below survives even without a vEcoli checkout on PYTHONPATH.
    """
    from v2ecoli.workflow.config import LIST_KEYS_TO_MERGE

    for key, value in overlay_config.items():
        if key in LIST_KEYS_TO_MERGE:
            base_config.setdefault(key, [])
            base_config[key].extend(value)
            if key == "engine_process_reports":
                base_config[key] = [tuple(path) for path in base_config[key]]
            base_config[key] = sorted(list(set(base_config[key])))
        elif (
            isinstance(value, dict)
            and key in base_config
            and isinstance(base_config[key], dict)
        ):
            _vecoli_reference_merge(base_config[key], value)
        else:
            base_config[key] = value


def test_engine_process_reports_merge_does_not_crash(tmp_path):
    """§2.6 regression: a chain carrying ``engine_process_reports`` (a list of
    path-lists) must merge without ``TypeError: unhashable type: 'list'``."""
    (tmp_path / "spatial.json").write_text(json.dumps({
        "engine_process_reports": [["boundary"], ["environment", "exchange"],
                                   ["listeners"]],
    }))
    (tmp_path / "child.json").write_text(json.dumps({
        "inherit_from": ["spatial.json"],
        # overlaps one entry (dedup) and adds a new one
        "engine_process_reports": [["listeners"], ["bulk"]],
    }))
    cfg = load_config_with_inheritance(
        str(tmp_path / "child.json"), config_dir=str(tmp_path))
    # merged, deduped, sorted list of path-tuples
    assert cfg["engine_process_reports"] == [
        ("boundary",), ("bulk",), ("environment", "exchange"), ("listeners",)]


def test_engine_process_reports_merge_matches_vecoli(tmp_path):
    """Parity test (§2.6): v2ecoli's ``_merge_configs`` must produce exactly what
    vEcoli's reference merger produces for ``engine_process_reports``."""
    base = {"engine_process_reports": [["boundary"], ["listeners"]],
            "add_processes": ["ecoli-shape"]}
    overlay = {"engine_process_reports": [["listeners"], ["environment", "exchange"]],
               "add_processes": ["gillespie"]}

    got = json.loads(json.dumps(base))  # deep copy
    _merge_configs(got, overlay)

    ref = json.loads(json.dumps(base))  # deep copy
    _vecoli_reference_merge(ref, overlay)

    assert got == ref
    assert got["engine_process_reports"] == [
        ("boundary",), ("environment", "exchange"), ("listeners",)]


def test_ported_two_generations_config_expands():
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "v2ecoli", "configs")
    cfg = load_config_with_inheritance(os.path.join(cfg_dir, "two_generations.json"))
    assert cfg["generations"] == 2
    assert cfg["n_init_sims"] == 2
    branches = expand_branches(cfg)
    # no variants block → baseline only × 2 seeds
    assert len(branches) == 2
    assert {b.seed for b in branches} == {0, 1}


def test_two_generations_config_has_multiscale_analyses():
    import os
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "v2ecoli", "configs")
    cfg = load_config_with_inheritance(os.path.join(cfg_dir, "two_generations.json"))
    opts = cfg["analysis_options"]
    assert "mass_fraction_summary" in opts["single"]
    assert "mass_growth_across_generations" in opts["multigeneration"]
    assert "doubling_time_distribution" in opts["multiseed"]
