import json
import pytest
from v2ecoli.workflow.config import load_config_with_inheritance, _merge_configs


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
