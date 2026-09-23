"""Unit tests for v2ecoli.library.config_bigraph.config_to_document.

Pure-logic coverage: a synthetic config in the generic vEcoli vocabulary
(``add_processes``/``swap_processes``/``topology``/...) maps to the loom's
state-document shape. No fork import, no simulator build, no fixture workspace
— the transform's graph shape needs none of those (fork_dir only enriches
node address/description, which these tests leave at the ``local:`` default).
"""

from __future__ import annotations

from v2ecoli.library.config_bigraph import config_to_document


def _synthetic_config() -> dict:
    return {
        "add_processes": ["proc_a", "proc_b"],
        "swap_processes": {"old_metab": "new_metab"},
        "exclude_processes": ["dropme"],
        "process_configs": {"proc_a": {"rate": 3}},
        "topology": {
            # flat store path -> wired
            "proc_a": {"bulk": ["bulk"], "listeners": ["listeners"]},
            # leading ".." walk-ups stripped to a root store
            "proc_b": {"fields": ["..", "..", "fields"]},
            # nested sub-port dict -> port renders but stays un-wired
            "new_metab": {"deep": {"sub": ["x"]}, "flat": ["metabolites"]},
        },
        "spatial_environment_config": {
            "reaction_diffusion": {"molecules": ["GLC", "drug"]}},
        "variants": {"dose_grid": {"value": [0, 1, 2]}},
    }


def test_process_nodes_created_for_added_and_swapped():
    doc = config_to_document(_synthetic_config())
    state = doc["state"]
    for name in ("proc_a", "proc_b", "new_metab"):
        assert state[name]["_type"] == "process"
        assert state[name]["_draft"] is True
    # swap annotation is preserved on the NEW process
    assert state["new_metab"]["_contract"]["swap_replaces"] == "old_metab"
    assert state["new_metab"]["description"].startswith("SWAP — replaces 'old_metab'")


def test_all_ports_render_from_inputs_schema():
    state = config_to_document(_synthetic_config())["state"]
    # every declared topology port shows up in _inputs (renders even unwired)
    assert set(state["proc_a"]["_inputs"]) == {"bulk", "listeners"}
    assert set(state["new_metab"]["_inputs"]) == {"deep", "flat"}


def test_flat_paths_wire_and_walkups_normalize():
    state = config_to_document(_synthetic_config())["state"]
    # flat paths become wires
    assert state["proc_a"]["inputs"]["bulk"] == ["bulk"]
    # leading ".." are stripped to a root store path
    assert state["proc_b"]["inputs"]["fields"] == ["fields"]
    # a root store node exists for each wire target
    assert state["bulk"] == {}
    assert state["fields"] == {}


def test_nested_subport_renders_but_is_not_wired():
    state = config_to_document(_synthetic_config())["state"]
    node = state["new_metab"]
    assert "deep" in node["_inputs"]          # port renders
    assert "deep" not in node.get("inputs", {})  # but is NOT wired
    assert node["inputs"]["flat"] == ["metabolites"]  # the flat sibling is


def test_process_config_carried_onto_node():
    state = config_to_document(_synthetic_config())["state"]
    assert state["proc_a"]["config"] == {"rate": 3}
    assert state["proc_b"]["config"] == {}  # absent -> empty, not missing


def test_annotation_nodes_present():
    state = config_to_document(_synthetic_config())["state"]
    assert state["excluded_processes"] == {"dropme": {}}
    assert set(state["environment"]) == {"GLC", "drug"}
    assert "dose_grid" in state["variants"]


def test_summary_counts():
    summary = config_to_document(_synthetic_config())["summary"]
    assert summary["n_process_nodes"] == 3   # 2 added + 1 swapped
    assert summary["added"] == ["proc_a", "proc_b"]
    assert summary["swapped"] == {"old_metab": "new_metab"}
    assert summary["has_spatial"] is True
    assert summary["n_variants"] == 1


def test_empty_config_is_safe():
    doc = config_to_document({})
    assert doc["state"] == {}
    assert doc["summary"]["n_process_nodes"] == 0


def test_store_node_never_clobbers_a_process_of_same_name():
    # a process wired to a store named like another process must not overwrite it
    cfg = {
        "add_processes": ["bulk"],  # process literally named "bulk"
        "topology": {"bulk": {"self": ["bulk"]}},
    }
    state = config_to_document(cfg)["state"]
    assert state["bulk"]["_type"] == "process"  # process wins, not an empty store
