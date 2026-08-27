# tests/test_config_to_composite.py
"""Pure-logic tests for the config→composite translator (no fork required).

Fork-backed executability tests (register + Composite-realize against real
vEcoli-fork antibiotic processes/configs) live downstream in sms-ecoli, where
the fork and its configs are wired — they cannot run in a generic v2ecoli
checkout and must not hardcode a private-fork path here.
"""
from v2ecoli.library.config_to_composite import config_to_composite


def _cfg():
    return {
        "add_processes": ["proc_a"],
        "swap_processes": {"old_m": "new_m"},
        "process_configs": {"proc_a": {"rate": 3}},
        "topology": {"proc_a": {"bulk": ["bulk"]}, "new_m": {"flux": ["metabolites"]}},
    }


def test_process_nodes_are_address_based_and_executable_shape():
    doc = config_to_composite(_cfg())
    assert set(doc) == {"schema", "state"}
    node = doc["state"]["proc_a"]
    assert node["_type"] == "process"
    assert node["address"] == "local:proc_a"      # local:<name> when no fork enriches
    assert node["config"] == {"rate": 3}
    assert "_draft" not in node                    # executable, not a draft view


def test_swap_node_annotated_and_present():
    state = config_to_composite(_cfg())["state"]
    assert state["new_m"]["_type"] == "process"
    assert state["new_m"]["_contract"]["swap_replaces"] == "old_m"


def test_store_nodes_exist_for_wire_targets():
    state = config_to_composite(_cfg())["state"]
    assert state["bulk"] == {}
    assert state["metabolites"] == {}
