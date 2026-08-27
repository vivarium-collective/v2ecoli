# tests/test_config_to_composite.py
import os, sys, pytest

# NOTE: the fork-first `ecoli` import (winning the sys.modules race against
# v2ecoli's own site-packages `ecoli` dependency) happens in tests/conftest.py,
# at conftest-import time — before any test module (this one included) is
# collected. That's required because conftest.py is the only place guaranteed
# to run ahead of collection order (e.g. test_config_bigraph.py sorts before
# this file alphabetically and would otherwise win the race first).

FORK = "/Users/eranagmon/code/vEcoli-private"

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


@pytest.mark.skipif(not os.path.isdir(FORK), reason="vEcoli-private fork absent")
def test_fork_enriches_address_and_registry_ports():
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    import ecoli.processes  # noqa: F401 — fork registry must load first
    from v2ecoli.library.config_to_composite import config_to_composite
    cfg = {"add_processes": ["pg-shape"], "topology": {}}  # no config topology
    node = config_to_composite(cfg, fork_dir=FORK)["state"]["pg-shape"]
    assert node["address"] == "local:PGShape"                 # real class name
    assert set(node["inputs"]) == {"bulk", "environment", "listeners"}  # from registry
