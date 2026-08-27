# tests/test_config_to_composite.py
import os, sys, pytest

FORK = "/Users/eranagmon/code/vEcoli-private"

# Fork enrichment needs the FORK's `ecoli` package to win the import race for
# the shared `ecoli` module name. `v2ecoli`'s own package __init__ transitively
# imports `ecoli.processes` too (its own engine dependency) the moment
# `v2ecoli` (below) is first imported, which would otherwise permanently bind
# `sys.modules["ecoli"]` to that *other* install before the fork-backed test
# gets a chance to run. Doing the fork import here, before `v2ecoli` is ever
# touched, ensures every test in this session (and this file's sibling test
# modules, collected afterward) sees the fork's registries.
if os.path.isdir(FORK):
    if FORK not in sys.path:
        sys.path.insert(0, FORK)
    import ecoli.processes  # noqa: F401 — fork registry must win the import race

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
