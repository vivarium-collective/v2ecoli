from v2ecoli.core import build_core
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)


def test_build_meta_composite_branches():
    config = {
        "experiment_id": "twovar",
        "n_init_sims": 2,
        "generations": 1,
        "single_daughters": True,
        "cache_dir": "out/cache",
        "out_dir": "out/twovar",
        "variants": {"kcat": {"target": "ecoli-metabolism.kcat", "value": [1, 2]}},
        "skip_baseline": True,
    }
    doc = build_meta_composite(config)
    branches = doc["state"]["branches"]
    assert len(branches) == 4  # 2 variants × 2 seeds

    # Each branch holds a LineageProcess node with the right config wiring.
    sample_key = next(iter(branches))
    node = branches[sample_key]["lineage"]
    assert node["_type"] == "process"
    assert node["address"] == "local:LineageProcess"
    assert node["config"]["generations"] == 1
    assert node["config"]["experiment_id"] == "twovar"
    assert "ecoli-metabolism.kcat" in node["config"]["config_overrides"]


def test_register_workflow_processes_resolves_address():
    core = build_core()
    register_workflow_processes(core)
    # Registration populates the link registry...
    from v2ecoli.workflow.lineage import LineageProcess
    assert core.link_registry["LineageProcess"] is LineageProcess

    # ...and a document referencing local:LineageProcess actually resolves
    # when a Composite is built (the failure mode that matters for the runner).
    from process_bigraph import Composite
    config = {
        "experiment_id": "resolve",
        "n_init_sims": 1,
        "generations": 1,
        "cache_dir": "out/cache",
        "out_dir": "out/resolve",
        "variants": {},
    }
    doc = build_meta_composite(config)
    composite = Composite(doc, core=core)
    branch = composite.state["branches"]["variant=0/seed=0"]
    assert isinstance(branch["lineage"]["instance"], LineageProcess)
