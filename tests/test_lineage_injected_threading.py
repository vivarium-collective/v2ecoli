"""Tests that injected_processes config threads from meta_composite -> lineage node."""


def test_meta_composite_carries_injected_processes():
    from v2ecoli.workflow.meta_composite import build_meta_composite
    cfg = {"experiment_id": "x", "n_init_sims": 1, "generations": 1,
           "single_daughters": True, "cache_dir": "out/cache",
           "out_dir": "out/x", "skip_baseline": False,
           "injected_processes": {"fork_repo": "/tmp/fork",
                                  "add_processes": ["example-secretion"]}}
    doc = build_meta_composite(cfg)
    (branch,) = doc["state"]["branches"].values()
    node_cfg = branch["lineage"]["config"]
    assert node_cfg["injected_processes"]["add_processes"] == ["example-secretion"]
