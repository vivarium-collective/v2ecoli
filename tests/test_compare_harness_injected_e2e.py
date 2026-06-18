import json, os
from scripts._compare.inject import resolve_injections
from scripts.compare_harness import build_injected_v2_config

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")
CFG = os.path.join(FORK, "configs", "example.json")


def test_build_injected_v2_config_embeds_block():
    with open(CFG) as f:
        vecoli_cfg = json.load(f)
    v2 = build_injected_v2_config(vecoli_cfg, fork_repo=FORK)
    inj = v2["injected_processes"]
    assert inj["fork_repo"] == FORK
    assert inj["add_processes"] == ["example-secretion"]
    # resolution succeeds against the fixture (fail-fast guards pass)
    specs = resolve_injections(FORK, inj)
    assert specs[0]["name"] == "example-secretion"
