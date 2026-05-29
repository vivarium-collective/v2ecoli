import os
import copy
import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")


def test_override_patches_process_config():
    from v2ecoli.core import build_core, load_cache_bundle
    from v2ecoli.composites.baseline import baseline

    # Pick any process+key actually present in the cached configs.
    bundle = load_cache_bundle(CACHE)
    proc = "ecoli-metabolism"
    cfg = bundle["configs"][proc]
    key = next(k for k, v in cfg.items() if isinstance(v, (int, float)))
    original = copy.deepcopy(cfg[key])
    sentinel = (original if isinstance(original, int) else 0.0)
    sentinel = sentinel + 12345 if isinstance(sentinel, int) else 99999.0

    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir=CACHE,
                   config_overrides={f"{proc}.{key}": sentinel})

    # The override must reach the instantiated process's parameters.
    # make_edge stores the process instance under edge["instance"]; since the
    # process classes do not set _raw_config, edge["config"] is empty ({}).
    # We therefore verify the override via instance.parameters[key].
    edge = doc["state"]["agents"]["0"][proc]
    instance = edge["instance"]
    assert instance.parameters[key] == sentinel

    # The cached bundle must NOT be mutated (lru_cache shares it).
    bundle2 = load_cache_bundle(CACHE)
    assert bundle2["configs"][proc][key] == original
