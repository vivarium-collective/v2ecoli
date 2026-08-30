from viva_superpowers.composite_generator import _REGISTRY
import v2ecoli.composites  # noqa: F401 — forces registration + aliases


def _clean_ids():
    return {k for k in _REGISTRY if k.startswith("v2ecoli.composites.")}


def test_both_composites_registered():
    ids = _clean_ids()
    assert "v2ecoli.composites.ecoli_baseline" in ids
    assert "v2ecoli.composites.ecoli_v1_hybrid" in ids


def test_distinct_generator_funcs():
    base = _REGISTRY["v2ecoli.composites.ecoli_baseline"]
    hyb = _REGISTRY["v2ecoli.composites.ecoli_v1_hybrid"]
    assert base.func is not hyb.func  # two real composites, not an alias
