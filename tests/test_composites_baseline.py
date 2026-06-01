"""Unit tests for v2ecoli.composites.baseline."""

import os

import pytest


@pytest.mark.fast
def test_baseline_function_is_registered():
    from pbg_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import baseline  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "baseline" in names


@pytest.mark.fast
def test_baseline_function_signature():
    """The generator takes (core, *, seed, cache_dir, config_overrides, bundle).

    ``bundle`` is an optional pre-loaded cache override so callers that
    need to mutate configs before document construction (e.g.
    scripts/run_plasmid_multiseed.py) can bypass the cache reload.
    """
    import inspect
    from v2ecoli.composites.baseline import baseline
    sig = inspect.signature(baseline)
    assert set(sig.parameters) == {
        "core", "seed", "cache_dir", "config_overrides", "bundle"}


@pytest.mark.sim
def test_baseline_returns_a_document():
    """End-to-end: call baseline() with the test fixture cache, assert it
    returns a process-bigraph document dict."""
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir="out/cache")
    assert isinstance(doc, dict)
    assert len(doc) > 0
