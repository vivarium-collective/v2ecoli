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
    """The generator function takes (core, *, seed, cache_dir)."""
    import inspect
    from v2ecoli.composites.baseline import baseline
    sig = inspect.signature(baseline)
    assert set(sig.parameters) == {"core", "seed", "cache_dir"}


@pytest.mark.sim
def test_baseline_returns_a_document():
    """End-to-end: call baseline() with the test fixture cache, assert it
    returns a process-bigraph document dict."""
    if not os.path.isdir("tests/fixtures/cache"):
        pytest.skip("test fixture cache not present")
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir="tests/fixtures/cache")
    assert isinstance(doc, dict)
    assert len(doc) > 0
