"""Unit tests for v2ecoli.composites.reconciled."""

import os

import pytest


@pytest.mark.fast
def test_reconciled_function_is_registered():
    from pbg_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import reconciled  # noqa: F401
    names = {e.name for e in _REGISTRY.values()}
    assert "reconciled" in names


@pytest.mark.fast
def test_reconciled_function_signature():
    import inspect
    from v2ecoli.composites.reconciled import reconciled
    sig = inspect.signature(reconciled)
    assert set(sig.parameters) == {"core", "seed", "cache_dir"}


@pytest.mark.sim
def test_reconciled_returns_a_document():
    if not os.path.isdir("tests/fixtures/cache"):
        pytest.skip("test fixture cache not present")
    from v2ecoli.core import build_core
    from v2ecoli.composites.reconciled import reconciled
    core = build_core()
    doc = reconciled(core=core, seed=0, cache_dir="tests/fixtures/cache")
    assert isinstance(doc, dict)
    assert len(doc) > 0
