"""Unit tests for v2ecoli.composites.colony."""

import os

import pytest


@pytest.mark.fast
def test_colony_function_is_registered():
    from pbg_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import colony  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "colony" in names


@pytest.mark.fast
def test_colony_function_signature():
    """The generator takes (core, *, seed, cache_dir, n_cells, env_size,
    physics_interval, ecoli_interval)."""
    import inspect
    from v2ecoli.composites.colony import colony
    sig = inspect.signature(colony)
    assert set(sig.parameters) == {
        "core", "seed", "cache_dir",
        "n_cells", "env_size",
        "physics_interval", "ecoli_interval",
    }


@pytest.mark.sim
def test_colony_returns_a_document():
    """End-to-end: call colony() with the test fixture cache; the document
    has a 'state' key with 'cells' and 'multibody' wired in."""
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; build via "
                    "`python scripts/build_cache.py`")
    try:
        from viva_munk import core_import  # noqa: F401
    except ImportError:
        pytest.skip("viva_munk package not installed; colony requires it")

    from v2ecoli.composites.colony import colony
    doc = colony(seed=0, cache_dir="out/cache", n_cells=1)
    assert isinstance(doc, dict)
    assert "state" in doc
    assert "cells" in doc["state"]
    assert "multibody" in doc["state"]
