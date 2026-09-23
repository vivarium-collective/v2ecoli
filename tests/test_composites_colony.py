"""Unit tests for v2ecoli.composites.ecoli_colony."""

import os

import pytest


@pytest.mark.fast
def test_colony_function_is_registered():
    from viva_superpowers.composite_generator import _REGISTRY
    from v2ecoli.composites import ecoli_colony  # noqa: F401 — fires decorator
    names = {e.name for e in _REGISTRY.values()}
    assert "ecoli_colony" in names


@pytest.mark.fast
def test_colony_function_signature():
    """The generator takes (core, *, seed, cache_dir, n_cells, env_size,
    physics_interval, ecoli_interval, transport)."""
    import inspect
    from v2ecoli.composites.ecoli_colony import colony
    sig = inspect.signature(colony)
    assert set(sig.parameters) == {
        "core", "seed", "cache_dir",
        "n_cells", "env_size",
        "physics_interval", "ecoli_interval",
        "transport",
    }


@pytest.mark.fast
def test_colony_transport_defaults_local_and_is_forwarded():
    """backlog item 88: make_colony_document already implements 'local'/'ray'
    transport; colony() must forward it, not silently hardcode 'local' --
    the whole point of exposing it through the registered composite."""
    import inspect
    from v2ecoli.composites.ecoli_colony import colony
    sig = inspect.signature(colony)
    assert sig.parameters["transport"].default == "local"

    import v2ecoli.composites.ecoli_colony as mod
    calls = []
    original = mod.make_colony_document
    try:
        mod.make_colony_document = lambda **kw: (calls.append(kw) or {"cells": {}})
        colony(core=object(), n_cells=1, transport="ray")
    finally:
        mod.make_colony_document = original
    assert calls[0]["transport"] == "ray"


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

    from v2ecoli.composites.ecoli_colony import colony
    doc = colony(seed=0, cache_dir="out/cache", n_cells=1)
    assert isinstance(doc, dict)
    assert "state" in doc
    assert "cells" in doc["state"]
    assert "multibody" in doc["state"]
