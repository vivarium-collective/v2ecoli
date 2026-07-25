"""Tests for the baseline ``media`` parameter — a lightweight media perturbation
from the existing ParCa cache (no per-condition re-fit).

Setting ``media`` to a condition in the cache's ``saved_media`` seeds the
environment's initial ``media_id`` so the ``media_update`` step shifts the cell
onto that condition on the first tick. Build-time tests only (cheap, cache-only);
the growth response is exercised by the demo seed investigation, not here.
"""
from __future__ import annotations

import os

import pytest

from v2ecoli.core import build_core

CACHE = "out/cache"
_needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE),
    reason="ParCa cache (out/cache) required for media resolution")


@_needs_cache
def test_media_sets_initial_environment_media_id():
    """A media condition in saved_media becomes the environment's initial id."""
    from v2ecoli.composites.baseline import baseline
    doc = baseline(core=build_core(), media="minimal_plus_amino_acids")
    env = doc["state"]["agents"]["0"]["environment"]
    assert env["media_id"] == "minimal_plus_amino_acids"


@_needs_cache
def test_media_default_minimal_is_unchanged():
    """The default leaves the cache's own 'minimal' initial condition in place."""
    from v2ecoli.composites.baseline import baseline
    doc = baseline(core=build_core(), media="minimal")
    assert doc["state"]["agents"]["0"]["environment"]["media_id"] == "minimal"


@_needs_cache
def test_unknown_media_fails_at_build():
    """A condition not in saved_media raises at build time, not silently mid-run."""
    from v2ecoli.composites.baseline import baseline
    with pytest.raises(ValueError, match="saved_media"):
        baseline(core=build_core(), media="not_a_real_medium")
