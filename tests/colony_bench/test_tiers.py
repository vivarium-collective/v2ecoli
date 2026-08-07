import pytest
from viva_munk.processes.multibody import make_rng

def test_simple_tier_has_port_contract():
    from v2ecoli.colony_bench.tiers import cell_factory
    rng = make_rng(0)
    aid, body = cell_factory("simple", rng=rng, env_size=30, seed=0,
                             x=15, y=15, angle=0.0)
    assert isinstance(aid, str)
    assert body["id"] == aid
    assert "location" in body and "mass" in body and "length" in body
    # simple tier drives division via an embedded grow_divide process
    assert body["grow_divide"]["_type"] == "process"
    assert body["grow_divide"]["outputs"]["agents"] == ["..", "..", "cells"]

def test_unknown_tier_raises():
    from v2ecoli.colony_bench.tiers import cell_factory
    rng = make_rng(0)
    with pytest.raises(ValueError, match="unknown tier"):
        cell_factory("bogus", rng=rng, env_size=30, seed=0, x=1, y=1, angle=0.0)
