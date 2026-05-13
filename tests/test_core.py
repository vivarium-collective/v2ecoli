"""Unit tests for v2ecoli.core."""

import os
import tempfile

import pytest


@pytest.mark.fast
def test_build_core_returns_core_with_ecoli_types():
    from v2ecoli.core import build_core
    from v2ecoli.types import ECOLI_TYPES
    core = build_core()
    # Spot-check that ECOLI_TYPES is registered — pick a known custom type
    # name from ECOLI_TYPES.
    sample_type = next(iter(ECOLI_TYPES))
    assert sample_type in core.registry


@pytest.mark.fast
def test_load_cache_bundle_returns_expected_keys(tmp_path, monkeypatch):
    """If the test fixture cache exists, load_cache_bundle returns a dict
    with the expected keys. If it doesn't exist (CI not yet primed),
    skip."""
    from v2ecoli.core import load_cache_bundle
    cache_dir = "tests/fixtures/cache"
    if not os.path.isdir(cache_dir):
        pytest.skip("test fixture cache not present")
    bundle = load_cache_bundle(cache_dir)
    assert isinstance(bundle, dict)
    # The bundle is created by save_cache; sanity-check it has at least
    # one of the keys that downstream generators consume.
    assert any(k in bundle for k in ("configs", "unique_names", "dry_mass_inc_dict", "initial_state"))


@pytest.mark.fast
def test_save_cache_writes_files(tmp_path, monkeypatch):
    """save_cache writes a sim_data_cache.dill + cache_version.json
    inside the target dir. Use the committed ParCa fixture as the source.

    The parca_state.pkl.gz is a gzipped parca state dict, not a plain
    simData pickle. We hydrate it into a plain dill file first, which is
    what save_cache (and LoadSimData) expect.
    """
    import dill
    from v2ecoli.core import save_cache
    fixture_path = "models/parca/parca_state.pkl.gz"
    if not os.path.exists(fixture_path):
        pytest.skip("ParCa fixture missing")
    # Hydrate the gzipped parca state into a plain simData pickle
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state,
        load_parca_state,
    )
    state = load_parca_state(fixture_path)
    sim_data = hydrate_sim_data_from_state(state)
    sd_path = str(tmp_path / "simData.cPickle")
    with open(sd_path, "wb") as f:
        dill.dump(sim_data, f)
    target = tmp_path / "cache"
    save_cache(sim_data_path=sd_path, cache_dir=str(target), seed=0)
    assert (target / "sim_data_cache.dill").exists()
    assert (target / "cache_version.json").exists()
