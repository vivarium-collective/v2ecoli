"""Tests for ``v2ecoli.composite.save_sim_input``.

The function takes a live ``SimulationDataEcoli`` and writes the
simulation-input bundle (initial_state.json + sim_data_cache.dill +
metadata.json + cache_version) directly — skipping the ~300 MB dill
round-trip that ``save_cache(sim_data_path, ...)`` performs to load
the same object from disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.fast


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / 'models' / 'parca' / 'parca_state.pkl.gz'


@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason=f'fixture absent at {FIXTURE_PATH}')
def test_save_sim_input_writes_full_bundle(tmp_path):
    """``save_sim_input`` produces all four bundle artifacts a downstream
    ``build_composite(cache_dir=...)`` expects."""
    from v2ecoli.core import save_sim_input
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state,
    )

    state = load_parca_state(str(FIXTURE_PATH))
    sim_data = hydrate_sim_data_from_state(state)

    bundle = tmp_path / 'bundle'
    save_sim_input(sim_data, str(bundle))

    expected = {'initial_state.json', 'sim_data_cache.dill',
                'metadata.json', 'cache_version.json'}
    actual = {p.name for p in bundle.iterdir()}
    missing = expected - actual
    assert not missing, f'bundle is missing files: {missing}'
    for name in expected:
        path = bundle / name
        assert path.stat().st_size > 0, f'{name} is empty'


def test_load_sim_data_rejects_no_input():
    """``LoadSimData`` requires at least one of sim_data_path or sim_data."""
    from v2ecoli.library.sim_data import LoadSimData

    with pytest.raises(ValueError, match='sim_data_path or sim_data'):
        LoadSimData()
