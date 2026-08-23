"""Tests for the mecillinam species injection in ``LoadSimData``.

Mirrors the ampicillin ``amp_lysis`` path: when ``LoadSimData`` is
constructed with ``mecillinam=True`` it injects the bulk species
``mecillinam[p]``, ``mecillinam_hydrolyzed[p]`` and the drug-target
complex ``mecillinam[p]-EG10606-MONOMER[i]`` into the bulk molecules
store at load time (no ParCa rebuild). This unblocks the mecillinam
antibiotic config (e.g. final_mec candidate arm), whose
``antibiotic_transport_odeint`` process otherwise raises
``ValueError: Names not found in bulk_names: ['mecillinam[p]', ...]``.

Uses the committed ParCa-state fixture (same source as
``tests/test_save_sim_input.py``) so the test is cache-light.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / 'models' / 'parca' / 'parca_state.pkl.gz'

MEC_SPECIES = (
    'mecillinam[p]',
    'mecillinam_hydrolyzed[p]',
    'mecillinam[p]-EG10606-MONOMER[i]',
)
PBP2_ID = 'EG10606-MONOMER[i]'


def _bulk_ids(loader):
    return list(
        loader.sim_data.internal_state.bulk_molecules.bulk_data.fullArray()['id']
    )


def _bulk_array(loader):
    return loader.sim_data.internal_state.bulk_molecules.bulk_data.fullArray()


@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason=f'fixture absent at {FIXTURE_PATH}')
def test_mecillinam_injection_adds_species():
    """``mecillinam=True`` injects the three species; ``False`` does not."""
    from v2ecoli.library.sim_data import LoadSimData
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state,
    )

    state = load_parca_state(str(FIXTURE_PATH))
    sim_data = hydrate_sim_data_from_state(state)

    # Default (mecillinam=False): none of the injected species present.
    # Run first because it does not mutate the bulk store.
    loader_off = LoadSimData(sim_data=sim_data)
    off_ids = _bulk_ids(loader_off)
    for name in MEC_SPECIES:
        assert name not in off_ids, (
            f'{name} should be absent when mecillinam=False'
        )
    # The free PBP2 target already exists in the baseline bulk store.
    assert PBP2_ID in off_ids, f'{PBP2_ID} (PBP2) missing from baseline bulk store'

    # mecillinam=True: all three species injected.
    loader_on = LoadSimData(sim_data=sim_data, mecillinam=True)
    on_arr = _bulk_array(loader_on)
    on_ids = list(on_arr['id'])
    for name in MEC_SPECIES:
        assert name in on_ids, f'{name} should be present when mecillinam=True'

    # PBP2 target still present exactly once (not re-added, not duplicated).
    assert on_ids.count(PBP2_ID) == 1, f'{PBP2_ID} should not be duplicated'

    # Every injected species carries a finite, strictly positive total mass.
    mass_by_id = {row['id']: np.asarray(row['mass'], dtype=float) for row in on_arr}
    for name in MEC_SPECIES:
        total = mass_by_id[name].sum()
        assert np.isfinite(total), f'{name} mass must be finite, got {total}'
        assert total > 0, f'{name} mass must be positive, got {total}'

    # The hydrolyzed form is heavier than mecillinam by the mass of water (18).
    mec = mass_by_id['mecillinam[p]'].sum()
    mec_hydro = mass_by_id['mecillinam_hydrolyzed[p]'].sum()
    assert mec_hydro == pytest.approx(mec + 18), (
        'hydrolyzed mecillinam should be mecillinam + 18 (water)'
    )

    # The drug-target complex mass is mecillinam mass + free PBP2 monomer mass.
    pbp2 = mass_by_id[PBP2_ID].sum()
    complex_mass = mass_by_id['mecillinam[p]-EG10606-MONOMER[i]'].sum()
    assert complex_mass == pytest.approx(mec + pbp2), (
        'complex mass should equal mecillinam mass + PBP2 monomer mass'
    )
