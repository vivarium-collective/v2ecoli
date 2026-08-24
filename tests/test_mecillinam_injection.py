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


# --- Generate-time re-injection (ecoli_baseline path) ----------------------
# The ecoli_baseline composite builds the CANDIDATE from a pre-built cache
# bundle (load_cache_bundle) and never re-runs LoadSimData, so the antibiotic
# flags never reached its bulk store. `inject_antibiotic_bulk_species`
# re-applies the SAME injection (single-sourced with LoadSimData) onto the
# bundle-loaded *columnar* initial-state bulk array — count 0, correct submass,
# no ParCa rebuild. These tests cover that helper directly against the columnar
# bulk store the bundle actually carries.

SUBMASS_COLS_HINT = 'metabolite_submass'


def _columnar_bulk(sim_data, **loader_kwargs):
    """Build the columnar initial-state ``bulk`` array (the bundle format) for a
    given sim_data via LoadSimData.generate_initial_state()."""
    from v2ecoli.library.sim_data import LoadSimData
    loader = LoadSimData(sim_data=sim_data, **loader_kwargs)
    return loader.generate_initial_state()['bulk']


@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason=f'fixture absent at {FIXTURE_PATH}')
def test_generate_path_reinjection_matches_loadsimdata():
    """`inject_antibiotic_bulk_species(mecillinam=True)` on the bundle-format
    columnar bulk store adds the three species with masses IDENTICAL to what
    LoadSimData(mecillinam=True) produces, and averts the bulk_names crash."""
    from v2ecoli.library.sim_data import inject_antibiotic_bulk_species
    from v2ecoli.library.schema import bulk_name_to_idx
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state,
    )

    state = load_parca_state(str(FIXTURE_PATH))

    # Baseline columnar bulk WITHOUT any antibiotic flag (the cache-bundle shape).
    bulk_off = _columnar_bulk(hydrate_sim_data_from_state(state))
    assert SUBMASS_COLS_HINT in bulk_off.dtype.names, (
        'columnar bulk store must carry per-submass columns')
    off_ids = list(bulk_off['id'])
    for name in MEC_SPECIES:
        assert name not in off_ids

    # Ground truth: LoadSimData(mecillinam=True) columnar bulk (separate sim_data
    # so its in-place bulk_molecules mutation does not leak into bulk_off).
    bulk_truth = _columnar_bulk(
        hydrate_sim_data_from_state(state), mecillinam=True)
    truth_by_id = {row['id']: row for row in bulk_truth}

    # Re-inject onto the plain columnar bulk store (the ecoli_baseline path).
    bulk_re = inject_antibiotic_bulk_species(bulk_off, mecillinam=True)
    re_by_id = {row['id']: row for row in bulk_re}

    submass_cols = [n for n in bulk_re.dtype.names if n.endswith('_submass')]
    for name in MEC_SPECIES:
        assert name in re_by_id, f'{name} missing after re-injection'
        # Counts start at 0, exactly like the LoadSimData injection.
        assert re_by_id[name]['count'] == 0
        # Submass columns match LoadSimData(mecillinam=True) to floating tol.
        for col in submass_cols:
            assert re_by_id[name][col] == pytest.approx(
                truth_by_id[name][col], rel=1e-9, abs=1e-30), (
                f'{name}.{col} must match the LoadSimData injection')

    # The exact crash the final_mec candidate arm hits: bulk_name_to_idx over a
    # bulk store MISSING the species raises; over the re-injected store it does
    # not. This is the acceptance condition (no ValueError: Names not found).
    with pytest.raises(ValueError, match='Names not found in bulk_names'):
        bulk_name_to_idx(list(MEC_SPECIES), bulk_off['id'])
    idx = bulk_name_to_idx(list(MEC_SPECIES), bulk_re['id'])
    assert len(idx) == len(MEC_SPECIES)
    # PBP2 target still present exactly once (not duplicated by re-injection).
    assert list(bulk_re['id']).count(PBP2_ID) == 1


@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason=f'fixture absent at {FIXTURE_PATH}')
def test_generate_path_reinjection_amp_lysis_and_idempotent():
    """amp_lysis re-injection adds the ampicillin species; re-applying either
    flag is idempotent (no duplicate rows); no flags is a no-op passthrough."""
    from v2ecoli.library.sim_data import inject_antibiotic_bulk_species
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state,
    )
    state = load_parca_state(str(FIXTURE_PATH))
    bulk_off = _columnar_bulk(hydrate_sim_data_from_state(state))

    # No flags = identity passthrough (same object).
    assert inject_antibiotic_bulk_species(bulk_off) is bulk_off

    amp_species = ('ampicillin[p]', 'ampicillin_hydrolyzed[p]')
    bulk_amp = inject_antibiotic_bulk_species(bulk_off, amp_lysis=True)
    for name in amp_species:
        assert name in list(bulk_amp['id'])

    # Both flags together add all five species.
    bulk_both = inject_antibiotic_bulk_species(
        bulk_off, mecillinam=True, amp_lysis=True)
    both_ids = list(bulk_both['id'])
    for name in (*MEC_SPECIES, *amp_species):
        assert name in both_ids

    # Idempotent: re-applying does not duplicate already-present species.
    bulk_twice = inject_antibiotic_bulk_species(bulk_both, mecillinam=True,
                                                amp_lysis=True)
    for name in (*MEC_SPECIES, *amp_species):
        assert list(bulk_twice['id']).count(name) == 1
