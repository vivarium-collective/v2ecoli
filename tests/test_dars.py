"""Phase 7 — DARS1/2 reactivation closes the DnaA nucleotide cycle.

The DARS Step in ``v2ecoli/processes/dars.py`` releases ADP from
DnaA-ADP, regenerating apo-DnaA. The still-active equilibrium reaction
``MONOMER0-160_RXN`` then re-loads apo-DnaA with cellular ATP, giving
back DnaA-ATP. Paired with Phase 5 (RIDA), this closes the cycle and
stabilizes the DnaA-ATP fraction inside the literature band.

Tests pin down:

  * The DARS Step is wired into the cell_state.
  * After ~5 min of sim, the DnaA-ATP fraction is solidly inside the
    literature band (30–70%) and stays there.
  * DnaA-ADP no longer monotonically grows — the cycle has closed.
  * The dars listener emits the per-tick reactivation flux.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


CACHE_DIR = 'out/cache'

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
        reason=f'cache dir {CACHE_DIR!r} not present; '
               f'rebuild with `python scripts/build_cache.py`',
    ),
]


@pytest.fixture(scope='module')
def composite():
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return make_replication_initiation_composite(cache_dir=CACHE_DIR, seed=0)


def _bulk_count(state, mol_id):
    bulk = state['agents']['0']['bulk']
    ids = bulk['id']
    counts = bulk['count']
    needle = mol_id.encode() if isinstance(ids[0], bytes) else mol_id
    matches = np.where(ids == needle)[0]
    assert len(matches) == 1, f'{mol_id} not found in bulk'
    return int(counts[matches[0]])


def _atp_fraction(state):
    apo = _bulk_count(state, 'PD03831[c]')
    atp = _bulk_count(state, 'MONOMER0-160[c]')
    adp = _bulk_count(state, 'MONOMER0-4565[c]')
    total = apo + atp + adp
    return atp / total if total else 0.0


def test_dars_step_in_cell_state(composite):
    cell = composite.state['agents']['0']
    assert 'dars' in cell, (
        f'dars step missing from cell_state; keys: '
        f'{[k for k in cell if not k.startswith("_")][:10]}...')


def test_atp_fraction_stabilizes_inside_literature_band(composite):
    """After 5 minutes the cycle has closed: DnaA-ATP fraction sits
    inside [0.30, 0.70] and is no longer trending toward zero."""
    composite.run(60.0)
    early = _atp_fraction(composite.state)
    composite.run(240.0)  # 4 more minutes -> t=300s total
    late = _atp_fraction(composite.state)
    assert 0.30 <= late <= 0.70, (
        f'DnaA-ATP fraction {late:.2f} outside literature band [0.30, 0.70] '
        f'at t=300s. RIDA + DARS should hold it inside the band.')
    # Phase 5 alone monotonically depleted DnaA-ATP toward 0 over this
    # window; with DARS the fraction should not drop more than ~30 pts.
    assert late >= early - 0.30, (
        f'DnaA-ATP fraction collapsed from {early:.2f} to {late:.2f}; '
        f'DARS may not be regenerating fast enough.')


def test_dars_listener_emits(composite):
    listeners = composite.state['agents']['0'].get('listeners', {})
    dars = listeners.get('dars', {})
    assert 'flux_adp_to_apo' in dars
    assert 'rate_constant' in dars
    assert dars['rate_constant'] > 0


def test_dnaA_adp_does_not_monotonically_grow(composite):
    """Without DARS, DnaA-ADP rises monotonically and DnaA-ATP depletes
    to zero. With DARS wired, the ADP pool is bounded — over a 5 min
    window the count should plateau, not double."""
    composite.run(60.0)
    adp_early = _bulk_count(composite.state, 'MONOMER0-4565[c]')
    composite.run(240.0)
    adp_late = _bulk_count(composite.state, 'MONOMER0-4565[c]')
    # Without DARS, adp_late would be much greater than adp_early.
    # Allow modest growth (< 50%) but flag a bounded steady state.
    assert adp_late <= adp_early * 1.5 + 20, (
        f'DnaA-ADP grew from {adp_early} to {adp_late}; DARS may not be '
        f'releasing ADP fast enough to balance RIDA flux.')
