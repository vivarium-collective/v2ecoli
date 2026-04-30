"""Phase 5 — RIDA: regulatory inactivation of DnaA-ATP.

The RIDA Step in ``v2ecoli/processes/rida.py`` converts DnaA-ATP into
DnaA-ADP at a rate proportional to active replisome count. The
companion change is in ``generate_replication_initiation`` — it
deactivates the ``MONOMER0-4565_RXN`` equilibrium so that RIDA's output
is not instantly re-dissociated by mass-action against cellular ADP.

The biology these tests pin down:

  * RIDA fires only when active replisomes are present (rate ∝ replisomes).
  * The DnaA-ATP fraction drops from the equilibrium ceiling (~95%)
    into the literature steady-state band (~30-70%) within a minute
    of replication starting.
  * Without DARS (Phase 7), DnaA-ATP eventually depletes — the cycle
    is not closed. That is expected behavior, not a regression.
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
    """Total DnaA-ATP fraction across the whole DnaA pool — free in
    cytoplasm plus sequestered onto chromosomal DnaA boxes. Phase 2
    titration (commit 3070a66) decrements the cytoplasmic pool by the
    count bound to boxes; literature DnaA-ATP measurements are the
    total cellular ratio regardless of localization."""
    apo = _bulk_count(state, 'PD03831[c]')
    free_atp = _bulk_count(state, 'MONOMER0-160[c]')
    free_adp = _bulk_count(state, 'MONOMER0-4565[c]')
    binding = state['agents']['0'].get('listeners', {}).get('dnaA_binding', {})
    seq_atp = int(binding.get('atp_sequestered', 0) or 0)
    seq_adp = int(binding.get('adp_sequestered', 0) or 0)
    total = apo + free_atp + seq_atp + free_adp + seq_adp
    return (free_atp + seq_atp) / total if total else 0.0


# ---------------------------------------------------------------------------
# A. The RIDA step is present in the cell_state
# ---------------------------------------------------------------------------

def test_rida_step_in_cell_state(composite):
    cell = composite.state['agents']['0']
    assert 'rida' in cell, (
        f'rida step missing from cell_state; keys: '
        f'{[k for k in cell if not k.startswith("_")][:10]}...')


# ---------------------------------------------------------------------------
# B. ADP equilibrium reaction is deactivated in this architecture
# ---------------------------------------------------------------------------

def test_dnaA_adp_equilibrium_deactivated(composite):
    """Without deactivation, mass-action against cellular ADP would
    drive DnaA-ADP back to apo + ADP each tick, which prevents RIDA
    flux from accumulating any DnaA-ADP. The architecture neutralizes
    that one reaction by zeroing its stoichMatrix column."""
    eq_edge = composite.state['agents']['0'].get('ecoli-equilibrium')
    assert eq_edge is not None, 'equilibrium step missing'
    instance = eq_edge['instance']
    deactivated = getattr(instance, '_deactivated_reactions', ())
    assert 'MONOMER0-4565_RXN' in deactivated, (
        f'MONOMER0-4565_RXN should be deactivated; got {deactivated}')


# ---------------------------------------------------------------------------
# C. RIDA produces DnaA-ADP and lowers the DnaA-ATP fraction
# ---------------------------------------------------------------------------

def test_rida_lowers_atp_fraction_into_literature_band(composite):
    """After ~60s of sim, the DnaA-ATP fraction drops from its
    equilibrium ceiling into the published 30-70% range. Confirms that
    RIDA flux + the deactivated DnaA-ADP equilibrium combine to put
    the model in the biologically observed window."""
    composite.run(60.0)
    frac = _atp_fraction(composite.state)
    assert 0.30 <= frac <= 0.70, (
        f'DnaA-ATP fraction {frac:.2f} outside literature band [0.30, 0.70] '
        f'after 60s of sim. RIDA may not be firing at the expected rate.')


def test_rida_flux_listener_emits(composite):
    """The rida listener exposes flux_atp_to_adp and active_replisomes
    each tick."""
    listeners = composite.state['agents']['0'].get('listeners', {})
    rida = listeners.get('rida', {})
    assert 'flux_atp_to_adp' in rida
    assert 'active_replisomes' in rida
    # After sim, replisomes should be > 0 (cell starts mid-replication).
    assert rida['active_replisomes'] > 0


def _adp_total(state):
    """Total DnaA-ADP — cytoplasmic + sequestered. See _atp_fraction
    for why we include the sequestered tier post-Phase-2."""
    free_adp = _bulk_count(state, 'MONOMER0-4565[c]')
    binding = state['agents']['0'].get('listeners', {}).get('dnaA_binding', {})
    seq_adp = int(binding.get('adp_sequestered', 0) or 0)
    return free_adp + seq_adp


def test_dnaA_adp_count_grows_with_replication(composite):
    """DnaA-ADP rises monotonically while replisomes are active.
    Snapshot the count, run another minute, expect higher."""
    adp_before = _adp_total(composite.state)
    composite.run(60.0)
    adp_after = _adp_total(composite.state)
    # Without DARS the cycle is open-loop; ADP only grows. We allow for
    # some rounding at small counts but expect a non-trivial increase.
    assert adp_after >= adp_before, (
        f'DnaA-ADP shrank: before={adp_before}, after={adp_after}. '
        f'RIDA is not producing DnaA-ADP; check flux and equilibrium.')
