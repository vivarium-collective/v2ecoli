"""Phase 1 — DnaA-ATP / DnaA-ADP nucleotide-pool partitioning.

Both nucleotide-bound forms of DnaA are already registered as bulk
molecules (``PD03831`` apo-DnaA, ``MONOMER0-160`` DnaA-ATP,
``MONOMER0-4565`` DnaA-ADP) and equilibrium reactions for both forms
fire every step (``MONOMER0-160_RXN`` and ``MONOMER0-4565_RXN`` in
``flat/equilibrium_reactions.tsv``). What Phase 1 wires is a listener
that exposes the three pool counts and a behavior test that confirms
the equilibrium actually drains apo-DnaA into the nucleotide-bound
forms over a short sim window.

DnaA-ADP stays near zero in this test because Phase 5 hasn't landed yet
— RIDA flux is the only source of DnaA-ADP, and its FBA flux is
currently zero.
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


def _listener_field(state, *path):
    listeners = state['agents']['0'].get('listeners', {})
    out = listeners
    for p in path:
        out = out[p]
    return out


# ---------------------------------------------------------------------------
# A. Init state — apo-DnaA exists, ATP/ADP forms haven't formed yet
# ---------------------------------------------------------------------------

def test_init_state_has_apo_dnaA_no_bound_forms(composite):
    """Before equilibrium runs, apo-DnaA holds the entire DnaA pool."""
    s = composite.state
    apo = _bulk_count(s, 'PD03831[c]')
    atp = _bulk_count(s, 'MONOMER0-160[c]')
    adp = _bulk_count(s, 'MONOMER0-4565[c]')
    assert apo > 0, 'expected positive apo-DnaA count at init'
    assert atp == 0, f'expected DnaA-ATP=0 at init, got {atp}'
    assert adp == 0, f'expected DnaA-ADP=0 at init, got {adp}'


# ---------------------------------------------------------------------------
# B. After a short sim — apo-DnaA drains into DnaA-ATP via equilibrium
# ---------------------------------------------------------------------------

def test_equilibrium_partitions_apo_into_nucleotide_bound_forms(composite):
    """One minute of sim is enough for the equilibrium reaction
    MONOMER0-160_RXN to drain apo-DnaA into the ATP-bound form. With
    Phase 5 (RIDA) also wired in this architecture, the ADP-bound form
    accumulates too."""
    composite.run(60.0)
    s = composite.state
    apo = _bulk_count(s, 'PD03831[c]')
    atp = _bulk_count(s, 'MONOMER0-160[c]')
    adp = _bulk_count(s, 'MONOMER0-4565[c]')
    assert apo == 0, (
        f'expected apo-DnaA drained by equilibrium MONOMER0-160_RXN, '
        f'got apo={apo} (atp={atp}, adp={adp})')
    assert atp + adp >= 100, (
        f'expected the nucleotide-bound DnaA pool to dominate; '
        f'atp+adp={atp + adp} (apo={apo})')
    assert adp > 0, (
        f'DnaA-ADP=0 after 60s; RIDA (Phase 5) should be producing '
        f'DnaA-ADP from DnaA-ATP at a rate ∝ active replisomes')


# ---------------------------------------------------------------------------
# C. The replication_data listener emits all three pool counts
# ---------------------------------------------------------------------------

def test_listener_emits_dnaA_pool_counts(composite):
    """``listeners.replication_data`` exposes apo / ATP / ADP pool counts
    so the trajectory can be analyzed without re-reading the bulk array."""
    s = composite.state
    rd = _listener_field(s, 'replication_data')
    assert 'dnaA_apo_count' in rd, \
        f'replication_data listener missing dnaA_apo_count; keys: {list(rd.keys())}'
    assert 'dnaA_atp_count' in rd
    assert 'dnaA_adp_count' in rd

    # The listener values match the bulk array within rounding from
    # in-tick step ordering — the listener captures values at its slot
    # in flow_order, while RIDA (later in the flow) modifies the bulk
    # before we read it from composite.state. ±5 molecules absorbs that.
    assert abs(rd['dnaA_apo_count'] - _bulk_count(s, 'PD03831[c]')) <= 5
    assert abs(rd['dnaA_atp_count'] - _bulk_count(s, 'MONOMER0-160[c]')) <= 5
    assert abs(rd['dnaA_adp_count'] - _bulk_count(s, 'MONOMER0-4565[c]')) <= 5


# ---------------------------------------------------------------------------
# D. Conservation across the equilibrium reaction
# ---------------------------------------------------------------------------

def test_total_dnaA_pool_size_is_consistent_with_translation(composite):
    """The sum apo + ATP-bound + ADP-bound is the total DnaA monomer count
    (modulo box-bound DnaA, which Phase 2 will track). Until Phase 2 lands
    DnaA isn't sequestered onto chromosomal boxes, so sum ≈ translated
    DnaA count over the sim window."""
    s = composite.state
    apo = _bulk_count(s, 'PD03831[c]')
    atp = _bulk_count(s, 'MONOMER0-160[c]')
    adp = _bulk_count(s, 'MONOMER0-4565[c]')
    total = apo + atp + adp
    # Initial DnaA monomer count was 124. Translation produces more
    # over 60s, but the total shouldn't spuriously shrink below ~100.
    assert total >= 100, (
        f'total DnaA monomer count dropped to {total} '
        f'(apo={apo}, atp={atp}, adp={adp}); equilibrium reactions may '
        f'be losing molecules to a side branch.')
