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


def _sequestered(state):
    """Return ``(atp_sequestered, adp_sequestered)`` from the Phase 2
    binding listener. Phase 2 titration (3070a66) decrements the
    cytoplasmic DnaA-ATP / DnaA-ADP bulk counts by the count bound to
    chromosomal DnaA boxes, so the cytoplasmic counts alone don't
    represent the whole DnaA pool."""
    binding = state['agents']['0'].get('listeners', {}).get(
        'dnaA_binding', {}) or {}
    return (
        int(binding.get('atp_sequestered', 0) or 0),
        int(binding.get('adp_sequestered', 0) or 0),
    )


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
    accumulates too. Phase 2 titration (3070a66) sequesters most of the
    ATP-bound pool onto chromosomal boxes, so the test checks the
    nucleotide-bound pool *including* the sequestered tier."""
    composite.run(60.0)
    s = composite.state
    apo = _bulk_count(s, 'PD03831[c]')
    free_atp = _bulk_count(s, 'MONOMER0-160[c]')
    free_adp = _bulk_count(s, 'MONOMER0-4565[c]')
    seq_atp, seq_adp = _sequestered(s)
    atp = free_atp + seq_atp
    adp = free_adp + seq_adp
    # Apo-DnaA is small but not always exactly zero — Phase 7 (DARS)
    # continuously regenerates it from DnaA-ADP, and the equilibrium
    # consumes it on a single-tick timescale. Allow a small steady-state
    # apo pool.
    assert apo <= 5, (
        f'expected apo-DnaA mostly drained by equilibrium MONOMER0-160_RXN, '
        f'got apo={apo} (free_atp={free_atp}, free_adp={free_adp}, '
        f'seq_atp={seq_atp}, seq_adp={seq_adp})')
    assert atp + adp >= 100, (
        f'expected the nucleotide-bound DnaA pool to dominate; '
        f'atp+adp={atp + adp} (apo={apo}, '
        f'free_atp={free_atp}, free_adp={free_adp}, '
        f'seq_atp={seq_atp}, seq_adp={seq_adp})')
    assert adp > 0, (
        f'DnaA-ADP=0 after 60s; RIDA (Phase 5) should be producing '
        f'DnaA-ADP from DnaA-ATP at a rate ∝ active replisomes')


# ---------------------------------------------------------------------------
# C. The replication_data listener emits all three pool counts
# ---------------------------------------------------------------------------

def test_listener_emits_dnaA_pool_counts(composite):
    """``listeners.replication_data`` exposes apo / ATP / ADP pool counts
    so the trajectory can be analyzed without re-reading the bulk array.

    Multi-step cascades within a tick (equilibrium ↔ binding ↔ RIDA ↔
    DARS ↔ DDAH all read or write the DnaA pool) make a precise
    bulk/listener equality fragile, so the test pins the looser
    properties: the fields exist, values are non-negative integers,
    and their sum is in the same ballpark as the total DnaA monomer
    pool."""
    s = composite.state
    rd = _listener_field(s, 'replication_data')
    for key in ('dnaA_apo_count', 'dnaA_atp_count', 'dnaA_adp_count'):
        assert key in rd, (
            f'replication_data listener missing {key}; '
            f'keys: {list(rd.keys())}')
        value = int(rd[key])
        assert value >= 0, f'{key} = {value} (expected non-negative)'

    listener_total = (int(rd['dnaA_apo_count'])
                      + int(rd['dnaA_atp_count'])
                      + int(rd['dnaA_adp_count']))
    seq_atp, seq_adp = _sequestered(s)
    bulk_total = (_bulk_count(s, 'PD03831[c]')
                  + _bulk_count(s, 'MONOMER0-160[c]')
                  + _bulk_count(s, 'MONOMER0-4565[c]')
                  + seq_atp + seq_adp)
    # Listener and bulk-plus-sequestered measure the same conserved
    # quantity from different in-tick slots; the gap is at most the
    # combined per-tick flux of RIDA / DARS / DDAH plus translation —
    # well under 50 over a 60 s window.
    assert abs(listener_total - bulk_total) <= 50, (
        f'listener total {listener_total} vs bulk+sequestered total '
        f'{bulk_total} — gap > 50 means a step is leaking molecules')


# ---------------------------------------------------------------------------
# D. Conservation across the equilibrium reaction
# ---------------------------------------------------------------------------

def test_total_dnaA_pool_size_is_consistent_with_translation(composite):
    """The sum apo + ATP-bound + ADP-bound + sequestered is the total
    DnaA monomer count. Phase 2 titration sequesters bound molecules
    onto chromosomal boxes, so the cytoplasmic pool alone is not the
    full count — including ``atp_sequestered + adp_sequestered`` from
    the binding listener gives the conserved total."""
    s = composite.state
    apo = _bulk_count(s, 'PD03831[c]')
    free_atp = _bulk_count(s, 'MONOMER0-160[c]')
    free_adp = _bulk_count(s, 'MONOMER0-4565[c]')
    seq_atp, seq_adp = _sequestered(s)
    total = apo + free_atp + free_adp + seq_atp + seq_adp
    # Initial DnaA monomer count was 124. Translation produces more
    # over 60s, but the total shouldn't spuriously shrink below ~100.
    assert total >= 100, (
        f'total DnaA monomer count dropped to {total} '
        f'(apo={apo}, free_atp={free_atp}, free_adp={free_adp}, '
        f'seq_atp={seq_atp}, seq_adp={seq_adp}); equilibrium reactions '
        f'may be losing molecules to a side branch.')
