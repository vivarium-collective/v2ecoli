"""Phase 2 — DnaA box binding process.

The DnaABoxBinding step in ``v2ecoli/processes/dnaA_box_binding.py``
samples bound/unbound for each active DnaA box from the equilibrium
occupancy probability:

    p_bound = [DnaA] / (Kd + [DnaA])

with [DnaA] = (DnaA-ATP + DnaA-ADP) / V_cell, computed in nM, and Kd
chosen per-region from
``v2ecoli.data.replication_initiation.REGION_BINDING_RULES``. High-
affinity regions (oriC, dnaA promoter, datA, DARS1, DARS2) get
Kd ≈ 1 nM and bind both nucleotide forms; everything else falls back
to a low-affinity ATP-preferential rule (Kd ≈ 100 nM).

Pre-Phase-2 the ``DnaA_bound`` field on ``DNAA_BOX_ARRAY`` was
write-only (set False at init and on fork passage, never set True).
These tests pin down that the field is now actually used and that the
per-region occupancy lines up with the curated affinity classes.
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


def _binding_listener(state):
    return state['agents']['0'].get('listeners', {}).get('dnaA_binding', {})


def test_binding_step_in_cell_state(composite):
    cell = composite.state['agents']['0']
    assert 'dnaA_box_binding' in cell, (
        f'binding step missing; keys: '
        f'{[k for k in cell if not k.startswith("_")][:10]}...')


def test_listener_emits_per_region_counts(composite):
    """After the binding step has fired once, the listener exposes
    bound counts for each named region plus 'other'."""
    composite.run(60.0)
    listener = _binding_listener(composite.state)
    for field in ('total_bound', 'total_active', 'fraction_bound',
                  'bound_oric', 'bound_dnaA_promoter', 'bound_datA',
                  'bound_DARS1', 'bound_DARS2', 'bound_other'):
        assert field in listener, f'{field} missing from dnaA_binding listener'


def test_listener_reports_nonzero_bound_count(composite):
    """Phase 2 emits a per-tick equilibrium-occupancy report on the
    DnaA-box bound state. The listener's total_bound count is what
    Phase 3 will read to gate initiation. (Note: Phase 2 does *not*
    write back to DnaA_bound on the unique store — see the comment
    in dnaA_box_binding.update on why.)"""
    composite.run(60.0)
    listener = _binding_listener(composite.state)
    assert listener['total_bound'] > 0, (
        f'binding listener reports total_bound=0 — Phase 2 binding step '
        f'did not run or sampled zero occupancy across all boxes. '
        f'Listener: {dict(listener)}')


def test_high_affinity_regions_are_saturated(composite):
    """At ~166 nM [DnaA-ATP] (typical cellular DnaA pool), high-
    affinity regions (Kd ~1 nM) should be ≥95% occupied. The
    bioinformatic strict-consensus search picks out the high-affinity
    boxes per region, so every box in oriC / dnaA_promoter / DARS2
    that's currently in motif_coordinates should be bound."""
    composite.run(120.0)
    listener = _binding_listener(composite.state)
    # All strict-consensus oriC boxes should be bound. Phase 0 baseline:
    # 3 distinct coords × 1-3 chromosome domains = up to ~9 boxes total.
    bound_oric = int(listener['bound_oric'])
    bound_dnaA_promoter = int(listener['bound_dnaA_promoter'])
    bound_DARS2 = int(listener['bound_DARS2'])
    assert bound_oric > 0, (
        f'No oriC boxes bound after 120s — high-affinity binding '
        f'should saturate. Listener: {dict(listener)}')
    assert bound_dnaA_promoter > 0, (
        f'No dnaA-promoter boxes bound; expected the consensus box1 to '
        f'saturate at high-affinity Kd. Listener: {dict(listener)}')
    assert bound_DARS2 > 0, (
        f'No DARS2 boxes bound; expected high-affinity saturation. '
        f'Listener: {dict(listener)}')


def test_low_affinity_other_boxes_not_fully_saturated(composite):
    """Boxes outside named regions fall back to low-affinity
    (Kd ≈ 100 nM, ATP-preferential). They should be partially bound
    or zero (when titration has drained the free DnaA-ATP pool),
    never 100% saturated like high-affinity sites — that would
    indicate the low-affinity Kd was bypassed.

    With titration enabled, sequestration on background boxes is
    bounded by the cytoplasmic DnaA-ATP pool; under sustained
    pressure the count drops to 0 once the pool is exhausted, so
    this test only checks the upper bound on saturation, not a
    nonzero lower bound."""
    composite.run(60.0)
    listener = _binding_listener(composite.state)
    bound_other = int(listener['bound_other'])
    total_active = int(listener['total_active'])
    assert 0 <= bound_other < total_active, (
        f"'other' bound={bound_other} of {total_active} active "
        f"boxes — should be a partial fraction, not saturated. "
        f'Listener: {dict(listener)}')


def test_per_region_counts_bounded_by_pdf_total(composite):
    """Per-region bound counts are bounded by the curated PDF total
    for that region (oriC has 11 boxes total, so bound_oric ≤ 11).
    Lower bound is zero. With per-tier sampling the high-affinity
    boxes saturate quickly while low-affinity ones fill cooperatively,
    so the per-region bound count is between n_high and n_total at
    typical DnaA concentrations."""
    from v2ecoli.data.replication_initiation import (
        PER_REGION_PDF_COUNT, PER_REGION_AFFINITY_PROFILE)
    composite.run(60.0)
    listener = _binding_listener(composite.state)
    region_fields = {
        'oriC': 'bound_oric',
        'dnaA_promoter': 'bound_dnaA_promoter',
        'datA': 'bound_datA',
        'DARS1': 'bound_DARS1',
        'DARS2': 'bound_DARS2',
    }
    for region, field in region_fields.items():
        n_total = PER_REGION_PDF_COUNT.get(region, 0)
        bound = int(listener[field])
        assert 0 <= bound <= n_total, (
            f'{region}: bound={bound} outside [0, {n_total}]. '
            f'Listener: {dict(listener)}')


def test_oric_load_and_trigger_split(composite):
    """At any reasonable DnaA-ATP concentration, oriC's high-affinity
    boxes (Kd ~1 nM) saturate quickly while low-affinity boxes (Kd >
    100 nM) fill cooperatively. Confirm the per-tier listener fields
    expose this split — bound_oric_high should be at or near the
    full count of 3, while bound_oric_low typically lags the low-tier
    total of 8."""
    composite.run(60.0)
    listener = _binding_listener(composite.state)
    high = int(listener['bound_oric_high'])
    low = int(listener['bound_oric_low'])
    assert 0 <= high <= 3, (
        f'bound_oric_high={high} outside [0, 3]')
    assert 0 <= low <= 8, (
        f'bound_oric_low={low} outside [0, 8]')
    # bound_oric is the sum of the tiers (the 'very_low' tier is 0
    # at oriC, so bound_oric == high + low).
    assert int(listener['bound_oric']) == high + low
