"""Phase 4 — SeqA sequestration of newly-replicated origins.

The SeqA protein (``EG12197-MONOMER``) is already expressed in the
baseline architecture (~1029 copies in init state). Phase 4 wires its
*activity*: after each initiation event, the DnaA-gated initiation
step's gate ratio is forced to 0 for ``seqA_sequestration_window_s``
seconds (default 600s = ~10 min, matching the curated PDF). Models
SeqA binding to hemimethylated GATC sites at the newly-replicated
origin, which prevents DnaA from reaching it.

These tests pin down:
  * SeqA the protein is in bulk (sanity check on the baseline
    architecture's expression pipeline).
  * The DnaA-gated step records ``seqA_sequestration_window_s = 600``
    when ``enable_seqA_sequestration=True`` (the architecture default).
  * The gate ratio collapses to 0 for the sequestration window after
    an initiation event.
  * After the window expires, the gate resumes its normal DnaA-ATP-
    per-oriC calculation.
  * ``enable_seqA_sequestration=False`` falls back to the unblocked
    DnaA gate (gate ratio is non-zero throughout post-init).
"""

from __future__ import annotations

import os

import numpy as np
import pytest


CACHE_DIR = 'out/cache'
SEQA_BULK_ID = 'EG12197-MONOMER[c]'

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
    if len(matches) == 0:
        return None
    return int(counts[matches[0]])


def test_seqA_protein_already_expressed(composite):
    """Sanity check — the SeqA protein is already in the bulk pool
    via the baseline transcription / translation pipeline. We don't
    add it; we just use it. ~1000 copies in init state."""
    n_seqA = _bulk_count(composite.state, SEQA_BULK_ID)
    assert n_seqA is not None and n_seqA > 100, (
        f'SeqA monomer (EG12197-MONOMER) missing or sparse: count={n_seqA}')


def test_sequestration_window_set_when_flag_on(composite):
    cell = composite.state['agents']['0']
    inst = cell['ecoli-chromosome-replication']['instance']
    assert getattr(inst, 'seqA_sequestration_window_s', 0) == 600.0, (
        f'expected 600s sequestration window with default flags, '
        f'got {getattr(inst, "seqA_sequestration_window_s", None)!r}')


def test_gate_closed_during_sequestration_window(composite):
    """After the t=0 initiation event, the gate ratio should be 0
    throughout the 600s sequestration window."""
    composite.run(60.0)
    rd = composite.state['agents']['0']['listeners'].get(
        'replication_data', {})
    ratio_60s = float(rd.get('critical_mass_per_oriC', -1))
    assert ratio_60s == 0.0, (
        f'gate ratio at t=60s = {ratio_60s} but should be 0 — '
        f'SeqA sequestration window not closing the gate')

    composite.run(540.0)  # total t = 600s, edge of window
    rd = composite.state['agents']['0']['listeners'].get(
        'replication_data', {})
    ratio_600s = float(rd.get('critical_mass_per_oriC', -1))
    assert ratio_600s == 0.0, (
        f'gate ratio at t=600s = {ratio_600s} but should still be 0 '
        f'(boundary of sequestration window)')


def test_gate_resumes_after_window_expires(composite):
    """At t=900s (300s past the 600s window), the gate ratio must
    reflect the actual DnaA-ATP-per-oriC again — non-zero, with the
    DnaA-ATP pool dynamics from RIDA + DARS visible."""
    composite.run(300.0)  # total t = 900s
    rd = composite.state['agents']['0']['listeners'].get(
        'replication_data', {})
    ratio_900s = float(rd.get('critical_mass_per_oriC', -1))
    assert ratio_900s > 0.0, (
        f'gate ratio at t=900s = {ratio_900s} but should be > 0 — '
        f'sequestration window should have expired')


def test_disabling_seqA_keeps_gate_open(composite):
    """``enable_seqA_sequestration=False`` skips the sequestration
    check; the gate ratio is the raw DnaA-ATP-per-oriC at all times."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    c = make_replication_initiation_composite(
        cache_dir=CACHE_DIR, seed=0,
        enable_seqA_sequestration=False)
    inst = c.state['agents']['0']['ecoli-chromosome-replication']['instance']
    assert getattr(inst, 'seqA_sequestration_window_s', None) == 0.0
    c.run(60.0)
    rd = c.state['agents']['0']['listeners'].get('replication_data', {})
    ratio = float(rd.get('critical_mass_per_oriC', -1))
    # Without SeqA, the gate at t=60s reads the DnaA-ATP-per-oriC ratio
    # — small post-init but non-zero.
    assert ratio > 0.0, (
        f'gate ratio at t=60s with SeqA disabled = {ratio} — should '
        f'reflect DnaA-ATP-per-oriC, not be 0')
