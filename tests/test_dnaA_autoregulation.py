"""Phase 8 — dnaA promoter autoregulation.

Closes the regulatory loop on the DnaA cycle: DnaA binding to the
high- and low-affinity DnaA boxes spanning the p1 / p2 promoters of
the dnaA gene represses transcription of dnaA itself. This phase wires
that feedback as a multiplicative scaling on the dnaA TU's
``basal_prob`` in the live ``TranscriptInitiation`` step, driven by
the ``listeners.dnaA_binding.bound_dnaA_promoter`` count emitted by
the Phase 2 binding step.

Tests pin down:
  * The autoregulation Step is wired into the cell_state.
  * Its listener emits the per-tick repression factor and applied
    basal_prob.
  * Disabling the autoregulation flag falls back to a composite
    without it.
  * The dnaA basal_prob is actually rescaled when boxes are bound.
  * The repression is fully reversible (basal_prob returns to baseline
    when occupancy drops, since the original value is captured at
    attach-time and re-applied each tick).
"""

from __future__ import annotations

import os

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


def test_autoregulation_step_in_cell_state(composite):
    cell = composite.state['agents']['0']
    assert 'dnaA_autoregulation' in cell, (
        f'dnaA_autoregulation step missing; '
        f'keys: {[k for k in cell if not k.startswith("_")][:10]}...')


def test_autoregulation_listener_emits_factor(composite):
    composite.run(60.0)
    listener = composite.state['agents']['0'].get('listeners', {})
    autoreg = listener.get('dnaA_autoregulation', {})
    assert 'repression_factor' in autoreg
    assert 'dnaA_basal_prob' in autoreg
    assert 'dnaA_basal_prob_baseline' in autoreg
    rf = float(autoreg['repression_factor'])
    # Repression factor is a fractional scale: 0 (full repression) to 1
    # (no repression). At default max_repression=0.7, the floor is 0.3.
    assert 0.0 <= rf <= 1.0
    baseline = float(autoreg['dnaA_basal_prob_baseline'])
    assert baseline > 0, 'baseline basal_prob should be positive (Parca-fit)'


def test_disabling_autoregulation_removes_step(composite):
    """``enable_dnaA_autoregulation=False`` skips the splice."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    c = make_replication_initiation_composite(
        cache_dir=CACHE_DIR, seed=0, enable_dnaA_autoregulation=False)
    cell = c.state['agents']['0']
    assert 'dnaA_autoregulation' not in cell, (
        'dnaA_autoregulation present despite enable=False')


def test_basal_prob_scaled_by_occupancy():
    """The Step rescales the dnaA TU's basal_prob in proportion to
    fractional occupancy. We bypass the full composite here and drive
    the Step directly with synthetic listener inputs to keep this test
    fast and deterministic."""
    import numpy as np

    from v2ecoli.processes.dnaA_autoregulation import (
        DnaAAutoregulation, KNOWN_DNAA_TU_IDS, N_DNAA_PROMOTER_BOXES,
    )

    class _StubTxi:
        def __init__(self):
            self.rna_data = np.array(
                [(KNOWN_DNAA_TU_IDS[0],), ('OTHER_RNA[c]',)],
                dtype=[('id', 'U32')])
            self.basal_prob = np.array([1e-4, 5e-5])

    txi = _StubTxi()
    step = DnaAAutoregulation({'time_step': 1.0, 'max_repression': 0.5})
    step.attach_transcript_initiation(txi)

    # Free: factor = 1.0, basal_prob unchanged.
    out = step.update({
        'listeners': {'dnaA_binding': {'bound_dnaA_promoter': 0}},
        'global_time': 0.0, 'timestep': 1.0,
    })
    listener = out['listeners']['dnaA_autoregulation']
    assert listener['repression_factor'] == pytest.approx(1.0)
    assert txi.basal_prob[0] == pytest.approx(1e-4)

    # Half-bound: factor = 1 - 0.5 * 0.5 = 0.75.
    half = N_DNAA_PROMOTER_BOXES // 2
    out = step.update({
        'listeners': {'dnaA_binding': {'bound_dnaA_promoter': half}},
        'global_time': 1.0, 'timestep': 1.0,
    })
    listener = out['listeners']['dnaA_autoregulation']
    f_bound_expected = half / N_DNAA_PROMOTER_BOXES
    rf_expected = 1.0 - 0.5 * f_bound_expected
    assert listener['repression_factor'] == pytest.approx(rf_expected)
    assert txi.basal_prob[0] == pytest.approx(1e-4 * rf_expected)

    # Drop back to 0 → fully reversible.
    out = step.update({
        'listeners': {'dnaA_binding': {'bound_dnaA_promoter': 0}},
        'global_time': 2.0, 'timestep': 1.0,
    })
    assert txi.basal_prob[0] == pytest.approx(1e-4)
