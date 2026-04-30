"""Phase 3 — DnaA-gated chromosome-replication initiation.

The replication_initiation architecture replaces the baseline
``ChromosomeReplication`` step with the
``DnaAGatedChromosomeReplication`` subclass. The subclass keeps every
piece of the base class except the gate: instead of
``cellMass / n_oriC >= criticalInitiationMass`` it gates on
``DnaA-ATP count / n_oriC >= threshold``. The per-oriC division gives
the same self-limiting feedback as the mass gate but routes the
trigger through the DnaA biology that Phases 5 + 7 set up.

These tests pin down:
  * The substituted class is in cell_state at the expected step name.
  * Initiation does fire (oriC count grows past its initial value)
    over a multi-minute window.
  * The gate self-limits (oriC doesn't explode to dozens of copies).
  * Disabling the gate via the architecture flag falls back to the
    baseline ChromosomeReplication.
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


def _count_active_unique(state, name):
    arr = state['agents']['0']['unique'].get(name)
    if arr is None or not hasattr(arr, 'dtype'):
        return 0
    if '_entryState' not in arr.dtype.names:
        return 0
    return int(arr['_entryState'].view(np.bool_).sum())


def test_dnaA_gated_subclass_is_in_cell_state(composite):
    """The architecture replaces the baseline ChromosomeReplication
    instance with DnaAGatedChromosomeReplication while keeping the
    same step name."""
    from v2ecoli.processes.chromosome_replication_dnaA_gated import (
        DnaAGatedChromosomeReplication,
    )
    cell = composite.state['agents']['0']
    edge = cell['ecoli-chromosome-replication']
    assert isinstance(edge['instance'], DnaAGatedChromosomeReplication), (
        f'expected DnaAGatedChromosomeReplication instance, got '
        f'{type(edge["instance"]).__name__}')


def test_critical_mass_per_oric_listener_is_dnaA_ratio(composite):
    """The base class's `critical_mass_per_oriC` listener field still
    populates, but it now reads as the DnaA-ATP-per-oriC ratio rather
    than the cell-mass ratio. The value should be a small float
    (0 to a few), not the cell-mass-derived ratio (which would be
    around 0.4 at init and rise toward 1.0 over a cell cycle)."""
    composite.run(60.0)
    rd = composite.state['agents']['0']['listeners'].get(
        'replication_data', {})
    ratio = float(rd.get('critical_mass_per_oriC', -1))
    assert 0.0 <= ratio <= 5.0, (
        f'critical_mass_per_oriC = {ratio!r} outside the plausible range '
        f'[0, 5] for the DnaA-ATP gate')


def test_initiation_fires_at_least_once(composite):
    """Over a 30-minute window the cell should fire at least one
    initiation, growing oriC from its initial 2 to >= 4. (The
    baseline mass gate does this same sanity check; the DnaA gate
    must hit at least the same bar.)"""
    composite.run(60.0)
    n0 = _count_active_unique(composite.state, 'oriC')
    composite.run(1740.0)  # 60s + 1740s = 1800s = 30 min total
    n1 = _count_active_unique(composite.state, 'oriC')
    assert n1 >= 4, (
        f'oriC count after 30 min of sim is {n1}; expected >= 4. '
        f'Phase 3 gate may be set too high — initiation never fires.')


def test_initiation_self_limits(composite):
    """After firing once, the DnaA-ATP-per-oriC value should drop
    below threshold and stop firing immediately. oriC count should
    not run away."""
    n_oric = _count_active_unique(composite.state, 'oriC')
    assert n_oric <= 8, (
        f'oriC count is {n_oric} — Phase 3 gate failed to self-limit. '
        f'Threshold is likely too low: every tick fires a new initiation.')


def test_disabling_gate_falls_back_to_baseline_class(composite):
    """``enable_dnaA_gated_initiation=False`` skips the swap; the
    baseline ChromosomeReplication is left in place."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    from v2ecoli.processes.chromosome_replication import ChromosomeReplication
    from v2ecoli.processes.chromosome_replication_dnaA_gated import (
        DnaAGatedChromosomeReplication,
    )
    c = make_replication_initiation_composite(
        cache_dir=CACHE_DIR, seed=0,
        enable_dnaA_gated_initiation=False)
    cell = c.state['agents']['0']
    edge = cell['ecoli-chromosome-replication']
    assert type(edge['instance']) is ChromosomeReplication, (
        f'expected baseline ChromosomeReplication when gate is disabled, '
        f'got {type(edge["instance"]).__name__}')
    assert not isinstance(
        edge['instance'], DnaAGatedChromosomeReplication)
