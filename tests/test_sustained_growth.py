"""Long-duration regression tests.

The existing test_architectures_grow tests run 60s — fast and catch total
stalls, but they miss regressions where the cell grows briefly and then
silently stalls mid-run.

Motivating incident: on pre-pint main, every tick after ~1000s emitted
    AttributeError: 'RnaDegradation' object has no attribute 'request_set'
    unum.IncompatibleUnitsError: [] can't be converted to [mol/L]
which multigeneration.py swallowed with a generic try/except. The cell
added only ~2 fg over 3600s, never replicated, never divided — and nothing
in the test suite caught it because the 60s growth window was clean.

These tests enforce:
1. Sustained growth: >=12 fg over 300s (healthy cell adds ~20 fg).
2. Chromosome replication: oriC count goes 1 -> 2 within 1500s. (slow)
"""

import os
import pytest
from v2ecoli.library.quantity_helpers import fg_magnitude


CACHE_DIR = 'out/cache'

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
        reason=f'cache dir {CACHE_DIR!r} not present; '
               f'rebuild with `python scripts/build_cache.py`',
    ),
]


def _dry_mass(composite):
    return fg_magnitude(composite.state['agents']['0']['listeners']['mass']['dry_mass'])


def _dna_mass(composite):
    return fg_magnitude(composite.state['agents']['0']['listeners']['mass']['dna_mass'])


def _n_oric(composite):
    """Count active oriC unique molecules.

    The ``replication_data.number_of_oric`` *listener* is 0 until the
    replication-data listener first runs, so reading it before/at build
    reports 0 even though the cell physically has origins. Count the oriC
    unique molecules directly (same access pattern as the Division step's
    full_chromosome read)."""
    unique = composite.state['agents']['0'].get('unique', {})
    oric = unique.get('oriC')
    if oric is None or not hasattr(oric, 'dtype'):
        return 0
    if '_entryState' in oric.dtype.names:
        return int(oric['_entryState'].sum())
    return int(len(oric))


@pytest.mark.timeout(600)
def test_baseline_sustained_growth_300s():
    """Cell must add >=12 fg over 300s. Exposes tick-level silent failures
    (unit errors, AttributeError on request_set, etc.) that only manifest
    after the first few chunks.

    Timeout override: 300 s of sim takes ~21 s wall locally (M2 Pro, ~14×
    realtime) but CI runners are ~3-5× slower, so we need headroom over
    the global 120 s cap. 600 s is still well below the refire-loop
    signature (>15 min wall per sim-second)."""
    from v2ecoli import build_composite
    composite = build_composite("ecoli_baseline", cache_dir=CACHE_DIR, seed=0)
    m0 = _dry_mass(composite)
    composite.run(300)
    m1 = _dry_mass(composite)
    delta = m1 - m0
    assert delta >= 12.0, (
        f'Baseline sustained growth too low: {m0:.2f} -> {m1:.2f} fg '
        f'(+{delta:.2f} fg in 300s; expected >= +12 fg). '
        f'Silent process failures may be stalling the cell.')


@pytest.mark.slow
def test_baseline_chromosome_replication_initiates():
    """Chromosome replication must be active over 1500s (~25 min).

    A viable cell starts with >= 1 oriC (the basal cache starts mid-cycle with
    overlapping rounds, so >= 2). Over 1500s a healthy cell replicates its DNA:
    either a new round opens (more origins) or DNA mass increases. A stalled
    cell — the regression this guards against — does neither."""
    from v2ecoli import build_composite
    composite = build_composite("ecoli_baseline", cache_dir=CACHE_DIR, seed=0)
    n0 = _n_oric(composite)
    dna0 = _dna_mass(composite)
    assert n0 >= 1, f'A viable cell starts with >= 1 oriC, got {n0}'
    composite.run(1500)
    n1 = _n_oric(composite)
    dna1 = _dna_mass(composite)
    assert dna1 > dna0 or n1 > n0, (
        f'No chromosome replication over 1500s: oriC {n0}->{n1}, '
        f'dna_mass {dna0:.3f}->{dna1:.3f} fg. A healthy cell replicates DNA.')
