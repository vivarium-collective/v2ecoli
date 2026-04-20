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
1. Sustained growth: >=20 fg over 500s (healthy cell adds ~30 fg).
2. Chromosome replication: oriC count goes 1 -> 2 within 1500s. (slow)
"""

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


def _dry_mass(composite):
    return float(composite.state['agents']['0']['listeners']['mass']['dry_mass'])


def _n_oric(composite):
    listeners = composite.state['agents']['0']['listeners']
    return int(listeners.get('replication_data', {}).get('number_of_oric', 1))


@pytest.mark.timeout(600)
def test_baseline_sustained_growth_500s():
    """Cell must add >=20 fg over 500s. Exposes tick-level silent failures
    (unit errors, AttributeError on request_set, etc.) that only manifest
    after the first few chunks.

    Timeout override: 500 s of sim takes ~35 s wall locally (M2 Pro, ~14×
    realtime) but CI runners are ~3-5× slower, so we need headroom over
    the global 120 s cap. 600 s is still well below the refire-loop
    signature (>15 min wall per sim-second)."""
    from v2ecoli.composite import make_composite
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)
    m0 = _dry_mass(composite)
    composite.run(500)
    m1 = _dry_mass(composite)
    delta = m1 - m0
    assert delta >= 20.0, (
        f'Baseline sustained growth too low: {m0:.2f} -> {m1:.2f} fg '
        f'(+{delta:.2f} fg in 500s; expected >= +20 fg). '
        f'Silent process failures may be stalling the cell.')


@pytest.mark.slow
def test_baseline_chromosome_replication_initiates():
    """Chromosome replication must initiate by 1500s (~25 min). The initial
    oriC count is 1; once replication initiates it becomes 2."""
    from v2ecoli.composite import make_composite
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)
    assert _n_oric(composite) == 1, (
        f'Expected initial oriC count = 1, got {_n_oric(composite)}')
    composite.run(1500)
    n = _n_oric(composite)
    assert n >= 2, (
        f'No chromosome replication after 1500s: oriC={n}. '
        f'Healthy cell initiates replication at ~23 min.')
