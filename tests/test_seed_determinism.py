"""Determinism under a fixed seed.

Two independent `make_composite(cache_dir=..., seed=0)` calls, each run
for the same duration, must produce identical simulation state. A failure
means something in the process-bigraph graph or in v2ecoli's processes is
reading from an unseeded source (dict iteration order, set iteration,
hash randomization, uninitialized RNG, etc.) — a silent hazard for every
reproducibility claim downstream.

Skipped if `out/cache` is not present, matching test_architectures_grow.py.

Only the baseline architecture is checked here; extend to departitioned /
reconciled if determinism in those variants also becomes a concern.
"""
from __future__ import annotations

import os

import pytest

from v2ecoli.composite import make_composite

from _state_equal import deep_equal


CACHE_DIR = 'out/cache'
DURATION = 60.0


pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
        reason=f'cache dir {CACHE_DIR!r} not present; '
               f'rebuild with `python scripts/build_cache.py`',
    ),
]


def _run_baseline(duration: float):
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)
    composite.run(duration)
    return composite.state


@pytest.mark.refire_blocked
def test_baseline_is_deterministic_under_fixed_seed():
    """Two fresh baseline runs with seed=0 produce identical state after
    the same duration. Localizes the first divergent leaf so the failure
    message points at the nondeterministic process."""
    state_a = _run_baseline(DURATION)
    state_b = _run_baseline(DURATION)

    ok, reason = deep_equal(state_a, state_b)
    assert ok, (
        f'Baseline simulation is nondeterministic under seed=0: {reason}. '
        f'The first divergent leaf is named above — look for unseeded RNG, '
        f'dict/set iteration order, or hash-dependent ordering in that '
        f'process or upstream producer.'
    )
