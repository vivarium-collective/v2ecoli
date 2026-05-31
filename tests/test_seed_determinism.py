"""Determinism under a fixed seed.

Two independent `build_composite("baseline", cache_dir=..., seed=0)` calls,
each run for the same duration, must produce identical simulation state. A
failure means something in the process-bigraph graph or in v2ecoli's processes
is reading from an unseeded source (dict iteration order, set iteration,
hash randomization, uninitialized RNG, etc.) — a silent hazard for every
reproducibility claim downstream.

Skipped if `out/cache` is not present, matching test_architectures_grow.py.

Only the baseline architecture is checked here; extend to other
architectures if determinism in those variants also becomes a concern.
"""
from __future__ import annotations

import os

import pytest

from v2ecoli import build_composite

from _state_equal import deep_equal


CACHE_DIR = 'out/cache'
# Nondeterminism (unseeded RNG, dict/set iteration order, hash-dependent
# ordering) diverges within the first few ticks, so a short run surfaces it
# just as well as a long one. Kept short deliberately: this test builds TWO
# full composites and holds both states for the deep compare, so on the
# memory-constrained CI behavior-tests runner a long duration inflates peak
# RAM (now real pint.Quantity leaves in the mass state) enough to OOM the
# worker mid-run — surfacing as a whole-job "operation was canceled".
DURATION = 30.0


pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
        reason=f'cache dir {CACHE_DIR!r} not present; '
               f'rebuild with `python scripts/build_cache.py`',
    ),
]


def _run_baseline(duration: float):
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=0)
    composite.run(duration)
    return composite.state


# Two full baseline sims (2 x DURATION s) + a deep state compare. The CI
# behavior-tests runner is ~14x slower than local (see test_sustained_growth),
# so this can approach the global 120 s pytest-timeout cap. That cap's thread
# method calls os._exit(1), which kills the WHOLE pytest run (reported as
# "operation was canceled"), so carry the same 600 s override the other
# multi-sim behavior tests use.
@pytest.mark.timeout(600)
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
