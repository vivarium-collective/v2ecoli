"""Ensemble diversity under varying seeds.

The companion of test_seed_determinism.py: two `build_composite("baseline",
seed=A)` and `build_composite("baseline", seed=B)` calls with A != B,
each run for the same duration, must produce DIFFERENT simulation state.
A failure means stochastic processes aren't picking up the master_seed
override — every process is sharing a single cache-derived seed, so a
multi-seed "ensemble" collapses to bit-identical trajectories (the bug
documented in workspace/studies/pdmp-00 planned_runs[fix-per-process-rng-seeding]).

Skipped if `out/cache` is not present, matching test_seed_determinism.py.

Only the baseline architecture is checked here.
"""
from __future__ import annotations

import os

import pytest

from v2ecoli import build_composite
from v2ecoli.composites.baseline import _derive_process_seed

from _state_equal import deep_equal


CACHE_DIR = 'out/cache'
DURATION = 40.0  # seed divergence appears within tens of seconds


pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
        reason=f'cache dir {CACHE_DIR!r} not present; '
               f'rebuild with `python scripts/build_cache.py`',
    ),
]


def _run_baseline(duration: float, seed: int):
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    composite.run(duration)
    return composite.state


def test_derive_process_seed_is_reproducible_and_diverse():
    """Pure-function smoke test for the per-process seed derivation."""
    assert _derive_process_seed(0, 'metabolism') == _derive_process_seed(0, 'metabolism')
    assert _derive_process_seed(0, 'metabolism') != _derive_process_seed(0, 'complexation')
    assert _derive_process_seed(0, 'metabolism') != _derive_process_seed(1, 'metabolism')
    assert 0 <= _derive_process_seed(99999, 'transcript_elongation') < 2**31


def test_baseline_diverges_under_different_seeds():
    """Two baseline runs with seed=0 and seed=1 produce DIFFERENT state
    after the same duration. Without per-process seeding, the trajectories
    would be bit-identical (every stochastic process inheriting the same
    cache-derived seed). Phase 0 ensemble work depends on this divergence."""
    state_a = _run_baseline(DURATION, seed=0)
    state_b = _run_baseline(DURATION, seed=1)

    ok, reason = deep_equal(state_a, state_b)
    assert not ok, (
        'Baseline runs with seed=0 and seed=1 produced bit-identical state '
        f'after {DURATION}s — the per-process seed override in '
        '_get_step_config is not flowing through. Multi-seed ensembles will '
        'collapse to a single trajectory until this is fixed.'
    )
