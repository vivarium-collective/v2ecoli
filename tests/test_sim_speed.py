"""Simulation speed floor: one simulated minute must run in bounded wall time.

The refire-loop regression (#26) made composite.run(1) take 15 min of CPU
and blow out 80 GB of RSS — but it was fixed by rebuilding the ParCa
fixture, not by fixing the scheduler. Future partial-cache or
self-retriggering-Step regressions could reintroduce the same class of
bug. ``tests/test_composite_run_no_refire.py`` bounds per-step invocation
counts; this test bounds wall time directly.

Current observed speed (post-15fe48b, post-cache-rebuild, local M2 Pro):
  composite.run(60): ~4 s wall (~15× realtime)

Budget below is calibrated so:
  - Refire loop (~2000× step invocations) → fails immediately.
  - A uniform 3× slowdown across all processes (e.g. a new pint-heavy
    hot path) → fails.
  - Normal CI-runner variance (slower CPU, cold caches) → passes with
    ~7× margin.
"""
from __future__ import annotations

import os
import time

import pytest

# Side-effect import: registers `nucleotide` / `amino_acid` / `count` on the
# shared pint registry before dill.load of the cache.
import v2ecoli.library.unit_bridge  # noqa: F401


pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir('out/cache') and not os.environ.get('CI'),
        reason="cache dir 'out/cache' not present; "
               "rebuild with `python scripts/build_cache.py`",
    ),
]


SIM_SECONDS = 60
WALL_BUDGET_S = 30.0


def test_composite_runs_at_positive_realtime_rate(sim_data_cache):
    """``composite.run(60)`` must complete in under 30 s wall.

    Not a performance benchmark — a floor. The refire loop took >15 min
    for one simulated second; the rp_ratio bug stayed fast but drifted
    biology. This test catches the former; test_growth_parity catches
    the latter.

    The warm-up ``composite.run(1)`` exercises lazy first-call caches
    (process instantiation, pint unit parsing, etc.) that the real budget
    shouldn't pay for on subsequent calls.
    """
    from v2ecoli.composite import make_composite

    composite = make_composite(cache_dir='out/cache', seed=0)
    composite.run(1)  # warm-up — first tick builds per-process caches

    t0 = time.time()
    composite.run(SIM_SECONDS)
    elapsed = time.time() - t0

    assert elapsed < WALL_BUDGET_S, (
        f'composite.run({SIM_SECONDS}) took {elapsed:.1f}s wall, over the '
        f'{WALL_BUDGET_S:.0f}s budget (historic baseline ≈ 4 s).\n'
        f'Causes to investigate:\n'
        f'  - Refire loop: check per-step invocation counts via '
        f'tests/test_composite_run_no_refire.py.\n'
        f'  - Per-tick allocation: a new pint / Unum conversion in a hot '
        f'update path compounds fast.\n'
        f'  - ODE-solver hangs: scipy solve_ivp against a stiff system '
        f'(Equilibrium, TwoComponentSystem, PolypeptideKinetics).'
    )
