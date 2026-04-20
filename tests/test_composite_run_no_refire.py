"""Regression guard against the step-refire loop that once blew up multigeneration.

On 2026-04-18 ``composite.run(1)`` (one simulated second) was invoking each
step ~2,294 times instead of ~1, burning 15 min of CPU and 30–80 GB RSS on
every sim-second. ``reports/multigeneration_report.py`` was OOM-killed at
30 min wall. The cause turned out to be an incomplete ParCa fixture: the
committed ``models/parca/parca_state.pkl.gz`` didn't populate
``metabolism.aa_enzymes``, so ``save_cache`` silently dropped the
``ecoli-polypeptide-elongation`` config. The resulting partial composite
didn't converge, and the bigraph scheduler re-fired every step until its
inputs stabilized — which never happened cleanly.

Rebuilding the fixture with a KB state where ``aa_enzymes`` is populated
(see the KB rollback + ``scripts/build_cache.py``) fixed it: sim-second now
costs ~0.2 s wall, each step fires 1–10 times, and sim tests complete in
seconds.

This test pins the healthy regime:
  - ``composite.run(1)`` completes in a bounded wall-clock budget (30 s,
    enough slack for CI's slower hardware).
  - No step is invoked more than ~100 times in 1 simulated second.
    Healthy bigraph composites converge in a few cycles; triple-digit
    cycle counts are a loud "refire loop" signal.

If a future change re-introduces the cycle (a partial save_cache, a new
Step that retriggers itself, etc.), this test will fail in 30 s instead
of the multi-hour OOM the original bug produced.
"""
from __future__ import annotations

import collections
import os
import threading
import time

import pytest

# Side-effect import: registers `nucleotide` / `amino_acid` / `count` on the
# shared pint registry before any dill.load of the cache.
import v2ecoli.library.unit_bridge  # noqa: F401


pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir('out/cache') and not os.environ.get('CI'),
        reason="cache dir 'out/cache' not present; "
               "rebuild with `python scripts/build_cache.py`",
    ),
]


# Budgets. Deliberately generous — the regression we're catching is ~2000×
# per sim-second, so anything up to 100× is not a false positive.
WALL_BUDGET_S = 30.0
PER_STEP_CAP = 100
SIM_DURATION = 1.0


def test_composite_run_one_sec_does_not_refire_loop():
    """1 simulated second must not invoke any step 100+ times.

    Pins the healthy convergence regime so a future partial-cache or
    self-retriggering step can't sneak the refire loop back in.
    See module docstring for the original symptom.
    """
    from process_bigraph.composite import Step

    counts: collections.Counter[str] = collections.Counter()
    orig_invoke = Step.invoke

    def counting_invoke(self, state, interval=None):
        counts[type(self).__name__] += 1
        return orig_invoke(self, state, interval)

    Step.invoke = counting_invoke
    try:
        from v2ecoli.composite import make_composite
        composite = make_composite(cache_dir='out/cache')

        # Wall-clock timeout via thread — a refire loop blocks the GIL in
        # C code, so a plain assertion wouldn't fire until the loop (maybe
        # never) finished.
        done = threading.Event()
        error: list[BaseException] = []

        def _run():
            try:
                composite.run(SIM_DURATION)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t0 = time.time()
        t.start()
        if not done.wait(WALL_BUDGET_S):
            top = counts.most_common(5)
            pytest.fail(
                f"composite.run({SIM_DURATION}) did not complete in "
                f"{WALL_BUDGET_S}s (still running after {time.time()-t0:.0f}s). "
                f"Top invocation counts so far: {dict(top)}. "
                f"Refire loop suspected — see tests/test_composite_run_no_refire.py."
            )
        if error:
            raise error[0]

        over_cap = {
            name: n for name, n in counts.items()
            if n > PER_STEP_CAP and name != 'UniqueUpdate'
        }
        # UniqueUpdate gets its own, higher cap because it's explicitly
        # placed once per execution layer (~11 layers). Even with generous
        # headroom for intra-layer re-firing, it should be under ~500.
        unique_count = counts.get('UniqueUpdate', 0)

        assert not over_cap and unique_count <= 5 * PER_STEP_CAP, (
            f"Step invocation counts exceeded the refire-loop bound for "
            f"{SIM_DURATION}s of simulation. Each step should fire at most "
            f"{PER_STEP_CAP} times per sim-second (a few convergence cycles). "
            f"UniqueUpdate cap is {5*PER_STEP_CAP} (once per exec layer × slack).\n"
            f"Over-cap steps: {over_cap}\n"
            f"UniqueUpdate: {unique_count}\n"
            f"Top 10 overall: {dict(counts.most_common(10))}"
        )
    finally:
        Step.invoke = orig_invoke
