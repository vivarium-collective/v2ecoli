"""Regression guard against the step-refire loop that blew up multigeneration.

Symptom (discovered 2026-04-18): ``reports/multigeneration_report.py`` at
default duration (3 generations, 3600 s/gen) was killed by the OS OOM
handler after ~30 min wall time. Earlier runs reportedly finished in 10-15
min. Per-process instrumentation during ``composite.run(1)`` (i.e. **one**
simulated second) showed:

  25,228  UniqueUpdate
   4,586  Allocator
   2,294  TwoComponentSystem, RnaMaturation, ProteinDegradation, Equilibrium,
          Complexation, ChromosomeReplication, PolypeptideInitiation, ...
   2,293  MassListener, RNACounts, MonomerCounts, RAMEmitter, ...

Each step ran roughly **2,000+× per simulated second**. One sim-second cost
~15 min of CPU (load_avg ~4, process pinned at 100% CPU, 30-80 GB RSS),
instead of the expected few seconds.

Root cause (hypothesis; not fixed in this PR): process-bigraph's cascading
trigger / ``cycle_step_state`` path re-fires every step whose inputs changed
after the previous layer's updates applied. v2ecoli's ``UniqueUpdate`` step
emits ``{mol: {"update": True}}`` on every call, which the scheduler treats
as a write to every unique-molecule port UniqueUpdate declares as output —
which in turn is read by most downstream steps, which feed back into the
reconciliation layer, which re-triggers UniqueUpdate, ad nauseam.

The gate in ``v2ecoli.library.ecoli_step.EcoliStep.perform_update`` (#20)
only skips updates for Steps that implement ``update_condition``.
``UniqueUpdate`` and most partitioning steps don't, so the cycle keeps
firing until (if ever) the scheduler's convergence check succeeds.

What this test pins:
  - ``composite.run(1)`` completes in a bounded wall-clock budget (30 s
    locally, enough slack for CI's slower hardware).
  - No step is invoked more than the convergence cap of ~100 times in 1
    simulated second. Healthy bigraph composites converge in a few cycles;
    triple-digit cycle counts are a loud "refire loop" signal.

The specific numeric bounds are the *regression thresholds*: ~2,000× is
clearly wrong, 10× is normal, 100× is the generous upper bound that lets
legitimate multi-pass convergence through while still catching the loop.

Until the scheduler-side fix lands, this test is expected to **fail** on
current main. That's the point — it pins the regression so the fix and the
test ship together.
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

    Pins the cascading-trigger regression that turned multigeneration from
    ~10 min into a 30-min OOM. See module docstring for the symptom trace.
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
