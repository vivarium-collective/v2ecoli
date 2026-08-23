"""Issue #495: single-cell ecoli_baseline division-stop.

Covers the #495 stopgap (docs + build-time warning) as superseded by the
Option A ``stop_at_division`` opt-in:

  (a) the corrected ``n_generations`` documentation states plainly that it is
      inert on a PLAIN single-cell (n_seeds=1, n_generations=1,
      stop_at_division=False) run, and that ``stop_at_division=True`` is the
      supported way to bound such a run to one cell cycle;
  (b) building a PLAIN single-cell baseline document emits a runtime warning
      that the run will continue past division;
  (c) ``stop_at_division=True`` routes the single-cell build through the
      lineage/batch machinery (LineageProcess stops at the first division),
      which changes the emit layout to the lineage ``batch_runner`` form;
  (d) ``stop_at_division=True`` + ``match_simdata`` raises a clear ValueError
      (Option A cannot support the build-time single-cell overlay); and
  (e) the #495 build-time warning is SUPPRESSED when ``stop_at_division=True``
      (that run DOES stop at division).

The doc, routing, guard, and warning-suppression assertions are cache-free and
always run. The two live assertions (the plain-single-cell warning fires, and a
stop_at_division run actually terminates at division) need the ParCa cache to
build/execute a real document, so they are skipped when it is absent; the
run-to-division check is additionally gated behind V2ECOLI_RUN_LIVE_DIVISION=1
because it simulates a whole cell cycle.
"""
import inspect
import os
import warnings

import pytest

import v2ecoli.composites.ecoli_baseline as ecoli_baseline
from v2ecoli.composites.ecoli_baseline import baseline

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")


def test_n_generations_doc_states_single_cell_has_no_division_stop():
    """Cache-free: the corrected #495 documentation is present."""
    src = inspect.getsource(ecoli_baseline)

    # Param-schema description for n_generations: must still say it has no effect
    # on the plain single-cell default, and now point at stop_at_division.
    assert "NO effect on the" in src
    assert "stop_at_division=True" in src

    # baseline() docstring: must state single-cell has no division-stop, cite the
    # issue, and document stop_at_division as the supported one-cell-cycle bound.
    doc = inspect.getdoc(baseline) or src
    assert "#495" in src
    assert "no division-stop" in doc.lower() or "no division-stop" in src.lower()
    assert "stop_at_division" in doc


def test_stop_at_division_is_a_declared_param():
    """Cache-free: stop_at_division is wired into signature AND param schema."""
    sig = inspect.signature(baseline)
    assert "stop_at_division" in sig.parameters
    assert sig.parameters["stop_at_division"].default is False

    # Declared in the @composite_generator param schema (default False, bool).
    src = inspect.getsource(ecoli_baseline)
    assert '"stop_at_division": {' in src


def test_stop_at_division_routes_through_lineage_path():
    """Cache-free: stop_at_division=True builds the lineage/batch document.

    The plain single-cell build returns a document whose state carries the flat
    ``agents/0`` cell; the lineage/batch path returns one whose state carries a
    ``batch_runner`` step (and no flat ``agents``). Asserting the shape proves
    the routing without needing to run the sim.
    """
    from v2ecoli.core import build_core
    core = build_core()
    doc = baseline(core=core, seed=0, stop_at_division=True)
    state = doc.get("state", {})
    assert "batch_runner" in state, "stop_at_division must route to the lineage path"
    assert "agents" not in state, (
        "stop_at_division must NOT build the flat single-cell agents layout")


def test_stop_at_division_with_match_simdata_raises():
    """Cache-free: Option A cannot support match_simdata; combining them errors."""
    from v2ecoli.core import build_core
    core = build_core()
    with pytest.raises(ValueError, match=r"incompatible with match_simdata"):
        baseline(core=core, seed=0, stop_at_division=True,
                 match_simdata="/nonexistent/simData.cPickle")


def test_stop_at_division_suppresses_no_division_stop_warning():
    """Cache-free: the #495 build-time warning does NOT fire under Option A.

    stop_at_division=True routes to the lineage path (which DOES stop at
    division), so the "no division-stop" note would be misleading and must not
    be emitted.
    """
    from v2ecoli.core import build_core
    core = build_core()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        baseline(core=core, seed=0, stop_at_division=True)
    assert not any("no division-stop" in str(w.message) for w in caught), (
        "stop_at_division=True must suppress the #495 no-division-stop warning")


@pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")
def test_single_cell_build_warns_no_division_stop():
    """A plain single-cell build surfaces the #495 no-division-stop note."""
    from v2ecoli.core import build_core
    core = build_core()
    with pytest.warns(UserWarning, match=r"no division-stop"):
        # Force the warning to fire even if a prior build in this process
        # already tripped warnings' once-per-location filter.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            baseline(core=core, seed=0, cache_dir=CACHE)


@pytest.mark.skipif(
    not (os.path.isdir(CACHE) and os.environ.get("V2ECOLI_RUN_LIVE_DIVISION")),
    reason="needs ParCa cache + V2ECOLI_RUN_LIVE_DIVISION=1 (runs a full cell "
           "cycle)")
def test_stop_at_division_run_terminates_at_first_division():
    """Live (opt-in): the generations=1 lineage halts at the first division.

    stop_at_division routes through the SAME LineageProcess the batch path uses,
    at n_seeds=1, generations=1. This exercises that process directly (its
    run-time division stop is the load-bearing mechanism for Option A; the
    BatchBaselineRunner fan-out around it needs the workflow deps that this venv
    lacks). We tick it exactly as meta_composite's wrapping Composite does — one
    call per ``time_step`` (the tick cadence, NOT one giant interval), where
    update() returns {} until the cell divides — and assert it stops at the FIRST
    division: complete=True, exactly ONE generation summary (divided=True), and
    the run ended well before the per-generation cap (i.e. it was bounded by
    division, not by max_duration_per_gen — no second generation, no overshoot to
    the cap).
    """
    from v2ecoli.workflow.lineage import LineageProcess
    from v2ecoli.core import build_core
    core = build_core()
    tick = 10.0            # LineageProcess tick cadence (meta_composite: time_step)
    cap = 6000.0           # per-gen cap, deliberately >> a ~2500s cell cycle
    proc = LineageProcess({
        "cache_dir": CACHE,
        "seed": 0,
        "generations": 1,
        "single_daughters": True,
        "time_step": tick,
        "max_duration_per_gen": cap,
        "emitter": "null",
    }, core=core)
    out = {}
    elapsed = 0.0
    n_ticks = int(cap / tick) + 5
    for _ in range(n_ticks):
        out = proc.update({}, tick)
        elapsed += tick
        if out.get("complete"):
            break
    assert out.get("complete") is True, f"never completed after {elapsed}s: {out!r}"
    summaries = out.get("summary", {}).get("generations", [])
    assert len(summaries) == 1, f"expected 1 generation, got {summaries!r}"
    assert summaries[0]["generation"] == 0
    assert summaries[0].get("divided") is True
    # Bounded by division, not by the cap: a real cell cycle here is ~2500s, so
    # completing well under the 6000s cap proves the stop was the division event
    # (and no second generation ran).
    assert elapsed < cap, (
        f"run reached the {cap}s cap without a division-bounded stop "
        f"(elapsed={elapsed}s)")
