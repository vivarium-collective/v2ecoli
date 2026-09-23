"""Tests for the LineageBookkeeper Step and the #588 chunk-independence fix.

Three groups:

A. Unit tests of the Step's pure logic + its helpers (``followed_lineage_id``,
   ``doublings_for``) — the prune target, the stateless doublings-from-id-length,
   and the no-op guard when ``single_daughters=False``.

B. A wiring test: ``add_population_aggregator(single_daughters=True)`` inserts
   the ``lineage_bookkeeper`` Step BEFORE ``population_aggregator`` in
   ``flow_order`` (and not at all when False). No ParCa cache needed.

C. The load-bearing one: a tick-by-tick simulation of what
   ``composite.run(chunk)`` does, reproducing #588. A reactor substrate
   accumulator is integrated every tick over the live agents. Doing the
   division bookkeeping every tick (the fix, via the REAL LineageBookkeeper)
   gives a substrate trajectory that is IDENTICAL across ``chunk`` values;
   doing it only at the chunk boundary (the pre-fix runner behavior) makes the
   trajectory depend on ``chunk`` — "no chunk value gives a reproducible reactor
   trajectory" (#588). The test asserts BOTH: the fix is chunk-independent AND
   the buggy path is not, so it genuinely discriminates.
"""

from __future__ import annotations

import pytest

from v2ecoli.steps.lineage_bookkeeper import (
    LineageBookkeeper,
    doublings_for,
    followed_lineage_id,
)


def _make_bookkeeper(single_daughters: bool) -> LineageBookkeeper:
    bk = LineageBookkeeper.__new__(LineageBookkeeper)
    bk.initialize({"single_daughters": single_daughters})
    return bk


# ---------------------------------------------------------------------------
# A. Unit — helpers
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_followed_prefers_zero_suffix():
    # A fresh division of "00" -> {"000","001"}: keep the all-zeros lineage.
    assert followed_lineage_id(["001", "000"]) == "000"
    # Founder.
    assert followed_lineage_id(["0"]) == "0"
    # Fallback when nothing ends in 0 (shouldn't happen in a single lineage).
    assert followed_lineage_id(["11", "1"]) == "1"
    # Empty population.
    assert followed_lineage_id([]) is None


@pytest.mark.fast
def test_doublings_is_id_length_minus_one():
    assert doublings_for("0") == 0.0       # gen 1 -> factor 2^0 = 1
    assert doublings_for("00") == 1.0      # gen 2
    assert doublings_for("000") == 2.0     # gen 3
    assert doublings_for("0000") == 3.0    # gen 4


# ---------------------------------------------------------------------------
# A. Unit — the Step
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_noop_when_not_single_daughters():
    """Default mode: a pure no-op, even with a sibling present. Guarantees every
    existing multi-agent / fixed / baseline composite is byte-identical."""
    bk = _make_bookkeeper(single_daughters=False)
    out = bk.next_update(1.0, {"agents": {"00": {}, "01": {}}, "lineage": {}})
    assert out == {}


@pytest.mark.fast
def test_prunes_sibling_and_sets_doublings_on_division_tick():
    """At a division tick two daughters are live; keep the all-zeros lineage,
    remove the sibling, and advance doublings from the followed id length."""
    bk = _make_bookkeeper(single_daughters=True)
    out = bk.next_update(1.0, {"agents": {"000": {}, "001": {}}, "lineage": {}})
    assert out["agents"] == {"_remove": ["001"]}
    assert out["lineage"]["doublings"] == 2.0     # len("000") - 1
    assert out["lineage"]["generation"] == 3.0


@pytest.mark.fast
def test_single_agent_no_prune_still_maintains_doublings():
    """Between divisions there is one agent: nothing to prune, but doublings is
    still asserted (idempotent) so it never lags a chunk behind."""
    bk = _make_bookkeeper(single_daughters=True)
    out = bk.next_update(1.0, {"agents": {"00": {}}, "lineage": {}})
    assert "agents" not in out              # no _remove
    assert out["lineage"]["doublings"] == 1.0
    assert out["lineage"]["generation"] == 2.0


@pytest.mark.fast
def test_empty_population_is_noop():
    bk = _make_bookkeeper(single_daughters=True)
    assert bk.next_update(1.0, {"agents": {}, "lineage": {}}) == {}


# ---------------------------------------------------------------------------
# B. Wiring — bookkeeper precedes the aggregator in flow_order
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_add_population_aggregator_wires_bookkeeper_before_aggregator(core):
    from v2ecoli.composites.ecoli_population import (
        LINEAGE_BOOKKEEPER_STEP_NAME,
        POPULATION_AGGREGATOR_STEP_NAME,
        add_population_aggregator,
    )

    doc = {"state": {"agents": {}}}
    add_population_aggregator(doc, core, single_daughters=True)

    flow = doc["flow_order"]
    assert LINEAGE_BOOKKEEPER_STEP_NAME in flow
    assert POPULATION_AGGREGATOR_STEP_NAME in flow
    # The bookkeeper must run BEFORE the aggregator so the pruned/advanced state
    # is what the aggregator (and the downstream coupler) read.
    assert flow.index(LINEAGE_BOOKKEEPER_STEP_NAME) < flow.index(
        POPULATION_AGGREGATOR_STEP_NAME
    )
    edge = doc["state"][LINEAGE_BOOKKEEPER_STEP_NAME]
    # The lineage leaves must be pinned to overwrite so the Step's absolute
    # doublings write replaces (a bare float leaf would ACCUMULATE: 2,4,6,...).
    assert edge["_outputs"]["lineage"] == {
        "doublings": "overwrite[float]",
        "generation": "overwrite[float]",
    }


@pytest.mark.fast
def test_add_population_aggregator_omits_bookkeeper_by_default(core):
    from v2ecoli.composites.ecoli_population import (
        LINEAGE_BOOKKEEPER_STEP_NAME,
        add_population_aggregator,
    )

    doc = {"state": {"agents": {}}}
    add_population_aggregator(doc, core)  # single_daughters defaults False

    assert LINEAGE_BOOKKEEPER_STEP_NAME not in doc.get("flow_order", [])
    assert LINEAGE_BOOKKEEPER_STEP_NAME not in doc["state"]


# ---------------------------------------------------------------------------
# C. Integration — chunk-independence of the integrated reactor state (#588)
# ---------------------------------------------------------------------------

# One "cell" = a growing dry mass. The reactor draws substrate proportional to
# live biomass every tick (what ReactorCellCoupler integrates). cells_per_agent
# in "fixed" mode (growth_factor 1) so a stray second daughter is a first-order
# 2x error in the draw — the crisp version of "medium concentration depends on
# chunk."
_MAX_STEPS = 120
_DIV_PERIOD = 25          # ticks between divisions (unaligned to the chunks below)
_K_UPTAKE = 1.0e-4        # substrate drawn per unit mass per tick
_CELLS_PER_AGENT = 1.0
_INITIAL_MASS = 100.0
_GROWTH = 1.0             # mass added per tick per agent


def _grow(agents):
    for st in agents.values():
        st["listeners"]["mass"]["dry_mass"] += _GROWTH


def _divide(agents):
    """Divide the followed (all-zeros) lineage into two daughters. Daughters
    DIVERGE slightly (a 55/45 mass split) so the two-daughter draw is not a
    trivial 2x of the followed — matching the real 'daughters diverge' regime."""
    followed = followed_lineage_id(list(agents.keys()))
    parent = agents.pop(followed)
    m = parent["listeners"]["mass"]["dry_mass"]
    agents[followed + "0"] = {"listeners": {"mass": {"dry_mass": 0.55 * m}}}
    agents[followed + "1"] = {"listeners": {"mass": {"dry_mass": 0.45 * m}}}


def _apply(agents, lineage, update):
    """Apply a LineageBookkeeper update to plain dicts (mimics the framework
    applying the Step's structural _remove + lineage write)."""
    if not update:
        return
    ag = update.get("agents")
    if ag and "_remove" in ag:
        for aid in ag["_remove"]:
            agents.pop(aid, None)
    lin = update.get("lineage")
    if lin:
        lineage.update(lin)


def _substrate_draw(agents):
    """What the reactor integrates this tick: total biomass draw, fixed mode."""
    total_mass = sum(
        st["listeners"]["mass"]["dry_mass"] for st in agents.values()
    )
    return total_mass * _CELLS_PER_AGENT * _K_UPTAKE


def _simulate(chunk: int, *, fix: bool) -> float:
    """Integrate reactor substrate to _MAX_STEPS, mirroring composite.run(chunk).

    fix=True  -> LineageBookkeeper runs EVERY tick (in-composite, the fix).
    fix=False -> bookkeeping only at the chunk boundary (pre-#588 runner).
    Returns the final substrate remaining.
    """
    agents = {"0": {"listeners": {"mass": {"dry_mass": _INITIAL_MASS}}}}
    lineage: dict = {"doublings": 0.0, "generation": 1.0}
    substrate = 1.0e6
    bk = _make_bookkeeper(single_daughters=True)

    tick = 0
    while tick < _MAX_STEPS:
        n = min(chunk, _MAX_STEPS - tick)
        for _ in range(n):
            tick += 1
            _grow(agents)
            if tick % _DIV_PERIOD == 0:
                _divide(agents)
            if fix:
                # In-composite bookkeeping fires on the division tick.
                _apply(agents, lineage, bk.next_update(1.0, {
                    "agents": agents, "lineage": lineage}))
            # Reactor integrates the (now correct, if fixed) live biomass.
            substrate -= _substrate_draw(agents)
        if not fix:
            # Pre-fix runner: prune + doublings only at the chunk boundary,
            # AFTER the whole chunk (both daughters) has already been integrated.
            _apply(agents, lineage, bk.next_update(1.0, {
                "agents": agents, "lineage": lineage}))
    return substrate


@pytest.mark.fast
def test_fix_makes_reactor_trajectory_chunk_independent():
    """With the in-composite bookkeeper, the integrated reactor substrate is
    identical across chunk values — the property #588 says was missing."""
    ref = _simulate(1, fix=True)
    for chunk in (2, 7, 10, 30, 100):
        got = _simulate(chunk, fix=True)
        assert got == pytest.approx(ref, abs=1e-9), (
            f"chunk={chunk} diverged from chunk=1 with the fix: {got} != {ref}")


@pytest.mark.fast
def test_without_fix_reactor_trajectory_depends_on_chunk():
    """Guard that the test is not vacuous: the pre-fix (chunk-boundary-only)
    bookkeeping DOES make the reactor trajectory chunk-dependent, so the
    assertion above is actually testing something."""
    at_1 = _simulate(1, fix=False)
    at_30 = _simulate(30, fix=False)
    assert at_1 != pytest.approx(at_30, abs=1e-6), (
        "expected chunk-dependence without the fix, but trajectories matched")
