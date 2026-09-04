"""Build + 1-step tests for the reactor_bird_coupled composite (mbp-03 req-2).

The composite wires:

  * ``baseline_population`` (single-cell WCM + PopulationAggregator) — the cell
    side, with the EnvironmentDriver in ``external_store`` mode (no-op; the
    coupler is the env source) and the EnvironmentMirror active.
  * a top-level ``BiRDTransportProcess`` edge (``local:BiRDTransportProcess``) —
    the reactor gas-liquid transport, emitting ADDITIVE deltas to
    ``reactor.dissolved_o2`` / ``reactor.dissolved_co2``.
  * a ``ReactorCellCoupler`` Step — the translator that writes the cell-side
    consumption as ADDITIVE deltas to the same dissolved stores (net change),
    passes biomass through to ``reactor.biomass`` (overwrite), and writes
    ``environment.external_concentrations`` (overwrite) for the mirror.

These tests only assert build + a single coupled step stays finite (the
behavior tests are mbp-03 T5).
"""

import math
import os

import pytest
from process_bigraph.composite import Composite

# Both tests build the reactor_bird_coupled composite, which wraps the baseline
# WCM and loads the ParCa cache (out/cache/initial_state.json). Skip the whole
# module when that cache isn't present (e.g. CI fast-tests), matching the
# convention used by the other cache-dependent tests.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.environ.get("V2ECOLI_CACHE", "out/cache")),
    reason="ParCa cache not present",
)


def test_reactor_bird_coupled_builds():
    """The generator is registered and builds a process_bigraph Composite."""
    from v2ecoli import build_composite

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")
    assert isinstance(c, Composite)


@pytest.mark.slow
def test_reactor_bird_coupled_runs_one_step():
    """One coupled WCM+reactor step runs without exception; dissolved O2 stays
    present and finite (additive transport/consumption net does not blow up)."""
    from v2ecoli import build_composite

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")
    c.run(1)

    reactor = c.state["reactor"]
    assert "dissolved_o2" in reactor
    do2 = reactor["dissolved_o2"]
    val = float(do2.magnitude) if hasattr(do2, "magnitude") else float(do2)
    assert math.isfinite(val), f"dissolved_o2 is not finite: {val!r}"


@pytest.mark.sim  # ticks the composite; NOT `slow` (see pyproject.toml:131)
def test_reactor_store_exposes_kla_co2_and_the_ammonium_pool():
    """The composite must EXPOSE what the study readouts declare.

    ⚠ Two gaps this pins, both of which were silent:

    * ``kla_co2`` is emitted by ``BiRDTransportProcess.outputs()`` exactly as
      ``kla_o2`` is, but was never declared or wired here — so the CO2 side of
      the gas transfer was unobservable while the O2 side was fully
      instrumented.
    * ``ammonium_medium_mM`` did not exist, so the nitrogen source was a static
      concentration the cell could never draw down.

    ⚠ And the trap that makes these silent: the coupler's ``InPlaceDict``
    reactor port DROPS any leaf the per-leaf output schema does not enumerate.
    A write to an unenumerated leaf goes nowhere and raises nothing, so a
    build-time presence check is the only thing that catches it.
    """
    from v2ecoli import build_composite

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache",
                        cells_per_agent=1e12, initial_glucose_mM=40.0)
    reactor = c.state["reactor"]
    for leaf in ("kla_o2", "kla_co2", "glucose_medium_mM", "ammonium_medium_mM"):
        assert leaf in reactor, f"reactor.{leaf} is not exposed by the composite"

    # ⚠ PRESENCE ALONE CANNOT FAIL. Each leaf exists if EITHER the `_reactor_store`
    # seed OR a port wiring names it, so no single production line is pinned by
    # the loop above: dropping the ammonium entry from the coupler's per-leaf
    # OUTPUT schema leaves the leaf present and merely freezes it forever, and
    # the whole suite stays green. That silent-freeze is the exact trap this
    # docstring describes, so the test has to watch the pool MOVE.
    def _f(x):
        return float(x.magnitude) if hasattr(x, "magnitude") else float(x)

    seeded = _f(reactor["ammonium_medium_mM"])
    c.run(2)
    after = _f(c.state["reactor"]["ammonium_medium_mM"])
    assert after != seeded, (
        f"ammonium_medium_mM never moved off its seed ({seeded} mM) after 2 "
        f"ticks at cells_per_agent=1e12. The leaf is present but inert — the "
        f"coupler's per-leaf output schema is not enumerating it, so the write "
        f"is dropped silently by the InPlaceDict reactor port.")


@pytest.mark.sim  # NOT `slow`: pyproject.toml:131 -- a `slow` test runs in NEITHER CI job
def test_kla_co2_is_written_by_transport():
    """Presence is not enough: a declared leaf can exist and never be written.

    The transport process already emits ``kla_co2``; only the wiring was
    missing, so after a tick the leaf must carry a real value rather than the
    seeded 0.0.
    """
    from v2ecoli import build_composite

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")
    c.run(1)

    def _f(x):
        return float(x.magnitude) if hasattr(x, "magnitude") else float(x)

    kla_co2 = _f(c.state["reactor"]["kla_co2"])
    kla_o2 = _f(c.state["reactor"]["kla_o2"])
    assert kla_co2 > 0.0, (
        f"kla_co2 exposed but never written (got {kla_co2!r}) — the transport "
        f"process emits it; the wiring is what was missing")

    # ⚠ `> 0.0` alone does NOT discriminate: swapping the kla_o2 / kla_co2
    # destinations leaves both positive and every assertion green, while
    # mbp-03 grades `cross_run_trend` on reactor.kla_co2 — i.e. CO2 transport
    # would be graded against O2's driver. Pin the RATIO instead.
    # CO2 diffuses more slowly than O2, so kla_co2 < kla_o2, and the ratio is
    # set by the transport correlation (measured 0.91839 on this composite).
    assert kla_co2 < kla_o2, (
        f"kla_co2 ({kla_co2}) must be BELOW kla_o2 ({kla_o2}) — CO2 is the "
        f"slower diffuser. Equal or inverted means the two destinations are "
        f"swapped or wired to the same source.")
    assert kla_co2 / kla_o2 == pytest.approx(0.9184, rel=1e-2), (
        f"kla_co2/kla_o2 = {kla_co2 / kla_o2} — expected ~0.9184. A swap or a "
        f"cross-wire lands far outside this band.")


@pytest.mark.sim  # NOT `slow`: pyproject.toml:131 -- a `slow` test runs in NEITHER CI job
def test_the_cell_sees_the_reactor_ammonium_pool():
    """The reactor's ammonium pool must reach ``boundary.external["AMMONIUM"]``.

    ⚠ This is the whole point of making ammonium a finite pool, and it is the
    one link that no other test covers.

    ⛔ WHAT THIS DOES **NOT** ESTABLISH (measured 2026-08-31, correcting an
    earlier claim in this file that a finite pool makes "nitrogen-limited growth
    reachable"). It does not. Seeded at 0.0 mM the boundary reads 2.1e-15 mM,
    the cell takes up NO nitrogen (per-tick ammonium exchange delta exactly
    0.0), and dry mass still climbs — 379.807 -> 380.104 fg over 5 ticks, a
    0.0012% difference from the 60 mM run. Exhausting the pool therefore yields
    no N-limitation; it yields a cell building mass at zero nitrogen, i.e. a
    silent nitrogen mass-conservation violation. The reason is in the repo:
    `is_carbon_starved` / `arrest_monomer_supply` (metabolism.py:143,163,
    #572/#592) exist because the WCM keeps polymerising after the exchange gate
    closes, and they are CARBON-ONLY, opt-in, default off. A nitrogen analogue
    is not wired.
    ⊕ Scope of the measurement: 5 ticks. Internal N pools would deplete
    eventually, so this shows the cliff produces no limitation and breaks the
    balance from the first tick past exhaustion — NOT that growth is never
    N-limited. The path is: the coupler writes
    ``AMMONIUM[c]`` into ``environment.external_concentrations`` ->
    ``EnvironmentMirror`` strips the compartment tag -> the agent's
    ``boundary.external["AMMONIUM"]``. Delete the coupler's write and the
    reactor pool still fills, still draws down, and every other assertion in
    this suite still passes — the cells simply never see it, and instead read
    the static media-recipe value (measured: 30.272 mM).

    The seed here is therefore deliberately NOT the default and NOT the recipe
    value, so a boundary reading either one fails.
    """
    from v2ecoli import build_composite

    seeded_mM = 17.5
    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache",
                        initial_ammonium_mM=seeded_mM)
    c.run(2)

    reactor = c.state["reactor"]
    pool = reactor["ammonium_medium_mM"]
    pool = float(pool.magnitude) if hasattr(pool, "magnitude") else float(pool)
    assert pool == pytest.approx(seeded_mM, rel=1e-6), (
        f"reactor pool did not seed at {seeded_mM}: {pool}")

    agents = c.state["agents"]
    assert agents, "no agents in the built composite"
    for agent_id, agent in agents.items():
        external = agent["boundary"]["external"]
        assert "AMMONIUM" in external, (
            f"agent {agent_id} has no AMMONIUM boundary concentration")
        seen = external["AMMONIUM"]
        seen = float(seen.magnitude) if hasattr(seen, "magnitude") else float(seen)
        assert seen == pytest.approx(pool, rel=1e-6), (
            f"agent {agent_id} sees AMMONIUM = {seen} mM while the reactor pool "
            f"holds {pool} mM — the coupler's env_concs write is not reaching "
            f"the cell, so the pool is decorative")
