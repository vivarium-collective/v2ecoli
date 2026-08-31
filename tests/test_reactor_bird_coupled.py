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

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")
    reactor = c.state["reactor"]
    for leaf in ("kla_o2", "kla_co2", "glucose_medium_mM", "ammonium_medium_mM"):
        assert leaf in reactor, f"reactor.{leaf} is not exposed by the composite"


@pytest.mark.slow
def test_kla_co2_is_written_by_transport():
    """Presence is not enough: a declared leaf can exist and never be written.

    The transport process already emits ``kla_co2``; only the wiring was
    missing, so after a tick the leaf must carry a real value rather than the
    seeded 0.0.
    """
    from v2ecoli import build_composite

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")
    c.run(1)

    kla_co2 = c.state["reactor"]["kla_co2"]
    val = float(kla_co2.magnitude) if hasattr(kla_co2, "magnitude") else float(kla_co2)
    assert val > 0.0, (
        f"kla_co2 exposed but never written (got {val!r}) — the transport "
        f"process emits it; the wiring is what was missing")


@pytest.mark.slow
def test_the_cell_sees_the_reactor_ammonium_pool():
    """The reactor's ammonium pool must reach ``boundary.external["AMMONIUM"]``.

    ⚠ This is the whole point of making ammonium a finite pool, and it is the
    one link that no other test covers. The path is: the coupler writes
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
