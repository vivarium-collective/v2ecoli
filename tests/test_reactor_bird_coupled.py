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


def test_reactor_store_exposes_the_full_gas_transfer_and_ledger_surface():
    """The composite must EXPOSE what the study readouts declare.

    ⚠ Three gaps this pins, all of which were silent:

    * ``kla_co2`` is emitted by ``BiRDTransportProcess.outputs()`` exactly as
      ``kla_o2`` is, but was never declared or wired here -- so the CO2 side of
      the gas transfer was unobservable while the O2 side was fully
      instrumented. (`reactor.kla_co2` is one of the readouts a study declares
      and the model did not expose.)
    * ``ammonium_medium_mM`` did not exist, so the nitrogen ledger had no input
      term and mbp-04's declared ``nitrogen_residual`` could not be evaluated.
    * ``reactor.diagnostics`` was not a subtree of the built composite AT ALL --
      zero of 1222 leaves matched "diagnostic" -- while mbp-03/04/05 declare
      seven ``reactor.diagnostics.*`` readouts between them.

    ⚠ And the trap that makes these silent: the coupler's ``InPlaceDict``
    reactor port DROPS any leaf the per-leaf output schema does not enumerate.
    A write to an unenumerated leaf goes nowhere and raises nothing, so a
    build-time presence check is the only thing that catches it.
    """
    from v2ecoli import build_composite
    from v2ecoli.steps.reactor_cell_coupler import DIAGNOSTIC_LEAVES

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache",
                        cells_per_agent=1.0e12, initial_glucose_mM=40.0)
    reactor = c.state["reactor"]

    for leaf in ("kla_o2", "kla_co2", "glucose_medium_mM", "ammonium_medium_mM"):
        assert leaf in reactor, f"reactor.{leaf} is not exposed by the composite"
    assert "diagnostics" in reactor, "reactor.diagnostics subtree does not exist"
    for leaf in DIAGNOSTIC_LEAVES:
        assert leaf in reactor["diagnostics"], (
            f"reactor.diagnostics.{leaf} declared in DIAGNOSTIC_LEAVES but not "
            f"seeded in the store -- study readouts naming it resolve to nothing")

    # Presence is not enough: a leaf can exist and never be written (that is
    # precisely how the medium leaves failed). Drive a tick and require the
    # coupler's writes to actually land.
    c.run(2)
    reactor = c.state["reactor"]
    assert reactor["kla_co2"] > 0.0, (
        f"kla_co2 exposed but never written (got {reactor['kla_co2']}) -- the "
        f"transport process emits it; the wiring is what was missing")
    assert reactor["ammonium_medium_mM"] > 0.0, "ammonium pool not seeded"
    assert reactor["diagnostics"]["carbon_in_mM"] >= 0.0
