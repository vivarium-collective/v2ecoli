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

import pytest
from process_bigraph.composite import Composite


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
