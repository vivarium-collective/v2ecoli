"""Build + run tests for the reactor_bird_coupled_millard composite (mbp-09 req-2).

The Millard analogue of ``reactor_bird_coupled``: the Millard kinetic-metabolism
cell (``baseline_millard``) + PopulationAggregator + the SAME BiRD reactor
coupling (BiRDTransportHours + ReactorCellCoupler + env-driver(external_store) +
EnvironmentMirror) that ``reactor_bird_coupled`` layers onto the WCM cell.

Milestone: the Millard cell demonstrably consumes reactor O2 end-to-end — the
Millard ODE's respiration (CYTBO) emits a negative
``agents.0.environment.exchange['OXYGEN-MOLECULE']`` (uptake), which the coupler
translates into a depletion of the shared ``reactor.dissolved_o2`` store.
"""

import math

import pytest
from process_bigraph.composite import Composite

from v2ecoli.steps.reactor_cell_coupler import O2_EXCHANGE_KEY


def _f(x) -> float:
    return float(x.magnitude) if hasattr(x, "magnitude") else float(x)


@pytest.mark.sim
def test_builds():
    """The generator is registered and builds a process_bigraph Composite."""
    from v2ecoli import build_composite

    c = build_composite(
        "reactor_bird_coupled_millard", seed=0, cache_dir="out/cache")
    assert isinstance(c, Composite)


@pytest.mark.sim
@pytest.mark.slow
def test_runs_and_consumes_o2():
    """A few coupled Millard+reactor steps run without divergence and the Millard
    cell consumes reactor O2.

    Asserts: no NaN; reactor.dissolved_o2 is present + finite; and the Millard
    cell drives O2 consumption — agents.0.environment.exchange['OXYGEN-MOLECULE']
    is negative (uptake) OR reactor.dissolved_o2 ends below o2_saturation under
    load (cells_per_agent scaled up so the population's O2 demand is non-trivial).
    """
    from v2ecoli import build_composite

    c = build_composite(
        "reactor_bird_coupled_millard", seed=0, cache_dir="out/cache",
        cells_per_agent=1.0e11,
        bird_reactor_config={"gas_flow_rate_Lpm": 2.0},
    )

    o2_exchange_min = 0.0
    for _ in range(5):
        c.run(1)
        do2 = _f(c.state["reactor"]["dissolved_o2"])
        assert math.isfinite(do2), f"dissolved_o2 not finite: {do2!r}"
        exch = c.state["agents"]["0"]["environment"]["exchange"]
        o2_exchange_min = min(o2_exchange_min, float(exch.get(O2_EXCHANGE_KEY, 0.0)))

    reactor = c.state["reactor"]
    assert "dissolved_o2" in reactor
    do_final = _f(reactor["dissolved_o2"])
    assert math.isfinite(do_final)

    cstar = _f(reactor["o2_saturation"])
    consumes = (o2_exchange_min < 0.0) or (do_final < cstar)
    assert consumes, (
        f"Millard cell did not consume reactor O2: "
        f"min O2 exchange={o2_exchange_min}, dissolved_o2={do_final}, "
        f"o2_saturation={cstar}")
