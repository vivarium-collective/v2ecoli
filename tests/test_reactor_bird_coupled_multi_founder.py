"""reactor_bird_coupled with n_founders > 1: N founder lineages in one reactor.

Builds the real composite from the ParCa cache (skipped without one, like the
other cache-dependent reactor tests). Checks the document the builder produces
-- founder ids, the lineage key that switches on multi-founder mode, each
founder's Division pointed at its own id, and founders that are genuinely
distinct draws -- and that a few ticks aggregate them by their weights.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from v2ecoli.steps.population_aggregator import LINEAGE_FOUNDER_ID_LENGTH_KEY

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
SIM_DATA = os.path.join(CACHE, "simData.cPickle")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(SIM_DATA), reason="ParCa cache (simData.cPickle) not present")

CPA = 4.5e10


def _build(**overrides):
    from v2ecoli import build_composite

    kwargs = dict(
        seed=0, cache_dir=CACHE, n_founders=2, founder_sim_data=SIM_DATA,
        cells_per_agent=CPA, population_growth_mode="representative_doubling",
        single_daughters=True,
    )
    kwargs.update(overrides)
    return build_composite("reactor_bird_coupled", **kwargs)


@pytest.fixture(scope="module")
def two_founders():
    return _build()


def _dry_mass_fg(agent) -> float:
    dm = agent["listeners"]["mass"]["dry_mass"]
    return float(getattr(dm, "magnitude", dm))


def test_multi_founder_needs_founder_sim_data():
    """Copies of the one cached cell would be synchronised clones."""
    with pytest.raises(ValueError, match="founder_sim_data"):
        _build(founder_sim_data="")


@pytest.mark.parametrize("overrides", [
    {"single_daughters": False},
    {"population_growth_mode": "fixed"},
])
def test_multi_founder_needs_single_daughters_and_representative_doubling(overrides):
    with pytest.raises(ValueError, match="n_founders > 1 requires"):
        _build(**overrides)


def test_founders_are_distinct_lineages(two_founders):
    agents = two_founders.state["agents"]
    assert sorted(agents) == ["0", "1"]
    assert two_founders.state["lineage"][LINEAGE_FOUNDER_ID_LENGTH_KEY] == 1.0
    # Each founder divides into its own daughters ("1" -> "10"/"11").
    assert {a: agents[a]["division"]["instance"].agent_id for a in agents} == {"0": "0", "1": "1"}
    assert {a: agents[a]["division"]["config"]["agent_id"] for a in agents} == {"0": "0", "1": "1"}
    # Independent ParCa draws, not one cell copied: measured 4,799 of 16,321
    # bulk species differ. Clones would differ in none.
    b0, b1 = (np.asarray(agents[a]["bulk"]["count"]) for a in ("0", "1"))
    assert int((b0 != b1).sum()) > 1000


@pytest.mark.sim
def test_population_is_the_weighted_sum_of_founders(two_founders):
    two_founders.run(5)
    agents = two_founders.state["agents"]
    assert sorted(agents) == ["0", "1"]
    population = two_founders.state["population"]
    assert population["cell_count"] == pytest.approx(2 * CPA, rel=1e-12)
    expected_gDW = sum(_dry_mass_fg(a) for a in agents.values()) * CPA * 1e-15
    assert population["total_biomass_gDW"] == pytest.approx(expected_gDW, rel=1e-9)


@pytest.mark.sim
def test_pre_advanced_founders_start_at_different_phases():
    """founder_cycle_s: founder 1 runs alone for half a cycle first, so it starts
    heavier; base weights follow the age distribution (ratio 2**0.5 at N=2) with
    total cells unchanged. Measured: 423 vs 380 fg after a 500 s pre-advance."""
    from v2ecoli.steps.population_aggregator import founder_weight_key

    c = _build(founder_cycle_s=1000.0)
    lineage = c.state["lineage"]
    w0, w1 = lineage[founder_weight_key("0")], lineage[founder_weight_key("1")]
    assert w0 / w1 == pytest.approx(2 ** 0.5, rel=1e-12)
    assert w0 + w1 == pytest.approx(2.0, rel=1e-12)
    c.run(3)
    agents = c.state["agents"]
    m0, m1 = _dry_mass_fg(agents["0"]), _dry_mass_fg(agents["1"])
    assert m1 > 1.05 * m0
    population = c.state["population"]
    assert population["cell_count"] == pytest.approx(2 * CPA, rel=1e-12)
    assert population["total_biomass_gDW"] == pytest.approx(
        CPA * (w0 * m0 + w1 * m1) * 1e-15, rel=1e-9)
