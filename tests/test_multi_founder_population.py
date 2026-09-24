"""Unit tests for multi-founder mode: N founder lineages sharing one environment.

Multi-founder mode is switched on by ``lineage.founder_id_length`` (L). Each
agent then represents ``cells_per_agent * 2**(len(id) - L)`` cells, derived from
its own phylogeny id, instead of the single scalar ``lineage.doublings``. Three
Steps honour it:

* ``PopulationAggregator`` weights each agent's dry mass by its own weight;
* ``LineageBookkeeper`` prunes WITHIN a lineage (one descendant per founder),
  and in single-founder mode refuses to silently prune extra founders;
* ``ReactorCellCoupler`` weights each agent's exchange by its share of the
  represented population instead of a uniform per-agent mean.

Group A: the single-founder path is unchanged, and multi-founder mode at N=1
reproduces it EXACTLY (``==``, not approx). Group B: the new behaviour at N>1,
each with a control that the old behaviour would fail.

Pure ``next_update`` math against synthetic states; no composite, no cache.
"""

from __future__ import annotations

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.lineage_bookkeeper import LineageBookkeeper
from v2ecoli.steps.population_aggregator import (
    GROWTH_MODE_DOUBLING,
    LINEAGE_FOUNDER_ID_LENGTH_KEY,
    PopulationAggregator,
    doublings_since_founder,
    founder_ids,
    founder_of,
    representative_weights,
)
from v2ecoli.steps.reactor_cell_coupler import (
    AVOGADRO,
    MW_O2,
    O2_EXCHANGE_KEY,
    ReactorCellCoupler,
)


@pytest.fixture(scope="module")
def core():
    return build_core()


def _mass_agent(dry_mass_fg: float) -> dict:
    return {"listeners": {"mass": {"dry_mass": dry_mass_fg}}}


def _o2_agent(o2_total: float) -> dict:
    return {"environment": {"exchange": {O2_EXCHANGE_KEY: o2_total}}}


def _aggregator(core, cells_per_agent: float, mode: str = GROWTH_MODE_DOUBLING):
    return PopulationAggregator(
        config={"cells_per_agent": cells_per_agent,
                "population_growth_mode": mode,
                "reactor_volume_L": 1.5},
        core=core,
    )


def _bookkeeper() -> LineageBookkeeper:
    bk = LineageBookkeeper.__new__(LineageBookkeeper)
    bk.initialize({"single_daughters": True})
    return bk


def _multi(id_length: int = 1, **extra) -> dict:
    return {LINEAGE_FOUNDER_ID_LENGTH_KEY: float(id_length), **extra}


# --- Helpers ---------------------------------------------------------------

def test_founder_ids_share_one_length_and_one_founder_is_the_legacy_id():
    assert founder_ids(1) == ["0"]
    assert founder_ids(10) == [str(k) for k in range(10)]
    eleven = founder_ids(11)
    assert eleven[0] == "00" and eleven[-1] == "10"
    assert {len(i) for i in eleven} == {2}


def test_founder_and_depth_are_read_off_the_id():
    assert founder_of("3010", 1) == "3"
    assert doublings_since_founder("3010", 1) == 3
    assert founder_of("07110", 2) == "07"
    assert doublings_since_founder("07110", 2) == 3
    with pytest.raises(ValueError):
        doublings_since_founder("0", 2)


def test_weights_double_per_division_since_founder():
    w = representative_weights(["3", "30", "300"], 1, 1.0e9)
    assert w == {"3": 1.0e9, "30": 2.0e9, "300": 4.0e9}


# --- A. N=1 is unchanged, and multi-founder mode reproduces it exactly -----

@pytest.mark.parametrize("agent_id", ["0", "00", "0000"])
def test_aggregator_n1_multi_founder_is_bit_identical_to_single_founder(core, agent_id):
    agg = _aggregator(core, 9.0e10)
    agents = {agent_id: _mass_agent(379.4)}
    doublings = float(len(agent_id) - 1)
    legacy = agg.next_update(1.0, {"agents": agents, "lineage": {"doublings": doublings}})
    multi = agg.next_update(1.0, {"agents": agents, "lineage": _multi(doublings=doublings)})
    assert multi == legacy


def test_coupler_n1_multi_founder_is_bit_identical_to_single_founder(core):
    def run(lineage):
        c = ReactorCellCoupler(
            config={"cells_per_agent": 9.0e10, "reactor_volume_L": 1.5}, core=core)
        base = {"reactor": {"dissolved_o2": 1.0e6, "dissolved_co2": 0.0, "volume_L": 1.5},
                "population": {"cell_count": 3.6e11}, "lineage": lineage}
        c.next_update(1.0, {**base, "agents": {"000": _o2_agent(-1.0e6)}})
        return c.next_update(1.0, {**base, "agents": {"000": _o2_agent(-4.3e6)}})

    assert run(_multi()) == run({"doublings": 2.0})


# --- B. N > 1 --------------------------------------------------------------

def test_aggregator_weights_each_lineage_by_its_own_doublings(core):
    """Two founders a generation apart: the deeper one stands for twice the cells.

    Control: the single-founder formula (one scalar doublings for everyone) gets
    it wrong for at least one of them, whatever scalar it is given.
    """
    cpa = 1.0e9
    agg = _aggregator(core, cpa)
    agents = {"00": _mass_agent(300.0), "1": _mass_agent(500.0)}
    out = agg.next_update(1.0, {"agents": agents, "lineage": _multi()})["population"]

    expected_mass_fg = 300.0 * cpa * 2 + 500.0 * cpa * 1
    assert out["total_biomass_gDW"] == pytest.approx(expected_mass_fg * 1e-15, rel=1e-12)
    assert out["cell_count"] == pytest.approx(3 * cpa, rel=1e-12)

    for scalar in (0.0, 1.0):
        legacy = agg.next_update(
            1.0, {"agents": agents, "lineage": {"doublings": scalar}})["population"]
        assert legacy["total_biomass_gDW"] != pytest.approx(out["total_biomass_gDW"], rel=1e-6)


def test_division_of_one_lineage_leaves_population_biomass_continuous(core):
    """Founder '3' divides and is pruned to '30' at half mass; founder '0' does not.

    Sum of weight * mass is continuous and only lineage '3' doubles its cell count.
    """
    cpa = 1.0e9
    agg = _aggregator(core, cpa)
    before = agg.next_update(1.0, {
        "agents": {"0": _mass_agent(400.0), "3": _mass_agent(760.0)},
        "lineage": _multi()})["population"]
    after = agg.next_update(1.0, {
        "agents": {"0": _mass_agent(400.0), "30": _mass_agent(380.0)},
        "lineage": _multi()})["population"]
    assert after["total_biomass_gDW"] == pytest.approx(before["total_biomass_gDW"], rel=1e-12)
    assert after["cell_count"] == pytest.approx(before["cell_count"] + cpa, rel=1e-12)


def test_initial_od_is_invariant_in_n_when_cells_per_agent_is_divided(core):
    """1 founder at 9e10 cells and 10 founders at 9e9 each give the same OD600.

    Control: 10 founders left at 9e10 read 10x too dense.
    """
    one = _aggregator(core, 9.0e10).next_update(
        1.0, {"agents": {"0": _mass_agent(379.4)}, "lineage": _multi()})
    ids = founder_ids(10)
    ten = {i: _mass_agent(379.4) for i in ids}
    compensated = _aggregator(core, 9.0e9).next_update(1.0, {"agents": ten, "lineage": _multi()})
    uncompensated = _aggregator(core, 9.0e10).next_update(1.0, {"agents": ten, "lineage": _multi()})

    od1 = one["population"]["OD600"]
    assert compensated["population"]["OD600"] == pytest.approx(od1, rel=1e-12)
    assert uncompensated["population"]["OD600"] == pytest.approx(10 * od1, rel=1e-12)


def test_multi_founder_mode_refuses_fixed_growth_mode(core):
    agg = _aggregator(core, 1.0e9, mode="fixed")
    with pytest.raises(ValueError, match="representative_doubling"):
        agg.next_update(1.0, {"agents": {"0": _mass_agent(1.0)}, "lineage": _multi()})


def test_bookkeeper_prunes_within_each_lineage_never_across():
    bk = _bookkeeper()
    lineage = _multi()
    assert bk.next_update(1.0, {"agents": {"0": {}, "1": {}}, "lineage": lineage}) == {}
    out = bk.next_update(1.0, {"agents": {"00": {}, "01": {}, "1": {}}, "lineage": lineage})
    assert out == {"agents": {"_remove": ["01"]}}
    out = bk.next_update(1.0, {"agents": {"00": {}, "10": {}, "11": {}}, "lineage": lineage})
    assert out == {"agents": {"_remove": ["11"]}}
    assert "lineage" not in out


def test_bookkeeper_refuses_to_prune_founders_in_single_founder_mode():
    """The silent failure this replaces: ten founders became one on tick 1.

    Control: a normal division (siblings '00'/'01') still prunes without error.
    """
    bk = _bookkeeper()
    with pytest.raises(ValueError, match="founders"):
        bk.next_update(1.0, {"agents": {"0": {}, "1": {}}, "lineage": {}})
    out = bk.next_update(1.0, {"agents": {"00": {}, "01": {}}, "lineage": {}})
    assert out["agents"] == {"_remove": ["01"]}


def test_coupler_weights_exchange_by_represented_cells(core):
    """Agents representing 1x and 2x cells_per_agent draw O2 in proportion 1:2.

    Control: the uniform per-agent mean (single-founder formula) splits the same
    population evenly, and gets a different answer.
    """
    cpa, volume_L = 1.0e9, 2.0
    d_shallow, d_deep = -3.0e6, -5.0e6          # per-tick counts, one cell each

    def second_tick(lineage):
        c = ReactorCellCoupler(
            config={"cells_per_agent": cpa, "reactor_volume_L": volume_L}, core=core)
        base = {"reactor": {"dissolved_o2": 1.0e6, "dissolved_co2": 0.0, "volume_L": volume_L},
                "population": {"cell_count": 3 * cpa}, "lineage": lineage}
        c.next_update(1.0, {**base, "agents": {"1": _o2_agent(0.0), "00": _o2_agent(0.0)}})
        return c.next_update(
            1.0, {**base, "agents": {"1": _o2_agent(d_shallow), "00": _o2_agent(d_deep)}}
        )["reactor"]["dissolved_o2"]

    per_count_mgL = 1.0 / AVOGADRO * 1000.0 / volume_L * MW_O2
    expected = (cpa * d_shallow + 2 * cpa * d_deep) * per_count_mgL
    assert second_tick(_multi()) == pytest.approx(expected, rel=1e-12)

    uniform = (1.5 * cpa) * (d_shallow + d_deep) * per_count_mgL
    assert second_tick({}) == pytest.approx(uniform, rel=1e-12)
    assert uniform != pytest.approx(expected, rel=1e-6)


def test_coupler_first_tick_fallback_uses_the_same_weights(core):
    """Before the aggregator's cell_count reaches the coupler, it falls back to
    the summed weights -- the same number the aggregator would publish."""
    cpa = 1.0e9
    c = ReactorCellCoupler(config={"cells_per_agent": cpa, "reactor_volume_L": 1.0}, core=core)
    states = {"reactor": {"dissolved_o2": 1.0e6, "dissolved_co2": 0.0, "volume_L": 1.0},
              "population": {}, "lineage": _multi(),
              "agents": {"1": _o2_agent(0.0), "00": _o2_agent(0.0)}}
    c.next_update(1.0, states)
    assert c.scale_fallbacks == 1
    states["agents"] = {"1": _o2_agent(-3.0e6), "00": _o2_agent(-5.0e6)}
    out = c.next_update(1.0, states)["reactor"]["dissolved_o2"]
    expected = (cpa * -3.0e6 + 2 * cpa * -5.0e6) / AVOGADRO * 1000.0 / 1.0 * MW_O2
    assert out == pytest.approx(expected, rel=1e-12)
