"""Unit tests for :class:`v2ecoli.steps.population_aggregator.PopulationAggregator`.

Targets the Step's ``next_update`` math directly against synthetic agent
dicts — no composite, no ParCa cache, no baseline document. Fast, runs
without the @pytest.mark.sim marker.

These do NOT substitute for the behavior tests in
``tests/test_mbp_01_time_varying_environment.py`` /
``tests/test_mbp_02_population_aggregation.py`` (which require the
composite wire-up to lift their @pytest.mark.skip). They validate the
new code's pure aggregation math under the chris_feedback_2026_05_26 §4
cells_per_agent design.
"""

from __future__ import annotations

import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.population_aggregator import (
    DEFAULT_OD_TO_GDW,
    FG_PER_GRAM,
    PopulationAggregator,
)


# Shared core (build_core is expensive).
@pytest.fixture(scope="module")
def core():
    return build_core()


def _agent(cell_mass_fg: float) -> dict:
    """Build a minimal agent state dict with just the cell_mass listener."""
    return {"listeners": {"mass": {"cell_mass": cell_mass_fg}}}


# --- Construction / config plumbing -----------------------------------------

def test_defaults_match_module_constants(core):
    p = PopulationAggregator(config={}, core=core)
    assert p.cells_per_agent == pytest.approx(1.0)
    assert p.od_to_gdw == pytest.approx(DEFAULT_OD_TO_GDW)   # 0.34 (Beulig)
    assert p.reactor_volume_L == pytest.approx(1.0)


def test_config_overrides_take_effect(core):
    """Regression guard: bigraph-schema config_schema must use type names
    (not {"_default": ...}) for overrides to propagate. See the schema
    comment block in population_aggregator.py."""
    p = PopulationAggregator(
        config={"cells_per_agent": 7.0, "od_to_gdw": 0.5, "reactor_volume_L": 2.0},
        core=core,
    )
    assert p.cells_per_agent == pytest.approx(7.0)
    assert p.od_to_gdw == pytest.approx(0.5)
    assert p.reactor_volume_L == pytest.approx(2.0)


# --- Empty / single / multi-agent aggregation -------------------------------

def test_empty_population_emits_zeros(core):
    """Used by mbp-03's no-cells-henry-equilibrium sim (force_zero_population)."""
    p = PopulationAggregator(config={}, core=core)
    out = p.next_update(1.0, {"agents": {}})
    pop = out["population"]
    assert pop["total_biomass_gDW"] == 0.0
    assert pop["cell_count"] == 0.0
    assert pop["biomass_concentration_gL"] == 0.0
    assert pop["OD600"] == 0.0


def test_single_cell_aggregation_matches_cell_mass(core):
    """At cells_per_agent=1.0 (default), total_biomass_gDW equals
    cell_mass × 1e-15 exactly. Mirrors mbp-02's
    single-cell-aggregation-matches-cell-mass behavior test under the
    unit-test framing.
    """
    cell_mass_fg = 1.0e15   # 1 g
    p = PopulationAggregator(config={}, core=core)
    out = p.next_update(1.0, {"agents": {"0": _agent(cell_mass_fg)}})
    pop = out["population"]
    assert pop["cell_count"] == pytest.approx(1.0)
    assert pop["total_biomass_gDW"] == pytest.approx(cell_mass_fg * FG_PER_GRAM)
    # biomass_concentration / OD600 derivations
    assert pop["biomass_concentration_gL"] == pytest.approx(1.0)   # vol = 1 L
    assert pop["OD600"] == pytest.approx(1.0 / DEFAULT_OD_TO_GDW)


def test_multi_agent_sum(core):
    """Aggregator sums cell_mass across agents (the literal-sum
    aggregator under cells_per_agent=1.0)."""
    p = PopulationAggregator(config={}, core=core)
    agents = {str(i): _agent(1.0e15) for i in range(5)}   # 5 agents, 1 g each
    out = p.next_update(1.0, {"agents": agents})
    pop = out["population"]
    assert pop["cell_count"] == pytest.approx(5.0)
    assert pop["total_biomass_gDW"] == pytest.approx(5.0)


# --- Pure consistency invariants (chris_feedback_2026_05_26 §4) -------------

@pytest.mark.parametrize("n_agents,cells_per_agent", [(1, 1.0), (4, 1.0), (4, 1.0e6), (16, 1.0e9)])
def test_population_count_equals_len_agents_times_scale(core, n_agents, cells_per_agent):
    """population.cell_count == len(agents.*) × cells_per_agent at every step.
    Pure aggregator-only check; mirrors mbp-02 behavior test
    `population_count_equals_len_agents` under the unit-test framing.
    """
    p = PopulationAggregator(config={"cells_per_agent": cells_per_agent}, core=core)
    agents = {str(i): _agent(1.0e15) for i in range(n_agents)}
    out = p.next_update(1.0, {"agents": agents})
    assert out["population"]["cell_count"] == pytest.approx(n_agents * cells_per_agent)


@pytest.mark.parametrize("cells_per_agent", [1.0, 1.0e3, 1.0e6, 1.0e9])
def test_total_biomass_equals_sum_cell_mass_times_scale(core, cells_per_agent):
    """population.total_biomass_gDW ==
       sum(agents.*.listeners.mass.cell_mass) × cells_per_agent × 1e-15.
    Pure aggregator-only check; mirrors mbp-02 behavior test
    `total_biomass_equals_sum_cell_mass_times_scale`.
    """
    cell_masses = [1.0e15, 2.0e15, 3.0e15, 4.0e15]
    agents = {str(i): _agent(m) for i, m in enumerate(cell_masses)}
    expected_gDW = sum(cell_masses) * cells_per_agent * FG_PER_GRAM
    p = PopulationAggregator(config={"cells_per_agent": cells_per_agent}, core=core)
    out = p.next_update(1.0, {"agents": agents})
    assert out["population"]["total_biomass_gDW"] == pytest.approx(expected_gDW, rel=1e-9)


# --- cells_per_agent scaling-factor checks ----------------------------------

def test_aggregator_output_scales_linearly_with_cells_per_agent(core):
    """Across cells_per_agent ∈ {1, 1e6, 1e9}, total_biomass_gDW and
    cell_count at matched simulation time scale linearly with cells_per_agent
    (no off-by-k error, no clipping). Validates the representative-sampling
    architectural decision (chris_feedback_2026_05_26 §4); mirrors mbp-02
    behavior test `aggregator-output-scales-linearly-with-cells-per-agent`.
    """
    agents = {str(i): _agent(1.0e15) for i in range(3)}
    state = {"agents": agents}

    p1 = PopulationAggregator(config={"cells_per_agent": 1.0}, core=core)
    p6 = PopulationAggregator(config={"cells_per_agent": 1.0e6}, core=core)
    p9 = PopulationAggregator(config={"cells_per_agent": 1.0e9}, core=core)

    out1 = p1.next_update(1.0, state)["population"]
    out6 = p6.next_update(1.0, state)["population"]
    out9 = p9.next_update(1.0, state)["population"]

    # cell_count scales linearly
    assert out6["cell_count"] == pytest.approx(out1["cell_count"] * 1.0e6, rel=1e-9)
    assert out9["cell_count"] == pytest.approx(out1["cell_count"] * 1.0e9, rel=1e-9)
    # total_biomass_gDW scales linearly
    assert out6["total_biomass_gDW"] == pytest.approx(out1["total_biomass_gDW"] * 1.0e6, rel=1e-9)
    assert out9["total_biomass_gDW"] == pytest.approx(out1["total_biomass_gDW"] * 1.0e9, rel=1e-9)


def test_aggregator_never_writes_agents_store(core):
    """The aggregator's outputs schema declares only `population` (per
    `topology` + `outputs()`); the next_update return MUST NOT contain an
    `agents` key. Regression guard against the aggregator leaking the
    scaling factor into per-cell processes (mirrors mbp-02's
    per-cell-observables-invariant-under-scaling test in the unit-test
    framing).
    """
    p = PopulationAggregator(config={"cells_per_agent": 1.0e9}, core=core)
    agents = {str(i): _agent(1.0e15) for i in range(3)}
    out = p.next_update(1.0, {"agents": agents})
    assert "agents" not in out


# --- Defensive: missing cell_mass / odd agent shapes -----------------------

def test_agent_without_cell_mass_is_skipped(core):
    """A snapshot mid-init might not yet have cell_mass populated. The
    aggregator should skip the agent (sum what it can find) rather than
    crash. cell_count still reflects len(agents) since the agent exists."""
    p = PopulationAggregator(config={}, core=core)
    agents = {
        "good": _agent(1.0e15),
        "no_mass": {"listeners": {"mass": {}}},
        "no_listeners": {},
    }
    out = p.next_update(1.0, {"agents": agents})
    pop = out["population"]
    assert pop["cell_count"] == pytest.approx(3.0)   # all 3 counted
    assert pop["total_biomass_gDW"] == pytest.approx(1.0e15 * FG_PER_GRAM)
