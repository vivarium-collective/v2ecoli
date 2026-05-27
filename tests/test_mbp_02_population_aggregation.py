"""Behavior tests for mbp-02-population-aggregation.

Each test corresponds 1:1 to a behavior_test entry in
``studies/mbp-02-population-aggregation/study.yaml`` (post
chris_feedback_2026_05_26 §4 reframe — cells_per_agent representative-
sampling scaling adopted; pure consistency invariants added alongside
calibration-dependent tests; OD600 declared cosmetic).

Tests that exercise a single short composite run (≤ 2 s) are implemented.
Tests that require a multi-generation sim (cell-count-doubles-per-generation,
total-biomass-grows-exponentially) remain @pytest.mark.skip with a
"needs slow multi-gen sim" reason — they belong in a nightly-job suite.

Test grouping mirrors the study.yaml structure:
  - Calibration-dependent tests (assume v2ecoli baseline μ as fixed prior)
  - Pure consistency invariants (decoupled from calibration)
  - Scaling-factor checks (the cells_per_agent architectural decision)
"""

from __future__ import annotations

import pytest

from process_bigraph import Composite

from v2ecoli.composites.baseline import baseline
from v2ecoli.composites.baseline_population import baseline_population
from v2ecoli.core import build_core
from v2ecoli.steps.population_aggregator import FG_PER_GRAM


pytestmark = pytest.mark.sim


# Shared core + 1-s run snapshot (build_core + composite.run are expensive).
@pytest.fixture(scope="module")
def core():
    return build_core()


def _run_one_second(core, *, builder=baseline_population, **kwargs) -> dict:
    """Build a composite, run for 1 second, return the live state dict."""
    doc = builder(core=core, seed=0, cache_dir="out/cache", **kwargs)
    comp = Composite(doc, core=core)
    comp.run(1)
    return comp.state


@pytest.fixture(scope="module")
def baseline_state(core):
    """1-second snapshot of the unaggregated baseline composite."""
    doc = baseline(core=core, seed=0, cache_dir="out/cache")
    comp = Composite(doc, core=core)
    comp.run(1)
    return comp.state


@pytest.fixture(scope="module")
def aggregated_state(core):
    """1-second snapshot of baseline_population at default cells_per_agent=1.0."""
    return _run_one_second(core)


# --- Calibration-dependent tests --------------------------------------------

@pytest.mark.skip(reason="Needs slow multi-gen sim (≥ 3 doublings ≈ 180 min); belongs in a nightly suite")
def test_cell_count_doubles_per_generation():
    """cell_count doubles approximately every doubling-time (~60 min on
    M9-glucose); ratio (final/initial) ∈ [6, 24] over a 240-min run absorbing
    stochasticity. Backs study.yaml behavior_test
    ``cell-count-doubles-per-generation``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Needs slow multi-gen sim (≥ 3 doublings ≈ 180 min); belongs in a nightly suite")
def test_total_biomass_grows_exponentially():
    """Fitted exponential rate of population.total_biomass_gDW over a multi-
    gen run matches the per-cell μ band [0.3, 0.9] /h.
    Backs study.yaml behavior_test ``total-biomass-grows-exponentially``.
    """
    raise NotImplementedError


def test_per_cell_observables_unchanged_vs_baseline(baseline_state, aggregated_state):
    """Regression guard: agents/0 state in baseline_population is identical
    to baseline at the same seed — aggregation is purely additive, doesn't
    perturb cell state. Backs study.yaml behavior_test
    ``per-cell-observables-unchanged-vs-baseline``.
    """
    cell_mass_base = float(baseline_state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    cell_mass_agg = float(aggregated_state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    assert cell_mass_agg == pytest.approx(cell_mass_base, rel=1e-12), (
        f"baseline cell_mass={cell_mass_base!r}, baseline_population cell_mass={cell_mass_agg!r}"
    )


def test_single_cell_aggregation_matches_cell_mass(aggregated_state):
    """At cells_per_agent=1.0 (default) with one agent,
    population.total_biomass_gDW == agents.0.listeners.mass.cell_mass × 1e-15.
    Backs study.yaml behavior_test ``single-cell-aggregation-matches-cell-mass``.
    """
    cell_mass_fg = float(aggregated_state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    pop_gdw = aggregated_state["population"]["total_biomass_gDW"]
    expected = cell_mass_fg * FG_PER_GRAM
    assert pop_gdw == pytest.approx(expected, rel=1e-9)


# --- Pure consistency invariants --------------------------------------------

def test_population_count_equals_len_agents(aggregated_state):
    """population.cell_count == len(agents.*) × cells_per_agent (default 1.0).
    Backs study.yaml behavior_test ``population_count_equals_len_agents``.
    """
    n_agents = len(aggregated_state["agents"])
    cell_count = aggregated_state["population"]["cell_count"]
    assert cell_count == pytest.approx(float(n_agents), rel=1e-12)


def test_total_biomass_equals_sum_cell_mass_times_scale(aggregated_state):
    """population.total_biomass_gDW ==
       sum(agents.*.listeners.mass.cell_mass) × cells_per_agent × 1e-15.
    Backs study.yaml behavior_test
    ``total_biomass_equals_sum_cell_mass_times_scale``.
    """
    sum_cell_mass_fg = sum(
        float(agent["listeners"]["mass"]["cell_mass"])
        for agent in aggregated_state["agents"].values()
    )
    expected = sum_cell_mass_fg * 1.0 * FG_PER_GRAM   # cells_per_agent=1.0 default
    actual = aggregated_state["population"]["total_biomass_gDW"]
    assert actual == pytest.approx(expected, rel=1e-9)


# --- cells_per_agent scaling-factor checks ----------------------------------

@pytest.mark.parametrize("cells_per_agent", [1.0, 1.0e6, 1.0e9])
def test_aggregator_output_scales_linearly_with_cells_per_agent(core, cells_per_agent):
    """At matched cell state, population.* scales linearly with
    cells_per_agent. Validates the representative-sampling architectural
    decision (chris_feedback_2026_05_26 §4). Backs study.yaml behavior_test
    ``aggregator-output-scales-linearly-with-cells-per-agent``.

    We exercise the actual composite (rather than only the unit-test path)
    so the test catches any wiring-side bug where cells_per_agent fails to
    propagate from the composite parameter to the Step instance config.
    """
    state = _run_one_second(core, cells_per_agent=cells_per_agent)
    cell_mass_fg = float(state["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    expected_gdw = cell_mass_fg * cells_per_agent * FG_PER_GRAM
    assert state["population"]["total_biomass_gDW"] == pytest.approx(expected_gdw, rel=1e-9)
    assert state["population"]["cell_count"] == pytest.approx(cells_per_agent, rel=1e-12)


def test_per_cell_observables_invariant_under_scaling(core):
    """Across cells_per_agent ∈ {1, 1e6}, per-cell cell_mass is byte-identical
    — cells_per_agent only affects aggregator output, never per-cell state.
    Backs study.yaml behavior_test ``per-cell-observables-invariant-under-scaling``.
    """
    s1 = _run_one_second(core, cells_per_agent=1.0)
    s2 = _run_one_second(core, cells_per_agent=1.0e6)
    m1 = float(s1["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    m2 = float(s2["agents"]["0"]["listeners"]["mass"]["cell_mass"])
    assert m1 == m2, (
        f"per-cell cell_mass changed when cells_per_agent was scaled "
        f"(1.0: {m1!r}, 1e6: {m2!r}) — aggregator is leaking the scaling "
        f"factor into per-cell processes"
    )
