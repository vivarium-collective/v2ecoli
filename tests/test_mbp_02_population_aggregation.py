"""Behavior tests for mbp-02-population-aggregation.

Scaffold for Build phase. Each test corresponds 1:1 to a behavior_test entry
in ``studies/mbp-02-population-aggregation/study.yaml`` (post
chris_feedback_2026_05_26 §4 reframe — cells_per_agent representative-
sampling scaling adopted; pure consistency invariants added alongside
calibration-dependent tests; OD600 declared cosmetic).

Tests are CURRENTLY SKIPPED with the reason "Build-phase scaffold —
PopulationAggregator wire-up TODO" so CI doesn't go red on the in-progress
work. As each TODO in
``v2ecoli/composites/baseline_population.py`` lands, the corresponding
``@pytest.mark.skip`` decorator should be removed.

Test grouping mirrors the study.yaml structure:
  - Calibration-dependent tests (assume v2ecoli baseline μ as fixed prior)
  - Pure consistency invariants (decoupled from calibration)
  - Scaling-factor checks (the cells_per_agent architectural decision)
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.sim


# --- Calibration-dependent tests --------------------------------------------

@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_cell_count_doubles_per_generation():
    """cell_count doubles approximately every doubling-time (~60 min on
    M9-glucose); ratio (final/initial) ≈ 2^(duration/doubling_time), with a
    band of [6, 24] over a 240-min run absorbing stochasticity.

    Implicitly assumes calibrated v2ecoli baseline μ as a fixed prior — drift
    here should first prompt a baseline-composite drift check, not just an
    aggregator check (per chris_feedback_2026_05_26 §4 framing).

    Backs study.yaml behavior_test ``cell-count-doubles-per-generation``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_total_biomass_grows_exponentially():
    """Fitted exponential rate of population.total_biomass_gDW over the run
    matches the per-cell μ band [0.3, 0.9] /h (v2ecoli-calibrated μ ≈ 0.5-0.7).

    Backs study.yaml behavior_test ``total-biomass-grows-exponentially``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_per_cell_observables_unchanged_vs_baseline():
    """Regression guard: per-cell DnaA count distribution matches the
    unaggregated v2ecoli.composites.baseline — aggregation is purely additive,
    doesn't perturb cell state.

    Backs study.yaml behavior_test
    ``per-cell-observables-unchanged-vs-baseline``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_single_cell_aggregation_matches_cell_mass():
    """With division disabled and cells_per_agent=1.0 (default),
    population.total_biomass_gDW == agents.0.listeners.mass.cell_mass × 1e-15
    every timestep. Identity at default scale.

    Backs study.yaml behavior_test ``single-cell-aggregation-matches-cell-mass``.
    """
    raise NotImplementedError


# --- Pure consistency invariants --------------------------------------------

@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_population_count_equals_len_agents():
    """At every emit step:
    population.cell_count == len(agents.*) × cells_per_agent.
    Pure aggregator-only check; independent of doubling-time calibration —
    isolates aggregator bugs from biology drift.

    Backs study.yaml behavior_test ``population_count_equals_len_agents``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_total_biomass_equals_sum_cell_mass_times_scale():
    """At every emit step:
    population.total_biomass_gDW ==
        sum(agents.*.listeners.mass.cell_mass) × cells_per_agent × 1e-15
    (relative tolerance 1e-9). Pure aggregator-only check; decoupled from
    calibration.

    Backs study.yaml behavior_test
    ``total_biomass_equals_sum_cell_mass_times_scale``.
    """
    raise NotImplementedError


# --- cells_per_agent scaling-factor checks ----------------------------------

@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_aggregator_output_scales_linearly_with_cells_per_agent():
    """Across cells_per_agent ∈ {1, 1e6, 1e9}, population.total_biomass_gDW
    and population.cell_count at matched simulation time scale linearly with
    cells_per_agent (no off-by-k error, no clipping; rtol 1e-9). Validates
    the representative-sampling architectural decision
    (chris_feedback_2026_05_26 §4).

    Backs study.yaml behavior_test
    ``aggregator-output-scales-linearly-with-cells-per-agent``.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="Build-phase scaffold — PopulationAggregator wire-up TODO")
def test_per_cell_observables_invariant_under_scaling():
    """Across cells_per_agent ∈ {1, 1e6, 1e9}, per-cell DnaA count and
    per-cell μ are byte-identical (cells_per_agent only affects aggregator
    output, never per-cell state). Regression guard against the aggregator
    leaking the scaling factor into per-cell processes.

    Backs study.yaml behavior_test
    ``per-cell-observables-invariant-under-scaling``.
    """
    raise NotImplementedError
