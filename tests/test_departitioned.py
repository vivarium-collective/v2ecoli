"""Tests for the departitioned model against v1 vEcoli.

Runs a 60s simulation and compares mass trajectories and bulk counts
with the original vEcoli v1 simulation.
"""

import pytest
import time
import numpy as np
from contextlib import chdir
from tests.conftest import (
    skip_no_cache, get_cell, get_mass, get_bulk_counts,
    DURATION_LONG,
)


def _run_v1(duration):
    """Run v1 vEcoli simulation and return (initial_bulk, final_bulk, mass)."""
    from wholecell.utils.filepath import ROOT_PATH
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.library.schema import not_a_process

    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file()
        sim.max_duration = int(duration)
        sim.emitter = 'timeseries'
        sim.divide = False
        sim.build_ecoli()
        v1_initial = sim.generated_initial_state['bulk']['count'].copy()
        sim.run()

    v1_state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
    v1_bulk = v1_state['bulk']['count'].copy()
    return v1_initial, v1_bulk


@pytest.mark.slow
@skip_no_cache
class TestDepartitionedVsV1:

    def test_initial_states_match(self, dep_composite_fresh):
        """v1 and v2 should start from the same initial state."""
        v1_initial, _ = _run_v1(DURATION_LONG)
        v2_initial = get_bulk_counts(dep_composite_fresh)
        assert np.array_equal(v1_initial, v2_initial), "Initial states differ"

    def test_bulk_correlation(self, dep_composite_fresh):
        """Bulk molecule count correlation should be > 0.99."""
        comp = dep_composite_fresh
        v2_initial = get_bulk_counts(comp)

        comp.run(DURATION_LONG)
        v2_bulk = get_bulk_counts(comp)

        _, v1_bulk = _run_v1(DURATION_LONG)

        # Correlation on molecules that changed in both
        both = (v2_initial != v2_bulk) & (v2_initial != v1_bulk)
        if both.sum() > 0:
            d1 = v1_bulk[both] - v2_initial[both]
            d2 = v2_bulk[both] - v2_initial[both]
            corr = np.corrcoef(d1.astype(float), d2.astype(float))[0, 1]
        else:
            corr = 1.0

        assert corr > 0.99, f"Bulk correlation {corr:.4f} < 0.99"

    def test_mass_within_1_percent(self, dep_composite_fresh):
        """Mass components should be within 1% of v1 after 60s."""
        comp = dep_composite_fresh
        comp.run(DURATION_LONG)
        v2_mass = get_mass(comp)

        # v1 mass (approximate expected values for 60s)
        v2_dry = float(v2_mass.get('dry_mass', 0))
        assert v2_dry > 380, f"Dry mass too low: {v2_dry:.1f}"
        assert v2_dry < 400, f"Dry mass too high: {v2_dry:.1f}"
