"""Integration tests for v2ecoli simulation.

Validates that the departitioned composite loads, runs, and produces
reasonable results.
"""

import pytest
import numpy as np
from tests.conftest import (
    skip_no_cache, get_cell, get_mass, get_bulk_counts,
    DURATION_SHORT,
)


@skip_no_cache
class TestCompositeLoad:

    def test_composite_has_steps(self, dep_composite):
        """Composite should have biological process steps."""
        assert len(dep_composite.step_paths) > 30

    def test_composite_has_bulk(self, dep_composite):
        """Cell should have bulk molecule array."""
        bulk = get_bulk_counts(dep_composite)
        assert len(bulk) > 10000

    def test_initial_dry_mass(self, dep_composite):
        """Initial dry mass should be ~380 fg."""
        mass = get_mass(dep_composite)
        dry = float(mass.get('dry_mass', 0))
        assert 350 < dry < 420, f"Initial dry_mass={dry:.1f}, expected ~380"


@skip_no_cache
class TestSimulationRun:

    def test_run_10s(self, dep_composite_fresh):
        """Run 10s simulation: bulk molecules should change."""
        comp = dep_composite_fresh
        bulk_before = get_bulk_counts(comp)

        comp.run(DURATION_SHORT)

        bulk_after = get_bulk_counts(comp)
        changed = (bulk_before != bulk_after).sum()
        assert changed > 0, "No bulk molecules changed after 10s"

    def test_mass_growth(self, dep_composite_fresh):
        """Dry mass should increase during simulation."""
        comp = dep_composite_fresh
        mass_before = float(get_mass(comp).get('dry_mass', 0))

        comp.run(DURATION_SHORT)

        mass_after = float(get_mass(comp).get('dry_mass', 0))
        assert mass_after > mass_before, (
            f"No growth: dry_mass {mass_before:.2f} -> {mass_after:.2f}")

    def test_global_time_advances(self, dep_composite_fresh):
        """Global time should advance by the run duration."""
        comp = dep_composite_fresh
        comp.run(DURATION_SHORT)
        assert comp.state['global_time'] == DURATION_SHORT
