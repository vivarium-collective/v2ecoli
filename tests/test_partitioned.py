"""Tests for the partitioned model.

Validates that the partitioned composite loads, runs, and produces
growth consistent with a working E. coli simulation.
"""

import pytest
import numpy as np
from tests.conftest import (
    skip_no_cache, get_cell, get_mass, get_bulk_counts,
    DURATION_SHORT, DURATION_LONG,
)


@skip_no_cache
class TestPartitionedLoad:

    def test_composite_has_more_steps(self, part_composite, dep_composite):
        """Partitioned should have more steps (requester+evolver pairs + allocators)."""
        assert len(part_composite.step_paths) > len(dep_composite.step_paths)

    def test_initial_dry_mass(self, part_composite):
        """Initial dry mass should be ~380 fg."""
        mass = get_mass(part_composite)
        dry = float(mass.get('dry_mass', 0))
        assert 350 < dry < 420, f"Initial dry_mass={dry:.1f}"


@pytest.mark.slow
@skip_no_cache
class TestPartitionedRun:

    def test_run_10s_bulk_changes(self, part_composite_fresh):
        """Bulk molecules should change after 10s."""
        comp = part_composite_fresh
        bulk_before = get_bulk_counts(comp)

        comp.run(DURATION_SHORT)

        bulk_after = get_bulk_counts(comp)
        changed = (bulk_before != bulk_after).sum()
        assert changed > 0, "No bulk molecules changed"

    def test_mass_growth(self, part_composite_fresh):
        """Dry mass should increase."""
        comp = part_composite_fresh
        mass_before = float(get_mass(comp).get('dry_mass', 0))

        comp.run(DURATION_SHORT)

        mass_after = float(get_mass(comp).get('dry_mass', 0))
        assert mass_after > mass_before, (
            f"No growth: dry_mass {mass_before:.2f} -> {mass_after:.2f}")
