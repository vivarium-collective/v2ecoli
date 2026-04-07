"""Cross-architecture comparison tests.

Runs both partitioned and departitioned models from the same initial
state and validates they produce equivalent results within tolerance.
"""

import pytest
import numpy as np
from tests.conftest import (
    skip_no_cache, get_cell, get_mass, get_bulk_counts,
    DURATION_LONG,
)


@pytest.mark.slow
@skip_no_cache
class TestArchitectureEquivalence:
    """Compare partitioned vs departitioned models."""

    MASS_FIELDS = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass']
    MAX_MASS_ERROR_PCT = 1.0  # max acceptable percent difference per field
    MIN_BULK_CORRELATION = 0.999

    def test_mass_trajectories_agree(self, cache_dir):
        """Mass components should agree within 1% over 60s."""
        from v2ecoli.composite import make_composite
        from v2ecoli.partitioned import make_partitioned_composite

        dep = make_composite(cache_dir=cache_dir)
        part = make_partitioned_composite(cache_dir=cache_dir)

        dep.run(DURATION_LONG)
        part.run(DURATION_LONG)

        dep_mass = get_mass(dep)
        part_mass = get_mass(part)

        for field in self.MASS_FIELDS:
            dv = float(dep_mass.get(field, 0))
            pv = float(part_mass.get(field, 0))
            ref = max(abs(dv), abs(pv), 1e-12)
            pct_diff = abs(dv - pv) / ref * 100
            assert pct_diff < self.MAX_MASS_ERROR_PCT, (
                f"{field}: dep={dv:.4f} part={pv:.4f} ({pct_diff:.4f}% > "
                f"{self.MAX_MASS_ERROR_PCT}%)")

    def test_bulk_correlation(self, cache_dir):
        """Final bulk counts should correlate > 0.999."""
        from v2ecoli.composite import make_composite
        from v2ecoli.partitioned import make_partitioned_composite

        dep = make_composite(cache_dir=cache_dir)
        part = make_partitioned_composite(cache_dir=cache_dir)

        dep.run(DURATION_LONG)
        part.run(DURATION_LONG)

        dep_bulk = get_bulk_counts(dep)
        part_bulk = get_bulk_counts(part)

        min_len = min(len(dep_bulk), len(part_bulk))
        assert min_len > 0, "No bulk data"

        db = dep_bulk[:min_len].astype(float)
        pb = part_bulk[:min_len].astype(float)
        mask = (db > 0) | (pb > 0)
        if mask.sum() > 1:
            corr = np.corrcoef(db[mask], pb[mask])[0, 1]
        else:
            corr = 1.0

        assert corr > self.MIN_BULK_CORRELATION, (
            f"Bulk correlation {corr:.6f} < {self.MIN_BULK_CORRELATION}")

    def test_both_grow(self, cache_dir):
        """Both architectures should show positive mass growth."""
        from v2ecoli.composite import make_composite
        from v2ecoli.partitioned import make_partitioned_composite

        dep = make_composite(cache_dir=cache_dir)
        part = make_partitioned_composite(cache_dir=cache_dir)

        dep_initial = float(get_mass(dep).get('dry_mass', 0))
        part_initial = float(get_mass(part).get('dry_mass', 0))

        dep.run(DURATION_LONG)
        part.run(DURATION_LONG)

        dep_final = float(get_mass(dep).get('dry_mass', 0))
        part_final = float(get_mass(part).get('dry_mass', 0))

        assert dep_final > dep_initial, (
            f"Departitioned: no growth {dep_initial:.2f} -> {dep_final:.2f}")
        assert part_final > part_initial, (
            f"Partitioned: no growth {part_initial:.2f} -> {part_final:.2f}")
