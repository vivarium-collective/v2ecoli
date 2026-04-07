"""Shared pytest fixtures for v2ecoli tests."""

import os
import pytest
import numpy as np

CACHE_DIR = os.environ.get('V2ECOLI_CACHE', 'out/cache')
DURATION_SHORT = 10.0
DURATION_LONG = 60.0


def _cache_available():
    return (os.path.isdir(CACHE_DIR) and
            os.path.exists(os.path.join(CACHE_DIR, 'sim_data_cache.dill')))


skip_no_cache = pytest.mark.skipif(
    not _cache_available(),
    reason=f"Cache not found at {CACHE_DIR}")


@pytest.fixture(scope='session')
def cache_dir():
    """Path to the simulation cache directory."""
    if not _cache_available():
        pytest.skip(f"Cache not found at {CACHE_DIR}")
    return CACHE_DIR


@pytest.fixture(scope='session')
def dep_composite(cache_dir):
    """Departitioned composite, built once per test session."""
    from v2ecoli.composite import make_composite
    return make_composite(cache_dir=cache_dir)


@pytest.fixture(scope='session')
def part_composite(cache_dir):
    """Partitioned composite, built once per test session."""
    from v2ecoli.partitioned import make_partitioned_composite
    return make_partitioned_composite(cache_dir=cache_dir)


@pytest.fixture
def dep_composite_fresh(cache_dir):
    """Fresh departitioned composite (not shared across tests)."""
    from v2ecoli.composite import make_composite
    return make_composite(cache_dir=cache_dir)


@pytest.fixture
def part_composite_fresh(cache_dir):
    """Fresh partitioned composite (not shared across tests)."""
    from v2ecoli.partitioned import make_partitioned_composite
    return make_partitioned_composite(cache_dir=cache_dir)


def get_cell(composite):
    """Extract the cell agent state from a composite."""
    return composite.state.get('agents', {}).get('0', {})


def get_mass(composite):
    """Extract mass listener values from a composite."""
    cell = get_cell(composite)
    return cell.get('listeners', {}).get('mass', {})


def get_bulk_counts(composite):
    """Extract bulk molecule count array from a composite."""
    cell = get_cell(composite)
    bulk = cell.get('bulk')
    if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
        return bulk['count'].copy()
    return np.array([])


def assert_mass_close(mass, field, expected, rtol=0.01, msg=''):
    """Assert a mass field is within rtol of expected value."""
    actual = float(mass.get(field, 0))
    diff_pct = abs(actual - expected) / max(abs(expected), 1e-12) * 100
    assert diff_pct < rtol * 100, (
        f"{msg}{field}: {actual:.4f} vs {expected:.4f} "
        f"({diff_pct:.4f}% > {rtol*100:.1f}%)"
    )
