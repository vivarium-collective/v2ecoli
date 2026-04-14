"""Regression tests for the three architectures: baseline, departitioned, reconciled.

Each commit that changes process classes or the generate*.py files should keep
all three architectures (a) instantiable, (b) able to run, and (c) producing
positive dry_mass growth over a short interval.

The motivating incident: commits 1a5c0ab and 0e282f5 promoted 8 processes
from PartitionedProcess to plain Step and updated baseline generate.py.
generate_departitioned.py and generate_reconciled.py silently dropped the
8 processes because they weren't registered as STANDALONE_STEPS — the
simulation ran but produced no proteins, so dry_mass stayed flat at ~386 fg.
"""

import os

import pytest


DURATION = 60.0
MIN_GROWTH_FG = 1.0  # over 60s a healthy cell adds several fg


def _resolve_cache_dir():
    """Same lookup as conftest.sim_data_cache, at module import time so
    pytestmark skipif can evaluate it. Checked into the repo under
    tests/fixtures/cache/ as gzipped files; locally-generated caches at
    out/cache/ still take precedence via the env override."""
    override = os.environ.get('V2ECOLI_CACHE_DIR')
    candidates = [override] if override else []
    fixture_dir = os.path.join(
        os.path.dirname(__file__), 'fixtures', 'cache')
    candidates += [fixture_dir, 'out/cache']
    for d in candidates:
        if d and os.path.isdir(d) and (
            os.path.exists(os.path.join(d, 'sim_data_cache.dill'))
            or os.path.exists(os.path.join(d, 'sim_data_cache.dill.gz'))
        ):
            return d
    return None


CACHE_DIR = _resolve_cache_dir()

pytestmark = pytest.mark.skipif(
    CACHE_DIR is None,
    reason='No ParCa cache found (tested V2ECOLI_CACHE_DIR, '
           'tests/fixtures/cache/, out/cache/).',
)


def _run_and_measure(make_composite_fn, duration=DURATION):
    composite = make_composite_fn(cache_dir=CACHE_DIR, seed=0)
    m0 = float(composite.state['agents']['0']['listeners']['mass']['dry_mass'])
    composite.run(duration)
    m1 = float(composite.state['agents']['0']['listeners']['mass']['dry_mass'])
    return m0, m1


def test_baseline_grows():
    from v2ecoli.composite import make_composite
    m0, m1 = _run_and_measure(make_composite)
    assert m1 - m0 >= MIN_GROWTH_FG, (
        f'Baseline dry_mass did not grow enough: {m0:.2f} -> {m1:.2f} fg')


def test_departitioned_grows():
    from v2ecoli.composite_departitioned import make_departitioned_composite
    m0, m1 = _run_and_measure(make_departitioned_composite)
    assert m1 - m0 >= MIN_GROWTH_FG, (
        f'Departitioned dry_mass did not grow enough: {m0:.2f} -> {m1:.2f} fg')


def test_reconciled_grows():
    from v2ecoli.composite_reconciled import make_reconciled_composite
    m0, m1 = _run_and_measure(make_reconciled_composite)
    assert m1 - m0 >= MIN_GROWTH_FG, (
        f'Reconciled dry_mass did not grow enough: {m0:.2f} -> {m1:.2f} fg')


def test_all_processes_instantiated_in_departitioned():
    """Every step in FLOW_ORDER must be instantiable. The previous silent-drop
    bug let processes disappear without any error — this test ensures the
    new RuntimeError guard fires during build_document and also catches any
    future promotion of a PartitionedProcess that isn't registered."""
    from v2ecoli.composite_departitioned import make_departitioned_composite
    from v2ecoli.generate_departitioned import FLOW_ORDER
    composite = make_departitioned_composite(cache_dir=CACHE_DIR, seed=0)
    cell_state = composite.state['agents']['0']
    # Every non-infra step in FLOW_ORDER should be present in cell_state
    for step_name in FLOW_ORDER:
        if step_name.startswith('allocator'):
            continue  # skipped by design
        assert step_name in cell_state, (
            f'Departitioned: step {step_name!r} missing from cell_state')


def test_all_processes_instantiated_in_reconciled():
    """Same contract for reconciled."""
    from v2ecoli.composite_reconciled import make_reconciled_composite
    from v2ecoli.generate_reconciled import FLOW_ORDER
    composite = make_reconciled_composite(cache_dir=CACHE_DIR, seed=0)
    cell_state = composite.state['agents']['0']
    for step_name in FLOW_ORDER:
        if step_name.startswith('allocator'):
            continue
        assert step_name in cell_state, (
            f'Reconciled: step {step_name!r} missing from cell_state')
