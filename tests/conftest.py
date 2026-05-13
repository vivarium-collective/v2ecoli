"""Shared fixtures for model-behavior tests.

These fixtures resume from saved simulation checkpoints so individual tests
can assert on post-replication / pre-division state in seconds instead of
re-running 40 minutes of simulation.

**Save-state policy:** all resumable state is loaded from bigraph-schema
documents in JSON form (see v2ecoli.cache.load_initial_state). The project
does not accept pickle/dill as a save-state format.

Checkpoint lookup order for the pre-division fixture:
  1. tests/fixtures/pre_division_state.json.gz   (blessed, committed)
  2. $V2ECOLI_PREDIV_CHECKPOINT_DIR/pre_division/pre_division_state.json.gz
  3. $V2ECOLI_PREDIV_CHECKPOINT_DIR/pre_division/pre_division_state.json

Canonical save state is bigraph-schema JSON (possibly gzipped). Any format
ending in .gz is transparently decompressed by v2ecoli.cache.load_json.
  - out/workflow/single_cell_meta.json
    (trajectory metadata: chromosome_snapshots, mass history)

Path overrides:
  V2ECOLI_PREDIV_CHECKPOINT_DIR   (default: out/workflow)
  V2ECOLI_CACHE_DIR               (default: out/cache)
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Global Unum patch — the merged ``v2ecoli.processes.parca.wholecell.utils``
# and the pypi ``wholecell.utils`` (from vEcoli[dev]) both define the same
# ``count`` / ``nucleotide`` / ``amino_acid`` unum units at module scope.
# Importing both in the same process raises ``NameConflictError`` because
# unum's unit table is global.  Make ``Unum.unit`` idempotent so re-import
# returns the existing symbol instead of crashing — applies across every
# test in this session.
# ---------------------------------------------------------------------------

def _patch_unum_idempotent():
    import unum  # type: ignore
    orig_unit = unum.Unum.unit

    def _idempotent_unit(cls, symbol, conv=None, name=''):
        existing = cls._unitTable.get(symbol)
        if existing is not None:
            return cls({symbol: 1}, 1, None, symbol)
        return orig_unit(symbol, conv, name)

    unum.Unum.unit = classmethod(_idempotent_unit)


_patch_unum_idempotent()


def _workflow_path(*parts):
    base = os.environ.get('V2ECOLI_PREDIV_CHECKPOINT_DIR', 'out/workflow')
    return os.path.join(base, *parts)


_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture(scope='session')
def predivision_state():
    """Pre-division cell state loaded from bigraph-schema JSON.

    Looks first at the blessed fixture committed under tests/fixtures/,
    then falls back to locally-generated checkpoints from reports/workflow_report.py.
    """
    candidates = [
        os.path.join(_FIXTURE_DIR, 'pre_division_state.json.gz'),
        _workflow_path('pre_division', 'pre_division_state.json.gz'),
        _workflow_path('pre_division', 'pre_division_state.json'),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        msg = (f'pre-division checkpoint not found. Tried: {candidates}. '
               f'Run `python reports/workflow_report.py` or set V2ECOLI_PREDIV_CHECKPOINT_DIR.')
        # Blessed fixture is committed; absence on CI signals a broken
        # workflow (checkout failure, LFS miss, etc.) — don't hide it.
        (pytest.fail if os.environ.get('CI') else pytest.skip)(msg)
    from v2ecoli.cache import load_initial_state
    return load_initial_state(path)


@pytest.fixture(scope='session')
def single_cell_trajectory():
    """Mass / chromosome / replisome trajectory from the single-cell run.

    Cheap JSON data source for "did replication initiate at the expected
    time" style assertions — no simulation needed.
    """
    path = _workflow_path('single_cell_meta.json')
    if not os.path.exists(path):
        pytest.skip(
            f'single_cell trajectory not found at {path!r}. '
            f'Run `python reports/workflow_report.py` to generate.')
    with open(path) as f:
        meta = json.load(f)
    snaps = meta.get('chromosome_snapshots')
    if not snaps:
        pytest.skip(f'trajectory JSON at {path!r} has no chromosome_snapshots')
    return {'snapshots': snaps, 'meta': meta}


@pytest.fixture(scope='session')
def sim_data_cache():
    """Path to the ParCa-derived sim_data_cache dir.

    Note: sim_data_cache.dill is a ParCa parameter artifact, not a
    simulation save state. The save-state format policy does not apply to
    it — ParCa outputs are compiled parameter objects produced once and
    reused across runs.

    Behavior when the cache is missing or stale:
      - Under CI (``CI`` env var truthy): pytest.fail with a rebuild
        command. A silent skip here means every sim-touching test is
        silently no-op'd on the PR, which is exactly how pre-existing
        breakage (e.g. the unum→pint migration) escaped review.
      - Locally: pytest.skip with the same rebuild message, so a
        developer without a cache can still run fast tests.
    """
    from v2ecoli.library.cache_version import (
        StaleCacheError, verify_cache_version,
    )

    cache_dir = os.environ.get('V2ECOLI_CACHE_DIR', 'out/cache')
    on_ci = bool(os.environ.get('CI'))

    rebuild_msg = (
        f'cache dir {cache_dir!r} not present. Rebuild with:\n'
        f'    python scripts/build_cache.py'
    )

    if not os.path.isdir(cache_dir):
        (pytest.fail if on_ci else pytest.skip)(rebuild_msg)

    try:
        verify_cache_version(cache_dir)
    except StaleCacheError as exc:
        (pytest.fail if on_ci else pytest.skip)(str(exc))

    return cache_dir


# ---------------------------------------------------------------------------
# Shared fixtures for emitter tests
# ---------------------------------------------------------------------------

@pytest.fixture
def core():
    """A bigraph-schema core for emitter construction tests."""
    from bigraph_schema import allocate_core
    return allocate_core()


@pytest.fixture
def minimal_xarray_config(tmp_path):
    """A minimum-valid config dict for v2ecoli.library.xarray_emitter.XArrayEmitter.

    Sufficient to build XarrayTransducer / ForestView / AsyncBufferWriter
    without raising. Used by tests that don't care about specific transducer
    behavior — only that the emitter constructs without error.

    The store path uses ``tmp_path`` so each test gets an isolated directory.
    No metadata is provided, so ``open_store`` is never called and no
    filesystem writes occur during construction.
    """
    store = str(tmp_path / "x.zarr")
    return {
        "emit": {},
        "out_uri": store,
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": [
            {
                "root": ("listeners",),
                "variables": {
                    "dummy_var": [{"path": "dummy/val", "dtype": "<f4"}],
                },
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {},
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }
