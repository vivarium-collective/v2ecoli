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
    then falls back to locally-generated checkpoints from workflow.py.
    """
    candidates = [
        os.path.join(_FIXTURE_DIR, 'pre_division_state.json.gz'),
        _workflow_path('pre_division', 'pre_division_state.json.gz'),
        _workflow_path('pre_division', 'pre_division_state.json'),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        pytest.skip(
            f'pre-division checkpoint not found. Tried: {candidates}. '
            f'Run `python workflow.py` or set V2ECOLI_PREDIV_CHECKPOINT_DIR.')
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
            f'Run `python workflow.py` to generate.')
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
    """
    cache_dir = os.environ.get('V2ECOLI_CACHE_DIR', 'out/cache')
    if not os.path.isdir(cache_dir):
        pytest.skip(f'cache dir {cache_dir!r} not present (run ParCa first)')
    return cache_dir
