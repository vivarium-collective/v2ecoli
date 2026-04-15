"""Workflow ↔ ParCa integration tests.

Exercise the three ``workflow.py`` steps that consume the merged
``v2ecoli.processes.parca`` subpackage:

  * ``step_raw_data`` — walks the vendored
    ``v2ecoli/processes/parca/reconstruction/ecoli/flat/`` directory and
    loads ``KnowledgeBaseEcoli`` from the merged subpackage.  Asserts
    that the knowledge-base statistics (counts of genes / RNAs /
    proteins / metabolites) are non-zero and stable, proving the
    workflow sees the merged flat files and not an external vEcoli tree.
  * ``step_parca`` (fixture fast-path) — hydrates the shipped
    ``models/parca/parca_state.pkl.gz`` via
    ``load_parca_state()``, dills its ``sim_data_root`` to
    ``out/workflow/simData.cPickle``, and asserts the produced file is
    a valid ``SimulationDataEcoli`` under the merged namespace.  This
    runs in under 10 s and doesn't touch vEcoli.
  * ``FLAT_DIR`` / ``BIOCYC_FILE_IDS`` structural sanity — the 10
    BioCyc TSVs exist in the merged flat directory, and nothing in
    ``workflow.py`` still references ``..vEcoli/reconstruction`` paths.

Full ``step_parca_composite`` path (running the 9-Step pipeline from
scratch, ~70 min) is intentionally *not* exercised here — that's slow
and already covered by ``test_parca_ports_and_wiring`` + the fixture
roundtrip.  If you want the integration test to actually run the
pipeline, set ``V2ECOLI_PARCA_RERUN=1`` and mark the test slow.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from unittest import mock

import dill
import pytest

# workflow.py lives at the repo root; tests/ sits alongside it.  Add the
# repo root to sys.path so ``import workflow`` resolves.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import workflow  # noqa: E402


FIXTURE_PATH = REPO_ROOT / 'models' / 'parca' / 'parca_state.pkl.gz'


# ---------------------------------------------------------------------------
# Path / constant sanity
# ---------------------------------------------------------------------------

def test_flat_dir_points_at_merged_parca():
    """workflow.FLAT_DIR should be inside the merged parca subpackage,
    never in an external vEcoli checkout."""
    assert 'v2ecoli/processes/parca/reconstruction/ecoli/flat' \
        in workflow.FLAT_DIR.replace(os.sep, '/')
    assert 'vEcoli' not in workflow.FLAT_DIR
    assert os.path.isdir(workflow.FLAT_DIR), \
        f'FLAT_DIR missing: {workflow.FLAT_DIR}'


def test_all_biocyc_tsvs_present_in_flat_dir():
    for fid in workflow.BIOCYC_FILE_IDS:
        path = os.path.join(workflow.FLAT_DIR, f'{fid}.tsv')
        assert os.path.isfile(path), f'missing BioCyc TSV: {path}'
        assert os.path.getsize(path) > 0


def test_workflow_has_no_external_vecoli_paths():
    """Structural smoke test — the rewritten workflow.py must not
    import from bare ``reconstruction`` / ``wholecell`` (the vEcoli top-
    level) and must not path-join into ``../vEcoli/``."""
    src = (REPO_ROOT / 'workflow.py').read_text()
    assert not re.search(
        r"^from (reconstruction|wholecell)\.", src, re.MULTILINE), \
        'workflow.py still imports bare vEcoli modules — should use ' \
        'v2ecoli.processes.parca.*'
    assert "'..', 'vEcoli'" not in src and '"../vEcoli"' not in src, \
        'workflow.py still path-joins into an external vEcoli checkout'


# ---------------------------------------------------------------------------
# step_raw_data — integrates with KnowledgeBaseEcoli from merged subpackage
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_step_raw_data_uses_merged_knowledge_base(tmp_path, monkeypatch):
    """Run the real ``step_raw_data`` against a clean metadata dir and
    assert it reports plausible counts from the merged KB."""
    monkeypatch.setattr(workflow, 'WORKFLOW_DIR', str(tmp_path))
    monkeypatch.setattr(workflow, 'META_DIR',
                        str(tmp_path / '_meta'), raising=False)
    # workflow.load_meta / save_meta path-join off WORKFLOW_DIR.  Force
    # a cache miss so step_raw_data actually runs KnowledgeBaseEcoli.
    monkeypatch.setattr(workflow, 'load_meta', lambda _n: None)
    saved = {}
    monkeypatch.setattr(workflow, 'save_meta',
                        lambda n, m: saved.setdefault(n, m))

    meta = workflow.step_raw_data()
    assert meta.get('skipped') is not True, \
        'step_raw_data should succeed with the merged subpackage; ' \
        f'got: {meta}'
    assert meta['n_genes'] > 4_000
    assert meta['n_proteins'] > 4_000
    assert meta['genome_length'] > 4_000_000  # E. coli is ~4.6 Mbp
    # The walked directory must be the vendored one.
    assert meta['n_files'] >= 130  # we ship 133 TSVs at time of writing


# ---------------------------------------------------------------------------
# step_parca fixture fast-path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason=f'fixture absent at {FIXTURE_PATH}')
def test_step_parca_hydrates_fixture(tmp_path, monkeypatch):
    """Run ``step_parca`` in fixture mode against a scratch workflow
    dir and assert it produces a usable ``simData.cPickle``."""
    workflow_dir = tmp_path / 'workflow'
    monkeypatch.setattr(workflow, 'WORKFLOW_DIR', str(workflow_dir))
    monkeypatch.setattr(workflow, 'SIM_DATA_PATH', None)
    monkeypatch.setattr(workflow, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(workflow, 'load_meta', lambda _n: None)
    monkeypatch.setattr(workflow, 'save_meta', lambda _n, _m: None)
    # Stub save_cache — we don't want to generate the full initial
    # state JSON during this test (that's a separate step with its own
    # coverage).  We assert the dill file is well-formed instead.
    monkeypatch.setattr(workflow, 'save_cache', lambda *a, **k: None)

    # Force the fixture path (not a composite rerun)
    assert workflow._OPTIONS.get('parca_rerun') is not True
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # fixture_path is resolved relative to cwd
    try:
        meta = workflow.step_parca()
    finally:
        os.chdir(cwd)

    assert meta['simdata_source'] == 'parca_fixture'
    assert meta['parca_ran'] is False
    sim_data_pkl = workflow_dir / 'simData.cPickle'
    assert sim_data_pkl.exists(), f'simData.cPickle not written'

    # The produced file must unpickle into a real SimulationDataEcoli.
    from v2ecoli.processes.parca.data_loader import (
        _install_legacy_pickle_aliases,
    )
    _install_legacy_pickle_aliases()
    with open(sim_data_pkl, 'rb') as f:
        sd = dill.load(f)
    assert type(sd).__name__ == 'SimulationDataEcoli'
    assert type(sd).__module__.startswith(
        'v2ecoli.processes.parca.reconstruction.ecoli'), \
        f'wrong namespace: {type(sd).__module__}'


@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason=f'fixture absent at {FIXTURE_PATH}')
def test_step_parca_fixture_is_fast(tmp_path, monkeypatch):
    """Fixture hydration should take < 15 seconds.  Guards against
    accidental regressions that fall off the fast path."""
    import time

    monkeypatch.setattr(workflow, 'WORKFLOW_DIR', str(tmp_path / 'workflow'))
    monkeypatch.setattr(workflow, 'SIM_DATA_PATH', None)
    monkeypatch.setattr(workflow, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(workflow, 'load_meta', lambda _n: None)
    monkeypatch.setattr(workflow, 'save_meta', lambda _n, _m: None)
    monkeypatch.setattr(workflow, 'save_cache', lambda *a, **k: None)

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        t0 = time.time()
        workflow.step_parca()
        elapsed = time.time() - t0
    finally:
        os.chdir(cwd)
    assert elapsed < 15, \
        f'fixture hydration took {elapsed:.1f}s, expected < 15s — ' \
        'the fast path likely regressed'
