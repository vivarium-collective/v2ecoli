"""Tests for ``ParcaBundleStep`` — wraps a pre-built ParCa cache bundle as a
content-addressed ``sim_data`` ``ArtifactRef``.

Fixture/pre-cached only: the ``bundle_dir`` used here is built once (per
module) from the shipped ParCa state fixture
(``models/parca/parca_state.pkl.gz``) via ``v2ecoli.core.save_sim_input`` —
the same helper ``tests/test_save_sim_input.py`` uses. This never runs full
ParCa. ``ParcaBundleStep`` itself does no fitting at all: it only reads and
hashes files that already exist at ``bundle_dir``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_PATH = REPO_ROOT / 'models' / 'parca' / 'parca_state.pkl.gz'


@pytest.fixture(scope='module')
def bundle_dir(tmp_path_factory):
    if not FIXTURE_PATH.exists():
        pytest.skip(f'fixture absent at {FIXTURE_PATH}')

    from v2ecoli.core import save_sim_input
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state,
    )

    state = load_parca_state(str(FIXTURE_PATH))
    sim_data = hydrate_sim_data_from_state(state)

    bundle = tmp_path_factory.mktemp('parca_bundle') / 'bundle'
    save_sim_input(sim_data, str(bundle))
    return str(bundle)


def _run_step(bundle_dir):
    from v2ecoli.core import build_core
    from v2ecoli.steps.parca_bundle import ParcaBundleStep

    core = build_core()
    step = ParcaBundleStep(
        {'mode': 'fixture', 'cpus': 1, 'condition': None,
         'bundle_dir': bundle_dir},
        core=core,
    )
    return step.update({})


def test_parca_bundle_step_emits_sim_data_ref(bundle_dir):
    update = _run_step(bundle_dir)
    ref = update['sim_data']

    assert ref['kind'] == 'sim_data'
    assert ref['hash']
    # store is the bundle DIRECTORY, not a file — downstream (T8/T9) injects
    # it directly as load_cache_bundle's cache_dir, which requires a dir.
    assert os.path.exists(os.path.join(ref['store'], 'sim_data_cache.dill'))


def test_parca_bundle_step_hash_is_deterministic(bundle_dir):
    """Two separate ``ParcaBundleStep`` runs over the SAME bundle must
    produce the SAME hash.

    Guards against (a) an XOR-style combine that self-cancels pairs of
    identical/complementary digests, and (b) filesystem enumeration-order
    nondeterminism (``os.listdir`` order is not guaranteed) leaking into the
    final hash.
    """
    hash_a = _run_step(bundle_dir)['sim_data']['hash']
    hash_b = _run_step(bundle_dir)['sim_data']['hash']

    assert hash_a == hash_b
