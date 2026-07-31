"""The vEcoli adapter, against the real CovertLab/vEcoli.

These resolve a real `git:` address, materialize a real checkout at a pinned
SHA, and check the resolved interface over the protocol's RPC boundary. What
they do *not* do is run a whole-cell simulation: that needs a ParCa pickle and
minutes of process construction, which is a different kind of test.

Everything that touches the network or a checkout is skipped when the local
CovertLab/vEcoli clone is absent, so the suite stays runnable anywhere.
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.fast

VECOLI_CHECKOUT = pathlib.Path.home() / 'code' / 'vEcoli'
VECOLI_SHA = 'd2f951296b3e47cb685192147c7b3e7d3c9aaa8e'
ADAPTER_DIR = pathlib.Path(__file__).resolve().parents[1] / 'v2ecoli' / 'comparison'

needs_vecoli = pytest.mark.skipif(
    not (VECOLI_CHECKOUT / '.git').is_dir(),
    reason='no local CovertLab/vEcoli checkout')


def _adapter():
    """Import the adapter the way the worker does — as a top-level module."""
    import sys
    if str(ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPTER_DIR))
    import vecoli_adapter
    return vecoli_adapter


def _git():
    import os

    from process_bigraph.protocols import git as gitproto
    gitproto.add_allowed_repo('CovertLab/vEcoli')
    gitproto.set_repo_url('CovertLab/vEcoli', str(VECOLI_CHECKOUT))

    # The worker subprocess inherits the environment, so this is how the
    # adapter becomes importable inside vEcoli's venv — `sys.path` on the
    # host does not reach it.
    existing = os.environ.get('PYTHONPATH', '')
    if str(ADAPTER_DIR) not in existing.split(os.pathsep):
        os.environ['PYTHONPATH'] = os.pathsep.join(
            [str(ADAPTER_DIR)] + ([existing] if existing else []))
    return gitproto


def _address(entry='make_process'):
    return f'CovertLab/vEcoli@{VECOLI_SHA}#vecoli_adapter:{entry}'


# ── the face, without needing vEcoli at all ─────────────────────────

def test_the_adapter_declares_the_whole_cell_face():
    """The face is plain data, so an address can be conformance-checked
    before anything is fetched."""
    adapter = _adapter()
    face = adapter.face()

    assert face['inputs'] == {}
    assert set(face['outputs']) == {
        'cell_mass', 'dry_mass', 'protein_mass', 'rna_mass', 'growth_rate'}
    assert all(kind == 'float' for kind in face['outputs'].values())


def test_the_adapter_answers_its_interface_without_building():
    """`interface()` must not construct a simulation — that is what makes a
    pre-run conformance check possible."""
    adapter = _adapter()
    process = adapter.make_process({})

    assert process.interface() == adapter.face()
    assert process.sim is None, 'interface() built the simulation'


def test_the_adapter_refuses_to_build_without_sim_data():
    adapter = _adapter()
    process = adapter.make_process({})

    with pytest.raises(ValueError, match='sim_data_path'):
        process._build()


def test_the_adapter_names_a_missing_sim_data_file(tmp_path):
    adapter = _adapter()
    process = adapter.make_process(
        {'sim_data_path': str(tmp_path / 'nope.cPickle')})

    with pytest.raises(FileNotFoundError, match='nope.cPickle'):
        process._build()


# ── the real address, the real repo ─────────────────────────────────

@needs_vecoli
def test_the_address_parses_and_is_allow_listed():
    gitproto = _git()
    address = gitproto.parse_git_address(_address())

    assert address.repo_slug == 'CovertLab/vEcoli'
    assert address.module == 'vecoli_adapter'
    assert address.entry == 'make_process'
    gitproto.check_allow_list(address)          # does not raise


@needs_vecoli
def test_an_unlisted_repo_is_refused():
    """Trust is an allow-list, and a repo outside it is refused rather than
    fetched."""
    gitproto = _git()
    address = gitproto.parse_git_address(
        'SomeoneElse/whatever@main#mod:entry')

    with pytest.raises(gitproto.AllowListError):
        gitproto.check_allow_list(address)


@needs_vecoli
def test_the_address_pins_a_real_commit():
    """A pinned SHA is the reproducibility unit — the address resolves to the
    commit, not to whatever the branch points at today."""
    gitproto = _git()
    pin = gitproto.resolve_and_pin(gitproto.parse_git_address(_address()))

    assert pin.sha == VECOLI_SHA


@needs_vecoli
@pytest.mark.slow
def test_the_resolved_interface_conforms_to_the_declared_face():
    """The core check: what the *real* vEcoli checkout exposes, read back
    over the protocol's RPC boundary, admits the face the comparison declares.

    Slow on a cold cache — it materializes a per-SHA venv and installs vEcoli
    into it.
    """
    from bigraph_schema import allocate_core
    from bigraph_schema.methods import load_protocol

    adapter = _adapter()
    gitproto = _git()
    core = gitproto.register_types(allocate_core())

    factory = load_protocol(core, core.access('git'), _address())
    assert factory.pin.sha == VECOLI_SHA

    edge = factory({}, core)
    try:
        resolved = edge.interface()
        # the interface came back from inside vEcoli's own venv
        gitproto.check_conformance(core, resolved, adapter.face())
    finally:
        edge.end()

    assert set(resolved['outputs']) == set(adapter.face()['outputs'])


@needs_vecoli
@pytest.mark.slow
def test_a_non_conforming_face_is_refused_by_name():
    """A run never starts against an address that does not conform, and the
    missing port is named."""
    from bigraph_schema import allocate_core
    from bigraph_schema.methods import load_protocol

    gitproto = _git()
    core = gitproto.register_types(allocate_core())
    factory = load_protocol(core, core.access('git'), _address())

    edge = factory({}, core)
    try:
        resolved = edge.interface()
    finally:
        edge.end()

    ok, reason = gitproto.conforms(
        core, resolved, {'inputs': {}, 'outputs': {'atp_flux': 'float'}})

    assert not ok
    assert 'atp_flux' in reason


# ── the ParCa compute entrypoint ────────────────────────────────────

def _parca_entry():
    import sys
    if str(ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPTER_DIR))
    import vecoli_parca
    return vecoli_parca


def test_the_parca_build_declares_a_reference_producing_face():
    """A build step produces a *reference*, not a trajectory — that is what
    keeps the bulk pickle inside the venv's checkout."""
    parca = _parca_entry()
    build = parca.build_sim_data({'outdir': '/tmp/whatever'})

    assert build.interface() == {
        'inputs': {}, 'outputs': {'artifact_ref': 'quote'}}


def test_the_parca_build_needs_an_outdir():
    parca = _parca_entry()
    with pytest.raises(ValueError, match='outdir'):
        parca.build_sim_data({}).update()


def test_the_parca_build_reports_an_existing_fit_without_refitting(tmp_path):
    """Idempotent on a warm checkout: an existing calibration object is
    fingerprinted and returned, not rebuilt."""
    parca = _parca_entry()

    kb = tmp_path / parca.KB_DIR
    kb.mkdir()
    (kb / parca.SIM_DATA_FILE).write_bytes(b'not really a fit, but bytes')

    ref = parca.build_sim_data(
        {'outdir': str(tmp_path)}).update()['artifact_ref']

    assert ref['kind'] == 'sim_data'
    assert ref['rebuilt'] is False
    assert ref['file'] == parca.SIM_DATA_FILE
    assert ref['bytes'] == len(b'not really a fit, but bytes')
    assert len(ref['fingerprint']) == 16


def test_the_reference_records_the_environment_that_built_it(tmp_path):
    """The artifact is only valid for the env that wrote it — recording that
    is what makes a stale pickle detectable rather than a crash at read."""
    parca = _parca_entry()

    kb = tmp_path / parca.KB_DIR
    kb.mkdir()
    (kb / parca.SIM_DATA_FILE).write_bytes(b'bytes')

    ref = parca.build_sim_data(
        {'outdir': str(tmp_path)}).update()['artifact_ref']

    assert ref['context']['built_in_venv'] is True
    assert 'scipy' in ref['context'] and 'numpy' in ref['context']


def test_the_fingerprint_follows_the_bytes(tmp_path):
    parca = _parca_entry()

    kb = tmp_path / parca.KB_DIR
    kb.mkdir()
    target = kb / parca.SIM_DATA_FILE

    target.write_bytes(b'one')
    first = parca.build_sim_data(
        {'outdir': str(tmp_path)}).update()['artifact_ref']['fingerprint']

    target.write_bytes(b'two')
    second = parca.build_sim_data(
        {'outdir': str(tmp_path)}).update()['artifact_ref']['fingerprint']

    assert first != second


def test_parcas_own_hash_is_used_as_the_fingerprint_when_available():
    """`run_parca` returns a SHA256 of the pickled bytes — a fingerprint of
    the *result*, which is exactly what detects a nondeterministic refit at
    the same address."""
    parca = _parca_entry()

    assert parca._fingerprint('/nonexistent', 'a' * 64) == 'a' * 16
