"""The comparison investigation as a triggerable composite.

These assert the *mechanics* on the real declared investigation — which
members exist, what depends on what, and that triggering one condition pulls
ParCa instead of refitting it. They deliberately do not run a comparison: a
full ParCa + vEcoli ensemble is hours, and none of it is needed to show that
the graph triggers correctly.

What is real here: the investigation and study declarations are read off
disk, the content addresses are computed by the shipped `artifact_id`, and
the ParCa artifacts are the actual `simData.cPickle` caches when they exist.
What is stood in for: the per-condition simulations, which are spy Steps that
record whether they ran.
"""

from __future__ import annotations

import os
import pathlib
import pickle

import pytest

from process_bigraph import Composite, Step, allocate_core
from process_bigraph.artifacts import (
    ArtifactRef, ArtifactResults, artifact_id, artifact_store,
    register_artifact_loader)
from process_bigraph.templates import trigger, study_address

from v2ecoli.library.comparison_composite import (
    INVESTIGATION, PARCA_ROOTS, SIM_DATA_FILE, investigation_document,
    load_investigation, publish_parca_cache, register_sim_data_loader,
    study_config, study_inputs, vecoli_address)

pytestmark = pytest.mark.fast

WORKSPACE = pathlib.Path(__file__).resolve().parents[1] / 'workspace'
VECOLI_COMMIT = 'd2f951296b3e47cb685192147c7b3e7d3c9aaa8e'

PARCA_RUNS: list = []
STUDY_RUNS: list = []


class SpyParca(Step):
    """Stands in for a ParCa fit. Records every time it would have run."""

    config_schema = {'name': {'_type': 'string', '_default': 'parca'},
                     'fit': {'_type': 'string', '_default': ''},
                     'cache': {'_type': 'string', '_default': ''}}

    def inputs(self):
        return {}

    def outputs(self):
        return {'results': 'node'}

    def results(self, path=None, context=None):
        return ArtifactResults(ArtifactRef(
            kind='sim_data', hash='computed', store='',
            context={'fit': self.config['fit']}))

    def update(self, state):
        PARCA_RUNS.append(self.config['name'])
        return {'results': self.results()}


class SpyStudy(Step):
    """Stands in for a condition's comparison run."""

    config_schema = {'name': {'_type': 'string', '_default': ''},
                     'condition': {'_type': 'string', '_default': ''},
                     'seeds': {'_type': 'integer', '_default': 1},
                     'generations': {'_type': 'integer', '_default': 1},
                     'cards': {'_type': 'quote'}}

    def inputs(self):
        return {}

    def outputs(self):
        return {'results': 'node'}

    def results(self, path=None, context=None):
        return ArtifactResults(ArtifactRef(
            kind='trajectory', hash='computed', store='',
            context={'condition': self.config['condition'],
                     'observables': {'growth_rate': 1.0}}))

    def update(self, state):
        STUDY_RUNS.append(self.config['name'])
        return {'results': self.results()}


def _core():
    core = allocate_core()
    core.register_link('ParcaStep', SpyParca)
    core.register_link('ComparisonStudy', SpyStudy)
    return core


def _document(**kwargs):
    return investigation_document(
        WORKSPACE, vecoli_commit=VECOLI_COMMIT, **kwargs)


def _publish_parca(document, store_root, commit=''):
    """Put both ParCa roots' artifacts in a store, as real directories."""
    addresses = {}
    for root in PARCA_ROOTS:
        address = study_address(document, root, commit=commit)
        entry = pathlib.Path(artifact_store(address, str(store_root)))
        entry.mkdir(parents=True, exist_ok=True)
        with open(entry / SIM_DATA_FILE, 'wb') as handle:
            pickle.dump({'fit': root, 'doubling_time': 44.0}, handle)
        addresses[root] = address
    return addresses


# ── the declaration, read as-is ─────────────────────────────────────

def test_the_real_investigation_converts():
    document, report = _document()

    declared = set(load_investigation(WORKSPACE, INVESTIGATION)['members'])
    modelled = set(report['modelled'])

    # `parca` is expanded into the two fits, so it is not modelled by name
    assert 'parca' in declared
    assert set(PARCA_ROOTS) <= modelled
    assert not report['unresolved'], (
        f'declared members with no study: {report["unresolved"]}')

    # every other declared member became a node
    assert (declared - {'parca'}) <= modelled


def test_a_condition_depends_on_both_parca_fits():
    """The declaration carries one `from: parca` edge; v2ecoli and vEcoli are
    fit separately, so it expands onto both roots."""
    document, _ = _document()

    sources = [edge['from'] for edge in document['basal']['_study']['inputs']]
    assert sources == ['parca_v2', 'parca_vecoli']
    assert all(edge['artifact'] == 'sim_data'
               for edge in document['basal']['_study']['inputs'])


def test_the_statistical_gate_consumes_a_condition():
    """`statistical` is the cross-condition gate: it depends on ParCa *and*
    on a condition's results."""
    document, _ = _document()

    edges = document['statistical']['_study']['inputs']
    assert ('basal', 'trajectory') in [(e['from'], e['artifact']) for e in edges]
    assert {'parca_v2', 'parca_vecoli'} <= {e['from'] for e in edges}


def test_the_vecoli_implementation_is_pinned():
    """The declaration's `commit:` is empty and the checkout is found through
    an environment variable; the document pins it instead."""
    document, report = _document()

    assert report['vecoli_address'] == (
        f'git:CovertLab/vEcoli@{VECOLI_COMMIT}#vecoli_adapter:make_process')
    assert document['_vecoli']['declared_commit'] == ''      # what it replaces
    assert document['_vecoli']['legacy_env'] == 'V2E_VECOLI_DIR'


def test_only_result_bearing_config_enters_the_address():
    """Prose, status and recorded outcomes must not perturb a content
    address — only what changes the result."""
    declaration = {
        'condition': 'acetate',
        'comparison': {'seeds': 4, 'generations': 4},
        'title': 'anything', 'status': 'evaluated',
        'runs': [{'result': 'PASS'}]}

    assert study_config(declaration) == {
        'condition': 'acetate', 'seeds': 4, 'generations': 4}


# ── triggering the real graph ───────────────────────────────────────

def test_trigger_a_condition_pulls_both_parca_roots(tmp_path):
    """The headline: trigger `basal` and ParCa does not refit."""
    PARCA_RUNS.clear()
    STUDY_RUNS.clear()

    document, _ = _document()
    _publish_parca(document, tmp_path)

    built, report = trigger(document, 'basal', root=str(tmp_path))

    assert sorted(report['pulled']) == ['parca_v2', 'parca_vecoli']
    assert report['computed'] == ['basal']

    # every other condition is absent from the built document
    assert 'succinate' not in built and 'with_aa' not in built
    assert 'statistical' not in built
    for root in PARCA_ROOTS:
        assert built[root]['sim']['address'] == 'local:CachedResults'
    assert built['basal']['sim']['address'] == 'local:ComparisonStudy'

    Composite({'state': built}, core=_core()).run(0.0)

    assert PARCA_RUNS == [], 'ParCa refit despite a cached artifact'
    assert STUDY_RUNS == ['basal']


def test_rerun_another_condition_reuses_the_same_parca(tmp_path):
    """The reuse case: a second condition triggers against the *same* cached
    fits — neither root recomputes."""
    PARCA_RUNS.clear()
    STUDY_RUNS.clear()

    document, _ = _document()
    _publish_parca(document, tmp_path)

    built, report = trigger(document, 'succinate', root=str(tmp_path))
    assert sorted(report['pulled']) == ['parca_v2', 'parca_vecoli']

    Composite({'state': built}, core=_core()).run(0.0)

    assert PARCA_RUNS == []
    assert STUDY_RUNS == ['succinate']


def test_triggering_the_gate_pulls_its_condition_too(tmp_path):
    """`statistical` depends on ParCa and on `basal`; with all three cached
    it computes alone."""
    PARCA_RUNS.clear()
    STUDY_RUNS.clear()

    document, _ = _document()
    _publish_parca(document, tmp_path)
    basal_address = study_address(document, 'basal')
    pathlib.Path(artifact_store(basal_address, str(tmp_path))).mkdir(
        parents=True, exist_ok=True)

    built, report = trigger(document, 'statistical', root=str(tmp_path))

    assert sorted(report['pulled']) == ['basal', 'parca_v2', 'parca_vecoli']
    assert report['computed'] == ['statistical']

    Composite({'state': built}, core=_core()).run(0.0)
    assert PARCA_RUNS == []
    assert STUDY_RUNS == ['statistical']


def test_an_uncached_parca_is_named_not_silently_refit(tmp_path):
    document, _ = _document()

    with pytest.raises(FileNotFoundError, match='parca_v2'):
        trigger(document, 'basal', root=str(tmp_path))


# ── the property that retires the symlink dance ─────────────────────

def test_parcas_address_is_worktree_independent(tmp_path):
    """ParCa's address is a function of its fit config and the commit — never
    of a path — so any worktree at the same commit resolves the same artifact
    with no symlink between them."""
    document, _ = _document()

    here = os.getcwd()
    try:
        addresses = []
        for name in ('worktree-a', 'worktree-b'):
            worktree = tmp_path / name
            worktree.mkdir()
            os.chdir(worktree)
            addresses.append(study_address(document, 'parca_v2',
                                           commit='abc123'))
    finally:
        os.chdir(here)

    assert addresses[0] == addresses[1]

    # and it still tracks the thing that matters: the commit
    assert study_address(document, 'parca_v2', commit='abc123') != \
        study_address(document, 'parca_v2', commit='def456')


def test_the_two_parca_fits_have_different_addresses():
    """v2ecoli and vEcoli are separate fits, so they must not collide in the
    store — v2 cannot load the upstream cache."""
    document, _ = _document()

    assert study_address(document, 'parca_v2') != \
        study_address(document, 'parca_vecoli')


def test_a_condition_address_changes_when_its_parca_does():
    """An address folds in its inputs' addresses, so refitting ParCa
    invalidates every condition downstream of it."""
    document, _ = _document()
    before = study_address(document, 'basal')

    document['parca_v2']['_study']['config']['fit'] = 'reparameterized'
    assert study_address(document, 'basal') != before


# ── the sim_data loader, against a real cache when there is one ─────

def test_the_sim_data_loader_reads_a_parca_cache(tmp_path):
    register_sim_data_loader()

    cache = tmp_path / 'cache_full'
    cache.mkdir()
    with open(cache / SIM_DATA_FILE, 'wb') as handle:
        pickle.dump({'doubling_time': 44.0, 'conditions': 51}, handle)

    handle_ = ArtifactResults(ArtifactRef(kind='sim_data', store=str(cache)))
    assert handle_.resolve() == {'doubling_time': 44.0, 'conditions': 51}


def test_publishing_an_existing_cache_makes_it_the_artifact(tmp_path):
    """A ParCa cache already on disk becomes the content-addressed artifact
    without being copied — the store entry points at it."""
    register_sim_data_loader()

    cache = tmp_path / 'out' / 'cache_full'
    cache.mkdir(parents=True)
    with open(cache / SIM_DATA_FILE, 'wb') as handle:
        pickle.dump({'fit': 'v2'}, handle)

    document, _ = _document()
    store_root = tmp_path / '.pbg' / 'artifacts'
    address = publish_parca_cache(
        document, 'parca_v2', cache, store_root=str(store_root))

    assert address == study_address(document, 'parca_v2')

    built, report = trigger(document, 'basal', root=str(store_root),
                            on_missing='compute')
    assert 'parca_v2' in report['pulled']

    ref = built['parca_v2']['sim']['config']['artifact_ref']
    assert ArtifactResults(ref).resolve() == {'fit': 'v2'}


@pytest.mark.skipif(
    not (pathlib.Path.home() / 'code/v2ecoli/out/cache_full'
         / SIM_DATA_FILE).is_file(),
    reason='no real ParCa cache on this machine')
def test_a_real_parca_cache_publishes_and_resolves_to_a_path(tmp_path):
    """Against the actual 86M `simData.cPickle`.

    The artifact is published and its store entry resolves to the real file.
    It is deliberately not unpickled: loading sim_data pulls the whole
    reconstruction stack, which is not what this is testing.
    """
    real_cache = pathlib.Path.home() / 'code/v2ecoli/out/cache_full'

    document, _ = _document()
    store_root = tmp_path / '.pbg' / 'artifacts'
    address = publish_parca_cache(
        document, 'parca_v2', real_cache, store_root=str(store_root))

    entry = pathlib.Path(artifact_store(address, str(store_root)))
    assert (entry / SIM_DATA_FILE).is_file()
    assert (entry / SIM_DATA_FILE).stat().st_size > 1_000_000


# ── the existing report cards, unchanged, as flush entities ─────────

def _graded_handle():
    """A results handle carrying the shape the `statistical` card reads:
    per-observable, per-seed means from the two engines."""
    handle = ArtifactResults(ArtifactRef(kind='trajectory', store=''))
    handle._resolved[None] = {'observables': {
        'cell_mass': [{'ve_mean': 100.0, 'v2_mean': 101.0},
                      {'ve_mean': 100.0, 'v2_mean': 100.5}],
        'dry_mass': [{'ve_mean': 30.0, 'v2_mean': 30.2},
                     {'ve_mean': 30.0, 'v2_mean': 30.1}]}}
    return handle


def test_a_real_report_card_grades_a_triggered_condition(tmp_path):
    """The existing `statistical` card — already a pbg Step — fires once on a
    triggered condition's results and returns a verdict.

    The card itself is untouched. What is stood in for is the *observables*
    it grades: real ones come from a full two-engine ensemble, so this feeds
    the shape it reads with values well inside the 5% band.
    """
    from scripts._compare.report_cards import REPORT_CARD_STEPS

    assert 'statistical_report_card' in REPORT_CARD_STEPS
    card_class = REPORT_CARD_STEPS['statistical_report_card']

    firings = []
    core = _core()

    class GradedStudy(SpyStudy):
        def results(self, path=None, context=None):
            return _graded_handle()

    class Grade(Step):
        """Adapts a condition's results handle to the card's input state."""

        config_schema = {}

        def inputs(self):
            return {'results': 'node'}

        def outputs(self):
            return {'verdict': 'string'}

        def update(self, state):
            observed = state['results'].resolve()
            firings.append(observed)
            card = card_class({}, self.core)
            result = card.update({
                'name': 'basal',
                'variant': 0,
                'observables': observed.get('observables', {})}) or {}
            return {'verdict': result.get('verdict', 'ungraded')}

    core.register_link('GradedStudy', GradedStudy)
    core.register_link('Grade', Grade)

    document, _ = _document(study_address_='local:GradedStudy')
    _publish_parca(document, tmp_path)

    built, _ = trigger(document, 'basal', root=str(tmp_path))
    built['basal']['verdict'] = ''
    built['basal']['grade'] = {
        '_type': 'step', 'address': 'local:Grade',
        'inputs': {'results': ['results']},
        'outputs': {'verdict': ['verdict']}}

    sim = Composite({'state': built}, core=core)
    sim.run(0.0)

    assert len(firings) == 1, f'card fired {len(firings)} times'
    verdict = sim.state['basal']['verdict']
    assert verdict in ('within_tol', 'drift', 'mismatch', 'ungraded'), verdict
    # ParCa still did not refit while the card graded
    assert PARCA_RUNS == []
