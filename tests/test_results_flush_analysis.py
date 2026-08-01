"""The results-native `analysis` flush kind (`ResultsAnalysis`).

Mirrors the viz/card coverage: an analysis Step consumes the ``results``
handle, resolves it to the trajectory, and fires ONCE to produce its data
artifact. The aggregation is exercised directly (pure over history rows); the
handle path is exercised with a fake handle so no engine/store is needed.
"""
from __future__ import annotations

import pytest
from process_bigraph import allocate_core

import v2ecoli.workflow.results_flush as results_flush
from v2ecoli.workflow.results_flush import ResultsAnalysis


@pytest.fixture(scope='module')
def core():
    return allocate_core()


class _FakeHandle:
    """A `results` handle whose resolve() returns a sentinel tree object.

    The DataTree->history flattening is stubbed (monkeypatched) so this test
    stays a unit test of the analysis step, not of the adapter.
    """

    def __init__(self, tree):
        self._tree = tree
        self.resolves = 0

    def resolve(self):
        self.resolves += 1
        return self._tree


def _history():
    # two generations, three ticks each, of one growing variable
    return [
        {'generation': 1, 'time': 0.0, 'dry_mass': 100.0},
        {'generation': 1, 'time': 1.0, 'dry_mass': 110.0},
        {'generation': 1, 'time': 2.0, 'dry_mass': 120.0},
        {'generation': 2, 'time': 3.0, 'dry_mass': 130.0},
        {'generation': 2, 'time': 4.0, 'dry_mass': 150.0},
        {'generation': 2, 'time': 5.0, 'dry_mass': 140.0},
    ]


def test_analyze_summarizes_per_generation(core):
    step = ResultsAnalysis(config={}, core=core)
    artifact = step.analyze(_history())

    assert artifact['filename'] == 'analysis.csv'
    summary = artifact['summary']
    # one row per generation, in order
    assert [r['generation'] for r in summary] == [1, 2]

    gen1, gen2 = summary
    assert gen1 == {'generation': 1, 'variable': 'dry_mass', 'n': 3,
                    'first': 100.0, 'last': 120.0, 'min': 100.0,
                    'max': 120.0, 'mean': 110.0}
    assert gen2['first'] == 130.0 and gen2['last'] == 140.0
    assert gen2['min'] == 130.0 and gen2['max'] == 150.0

    # the CSV artifact carries a header + one line per generation
    csv_lines = artifact['csv'].strip().splitlines()
    assert csv_lines[0] == 'generation,variable,n,first,last,min,max,mean'
    assert len(csv_lines) == 1 + len(summary)


def test_analyze_ignores_ticks_missing_the_variable(core):
    step = ResultsAnalysis(config={'variable': 'protein_mass'}, core=core)
    # only some ticks carry the requested variable
    history = [
        {'generation': 1, 'dry_mass': 100.0},
        {'generation': 1, 'protein_mass': 5.0},
        {'generation': 1, 'protein_mass': 7.0},
    ]
    summary = step.analyze(history)['summary']
    assert len(summary) == 1
    assert summary[0]['n'] == 2 and summary[0]['mean'] == 6.0


def test_update_fires_once_and_produces_the_artifact(monkeypatch, core):
    monkeypatch.setattr(results_flush, 'datatree_to_history',
                        lambda tree: _history())
    handle = _FakeHandle(tree=object())
    step = ResultsAnalysis(config={'filename': 'growth_summary.csv'}, core=core)

    out = step.update({'results': handle})

    # fired once: the handle was resolved exactly once
    assert handle.resolves == 1
    artifact = out['artifact']
    assert artifact['filename'] == 'growth_summary.csv'
    assert [r['generation'] for r in artifact['summary']] == [1, 2]
    assert artifact['csv'].startswith('generation,variable,n,')


def test_update_with_no_handle_is_empty(core):
    step = ResultsAnalysis(config={}, core=core)
    assert step.update({'results': None}) == {'artifact': {}}
    assert step.update({}) == {'artifact': {}}
