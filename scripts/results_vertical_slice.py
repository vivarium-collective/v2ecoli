"""End-to-end: a study executes, and its results drive a figure, a card, and
an analysis.

The study is the higher-order DAG:

    sim      : SimulationStep(inner = a v2ecoli sim + XArrayEmitter) -> results
    figure   : ResultsVisualization <- results   -> a real HTML figure
    card     : GrowthCard           <- results   -> a verdict JSON
    analysis : ResultsAnalysis      <- results   -> a data artifact (CSV)

``figure``, ``card`` and ``analysis`` wire to ``['..', 'results']`` and each
fire ONCE after the simulation completes, because the simulation is a single
node of the outer network rather than a sibling of these steps.

Everything here executes live: the inner simulation ticks through the engine
and writes a real zarr store; the handle resolves by reading that store back.

    PYTHONPATH=<v2ecoli worktree>:<process-bigraph> \\
        ~/code/v2ecoli/.venv/bin/python scripts/results_vertical_slice.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import EmitterResults

import v2ecoli.library.xarray_emitter  # noqa: F401  (registers XArrayEmitter)
from v2ecoli.library.xarray_run import build_emitter_config


DRY_MASS = ['listeners', 'mass', 'dry_mass']


def inner_simulation(store_path: Path, rate: float, initial_mass: float) -> dict:
    """A v2ecoli-shaped simulation that writes a real zarr store.

    A growth process drives ``listeners/mass/dry_mass``; an ``XArrayEmitter``
    records it, with its ``results`` port wired so the handle leaves the node.
    """
    emitter_config = build_emitter_config(
        store_path=store_path,
        view=[{
            'root': ('listeners',),
            'variables': {
                'mass': {'dry_mass': [{'path': 'dry_mass',
                                       'dtype': 'float64'}]}}}],
        metadata_base={},
        generation=1,
        agent_id='0',
        buffer_size=3)
    emitter_config['strategy'] = 'flat'
    emitter_config['emit_root'] = []
    emitter_config['emit'] = {'global_time': 'node', 'listeners': 'node'}

    return {
        'global_time': 0.0,
        'listeners': {'mass': {'dry_mass': initial_mass}},
        'grow': {
            '_type': 'process',
            'address': 'local:IncreaseProcess',
            'config': {'rate': rate},
            'interval': 1.0,
            'inputs': {'level': DRY_MASS},
            'outputs': {'level': DRY_MASS}},
        'emitter': {
            '_type': 'step',
            'address': 'local:XArrayEmitter',
            'config': emitter_config,
            'inputs': {'listeners': ['listeners'],
                       'global_time': ['global_time']},
            'outputs': {'results': ['results']}}}


def study_document(store_path: Path, runtime: float, rate: float,
                   initial_mass: float) -> dict:
    """The outer study: one simulation node, two flush entities."""
    return {
        'sim': {
            '_type': 'step',
            'address': 'local:SimulationStep',
            'config': {
                'state': inner_simulation(store_path, rate, initial_mass),
                'runtime': runtime},
            'inputs': {},
            'outputs': {'results': ['results']}},
        'figure': {
            'html': '',
            'render': {
                '_type': 'step',
                'address': 'local:ResultsVisualization',
                'config': {'title': 'v2ecoli growth (live run)'},
                'inputs': {'results': ['..', 'results']},
                'outputs': {'html': ['html']}}},
        'card': {
            'view': '', 'data': {},
            'grade': {
                '_type': 'step',
                'address': 'local:GrowthCard',
                'config': {'variable': 'dry_mass', 'title': 'Growth'},
                'inputs': {'results': ['..', 'results']},
                'outputs': {'view': ['view'], 'data': ['data']}}},
        'analysis': {
            'artifact': {},
            'analyze': {
                '_type': 'step',
                'address': 'local:ResultsAnalysis',
                'config': {'variable': 'dry_mass',
                           'filename': 'growth_summary.csv'},
                'inputs': {'results': ['..', 'results']},
                'outputs': {'artifact': ['artifact']}}}}


def build_core():
    from v2ecoli.workflow.results_flush import (
        ResultsVisualization, GrowthCard, ResultsAnalysis)
    core = allocate_core()
    core.register_link('ResultsVisualization', ResultsVisualization)
    core.register_link('GrowthCard', GrowthCard)
    core.register_link('ResultsAnalysis', ResultsAnalysis)
    return core


def run_slice(out_dir: Path, runtime: float = 5.0, rate: float = 0.1,
              initial_mass: float = 100.0) -> dict:
    """Execute the study and return everything the assertions need."""
    store_path = out_dir / 'run.zarr'
    if store_path.exists():
        shutil.rmtree(store_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    core = build_core()
    study = Composite({'state': study_document(
        store_path, runtime, rate, initial_mass)}, core=core)
    study.run(0.0)

    handle = study.state['results']
    html = study.state['figure']['html']
    verdict = study.state['card']['data']
    card_html = study.state['card']['view']
    artifact = study.state['analysis']['artifact']

    (out_dir / 'figure.html').write_text(html, encoding='utf-8')
    (out_dir / 'growth.verdict.json').write_text(
        json.dumps(verdict, indent=1), encoding='utf-8')
    (out_dir / 'growth.html').write_text(card_html, encoding='utf-8')
    if artifact.get('csv'):
        (out_dir / artifact.get('filename', 'analysis.csv')).write_text(
            artifact['csv'], encoding='utf-8')

    return {
        'study': study,
        'store_path': store_path,
        'handle': handle,
        'html': html,
        'verdict': verdict,
        'artifact': artifact,
        'out_dir': out_dir}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/tmp/v2ecoli-results-slice')
    parser.add_argument('--runtime', type=float, default=5.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    result = run_slice(out_dir, runtime=args.runtime)

    handle = result['handle']
    history = handle.resolve()
    from v2ecoli.library.results_adapter import datatree_to_history
    rows = datatree_to_history(history)

    print('=' * 66)
    print('ASSERTIONS')
    print('=' * 66)

    assert result['store_path'].exists(), 'no zarr store on disk'
    n_files = sum(1 for _ in result['store_path'].rglob('*'))
    print(f'  zarr store written           : {result["store_path"]} '
          f'({n_files} entries)')

    assert isinstance(handle, EmitterResults), f'not a handle: {handle!r}'
    assert len(rows) > 1, f'trajectory has {len(rows)} row(s)'
    print(f'  results handle               : {handle}')
    print(f'  resolved trajectory          : {len(rows)} rows, '
          f'dry_mass {rows[0]["dry_mass"]:.3f} -> {rows[-1]["dry_mass"]:.3f}')

    html = result['html']
    assert html and '<' in html, 'figure is empty'
    assert 'base64' in html or '<svg' in html, 'figure carries no image'
    print(f'  figure artifact              : {len(html)} bytes '
          f'-> {result["out_dir"] / "figure.html"}')

    verdict = result['verdict']
    assert verdict.get('status') in ('pass', 'fail'), f'no verdict: {verdict}'
    assert verdict['n_points'] == len(rows)
    print(f'  card verdict                 : {verdict["status"]!r} '
          f'(fold_change={verdict["fold_change"]:.3f}, '
          f'n_points={verdict["n_points"]})')
    print(f'  card artifacts               : '
          f'{result["out_dir"] / "growth.verdict.json"}')

    artifact = result['artifact']
    assert artifact.get('csv'), f'analysis produced no data artifact: {artifact}'
    assert artifact['summary'], 'analysis summary is empty'
    assert artifact['csv'].splitlines()[0] == (
        'generation,variable,n,first,last,min,max,mean')
    print(f'  analysis artifact            : {len(artifact["summary"])} '
          f'generation row(s) -> '
          f'{result["out_dir"] / artifact["filename"]}')

    # The higher-order-DAG signature: the flush entities are downstream of a
    # single simulation node, so how long the simulation runs cannot change
    # how many times they fire.
    from v2ecoli.workflow.results_flush import (
        ResultsVisualization, GrowthCard, ResultsAnalysis)

    firings = {'figure': 0, 'card': 0, 'analysis': 0}
    original_figure = ResultsVisualization.update
    original_card = GrowthCard.update
    original_analysis = ResultsAnalysis.update

    def count_figure(self, state):
        firings['figure'] += 1
        return original_figure(self, state)

    def count_card(self, state):
        firings['card'] += 1
        return original_card(self, state)

    def count_analysis(self, state):
        firings['analysis'] += 1
        return original_analysis(self, state)

    ResultsVisualization.update = count_figure
    GrowthCard.update = count_card
    ResultsAnalysis.update = count_analysis
    try:
        counts = {}
        # NOTE: buffer-aligned lengths. XArrayEmitter.query() flushes on
        # read and its transducer asserts the buffer is exactly full, so a
        # tick count that is not a multiple of buffer_size(=3) raises. That
        # is a pbg-emitters defect, not a property of this slice — see the
        # branch report.
        for runtime in (5.0, 11.0):
            firings['figure'] = firings['card'] = firings['analysis'] = 0
            probe = run_slice(out_dir / f'len{int(runtime)}', runtime=runtime)
            rows = datatree_to_history(probe['handle'].resolve())
            counts[runtime] = (firings['figure'], firings['card'],
                               firings['analysis'], len(rows))
    finally:
        ResultsVisualization.update = original_figure
        GrowthCard.update = original_card
        ResultsAnalysis.update = original_analysis

    for runtime, (n_figure, n_card, n_analysis, n_rows) in counts.items():
        assert n_figure == 1, f'figure fired {n_figure}x at runtime={runtime}'
        assert n_card == 1, f'card fired {n_card}x at runtime={runtime}'
        assert n_analysis == 1, (
            f'analysis fired {n_analysis}x at runtime={runtime}')
    lengths = {runtime: n_rows for runtime, (*_, n_rows) in counts.items()}
    assert len(set(lengths.values())) == len(lengths), (
        f'runs did not differ in length: {lengths}')
    print(f'  firing counts                : '
          + ', '.join(
              f'runtime={r} -> {n_rows} rows, figure x{f}, card x{c}, '
              f'analysis x{a}'
              for r, (f, c, a, n_rows) in counts.items()))

    print('  ALL ASSERTIONS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
