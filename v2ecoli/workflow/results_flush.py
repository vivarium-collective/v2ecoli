"""Flush entities that grade and draw **this run**, from its results handle.

These are the downstream half of the higher-order DAG: a simulation node
produces an ``EmitterResults`` handle, and these steps consume it. Each fires
once, after the run, and sees the whole trajectory — not a store value caught
mid-tick.

That is the difference from the existing post-sim cards, which read a
precomputed JSON or re-open a parquet directory. These grade the data the run
just produced.
"""

from __future__ import annotations


from process_bigraph import Step

from v2ecoli.library.results_adapter import (
    results_to_visualization_inputs, datatree_to_history)


class ResultsVisualization(Step):
    """Adapt a ``results`` handle to an existing v2ecoli visualization.

    The visualization itself is untouched: it still consumes ``history`` and
    ``metadata`` exactly as it does when fed from parquet. Only the source
    changes — this run's results rather than a re-read of disk.
    """

    config_schema = {
        'address': {'_type': 'string', '_default': 'local:multigeneration'},
        'title': {'_type': 'string', '_default': 'v2ecoli run'}}

    def inputs(self):
        return {'results': 'node'}

    def outputs(self):
        return {'html': 'string'}

    def _visualization(self):
        from v2ecoli.visualizations.multigeneration import (
            MultigenerationVisualization)
        return MultigenerationVisualization(
            config={'title': self.config['title']}, core=self.core)

    def update(self, state):
        handle = state.get('results')
        if handle is None:
            return {'html': ''}

        inputs = results_to_visualization_inputs(handle)
        rendered = self._visualization().update(inputs) or {}
        return {'html': rendered.get('html', '')}


class GrowthCard(Step):
    """Grade whether the cell actually grew over the run.

    A deliberately cheap, real check: dry mass must end above where it
    started, and must not go backwards by more than ``tolerance`` between
    consecutive ticks. It needs nothing but the trajectory — no ``sim_data``,
    no validation bundle — which is the point: the handle alone is enough to
    grade a run.
    """

    config_schema = {
        'variable': {'_type': 'string', '_default': 'dry_mass'},
        'tolerance': {'_type': 'float', '_default': 0.0},
        'title': {'_type': 'string', '_default': 'Growth'}}

    def inputs(self):
        return {'results': 'node'}

    def outputs(self):
        return {'view': 'string', 'data': 'quote'}

    def build(self, handle) -> tuple[dict, str]:
        """Grade the resolved trajectory. Returns ``(verdict, html)``."""
        variable = self.config['variable']
        tolerance = float(self.config['tolerance'])

        history = datatree_to_history(handle.resolve())
        series = [
            float(row[variable]) for row in history if variable in row]

        if len(series) < 2:
            verdict = {
                'title': self.config['title'],
                'status': 'fail',
                'reason': f'{variable!r} has {len(series)} point(s); '
                          f'a growth check needs at least 2',
                'n_points': len(series)}
            return verdict, self._html(verdict)

        start, end = series[0], series[-1]
        biggest_drop = max(
            (series[i] - series[i + 1] for i in range(len(series) - 1)),
            default=0.0)

        grew = end > start
        monotone = biggest_drop <= tolerance
        status = 'pass' if (grew and monotone) else 'fail'

        verdict = {
            'title': self.config['title'],
            'status': status,
            'variable': variable,
            'n_points': len(series),
            'start': start,
            'end': end,
            'fold_change': (end / start) if start else None,
            'largest_decrease': biggest_drop,
            'grew': grew,
            'monotone_within_tolerance': monotone}
        return verdict, self._html(verdict)

    def _html(self, verdict: dict) -> str:
        rows = ''.join(
            f'<tr><th style="text-align:left">{key}</th>'
            f'<td>{value}</td></tr>'
            for key, value in verdict.items())
        colour = '#1a7f37' if verdict.get('status') == 'pass' else '#b3261e'
        return (
            f'<section><h2 style="color:{colour}">'
            f'{verdict.get("title", "Growth")}: '
            f'{verdict.get("status", "?").upper()}</h2>'
            f'<table>{rows}</table></section>')

    def update(self, state):
        handle = state.get('results')
        if handle is None:
            return {'view': '', 'data': {}}
        verdict, html = self.build(handle)
        return {'view': html, 'data': verdict}


class ResultsAnalysis(Step):
    """Aggregate this run's trajectory into an analysis **data artifact**.

    The third flush kind. Where a visualization emits a figure and a report
    card emits a verdict, an *analysis* emits a data file — a table derived
    from the whole trajectory. This is the results-native analogue of the
    DuckDB/parquet analyses in ``v2ecoli/workflow/analyses/`` (e.g. the mass
    analyses): the same per-generation aggregate, but fed from this run's
    resolved ``results`` handle rather than a parquet re-read.

    Kept deliberately thin: a per-generation summary (n, first, last, min,
    max, mean) of one variable, emitted as CSV — the ``{filename, csv}`` shape
    an analysis artifact takes. Like the other flush steps it needs nothing but
    the handle, and fires once after the run.
    """

    config_schema = {
        'variable': {'_type': 'string', '_default': 'dry_mass'},
        'filename': {'_type': 'string', '_default': 'analysis.csv'}}

    #: Columns of the emitted per-generation summary table, in order.
    COLUMNS = ('generation', 'variable', 'n', 'first', 'last',
               'min', 'max', 'mean')

    def inputs(self):
        return {'results': 'node'}

    def outputs(self):
        return {'artifact': 'quote'}

    def analyze(self, history: list[dict]) -> dict:
        """Aggregate flat per-tick rows into a per-generation summary artifact.

        Pure over ``history`` (the shape ``datatree_to_history`` returns), so
        the aggregation is unit-testable without an engine or a store.
        """
        variable = self.config['variable']

        by_generation: dict[int, list[float]] = {}
        for row in history:
            if variable in row:
                generation = int(row.get('generation', 0))
                by_generation.setdefault(generation, []).append(
                    float(row[variable]))

        summary = []
        for generation in sorted(by_generation):
            values = by_generation[generation]
            summary.append({
                'generation': generation,
                'variable': variable,
                'n': len(values),
                'first': values[0],
                'last': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values)})

        lines = [','.join(self.COLUMNS)]
        lines.extend(
            ','.join(str(row[column]) for column in self.COLUMNS)
            for row in summary)

        return {
            'filename': self.config['filename'],
            'csv': '\n'.join(lines) + '\n',
            'summary': summary}

    def update(self, state):
        handle = state.get('results')
        if handle is None:
            return {'artifact': {}}
        history = datatree_to_history(handle.resolve())
        return {'artifact': self.analyze(history)}
