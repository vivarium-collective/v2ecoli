"""
Three-way architecture comparison: Baseline vs Departitioned vs Reconciled.

Runs all three simulations in parallel using multiprocessing, extracts the
structural graph data for each architecture, then dispatches to
``CompareVisualization`` to render a side-by-side Cytoscape HTML report.

v0 (structural-only)
--------------------
This wrapper produces a **structural** report: three interactive Cytoscape.js
composition diagrams side-by-side (baseline / departitioned / reconciled).

The full v1 report (mass trajectories, divergence analysis, molecule
divergence table, interactive JSON state viewer) produced by the original
compare_report.py is deferred. The time-series / metrics code from the
original script has been removed from the main entry point but retained in
the module as private helpers (``_run_one_model``, ``compute_metrics``,
``plot_*``) so it can be revived in a follow-up.

Usage:
    python reports/compare_report.py                        # default 0s sim (structural-only)
    python reports/compare_report.py --no-parallel          # sequential fallback
    python reports/compare_report.py --output out/compare.html
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DURATION = 0      # structural-only by default in v0
SNAPSHOT_INTERVAL = 10

MODELS: dict[str, dict] = {
    'baseline':       {'label': 'Baseline (Partitioned)',  'color': '#2563eb'},
    'departitioned':  {'label': 'Departitioned',            'color': '#dc2626'},
    'reconciled':     {'label': 'Reconciled',               'color': '#16a34a'},
}


# ---------------------------------------------------------------------------
# Composite build + graph extraction (one per worker)
# ---------------------------------------------------------------------------

def _build_one_model(args: tuple) -> dict:
    """Build a composite and extract the structural graph spec.

    Runs in a subprocess (or inline if sequential).  Returns a plain-dict
    payload that crosses the multiprocessing boundary safely.
    """
    model_key, cache_dir, seed = args
    import warnings
    warnings.filterwarnings('ignore')

    from v2ecoli import build_composite

    t0 = time.time()
    composite = build_composite(model_key, cache_dir=cache_dir, seed=seed)
    build_time = time.time() - t0

    n_steps = len(composite.step_paths)
    print(f'  [{model_key}] built in {build_time:.1f}s, {n_steps} steps')

    # Extract structural graph data
    graph_data: dict = {}
    try:
        if model_key == 'baseline':
            from v2ecoli.composites.baseline import (
                build_execution_layers, DEFAULT_FEATURES,
            )
        elif model_key == 'departitioned':
            from v2ecoli.composites.departitioned import (
                build_execution_layers, DEFAULT_FEATURES,
            )
        elif model_key == 'reconciled':
            from v2ecoli.composites.reconciled import (
                build_execution_layers, DEFAULT_FEATURES,
            )
        else:
            raise ValueError(f'Unknown model key: {model_key}')

        from v2ecoli.visualizations._helpers import build_graph
        layers = build_execution_layers(DEFAULT_FEATURES)
        graph_data = build_graph(composite, layers)
    except Exception as exc:
        print(f'  [{model_key}] graph extraction failed: {exc}')

    # Merge architecture key so CompareVisualization can label each panel.
    graph_data['architecture'] = model_key
    graph_data.setdefault('nodes', [])
    graph_data.setdefault('edges', [])
    graph_data.setdefault('layers', [])
    graph_data.setdefault('legend', [])

    return {
        'key':        model_key,
        'build_time': build_time,
        'n_steps':    n_steps,
        'graph_data': graph_data,
    }


# ---------------------------------------------------------------------------
# Study spec + main entry point
# ---------------------------------------------------------------------------

def _load_study(study_path: str) -> dict:
    """Load + lightly validate a v3-shape study.yaml driving this report.

    Expected shape:
        baseline:
        - name: <one of MODELS>           # used as the build_composite key
          composite: <dotted ref>          # informational; matched against ``name``
          params: { seed: int, cache_dir: str }
        - ...
        visualizations:
        - name: comparison                 # first viz used
          address: local:CompareVisualization
          config: { title: str, ... }
    """
    import yaml as _yaml
    with open(study_path) as fh:
        spec = _yaml.safe_load(fh) or {}

    baseline_entries = spec.get('baseline') or []
    if not baseline_entries:
        raise ValueError(
            f"study {study_path!r}: `baseline:` list is empty — nothing to compare"
        )
    unknown = [e.get('name') for e in baseline_entries if e.get('name') not in MODELS]
    if unknown:
        raise ValueError(
            f"study {study_path!r}: baseline entry name(s) {unknown!r} not in "
            f"{sorted(MODELS)} — compare_report only knows these architectures"
        )
    return spec


def run_comparison(
    seed: int = 0,
    cache_dir: str = 'out/cache',
    output: str = 'out/comparison_report.html',
    parallel: bool = True,
    study: dict | None = None,
) -> str:
    """Build the three architecture composites, extract their graph_data, and
    render via CompareVisualization.

    When ``study`` is provided (a parsed study.yaml dict, v3 shape):
      - the model list comes from ``study.baseline[].name``
      - per-model seed/cache_dir come from ``study.baseline[i].params``
        (CLI ``seed`` / ``cache_dir`` are still applied as defaults when a
        params entry omits them)
      - the visualization config comes from
        ``study.visualizations[0].config`` if a ``CompareVisualization`` entry
        is present; otherwise the title/options default to in-code values
    Otherwise the legacy behavior runs all three MODELS at the supplied
    seed + cache_dir.
    """
    print('=== v2ecoli Architecture Comparison (v0 — structural) ===')

    # Resolve per-model build args from the study (or fall back to defaults).
    if study is not None:
        baseline_entries = study['baseline']
        model_keys = [e['name'] for e in baseline_entries]
        per_model_args = []
        for entry in baseline_entries:
            p = entry.get('params') or {}
            per_model_args.append((
                entry['name'],
                p.get('cache_dir', cache_dir),
                int(p.get('seed', seed)),
            ))
        print(f'    Study: {study.get("name", "(unnamed)")}  '
              f'Models: {", ".join(model_keys)}  Parallel: {parallel}')
    else:
        model_keys = list(MODELS)
        per_model_args = [(k, cache_dir, seed) for k in model_keys]
        print(f'    Seed: {seed}  Cache: {cache_dir}  Parallel: {parallel}')

    # Build composites + extract graph specs
    print(f'\nStep 1: Building {len(per_model_args)} composite(s) '
          f'{"in parallel" if parallel else "sequentially"}...')
    t0 = time.time()
    if parallel:
        ctx = mp.get_context('spawn')
        with ctx.Pool(len(per_model_args)) as pool:
            results = pool.map(_build_one_model, per_model_args)
    else:
        results = [_build_one_model(args) for args in per_model_args]
    print(f'  All {len(results)} built in {time.time() - t0:.1f}s wall')

    build_data = {r['key']: r for r in results}
    composite_specs = [build_data[k]['graph_data'] for k in model_keys if k in build_data]

    # Visualization config: prefer study's CompareVisualization entry, else default.
    viz_config = {"title": "v2ecoli Architecture Comparison"}
    if study is not None:
        for v in (study.get('visualizations') or []):
            if isinstance(v, dict) and 'CompareVisualization' in (v.get('address') or ''):
                viz_config = dict(v.get('config') or viz_config)
                break

    # Dispatch to CompareVisualization
    print('\nStep 2: Rendering HTML via CompareVisualization...')
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.compare import CompareVisualization

    viz = CompareVisualization(
        config=viz_config,
        core=allocate_core(),
    )
    result = viz.update({"composite_specs": composite_specs})
    html = result["html"]

    # Write output
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as fh:
        fh.write(html)
    print(f'\n=== Done. Report: {output} ===')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='v2ecoli architecture comparison (structural, v0)')
    parser.add_argument('--study',      default=None,
        help='Path to a v3-shape study.yaml driving the comparison (overrides '
             'the default model list; CLI seed/cache-dir become per-model '
             'defaults when the study omits them).')
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--cache-dir',  default='out/cache')
    parser.add_argument('--output',     default='out/comparison_report.html')
    parser.add_argument('--no-parallel', action='store_true')
    args = parser.parse_args()
    study_spec = _load_study(args.study) if args.study else None
    run_comparison(
        seed=args.seed,
        cache_dir=args.cache_dir,
        output=args.output,
        parallel=not args.no_parallel,
        study=study_spec,
    )
