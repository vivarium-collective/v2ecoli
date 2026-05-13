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


def _build_all_parallel(cache_dir: str, seed: int) -> dict[str, dict]:
    print('\nStep 1: Building 3 composites in parallel...')
    t0 = time.time()
    args_list = [(k, cache_dir, seed) for k in MODELS]
    ctx = mp.get_context('spawn')
    with ctx.Pool(3) as pool:
        results = pool.map(_build_one_model, args_list)
    print(f'  All 3 built in {time.time() - t0:.1f}s wall (parallel)')
    return {r['key']: r for r in results}


def _build_all_sequential(cache_dir: str, seed: int) -> dict[str, dict]:
    print('\nStep 1: Building 3 composites sequentially...')
    t0 = time.time()
    results = {}
    for key in MODELS:
        r = _build_one_model((key, cache_dir, seed))
        results[r['key']] = r
    print(f'  All 3 built in {time.time() - t0:.1f}s wall (sequential)')
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_comparison(
    seed: int = 0,
    cache_dir: str = 'out/cache',
    output: str = 'out/comparison_report.html',
    parallel: bool = True,
) -> str:
    print('=== v2ecoli Architecture Comparison (v0 — structural) ===')
    print(f'    Seed: {seed}  Cache: {cache_dir}  Parallel: {parallel}')

    # Build composites + extract graph specs
    if parallel:
        build_data = _build_all_parallel(cache_dir, seed)
    else:
        build_data = _build_all_sequential(cache_dir, seed)

    # Collect specs in model order
    composite_specs = [build_data[k]['graph_data'] for k in MODELS if k in build_data]

    # Dispatch to CompareVisualization
    print('\nStep 2: Rendering HTML via CompareVisualization...')
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.compare import CompareVisualization

    viz = CompareVisualization(
        config={"title": "v2ecoli Architecture Comparison"},
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
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--cache-dir',  default='out/cache')
    parser.add_argument('--output',     default='out/comparison_report.html')
    parser.add_argument('--no-parallel', action='store_true')
    args = parser.parse_args()
    run_comparison(
        seed=args.seed,
        cache_dir=args.cache_dir,
        output=args.output,
        parallel=not args.no_parallel,
    )
