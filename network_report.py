"""Standalone composition-diagram viewer for a v2ecoli composite.

Loads one architecture (baseline by default), builds the execution layers,
extracts the composition graph via ``v2ecoli.viz.build_graph``, writes an
HTML page with the interactive Cytoscape network, and opens it.

Usage:
    python network_report.py                          # baseline (default)
    python network_report.py --model departitioned
    python network_report.py --model reconciled
    python network_report.py --model baseline --no-open
    python network_report.py --output out/network_baseline.html
"""

import os
import sys
import argparse
import subprocess
import warnings

warnings.filterwarnings('ignore')


MODELS = {
    'baseline': {
        'factory': 'v2ecoli.composite.make_composite',
        'layers': 'v2ecoli.generate',
        'label': 'Baseline (partitioned)',
    },
    'departitioned': {
        'factory': 'v2ecoli.composite_departitioned.make_departitioned_composite',
        'layers': 'v2ecoli.generate_departitioned',
        'label': 'Departitioned (no allocator)',
    },
    'reconciled': {
        'factory': 'v2ecoli.composite_reconciled.make_reconciled_composite',
        'layers': 'v2ecoli.generate_reconciled',
        'label': 'Reconciled (grouped allocator)',
    },
}


def _resolve(dotted):
    mod_path, name = dotted.rsplit('.', 1)
    import importlib
    return getattr(importlib.import_module(mod_path), name)


def main():
    parser = argparse.ArgumentParser(description='v2ecoli composition-diagram viewer')
    parser.add_argument('--model', choices=list(MODELS), default='baseline',
                        help='which architecture to visualize')
    parser.add_argument('--cache', default='out/cache',
                        help='simdata cache directory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None,
                        help='output HTML path (default: out/network_<model>.html)')
    parser.add_argument('--no-open', action='store_true',
                        help='skip opening the report in a browser')
    args = parser.parse_args()

    spec = MODELS[args.model]
    make_composite = _resolve(spec['factory'])

    import importlib
    layers_mod = importlib.import_module(spec['layers'])
    build_execution_layers = layers_mod.build_execution_layers
    DEFAULT_FEATURES = layers_mod.DEFAULT_FEATURES

    from v2ecoli.viz import build_graph, render_html

    print(f'Building {args.model} composite ...')
    composite = make_composite(cache_dir=args.cache, seed=args.seed)
    layers = build_execution_layers(DEFAULT_FEATURES)

    print('Extracting composition graph ...')
    data = build_graph(composite, layers)

    n_proc = sum(1 for n in data['nodes'] if n['data']['kind'] == 'process')
    n_store = sum(1 for n in data['nodes'] if n['data']['kind'] == 'store')
    n_edges = len(data['edges'])
    title = f'v2ecoli · {spec["label"]}'
    subtitle = f'{n_proc} processes · {n_store} stores · {n_edges} edges'

    output = args.output or f'out/network_{args.model}.html'
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        f.write(render_html(data, title, subtitle))

    # Mirror to docs/ so GitHub Pages stays in sync.
    import shutil
    docs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'docs')
    if os.path.isdir(docs_dir):
        shutil.copy2(output, os.path.join(
            docs_dir, f'network_{args.model}.html'))

    print(f'Wrote {output}')
    print(f'  {subtitle}')

    if not args.no_open:
        subprocess.run(['open', output], capture_output=True)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
