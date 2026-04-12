#!/usr/bin/env python3
"""Interactive network visualization for any v2ecoli architecture.

Supports baseline (default), departitioned, and reconciled. Each renders
to its own HTML file in ``out/viz/``. Multiple architectures can be built
in one invocation.

Usage:
    # Single architecture (opens in browser)
    python scripts/viz_network.py --arch baseline
    python scripts/viz_network.py --arch departitioned
    python scripts/viz_network.py --arch reconciled

    # All three at once
    python scripts/viz_network.py --arch baseline departitioned reconciled
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2ecoli.viz import build_graph, write_outputs  # noqa: E402


OUT_DIR = REPO_ROOT / 'out' / 'viz'


# Architecture registry: name -> (composite_factory, layer_builder, title_prefix)
def _load_arch(name: str):
    if name == 'baseline':
        from v2ecoli.composite import make_composite
        from v2ecoli.generate import build_execution_layers, DEFAULT_FEATURES
        return {
            'factory': lambda cache_dir, features: make_composite(
                cache_dir=cache_dir,
                features=features or DEFAULT_FEATURES,
            ),
            'layer_builder': build_execution_layers,
            'default_features': DEFAULT_FEATURES,
            'title': 'Baseline (partitioned)',
            'name': 'baseline_network',
        }
    if name == 'departitioned':
        from v2ecoli.composite_departitioned import make_departitioned_composite
        from v2ecoli.generate_departitioned import (
            build_execution_layers, DEFAULT_FEATURES,
        )
        return {
            'factory': lambda cache_dir, features: make_departitioned_composite(
                cache_dir=cache_dir,
            ),
            'layer_builder': build_execution_layers,
            'default_features': DEFAULT_FEATURES,
            'title': 'Departitioned (no allocator)',
            'name': 'departitioned_network',
        }
    if name == 'reconciled':
        from v2ecoli.composite_reconciled import make_reconciled_composite
        from v2ecoli.generate_reconciled import (
            build_execution_layers, DEFAULT_FEATURES,
        )
        return {
            'factory': lambda cache_dir, features: make_reconciled_composite(
                cache_dir=cache_dir,
            ),
            'layer_builder': build_execution_layers,
            'default_features': DEFAULT_FEATURES,
            'title': 'Reconciled (grouped allocator)',
            'name': 'reconciled_network',
        }
    raise ValueError(f'Unknown architecture: {name}')


ARCH_CHOICES = ['baseline', 'departitioned', 'reconciled']


def build_and_write(arch: str, cache_dir: Path, features: list[str] | None) -> Path:
    spec = _load_arch(arch)
    layer_builder = spec['layer_builder']
    features_list = features if features is not None else spec['default_features']

    print(f'[{arch}] building composite from {cache_dir}...')
    composite = spec['factory'](str(cache_dir), features_list)

    print(f'[{arch}] extracting graph...')
    layers = layer_builder(features_list)
    data = build_graph(composite, layers)

    n_proc = sum(1 for n in data['nodes'] if n['data']['kind'] == 'process')
    n_store = sum(1 for n in data['nodes'] if n['data']['kind'] == 'store')
    n_edges = len(data['edges'])
    feat_str = ', '.join(features_list) or 'none'
    subtitle = (f'{n_proc} processes · {n_store} stores · '
                f'{n_edges} edges · features: {feat_str}')

    print(f'[{arch}] {subtitle}')

    json_path, html_path = write_outputs(
        data,
        out_dir=OUT_DIR,
        name=spec['name'],
        title=f'v2ecoli · {spec["title"]}',
        subtitle=subtitle,
    )
    print(f'[{arch}] wrote {json_path}')
    print(f'[{arch}] wrote {html_path}')
    return html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', nargs='+', default=['baseline'],
                        choices=ARCH_CHOICES,
                        help='Architecture(s) to visualize')
    parser.add_argument('--cache-dir', default='out/cache')
    parser.add_argument('--features', nargs='*', default=None,
                        help='Feature modules to enable (default: per-arch defaults)')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not open the HTML viewers in a browser')
    args = parser.parse_args()

    cache_dir = REPO_ROOT / args.cache_dir
    if not cache_dir.is_dir():
        print(f'Cache directory not found: {cache_dir}', file=sys.stderr)
        sys.exit(1)

    html_paths = []
    for arch in args.arch:
        try:
            html_paths.append(build_and_write(arch, cache_dir, args.features))
        except Exception as e:
            print(f'[{arch}] failed: {e}', file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not args.no_open:
        for p in html_paths:
            webbrowser.open(f'file://{p}')


if __name__ == '__main__':
    main()
