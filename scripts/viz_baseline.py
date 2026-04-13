#!/usr/bin/env python3
"""Render a large, navigable bigraph visualization of the baseline composite.

Output:
    out/viz/baseline_bigraph.svg          -- raw Graphviz SVG
    out/viz/baseline_bigraph.html         -- zoom/pan HTML viewer with sidebar

Usage:
    python scripts/viz_baseline.py [--cache-dir CACHE_DIR] [--no-open]

The HTML viewer is opened in the default browser unless --no-open is given.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import webbrowser
from pathlib import Path
from typing import Any

# Make the repo importable when running the script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bigraph_viz import plot_bigraph  # noqa: E402

from v2ecoli.composite import make_composite  # noqa: E402
from v2ecoli.generate import (  # noqa: E402
    BASE_EXECUTION_LAYERS,
    build_execution_layers,
    DEFAULT_FEATURES,
)


OUT_DIR = REPO_ROOT / 'out' / 'viz'
SVG_NAME = 'baseline_bigraph'


# ---------------------------------------------------------------------------
# Subsystem classification
# ---------------------------------------------------------------------------
# Order matters: first matching rule wins.
BIO_COLORS: list[tuple[str, str, str, Any]] = [
    # (subsystem_key, display_label, hex_color, matcher(name))
    ('replication',  'DNA replication',        '#F4A7A1',
        lambda n: 'chromosome' in n or 'replication' in n or 'oriC' in n),
    ('transcription','Transcription',          '#9CC8E3',
        lambda n: 'transcript' in n or 'rna-synth' in n
                  or 'rna_synth' in n or 'rnap' in n),
    ('rna',          'RNA metabolism',         '#B5D9F0',
        lambda n: n.startswith('RNA') or 'rna-' in n or 'rna_' in n
                  or 'rna-mat' in n or 'rna-deg' in n),
    ('translation',  'Translation',            '#A4D4A4',
        lambda n: 'polypeptide' in n or 'ribosome' in n
                  or 'protein-deg' in n),
    ('regulation',   'Regulation (TFs, ppGpp)','#D9B8E0',
        lambda n: 'tf-' in n or 'tf_' in n or 'ppgpp' in n
                  or 'attenuation' in n),
    ('signaling',    'Signaling / equilibrium','#FFD58C',
        lambda n: 'equilibrium' in n or 'two-component' in n
                  or 'complexation' in n),
    ('metabolism',   'Metabolism (FBA)',       '#F7D488',
        lambda n: 'metabolism' in n),
    ('alloc',        'Partition / allocator',  '#FFAE80',
        lambda n: 'allocator' in n or '_requester' in n
                  or '_evolver' in n),
    ('listen',       'Listeners',              '#D5D5D5',
        lambda n: 'listener' in n or n.endswith('_listener')),
    ('infra',        'Infrastructure / flow',  '#E8E8E8',
        lambda n: any(s in n for s in (
            'unique_update', 'global_clock', 'emitter',
            'mark_d_period', 'division', 'metabolic_kinetics',
            'media_update', 'post-division'))),
]


def classify(name: str) -> tuple[str, str, str] | None:
    for key, label, color, matcher in BIO_COLORS:
        if matcher(name):
            return key, label, color
    return None


# ---------------------------------------------------------------------------
# Composite → filtered viz state
# ---------------------------------------------------------------------------

STORE_NODES = (
    'bulk', 'unique', 'environment', 'boundary', 'listeners',
    'request', 'allocate', 'process_state', 'next_update_time',
    'global_time', 'timestep', 'ppgpp_state', 'attenuation_config',
)


def build_viz_state(composite) -> tuple[dict, dict, dict[str, list]]:
    """Return (viz_state, fill_colors, groups_by_subsystem).

    Strips internal bookkeeping keys (_layer, _flow) and keeps each
    process node's visible port wiring.
    """
    cell = composite.state.get('agents', {}).get('0', composite.state)
    viz: dict[str, Any] = {}

    for name, edge in cell.items():
        if not isinstance(edge, dict):
            continue
        if '_type' in edge:
            inputs = {
                p: w for p, w in edge.get('inputs', {}).items()
                if not p.startswith('_layer') and not p.startswith('_flow')
            }
            outputs = {
                p: w for p, w in edge.get('outputs', {}).items()
                if not p.startswith('_layer') and not p.startswith('_flow')
            }
            clean = name.replace('ecoli-', '')
            viz[clean] = {
                '_type': edge['_type'],
                'inputs': inputs,
                'outputs': outputs,
            }
        elif name == 'unique' and isinstance(edge, dict):
            viz[name] = {k: {} for k in edge.keys() if not k.startswith('_')}
        elif name in STORE_NODES:
            viz[name] = {}

    prefix = ('agents', '0')
    fill_colors: dict = {}
    groups: dict[str, list] = {key: [] for (key, *_rest) in BIO_COLORS}

    for n, node in viz.items():
        if not isinstance(node, dict) or '_type' not in node:
            continue
        c = classify(n)
        if c is None:
            continue
        key, _label, color = c
        path = prefix + (n,)
        fill_colors[path] = color
        groups[key].append(path)

    return {'agents': {'0': viz}}, fill_colors, groups


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def render_svg(composite, out_dir: Path) -> str:
    """Call plot_bigraph and return the SVG string (cleaned for embedding)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    viz_state, fill_colors, groups = build_viz_state(composite)

    plot_bigraph(
        viz_state,
        # Layout: horizontal flow reads like the execution pipeline
        rankdir='LR',
        engine='dot',
        # Large, high-DPI canvas so detail survives zoom
        dpi='96',
        size='40,28',
        # Declutter
        remove_process_place_edges=True,
        port_labels=False,
        # Readability — large labels for easy scanning at any zoom
        node_label_size='40pt',
        process_label_size='36pt',
        label_margin='0.15',
        # Color-coded subsystems
        node_groups=[g for g in groups.values() if g],
        node_fill_colors=fill_colors,
        # Output
        out_dir=str(out_dir),
        filename=SVG_NAME,
        file_format='svg',
    )

    svg_path = out_dir / f'{SVG_NAME}.svg'
    svg = svg_path.read_text()
    # Strip the hard-coded width/height so the viewer can resize freely
    svg = re.sub(r'width="[^"]*pt"', '', svg, count=1)
    svg = re.sub(r'height="[^"]*pt"', '', svg, count=1)
    return svg


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def _layer_index_html(features: list[str] | None) -> str:
    """Render the sidebar: execution layers with clickable step names."""
    layers = build_execution_layers(features or DEFAULT_FEATURES)

    rows = []
    for idx, layer in enumerate(layers):
        names = []
        for step in layer:
            clean = step.replace('ecoli-', '')
            c = classify(clean)
            color = c[2] if c else '#EEE'
            names.append(
                f'<span class="chip" data-step="{html.escape(clean)}" '
                f'style="background:{color}">{html.escape(clean)}</span>'
            )
        rows.append(
            f'<div class="layer-row">'
            f'<div class="layer-num">L{idx:02d}</div>'
            f'<div class="layer-chips">{"".join(names)}</div>'
            f'</div>'
        )
    return '\n'.join(rows)


def _legend_html() -> str:
    items = []
    for (_key, label, color, _m) in BIO_COLORS:
        items.append(
            f'<div class="legend-row">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f'<span class="legend-label">{html.escape(label)}</span>'
            f'</div>'
        )
    return '\n'.join(items)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>v2ecoli — Baseline Bigraph</title>
  <style>
    :root {{ --sidebar-w: 360px; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fafafa; color: #222; }}
    header {{ padding: 10px 16px; border-bottom: 1px solid #ddd; background: #fff; display: flex; align-items: baseline; gap: 16px; }}
    header h1 {{ font-size: 16px; margin: 0; }}
    header .subtitle {{ font-size: 12px; color: #666; }}
    header .actions {{ margin-left: auto; font-size: 12px; color: #666; }}
    .app {{ display: flex; height: calc(100vh - 44px); }}
    aside {{
      width: var(--sidebar-w); flex: 0 0 var(--sidebar-w);
      overflow-y: auto; border-right: 1px solid #ddd; background: #fff;
      padding: 12px 14px; font-size: 12px;
    }}
    aside h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #555; margin: 18px 0 8px; }}
    aside h2:first-child {{ margin-top: 0; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
    .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #888; }}
    .layer-row {{ display: flex; align-items: flex-start; gap: 6px; padding: 4px 0; border-top: 1px solid #f0f0f0; }}
    .layer-row:first-child {{ border-top: none; }}
    .layer-num {{ flex: 0 0 32px; font-family: 'SF Mono', Menlo, monospace; font-size: 11px; color: #888; padding-top: 3px; }}
    .layer-chips {{ display: flex; flex-wrap: wrap; gap: 3px; }}
    .chip {{
      display: inline-block; padding: 2px 6px; border-radius: 10px;
      font-size: 11px; border: 1px solid rgba(0,0,0,0.12);
      cursor: pointer; user-select: none;
    }}
    .chip:hover {{ outline: 2px solid #333; }}
    .chip.active {{ outline: 3px solid #000; font-weight: 600; }}
    main {{ flex: 1; position: relative; overflow: hidden; background: #fff; }}
    #svg-host {{ position: absolute; inset: 0; }}
    #svg-host > svg {{ width: 100%; height: 100%; }}
    .toolbar {{
      position: absolute; top: 10px; right: 10px; z-index: 10;
      background: rgba(255,255,255,0.92); border: 1px solid #ccc;
      border-radius: 6px; padding: 4px; display: flex; gap: 2px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .toolbar button {{
      border: none; background: #f4f4f4; padding: 6px 10px;
      font-size: 13px; cursor: pointer; border-radius: 4px;
    }}
    .toolbar button:hover {{ background: #e8e8e8; }}
    .counts {{ font-family: 'SF Mono', Menlo, monospace; font-size: 11px; color: #666; }}
    /* Highlight matching nodes (by title prefix) */
    #svg-host g.node.highlighted > ellipse,
    #svg-host g.node.highlighted > polygon,
    #svg-host g.node.highlighted > rect {{
      stroke: #000 !important; stroke-width: 4px !important;
    }}
  </style>
</head>
<body>
  <header>
    <h1>v2ecoli · Baseline composite</h1>
    <span class="subtitle">{n_processes} processes · {n_layers} execution layers · features: {features}</span>
    <span class="actions"><kbd>scroll</kbd> to zoom · <kbd>drag</kbd> to pan · click a chip to highlight</span>
  </header>
  <div class="app">
    <aside>
      <h2>Subsystem legend</h2>
      {legend}
      <h2>Execution order (top → bottom)</h2>
      {layers}
      <h2>Tips</h2>
      <p style="color:#666; line-height:1.4;">
        Each layer runs as a parallel group; the next layer starts only
        when all steps in the current one have finished. Partitioned
        processes appear as <code>_requester</code> / <code>_evolver</code>
        pairs flanking their allocator.
      </p>
    </aside>
    <main>
      <div class="toolbar">
        <button id="btn-reset" title="Reset view">⟲ Reset</button>
        <button id="btn-fit" title="Fit to screen">⤢ Fit</button>
        <button id="btn-zoom-in" title="Zoom in">+</button>
        <button id="btn-zoom-out" title="Zoom out">−</button>
      </div>
      <div id="svg-host">{svg}</div>
    </main>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
  <script>
    const host = document.getElementById('svg-host');
    const svg = host.querySelector('svg');
    // Normalize the SVG so svg-pan-zoom can manipulate it
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.removeAttribute('width');
    svg.removeAttribute('height');

    const panZoom = svgPanZoom(svg, {{
      zoomEnabled: true, controlIconsEnabled: false,
      fit: true, center: true, minZoom: 0.1, maxZoom: 20,
      zoomScaleSensitivity: 0.3,
    }});

    document.getElementById('btn-reset').onclick = () => panZoom.reset();
    document.getElementById('btn-fit').onclick = () => {{ panZoom.resize(); panZoom.fit(); panZoom.center(); }};
    document.getElementById('btn-zoom-in').onclick = () => panZoom.zoomBy(1.3);
    document.getElementById('btn-zoom-out').onclick = () => panZoom.zoomBy(1 / 1.3);

    // Chip click → highlight node
    function clearHighlights() {{
      document.querySelectorAll('#svg-host g.node.highlighted').forEach(
        el => el.classList.remove('highlighted')
      );
      document.querySelectorAll('.chip.active').forEach(
        el => el.classList.remove('active')
      );
    }}
    function findNode(name) {{
      // Graphviz writes the node label into <title>
      const titles = Array.from(host.querySelectorAll('g.node > title'));
      return titles.find(t => t.textContent.includes(name))?.parentElement;
    }}
    document.querySelectorAll('.chip').forEach(chip => {{
      chip.addEventListener('click', () => {{
        const name = chip.dataset.step;
        clearHighlights();
        chip.classList.add('active');
        const node = findNode(name);
        if (!node) return;
        node.classList.add('highlighted');
        // Pan to the node
        const bbox = node.getBBox();
        const viewBox = svg.viewBox.baseVal;
        const cx = bbox.x + bbox.width / 2 - viewBox.x;
        const cy = bbox.y + bbox.height / 2 - viewBox.y;
        // Rough center-on-node
        panZoom.resize();
        panZoom.reset();
        const sizes = panZoom.getSizes();
        panZoom.zoom(2.5);
        panZoom.pan({{
          x: sizes.width / 2 - cx * panZoom.getZoom(),
          y: sizes.height / 2 - cy * panZoom.getZoom(),
        }});
      }});
    }});

    // Escape clears highlights
    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') clearHighlights();
    }});
  </script>
</body>
</html>
"""


def render_html(svg: str, composite, features: list[str] | None,
                out_path: Path) -> None:
    cell = composite.state.get('agents', {}).get('0', composite.state)
    n_processes = sum(
        1 for e in cell.values() if isinstance(e, dict) and '_type' in e
    )
    n_layers = len(build_execution_layers(features or DEFAULT_FEATURES))
    features_label = ', '.join(features or DEFAULT_FEATURES) or 'none'

    html_doc = HTML_TEMPLATE.format(
        svg=svg,
        legend=_legend_html(),
        layers=_layer_index_html(features),
        n_processes=n_processes,
        n_layers=n_layers,
        features=html.escape(features_label),
    )
    out_path.write_text(html_doc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='out/cache',
                        help='Directory containing initial_state.json + sim_data_cache.dill')
    parser.add_argument('--features', nargs='*', default=None,
                        help='Feature modules to enable (default: DEFAULT_FEATURES)')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not open the HTML viewer in a browser')
    args = parser.parse_args()

    cache_dir = REPO_ROOT / args.cache_dir
    if not cache_dir.is_dir():
        print(f'Cache directory not found: {cache_dir}', file=sys.stderr)
        print('Run a simulation first (or pass --cache-dir).', file=sys.stderr)
        sys.exit(1)

    print(f'Building composite from {cache_dir}...')
    composite = make_composite(
        cache_dir=str(cache_dir),
        features=args.features,
    )

    out_dir = OUT_DIR
    print(f'Rendering SVG to {out_dir / (SVG_NAME + ".svg")}...')
    svg = render_svg(composite, out_dir)

    html_path = out_dir / f'{SVG_NAME}.html'
    print(f'Writing HTML viewer to {html_path}...')
    render_html(svg, composite, args.features, html_path)

    print(f'\nDone. Open:')
    print(f'  {html_path}')

    if not args.no_open:
        webbrowser.open(f'file://{html_path}')


if __name__ == '__main__':
    main()
