"""Interactive Cytoscape.js network visualization for any v2ecoli composite.

Public API:
    build_graph(composite, layers) -> dict
        Pure extractor. Returns {nodes, edges, layers, legend}.
    render_html(data, title, subtitle) -> str
        Renders the HTML viewer as a string.
    write_outputs(data, out_dir, name, title, subtitle) -> (json_path, html_path)
        Convenience: write the JSON + HTML side-by-side.

Architecture-agnostic: accepts any ``Composite`` instance and the list of
execution layers used to lay out its steps. Feed it the baseline,
departitioned, or reconciled composite and the matching layers list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence


# ---------------------------------------------------------------------------
# Subsystem classification
# ---------------------------------------------------------------------------
# (key, display_label, hex_color, matcher(name: str) -> bool)
BIO_COLORS: list[tuple[str, str, str, Callable[[str], bool]]] = [
    ('replication',  'DNA replication',        '#F4A7A1',
        lambda n: 'chromosome' in n or 'replication' in n or 'oriC' in n),
    ('transcription','Transcription',          '#9CC8E3',
        lambda n: 'transcript' in n or 'rnap' in n),
    ('rna',          'RNA metabolism',         '#B5D9F0',
        lambda n: n.startswith('RNA') or 'rna-' in n or 'rna_' in n),
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
        lambda n: 'allocator' in n),
    ('listen',       'Listeners',              '#D5D5D5',
        lambda n: 'listener' in n or n.endswith('_listener')),
    ('infra',        'Infrastructure / flow',  '#E8E8E8',
        lambda n: any(s in n for s in (
            'unique_update', 'global_clock', 'emitter', 'mark_d_period',
            'division', 'exchange_data', 'media_update', 'post-division',
            'reconciled_', 'allocator_'))),
]


def classify(name: str) -> tuple[str, str]:
    """Return (subsystem_key, color) for a process name."""
    for key, _label, color, matcher in BIO_COLORS:
        if matcher(name):
            return key, color
    return 'other', '#CCCCCC'


# ---------------------------------------------------------------------------
# Composite → graph data
# ---------------------------------------------------------------------------

_STORE_NODES = (
    'bulk', 'unique', 'environment', 'boundary', 'listeners',
    'request', 'allocate', 'process_state', 'next_update_time',
    'global_time', 'timestep', 'ppgpp_state', 'attenuation_config',
)


def _wire_to_store_id(wire: Any) -> str | None:
    """Return a stable store-node ID from a wire path (first segment)."""
    if isinstance(wire, (list, tuple)) and wire:
        head = wire[0]
        if isinstance(head, str) and head and not head.startswith('_'):
            return head
    return None


def build_graph(composite, layers: Sequence[Sequence[str]]) -> dict:
    """Extract {nodes, edges, layers, legend} from a Composite.

    Args:
        composite: a process-bigraph Composite
        layers: list of lists, one per execution layer, containing step
            names in execution order. Used for the sidebar layer index
            and for per-node layer metadata.

    Returns:
        Plain-dict payload suitable for JSON serialization and for the
        Cytoscape viewer.
    """
    cell = composite.state.get('agents', {}).get('0', composite.state)

    # Layer index: map each step name to its execution layer
    layer_of: dict[str, int] = {}
    for i, layer in enumerate(layers):
        for step in layer:
            layer_of[step] = i
            layer_of[step.replace('ecoli-', '')] = i

    nodes: dict[str, dict] = {}
    edge_index: dict[tuple[str, str, str], dict] = {}

    for raw_name, edge in cell.items():
        if not isinstance(edge, dict):
            continue
        if '_type' not in edge:
            continue

        name = raw_name.replace('ecoli-', '')
        subsystem, color = classify(name)
        nodes[name] = {
            'data': {
                'id': name,
                'label': name,
                'kind': 'process',
                'subsystem': subsystem,
                'color': color,
                'layer': layer_of.get(raw_name, layer_of.get(name, -1)),
                'klass': edge.get('_type', 'process'),
            },
        }

        for direction, port_dict in (
                ('input', edge.get('inputs', {}) or {}),
                ('output', edge.get('outputs', {}) or {})):
            for port, wire in port_dict.items():
                if port.startswith('_layer') or port.startswith('_flow'):
                    continue
                store_id = _wire_to_store_id(wire)
                if store_id is None:
                    continue
                if store_id not in nodes:
                    nodes[store_id] = {
                        'data': {
                            'id': store_id,
                            'label': store_id,
                            'kind': 'store',
                            'subsystem': 'store',
                            'color': '#FFFFFF',
                        },
                    }
                src, dst = (store_id, name) if direction == 'input' \
                    else (name, store_id)
                key = (src, dst, direction)
                if key in edge_index:
                    edge_index[key]['data']['ports'].append(port)
                else:
                    edge_index[key] = {
                        'data': {
                            'id': f'{src}__{dst}__{direction}',
                            'source': src,
                            'target': dst,
                            'direction': direction,
                            'ports': [port],
                        },
                    }

    # Capture top-level store nodes even if no process references them
    for name, entry in cell.items():
        if name.startswith('_') or name in nodes:
            continue
        if isinstance(entry, dict) and '_type' not in entry:
            nodes[name] = {
                'data': {
                    'id': name,
                    'label': name,
                    'kind': 'store',
                    'subsystem': 'store',
                    'color': '#FFFFFF',
                },
            }

    # Compact edge labels
    for e in edge_index.values():
        ports = sorted(set(e['data']['ports']))
        e['data']['ports'] = ports
        e['data']['label'] = (
            ', '.join(ports) if len(ports) <= 3
            else f'{ports[0]} (+{len(ports) - 1} more)'
        )

    return {
        'nodes': list(nodes.values()),
        'edges': list(edge_index.values()),
        'layers': [
            [step.replace('ecoli-', '') for step in layer]
            for layer in layers
        ],
        'legend': [
            {'key': k, 'label': lbl, 'color': c}
            for (k, lbl, c, _m) in BIO_COLORS
        ],
    }


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #222; }}
    header {{ padding: 10px 16px; border-bottom: 1px solid #ddd; background: #fff; display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
    header h1 {{ font-size: 16px; margin: 0; }}
    header .subtitle {{ font-size: 12px; color: #666; }}
    .app {{ display: flex; height: calc(100vh - 44px); }}
    aside {{
      width: 300px; flex: 0 0 300px;
      overflow-y: auto; border-right: 1px solid #ddd; background: #fafafa;
      padding: 12px 14px; font-size: 12px;
    }}
    aside h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #555; margin: 16px 0 6px; }}
    aside h2:first-child {{ margin-top: 0; }}
    aside input[type=search] {{
      width: 100%; padding: 6px 8px; font-size: 13px;
      border: 1px solid #bbb; border-radius: 4px;
    }}
    .filter-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; cursor: pointer; user-select: none; }}
    .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #888; flex: 0 0 14px; }}
    .filter-row.disabled .legend-label {{ color: #aaa; text-decoration: line-through; }}
    .filter-row.disabled .legend-swatch {{ opacity: 0.3; }}
    .layer-row {{ padding: 3px 0; border-top: 1px solid #eee; }}
    .layer-row:first-child {{ border-top: none; }}
    .layer-num {{ font-family: 'SF Mono', Menlo, monospace; font-size: 10px; color: #888; }}
    .chip {{
      display: inline-block; padding: 1px 6px; margin: 2px 2px 0 0;
      border-radius: 9px; font-size: 11px; cursor: pointer;
      border: 1px solid rgba(0,0,0,0.15); user-select: none;
    }}
    .chip:hover {{ outline: 1px solid #333; }}
    main {{ flex: 1; position: relative; overflow: hidden; }}
    #cy {{ position: absolute; inset: 0; background: #fff; }}
    .toolbar {{
      position: absolute; top: 10px; right: 10px; z-index: 10;
      background: rgba(255,255,255,0.96); border: 1px solid #ccc;
      border-radius: 6px; padding: 6px; display: flex; gap: 6px; align-items: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 12px;
    }}
    .toolbar label {{ color: #666; }}
    .toolbar select, .toolbar button {{
      border: 1px solid #ccc; background: #fff; padding: 4px 8px;
      font-size: 12px; border-radius: 4px; cursor: pointer;
    }}
    .toolbar button:hover, .toolbar select:hover {{ background: #f0f0f0; }}
    #details {{
      position: absolute; bottom: 10px; left: 10px; z-index: 10;
      max-width: 420px; max-height: 40vh; overflow-y: auto;
      background: rgba(255,255,255,0.98); border: 1px solid #ccc;
      border-radius: 6px; padding: 10px 12px; font-size: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      display: none;
    }}
    #details h3 {{ margin: 0 0 6px 0; font-size: 13px; }}
    #details .meta {{ color: #666; font-size: 11px; margin-bottom: 6px; }}
    #details table {{ width: 100%; border-collapse: collapse; }}
    #details td {{ padding: 2px 6px; border-top: 1px solid #eee; font-size: 11px; }}
    #details td:first-child {{ color: #666; white-space: nowrap; }}
    #details code {{ font-size: 11px; background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
    .hotkey {{ font-family: 'SF Mono', Menlo, monospace; background: #eee; padding: 1px 4px; border-radius: 3px; color: #444; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <span class="subtitle">{subtitle}</span>
    <span class="subtitle">· <span class="hotkey">click</span> to focus  <span class="hotkey">esc</span> to clear  <span class="hotkey">drag</span> to move nodes</span>
  </header>
  <div class="app">
    <aside>
      <h2>Search</h2>
      <input type="search" id="q" placeholder="Filter nodes by name…">

      <h2>Subsystems</h2>
      <div id="legend"></div>

      <h2>Execution layers</h2>
      <div id="layers"></div>
    </aside>
    <main>
      <div class="toolbar">
        <label>Layout:</label>
        <select id="layout-select">
          <option value="dagre" selected>Dagre (hierarchy)</option>
          <option value="fcose">fCoSE (force)</option>
          <option value="grid">Grid</option>
          <option value="circle">Circle</option>
          <option value="concentric">Concentric</option>
          <option value="breadthfirst">Breadth-first</option>
        </select>
        <button id="btn-fit">Fit</button>
        <button id="btn-reset">Reset</button>
      </div>
      <div id="cy"></div>
      <div id="details"></div>
    </main>
  </div>
  <script>
    const DATA = {data_json};

    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements: {{ nodes: DATA.nodes, edges: DATA.edges }},
      minZoom: 0.1, maxZoom: 5, wheelSensitivity: 0.2,
      style: [
        {{ selector: 'node', style: {{
            'label': 'data(label)',
            'background-color': 'data(color)',
            'border-color': '#333', 'border-width': 1,
            'font-size': '16px',
            'text-valign': 'center', 'text-halign': 'center',
            'text-wrap': 'wrap', 'text-max-width': 120,
            'width': 'label', 'height': 'label',
            'padding': '8px', 'shape': 'round-rectangle',
        }} }},
        {{ selector: 'node[kind="store"]', style: {{
            'shape': 'ellipse', 'background-color': '#ffffff',
            'border-color': '#666', 'border-width': 2, 'border-style': 'dashed',
            'font-weight': 600, 'color': '#333',
        }} }},
        {{ selector: 'edge', style: {{
            'curve-style': 'bezier', 'width': 1.5,
            'line-color': '#bbb', 'target-arrow-shape': 'triangle',
            'target-arrow-color': '#bbb', 'arrow-scale': 0.9, 'opacity': 0.75,
        }} }},
        {{ selector: 'edge[direction="input"]', style: {{
            'line-color': '#7BA7D9', 'target-arrow-color': '#7BA7D9' }} }},
        {{ selector: 'edge[direction="output"]', style: {{
            'line-color': '#D88A7B', 'target-arrow-color': '#D88A7B' }} }},
        {{ selector: '.faded', style: {{ 'opacity': 0.1, 'text-opacity': 0.1 }} }},
        {{ selector: '.highlighted', style: {{ 'border-color': '#000', 'border-width': 3 }} }},
        {{ selector: 'edge.highlighted', style: {{ 'width': 3, 'opacity': 1 }} }},
      ],
    }});

    const layoutOpts = {{
      dagre:        {{ name: 'dagre', rankDir: 'LR', nodeSep: 40, rankSep: 120, edgeSep: 20, animate: true }},
      fcose:        {{ name: 'fcose', quality: 'proof', randomize: false, animate: true, idealEdgeLength: 90, nodeRepulsion: 4500 }},
      grid:         {{ name: 'grid', avoidOverlap: true, padding: 24, animate: true }},
      circle:       {{ name: 'circle', padding: 24, animate: true }},
      concentric:   {{ name: 'concentric', padding: 24, animate: true, concentric: n => (n.data('kind') === 'store' ? 1 : 10) }},
      breadthfirst: {{ name: 'breadthfirst', directed: true, padding: 24, animate: true }},
    }};
    function applyLayout(name) {{ cy.layout(layoutOpts[name] || layoutOpts.dagre).run(); }}
    applyLayout('dagre');

    document.getElementById('layout-select').addEventListener('change', e => applyLayout(e.target.value));
    document.getElementById('btn-fit').onclick = () => cy.fit(null, 24);
    document.getElementById('btn-reset').onclick = () => {{
      cy.elements().removeClass('faded highlighted'); hideDetails();
      applyLayout(document.getElementById('layout-select').value);
    }};

    function focusNode(node) {{
      cy.elements().addClass('faded').removeClass('highlighted');
      const nh = node.closedNeighborhood();
      nh.removeClass('faded').addClass('highlighted');
      showDetails(node);
    }}
    cy.on('tap', 'node', evt => focusNode(evt.target));
    cy.on('tap', evt => {{
      if (evt.target === cy) {{ cy.elements().removeClass('faded highlighted'); hideDetails(); }}
    }});
    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') {{ cy.elements().removeClass('faded highlighted'); hideDetails(); }}
    }});

    const detailsEl = document.getElementById('details');
    function showDetails(node) {{
      const d = node.data();
      const incoming = node.incomers('edge').map(e => ({{ peer: e.source().id(), ports: e.data('ports') }}));
      const outgoing = node.outgoers('edge').map(e => ({{ peer: e.target().id(), ports: e.data('ports') }}));
      let html = `<h3>${{escapeHtml(d.label)}}</h3>`;
      const meta = [];
      if (d.kind) meta.push(d.kind);
      if (d.subsystem) meta.push(d.subsystem);
      if (d.layer !== undefined && d.layer !== -1) meta.push(`layer L${{String(d.layer).padStart(2,'0')}}`);
      if (d.klass) meta.push(`<code>${{escapeHtml(d.klass)}}</code>`);
      html += `<div class="meta">${{meta.join(' · ')}}</div><table>`;
      if (incoming.length) {{
        html += `<tr><td colspan=2 style="color:#7BA7D9;font-weight:600">Inputs (${{incoming.length}})</td></tr>`;
        for (const e of incoming) html += `<tr><td>${{escapeHtml(e.peer)}}</td><td>${{escapeHtml(e.ports.join(', '))}}</td></tr>`;
      }}
      if (outgoing.length) {{
        html += `<tr><td colspan=2 style="color:#D88A7B;font-weight:600">Outputs (${{outgoing.length}})</td></tr>`;
        for (const e of outgoing) html += `<tr><td>${{escapeHtml(e.peer)}}</td><td>${{escapeHtml(e.ports.join(', '))}}</td></tr>`;
      }}
      html += `</table>`;
      detailsEl.innerHTML = html;
      detailsEl.style.display = 'block';
    }}
    function hideDetails() {{ detailsEl.style.display = 'none'; }}
    function escapeHtml(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]); }}

    const legendEl = document.getElementById('legend');
    const hiddenSubs = new Set();
    DATA.legend.forEach(({{key, label, color}}) => {{
      const row = document.createElement('div');
      row.className = 'filter-row';
      row.dataset.sub = key;
      row.innerHTML = `<span class="legend-swatch" style="background:${{color}}"></span><span class="legend-label">${{escapeHtml(label)}}</span>`;
      row.onclick = () => {{
        if (hiddenSubs.has(key)) {{ hiddenSubs.delete(key); row.classList.remove('disabled'); }}
        else {{ hiddenSubs.add(key); row.classList.add('disabled'); }}
        applyFilter();
      }};
      legendEl.appendChild(row);
    }});
    function applyFilter() {{
      const q = (document.getElementById('q').value || '').trim().toLowerCase();
      cy.batch(() => {{
        cy.nodes().forEach(n => {{
          const d = n.data();
          const hideSub = d.kind === 'process' && hiddenSubs.has(d.subsystem);
          const hideSearch = q && !(d.label || '').toLowerCase().includes(q);
          n.style('display', hideSub || hideSearch ? 'none' : 'element');
        }});
      }});
    }}
    document.getElementById('q').addEventListener('input', applyFilter);

    const layersEl = document.getElementById('layers');
    DATA.layers.forEach((layer, idx) => {{
      const row = document.createElement('div');
      row.className = 'layer-row';
      const num = `L${{String(idx).padStart(2,'0')}}`;
      const chips = layer.map(step => {{
        const node = cy.getElementById(step);
        const color = node.nonempty() ? node.data('color') : '#eee';
        return `<span class="chip" data-step="${{escapeHtml(step)}}" style="background:${{color}}">${{escapeHtml(step)}}</span>`;
      }}).join('');
      row.innerHTML = `<div class="layer-num">${{num}}</div><div>${{chips}}</div>`;
      layersEl.appendChild(row);
    }});
    layersEl.addEventListener('click', e => {{
      const chip = e.target.closest('.chip');
      if (!chip) return;
      const node = cy.getElementById(chip.dataset.step);
      if (node.nonempty()) {{ focusNode(node); cy.animate({{ center: {{ eles: node }}, zoom: 1.5 }}, {{ duration: 400 }}); }}
    }});
  </script>
</body>
</html>
"""


def render_html(data: dict, title: str, subtitle: str) -> str:
    """Return the interactive HTML viewer as a string."""
    import html as _html
    return _HTML_TEMPLATE.format(
        title=_html.escape(title),
        subtitle=_html.escape(subtitle),
        data_json=json.dumps(data),
    )


def write_outputs(
    data: dict,
    out_dir: str | Path,
    name: str,
    title: str,
    subtitle: str,
) -> tuple[Path, Path]:
    """Write ``<name>.json`` and ``<name>.html`` side-by-side, return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f'{name}.json'
    json_path.write_text(json.dumps(data, indent=2))

    html_path = out_dir / f'{name}.html'
    html_path.write_text(render_html(data, title, subtitle))

    return json_path, html_path
