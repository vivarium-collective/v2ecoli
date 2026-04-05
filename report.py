"""
Comprehensive v2ecoli comparison report.

Runs v1 and v2 simulations side-by-side, generates plots including
mass fraction summaries, an improved bigraph visualization, and an
interactive HTML report with the JSON document.
"""

import os
import io
import json
import time
import base64
import html as html_lib

import dill
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contextlib import chdir
from wholecell.utils.filepath import ROOT_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.library.schema import not_a_process

from v2ecoli.generate import generate_document
from v2ecoli.composite import load_simulation, EcoliSimulation

from bigraph_viz import plot_bigraph


DURATION = 10.0
SIM_DATA_PATH = os.path.join(ROOT_PATH, 'out', 'kb', 'simData.cPickle')
OUT_DIR = 'out/report'


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_v1():
    """Run the v1 vEcoli simulation with timeseries emitter."""
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file()
        sim.max_duration = int(DURATION)
        sim.emitter = 'timeseries'
        sim.divide = False
        sim.build_ecoli()
        initial = sim.generated_initial_state['bulk']['count'].copy()
        t0 = time.time()
        sim.run()
        runtime = time.time() - t0

    state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
    bulk = state['bulk']['count'].copy()
    timeseries = sim.query()

    return {
        'initial': initial,
        'final': bulk,
        'runtime': runtime,
        'timeseries': timeseries,
    }


def run_v2():
    """Run the v2ecoli simulation."""
    doc_path = os.path.join(OUT_DIR, 'v2_ecoli.pickle')
    generate_document(doc_path, sim_data_path=SIM_DATA_PATH)

    ecoli = load_simulation(doc_path)
    initial = np.array(ecoli.state['agents']['0']['bulk']['count'], copy=True)
    t0 = time.time()
    ecoli.run(DURATION)
    runtime = time.time() - t0
    bulk = ecoli.state['agents']['0']['bulk']['count'].copy()

    return {
        'initial': initial,
        'final': bulk,
        'runtime': runtime,
        'state': ecoli.state,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Mass fraction plots
# ---------------------------------------------------------------------------

MASS_COMPONENTS = {
    'Protein': 'protein_mass',
    'tRNA': 'tRna_mass',
    'rRNA': 'rRna_mass',
    'mRNA': 'mRna_mass',
    'DNA': 'dna_mass',
    'Small Mol': 'smallMolecule_mass',
    'Dry Mass': 'dry_mass',
}

MASS_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#ffff33', '#a65628',
]


def v1_query_to_mass_timeseries(timeseries):
    """Convert v1 sim.query() output to mass timeseries."""
    times = sorted(t for t in timeseries.keys() if isinstance(t, (int, float)) and t > 0)
    result = {'time': np.array(times)}
    for label, key in MASS_COMPONENTS.items():
        values = []
        for t in times:
            snapshot = timeseries[t]
            mass = snapshot.get('listeners', {}).get('mass', {})
            values.append(mass.get(key, 0.0))
        result[label] = np.array(values)
    return result


def v2_state_to_mass_timeseries(state):
    """Extract mass data from v2 cell state after simulation."""
    cell = state['agents']['0']
    listeners = cell.get('listeners', {})
    mass = listeners.get('mass', {})

    # v2 doesn't have a timeseries emitter yet — just use final values
    result = {'time': np.array([0.0, state.get('global_time', DURATION)])}
    for label, key in MASS_COMPONENTS.items():
        val = mass.get(key, 0.0)
        result[label] = np.array([val, val])  # flat line (single snapshot)
    return result


def plot_mass_fractions(v1_ts, v2_ts):
    """Plot mass fraction comparison: v1 timeseries vs v2 final state."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, (label, ts) in enumerate([('vEcoli (v1)', v1_ts), ('v2ecoli (v2)', v2_ts)]):
        ax = axes[col]
        t = ts['time']
        if len(t) < 2:
            ax.text(0.5, 0.5, 'No timeseries data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='#94a3b8')
            ax.set_title(label)
            continue

        time_min = (t - t[0]) / 60.0
        for i, (comp_label, key) in enumerate(MASS_COMPONENTS.items()):
            values = ts.get(comp_label)
            if values is None or len(values) == 0 or values[0] == 0:
                continue
            fold_change = values / values[0]
            dry = ts.get('Dry Mass')
            fraction = np.mean(values / dry) if dry is not None and np.all(dry > 0) else 0
            ax.plot(time_min, fold_change,
                    color=MASS_COLORS[i % len(MASS_COLORS)],
                    linewidth=2,
                    label=f'{comp_label} ({fraction:.3f})')

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Mass (normalized to t=0)')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    fig.suptitle('Biomass Components (avg fraction of dry mass)', fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Bulk molecule plots
# ---------------------------------------------------------------------------

def plot_bulk_scatter(v1, v2):
    v1_delta = v1['final'].astype(float) - v1['initial'].astype(float)
    v2_delta = v2['final'].astype(float) - v2['initial'].astype(float)
    both_changed = (v1_delta != 0) & (v2_delta != 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(v1_delta[both_changed], v2_delta[both_changed],
               alpha=0.5, s=10, c='#2563eb')
    lim = max(abs(v1_delta[both_changed]).max(), abs(v2_delta[both_changed]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='y=x')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('v1 (vEcoli) count change')
    ax.set_ylabel('v2 (v2ecoli) count change')
    ax.set_title(f'Bulk Molecule Count Changes ({both_changed.sum()} molecules)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.2)
    return fig_to_base64(fig)


def plot_bulk_histogram(v1, v2):
    v1_delta = v1['final'].astype(float) - v1['initial'].astype(float)
    v2_delta = v2['final'].astype(float) - v2['initial'].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    v1_nz = v1_delta[v1_delta != 0]
    v2_nz = v2_delta[v2_delta != 0]
    if len(v1_nz) > 0:
        ax1.hist(v1_nz, bins=50, alpha=0.7, color='#dc2626', label='v1')
    if len(v2_nz) > 0:
        ax1.hist(v2_nz, bins=50, alpha=0.7, color='#2563eb', label='v2')
    ax1.set_xlabel('Count change')
    ax1.set_ylabel('Number of molecules')
    ax1.set_title('Distribution of Count Changes')
    ax1.legend()
    ax1.set_yscale('log')

    both = (v1_delta != 0) & (v2_delta != 0)
    corr = np.corrcoef(v1_delta[both], v2_delta[both])[0, 1] if both.sum() > 0 else 0.0
    ax2.bar(['v1 changed', 'v2 changed', 'Both changed'],
            [(v1_delta != 0).sum(), (v2_delta != 0).sum(), both.sum()],
            color=['#dc2626', '#2563eb', '#7c3aed'])
    ax2.set_title(f'Molecule Counts (Pearson r = {corr:.4f})')
    ax2.set_ylabel('Number of molecules')
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_runtime_comparison(v1, v2):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['vEcoli (v1)', 'v2ecoli (v2)'],
                  [v1['runtime'], v2['runtime']],
                  color=['#dc2626', '#2563eb'])
    for bar, val in zip(bars, [v1['runtime'], v2['runtime']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=12)
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title(f'{DURATION}s Simulated Time')
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Bigraph visualization (improved from genEcoli)
# ---------------------------------------------------------------------------

ALWAYS_SKIP = {'unique_update', 'global_clock', 'bulk-timeline',
               'mark_d_period', 'division', 'exchange_data', 'media_update',
               'post-division-mass-listener'}
SKIP_PORTS = {'timestep', 'global_time', 'next_update_time', 'process'}

COLOR_GROUPS = {
    'dna':      ('#FFB6C1', lambda n: 'chromosome' in n),
    'rna':      ('#ADD8E6', lambda n: any(s in n for s in ('transcript', 'rna-', 'RNA', 'rnap'))),
    'protein':  ('#90EE90', lambda n: any(s in n for s in ('polypeptide', 'protein', 'ribosome'))),
    'meta':     ('#FFD700', lambda n: any(s in n for s in ('metabolism', 'equilibrium', 'complexation', 'two-component'))),
    'reg':      ('#DDA0DD', lambda n: any(s in n for s in ('tf-', 'tf_'))),
    'listen':   ('#D3D3D3', lambda n: 'listener' in n),
    'allocate': ('#FFA07A', lambda n: 'allocator' in n),
}


def _build_viz_cell(cell):
    """Build a filtered visualization dict from cell state."""
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict):
            continue

        if '_type' in edge:
            if any(s in name for s in ALWAYS_SKIP):
                continue
            if '_requester' in name:
                continue

            inputs = {}
            for port, wire in edge.get('inputs', {}).items():
                if port.startswith('_flow') or port in SKIP_PORTS:
                    continue
                if isinstance(wire, list) and wire and wire[0] in ('request', 'allocate'):
                    continue
                if isinstance(wire, list):
                    inputs[port] = wire
                elif isinstance(wire, dict) and '_path' in wire:
                    inputs[port] = wire['_path']

            clean = name.replace('ecoli-', '').replace('_evolver', '')
            viz[clean] = {'_type': edge['_type'], 'inputs': inputs}

        elif name == 'unique' and isinstance(edge, dict):
            viz[name] = {k: {} for k in edge.keys()}
        elif name in ('bulk', 'listeners', 'environment', 'boundary'):
            viz[name] = {}

    return viz


def generate_bigraph_svg(v2_state):
    """Generate clean bigraph SVG with color-coded processes."""
    cell = v2_state['agents']['0']
    viz_cell = _build_viz_cell(cell)
    viz_state = {'agents': {'0': viz_cell}}
    prefix = ('agents', '0')

    colors = {}
    groups_dict = {k: [] for k in COLOR_GROUPS}
    for name in viz_cell:
        if '_type' not in viz_cell.get(name, {}):
            continue
        path = prefix + (name,)
        for gk, (color, matcher) in COLOR_GROUPS.items():
            if matcher(name):
                colors[path] = color
                groups_dict[gk].append(path)
                break

    groups = [g for g in groups_dict.values() if g]

    try:
        plot_bigraph(
            viz_state,
            remove_process_place_edges=True,
            node_groups=groups,
            node_fill_colors=colors,
            size='20,16',
            rankdir='TB',
            dpi='150',
            port_labels=False,
            node_label_size='24pt',
            label_margin='0.08',
            out_dir=OUT_DIR,
            filename='v2ecoli_bigraph',
            file_format='svg',
        )
        svg_path = os.path.join(OUT_DIR, 'v2ecoli_bigraph.svg')
        if os.path.exists(svg_path):
            with open(svg_path) as f:
                return f.read()
    except Exception as e:
        return f'<p>Bigraph visualization failed: {html_lib.escape(str(e))}</p>'
    return '<p>No SVG generated</p>'


# ---------------------------------------------------------------------------
# JSON document serializer
# ---------------------------------------------------------------------------

def state_to_json(state, max_depth=6, max_items=50):
    if max_depth <= 0:
        return "..."
    if isinstance(state, np.ndarray):
        if state.size <= 10:
            return state.tolist()
        info = {'_type': 'ndarray', 'dtype': str(state.dtype), 'shape': list(state.shape)}
        if np.issubdtype(state.dtype, np.number):
            info['min'] = float(np.min(state))
            info['max'] = float(np.max(state))
        return info
    elif isinstance(state, dict):
        result = {}
        for i, (k, v) in enumerate(state.items()):
            if i >= max_items:
                result['...'] = f'{len(state) - max_items} more keys'
                break
            result[k] = f"<{type(v).__name__}>" if k == 'instance' else state_to_json(v, max_depth - 1, max_items)
        return result
    elif isinstance(state, (list, tuple)):
        if len(state) <= 10:
            return [state_to_json(v, max_depth - 1, max_items) for v in state]
        return {'_type': 'list', 'length': len(state),
                'sample': [state_to_json(v, max_depth - 1, max_items) for v in state[:3]]}
    elif isinstance(state, (np.integer,)): return int(state)
    elif isinstance(state, (np.floating,)): return float(state)
    elif isinstance(state, np.bool_): return bool(state)
    elif isinstance(state, (int, float, str, bool, type(None))): return state
    elif isinstance(state, set): return sorted(list(state))[:max_items]
    elif isinstance(state, bytes): return f"<bytes len={len(state)}>"
    elif hasattr(state, 'asNumber'): return f"{state}"
    else: return f"<{type(state).__name__}>"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'_type': 'ndarray', 'dtype': str(obj.dtype), 'shape': list(obj.shape)} if obj.size > 10 else obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, set): return sorted(list(obj))
        if isinstance(obj, bytes): return f"<bytes len={len(obj)}>"
        return f"<{type(obj).__name__}>"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_report():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Running v1 simulation...")
    v1 = run_v1()
    print(f"  v1 done: {v1['runtime']:.2f}s, {(v1['initial'] != v1['final']).sum()} changed")

    print("Running v2 simulation...")
    v2 = run_v2()
    print(f"  v2 done: {v2['runtime']:.2f}s, {(v2['initial'] != v2['final']).sum()} changed")

    # Statistics
    v1_changed = (v1['initial'] != v1['final']).sum()
    v2_changed = (v2['initial'] != v2['final']).sum()
    v1_delta = v1['final'].astype(float) - v1['initial'].astype(float)
    v2_delta = v2['final'].astype(float) - v2['initial'].astype(float)
    both = (v1_delta != 0) & (v2_delta != 0)
    corr = np.corrcoef(v1_delta[both], v2_delta[both])[0, 1] if both.sum() > 0 else 0.0

    # Mass fraction timeseries
    print("Generating mass fraction plots...")
    v1_mass = v1_query_to_mass_timeseries(v1['timeseries'])
    v2_mass = v2_state_to_mass_timeseries(v2['state'])
    mass_b64 = plot_mass_fractions(v1_mass, v2_mass)

    # Other plots
    print("Generating comparison plots...")
    scatter_b64 = plot_bulk_scatter(v1, v2)
    hist_b64 = plot_bulk_histogram(v1, v2)
    runtime_b64 = plot_runtime_comparison(v1, v2)

    # Bigraph viz
    print("Generating bigraph visualization...")
    bigraph_svg = generate_bigraph_svg(v2['state'])

    # JSON
    print("Serializing JSON document...")
    doc_json = json.dumps(state_to_json(v2['state']), indent=2, cls=NumpyEncoder)

    # Build HTML
    report_path = os.path.join(OUT_DIR, 'comparison_report.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli Comparison Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 2em; margin: 20px 0; color: #0f172a; }}
  h2 {{ font-size: 1.4em; margin: 30px 0 15px; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: white; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat .label {{ font-size: 0.8em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 1.6em; font-weight: 700; margin-top: 4px; }}
  .stat .value.green {{ color: #16a34a; }}
  .stat .value.blue {{ color: #2563eb; }}
  .stat .value.red {{ color: #dc2626; }}
  .stat .value.purple {{ color: #7c3aed; }}
  .plot {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .plot img {{ max-width: 100%; height: auto; }}
  .bigraph {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: auto; max-height: 900px; }}
  .bigraph svg {{ max-width: 100%; height: auto; }}
  .json-viewer {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-height: 700px; overflow: auto; }}
  .json-tree {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.5; }}
  .json-tree details {{ margin-left: 18px; }}
  .json-tree summary {{ cursor: pointer; color: #475569; padding: 1px 0; }}
  .json-tree summary:hover {{ color: #0f172a; background: #f1f5f9; }}
  .json-tree .key {{ color: #7c3aed; font-weight: 500; }}
  .json-tree .string {{ color: #16a34a; }}
  .json-tree .number {{ color: #2563eb; }}
  .json-tree .null {{ color: #94a3b8; }}
  .json-tree .type-tag {{ color: #94a3b8; font-style: italic; font-size: 0.85em; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; padding: 10px;
             background: #f8fafc; border-radius: 8px; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid #ccc; }}
  footer {{ margin-top: 40px; padding: 20px 0; border-top: 1px solid #e2e8f0;
            color: #94a3b8; font-size: 0.85em; text-align: center; }}
</style>
</head>
<body>

<h1>v2ecoli Comparison Report</h1>
<p style="color: #64748b; margin-bottom: 20px;">{DURATION}s simulated &middot; {time.strftime('%Y-%m-%d %H:%M')}</p>

<h2>Summary</h2>
<div class="summary">
  <div class="stat"><div class="label">Correlation</div><div class="value green">{corr:.4f}</div></div>
  <div class="stat"><div class="label">v1 Runtime</div><div class="value red">{v1['runtime']:.2f}s</div></div>
  <div class="stat"><div class="label">v2 Runtime</div><div class="value blue">{v2['runtime']:.2f}s</div></div>
  <div class="stat"><div class="label">v1 Changed</div><div class="value red">{v1_changed}</div></div>
  <div class="stat"><div class="label">v2 Changed</div><div class="value blue">{v2_changed}</div></div>
  <div class="stat"><div class="label">Both Changed</div><div class="value purple">{both.sum()}</div></div>
  <div class="stat"><div class="label">Total Molecules</div><div class="value">{len(v1['initial'])}</div></div>
  <div class="stat"><div class="label">Speedup</div><div class="value blue">{v1['runtime']/v2['runtime']:.1f}x</div></div>
</div>

<h2>Mass Fraction Summary</h2>
<div class="plot">
  <img src="data:image/png;base64,{mass_b64}" alt="Mass fractions">
</div>

<h2>Runtime</h2>
<div class="plot">
  <img src="data:image/png;base64,{runtime_b64}" alt="Runtime">
</div>

<h2>Bulk Molecule Count Changes</h2>
<div class="plot">
  <img src="data:image/png;base64,{scatter_b64}" alt="Scatter">
</div>
<div class="plot">
  <img src="data:image/png;base64,{hist_b64}" alt="Histogram">
</div>

<h2>v2ecoli Bigraph</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#FFB6C1"></div> DNA</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ADD8E6"></div> RNA</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#90EE90"></div> Protein</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#FFD700"></div> Metabolism</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#DDA0DD"></div> Regulation</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#D3D3D3"></div> Listeners</div>
</div>
<div class="bigraph">
  {bigraph_svg}
</div>

<h2>v2ecoli Document (JSON)</h2>
<div class="json-viewer">
  <div class="json-tree" id="json-root"></div>
</div>

<footer>
  v2ecoli &middot;
  <a href="https://github.com/vivarium-collective/v2ecoli">github.com/vivarium-collective/v2ecoli</a>
</footer>

<script>
const docData = {doc_json};

function renderJson(data, container, depth) {{
  if (depth > 8) {{ container.textContent = '...'; return; }}
  if (data === null) {{
    container.innerHTML = '<span class="null">null</span>';
  }} else if (typeof data === 'string') {{
    if (data.startsWith('<') && data.endsWith('>'))
      container.innerHTML = '<span class="type-tag">' + esc(data) + '</span>';
    else
      container.innerHTML = '<span class="string">"' + esc(trunc(data, 80)) + '"</span>';
  }} else if (typeof data === 'number') {{
    container.innerHTML = '<span class="number">' + data + '</span>';
  }} else if (typeof data === 'boolean') {{
    container.innerHTML = '<span class="number">' + data + '</span>';
  }} else if (Array.isArray(data)) {{
    const d = document.createElement('details');
    if (depth < 1) d.open = true;
    const s = document.createElement('summary');
    s.textContent = '[' + data.length + ' items]';
    d.appendChild(s);
    data.forEach((item, i) => {{
      const r = document.createElement('div');
      r.style.marginLeft = '18px';
      r.innerHTML = '<span class="key">' + i + '</span>: ';
      const v = document.createElement('span');
      renderJson(item, v, depth + 1);
      r.appendChild(v);
      d.appendChild(r);
    }});
    container.appendChild(d);
  }} else if (typeof data === 'object') {{
    const keys = Object.keys(data);
    const d = document.createElement('details');
    if (depth < 2) d.open = true;
    const s = document.createElement('summary');
    s.textContent = '{{' + keys.length + ' keys}}';
    d.appendChild(s);
    keys.forEach(k => {{
      const r = document.createElement('div');
      r.style.marginLeft = '18px';
      r.innerHTML = '<span class="key">' + esc(k) + '</span>: ';
      const v = document.createElement('span');
      renderJson(data[k], v, depth + 1);
      r.appendChild(v);
      d.appendChild(r);
    }});
    container.appendChild(d);
  }} else {{
    container.textContent = String(data);
  }}
}}

function esc(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function trunc(s, n) {{ return s.length > n ? s.slice(0, n) + '...' : s; }}

renderJson(docData, document.getElementById('json-root'), 0);
</script>
</body>
</html>""")

    print(f"\nReport saved to {report_path}")
    return report_path


if __name__ == '__main__':
    generate_report()
