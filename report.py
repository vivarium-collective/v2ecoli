"""
Comprehensive v2ecoli comparison report.

Runs v1 and v2 simulations side-by-side, generates plots, bigraph
visualization, and an interactive HTML report with the JSON document.
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


DURATION = 10.0
SIM_DATA_PATH = os.path.join(ROOT_PATH, 'out', 'kb', 'simData.cPickle')
OUT_DIR = 'out/report'


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_v1():
    """Run the v1 vEcoli simulation."""
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
# Plots
# ---------------------------------------------------------------------------

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_bulk_scatter(v1, v2):
    """Scatter plot of bulk molecule count changes: v1 vs v2."""
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
    """Histogram of count changes for v1 and v2."""
    v1_delta = v1['final'].astype(float) - v1['initial'].astype(float)
    v2_delta = v2['final'].astype(float) - v2['initial'].astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    v1_nonzero = v1_delta[v1_delta != 0]
    v2_nonzero = v2_delta[v2_delta != 0]

    if len(v1_nonzero) > 0:
        ax1.hist(v1_nonzero, bins=50, alpha=0.7, color='#dc2626', label='v1')
    if len(v2_nonzero) > 0:
        ax1.hist(v2_nonzero, bins=50, alpha=0.7, color='#2563eb', label='v2')
    ax1.set_xlabel('Count change')
    ax1.set_ylabel('Number of molecules')
    ax1.set_title('Distribution of Count Changes')
    ax1.legend()
    ax1.set_yscale('log')

    # Correlation plot
    both = (v1_delta != 0) & (v2_delta != 0)
    if both.sum() > 0:
        corr = np.corrcoef(v1_delta[both], v2_delta[both])[0, 1]
        ax2.bar(['v1 changed', 'v2 changed', 'Both changed'],
                [(v1_delta != 0).sum(), (v2_delta != 0).sum(), both.sum()],
                color=['#dc2626', '#2563eb', '#7c3aed'])
        ax2.set_title(f'Molecule Counts (Pearson r = {corr:.4f})')
    ax2.set_ylabel('Number of molecules')

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_runtime_comparison(v1, v2):
    """Bar chart of runtimes."""
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
# Bigraph visualization
# ---------------------------------------------------------------------------

def generate_bigraph_svg(v2_state):
    """Generate bigraph visualization as SVG."""
    from bigraph_viz import plot_bigraph

    # Build a simplified state for visualization (remove numpy arrays)
    cell = v2_state['agents']['0']
    viz_state = {}
    for key, value in cell.items():
        if isinstance(value, dict) and 'instance' in value:
            viz_state[key] = {
                '_type': value.get('_type', 'step'),
                'inputs': {k: v for k, v in value.get('inputs', {}).items()
                           if not k.startswith('_flow')},
                'outputs': {k: v for k, v in value.get('outputs', {}).items()
                            if not k.startswith('_flow')},
            }
        elif isinstance(value, np.ndarray):
            viz_state[key] = f"<array {value.shape}>"
        elif isinstance(value, dict):
            viz_state[key] = {k: f"<{type(v).__name__}>" if not isinstance(v, (str, int, float, dict)) else v
                              for k, v in list(value.items())[:10]}
        else:
            viz_state[key] = value

    try:
        graph = plot_bigraph(
            viz_state,
            out_dir=OUT_DIR,
            filename='v2ecoli_bigraph',
            file_format='svg',
            show_compiled_state=False,
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
    """Convert state to JSON-serializable dict, truncating large arrays."""
    if max_depth <= 0:
        return "..."

    if isinstance(state, np.ndarray):
        if state.size <= 10:
            return state.tolist()
        return {
            '_type': 'ndarray',
            'dtype': str(state.dtype),
            'shape': list(state.shape),
            'sample': state.flat[:5].tolist(),
            'min': float(np.min(state)) if np.issubdtype(state.dtype, np.number) else None,
            'max': float(np.max(state)) if np.issubdtype(state.dtype, np.number) else None,
        }
    elif isinstance(state, dict):
        result = {}
        for i, (k, v) in enumerate(state.items()):
            if i >= max_items:
                result['...'] = f'{len(state) - max_items} more keys'
                break
            if k == 'instance':
                result[k] = f"<{type(v).__name__}>"
            else:
                result[k] = state_to_json(v, max_depth - 1, max_items)
        return result
    elif isinstance(state, (list, tuple)):
        if len(state) <= 10:
            return [state_to_json(v, max_depth - 1, max_items) for v in state]
        return {
            '_type': 'list',
            'length': len(state),
            'sample': [state_to_json(v, max_depth - 1, max_items) for v in state[:3]],
        }
    elif isinstance(state, (np.integer,)):
        return int(state)
    elif isinstance(state, (np.floating,)):
        return float(state)
    elif isinstance(state, np.bool_):
        return bool(state)
    elif isinstance(state, (int, float, str, bool, type(None))):
        return state
    elif isinstance(state, set):
        return list(state)[:max_items]
    elif isinstance(state, bytes):
        return f"<bytes len={len(state)}>"
    elif hasattr(state, 'asNumber'):
        return f"{state}"
    else:
        return f"<{type(state).__name__}>"


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

    # Generate plots
    print("Generating plots...")
    scatter_b64 = plot_bulk_scatter(v1, v2)
    hist_b64 = plot_bulk_histogram(v1, v2)
    runtime_b64 = plot_runtime_comparison(v1, v2)

    # Generate bigraph viz
    print("Generating bigraph visualization...")
    bigraph_svg = generate_bigraph_svg(v2['state'])

    # Generate JSON document
    print("Serializing JSON document...")
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                if obj.size <= 10:
                    return obj.tolist()
                return {'_type': 'ndarray', 'dtype': str(obj.dtype),
                        'shape': list(obj.shape)}
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, set): return list(obj)
            if isinstance(obj, bytes): return f"<bytes len={len(obj)}>"
            return f"<{type(obj).__name__}>"

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
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 2em; margin: 20px 0; color: #0f172a; }}
  h2 {{ font-size: 1.4em; margin: 30px 0 15px; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 15px; margin: 20px 0; }}
  .stat {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat .label {{ font-size: 0.85em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 1.8em; font-weight: 700; margin-top: 5px; }}
  .stat .value.green {{ color: #16a34a; }}
  .stat .value.blue {{ color: #2563eb; }}
  .stat .value.red {{ color: #dc2626; }}
  .stat .value.purple {{ color: #7c3aed; }}
  .plot {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .plot img {{ max-width: 100%; height: auto; }}
  .bigraph {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow-x: auto; }}
  .bigraph svg {{ max-width: 100%; height: auto; }}
  .json-viewer {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .json-tree {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.5; }}
  .json-tree details {{ margin-left: 20px; }}
  .json-tree summary {{ cursor: pointer; color: #475569; }}
  .json-tree summary:hover {{ color: #0f172a; }}
  .json-tree .key {{ color: #7c3aed; }}
  .json-tree .string {{ color: #16a34a; }}
  .json-tree .number {{ color: #2563eb; }}
  .json-tree .null {{ color: #94a3b8; }}
  .json-tree .type-tag {{ color: #94a3b8; font-style: italic; font-size: 0.85em; }}
  footer {{ margin-top: 40px; padding: 20px 0; border-top: 1px solid #e2e8f0;
            color: #94a3b8; font-size: 0.85em; text-align: center; }}
</style>
</head>
<body>

<h1>v2ecoli Comparison Report</h1>
<p style="color: #64748b; margin-bottom: 20px;">{DURATION}s simulated &middot; Generated {time.strftime('%Y-%m-%d %H:%M')}</p>

<h2>Summary</h2>
<div class="summary">
  <div class="stat">
    <div class="label">Correlation</div>
    <div class="value green">{corr:.4f}</div>
  </div>
  <div class="stat">
    <div class="label">v1 Runtime</div>
    <div class="value red">{v1['runtime']:.2f}s</div>
  </div>
  <div class="stat">
    <div class="label">v2 Runtime</div>
    <div class="value blue">{v2['runtime']:.2f}s</div>
  </div>
  <div class="stat">
    <div class="label">v1 Molecules Changed</div>
    <div class="value red">{v1_changed}</div>
  </div>
  <div class="stat">
    <div class="label">v2 Molecules Changed</div>
    <div class="value blue">{v2_changed}</div>
  </div>
  <div class="stat">
    <div class="label">Both Changed</div>
    <div class="value purple">{both.sum()}</div>
  </div>
  <div class="stat">
    <div class="label">Total Bulk Molecules</div>
    <div class="value">{len(v1['initial'])}</div>
  </div>
  <div class="stat">
    <div class="label">Speedup</div>
    <div class="value blue">{v1['runtime']/v2['runtime']:.1f}x</div>
  </div>
</div>

<h2>Runtime Comparison</h2>
<div class="plot">
  <img src="data:image/png;base64,{runtime_b64}" alt="Runtime comparison">
</div>

<h2>Bulk Molecule Count Changes</h2>
<div class="plot">
  <img src="data:image/png;base64,{scatter_b64}" alt="Scatter plot">
</div>
<div class="plot">
  <img src="data:image/png;base64,{hist_b64}" alt="Histogram">
</div>

<h2>v2ecoli Bigraph Visualization</h2>
<div class="bigraph">
  {bigraph_svg}
</div>

<h2>v2ecoli Document (JSON)</h2>
<div class="json-viewer">
  <div class="json-tree" id="json-root"></div>
</div>

<footer>
  Generated by v2ecoli report.py &middot;
  <a href="https://github.com/vivarium-collective/v2ecoli">github.com/vivarium-collective/v2ecoli</a>
</footer>

<script>
const docData = {doc_json};

function renderJson(data, container, depth) {{
  if (depth > 8) {{ container.textContent = '...'; return; }}

  if (data === null) {{
    container.innerHTML = '<span class="null">null</span>';
  }} else if (typeof data === 'string') {{
    if (data.startsWith('<') && data.endsWith('>')) {{
      container.innerHTML = '<span class="type-tag">' + escHtml(data) + '</span>';
    }} else {{
      container.innerHTML = '<span class="string">"' + escHtml(truncate(data, 80)) + '"</span>';
    }}
  }} else if (typeof data === 'number') {{
    container.innerHTML = '<span class="number">' + data + '</span>';
  }} else if (typeof data === 'boolean') {{
    container.innerHTML = '<span class="number">' + data + '</span>';
  }} else if (Array.isArray(data)) {{
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = '[' + data.length + ' items]';
    details.appendChild(summary);
    data.forEach((item, i) => {{
      const row = document.createElement('div');
      row.style.marginLeft = '20px';
      const idx = document.createElement('span');
      idx.className = 'key';
      idx.textContent = i + ': ';
      row.appendChild(idx);
      const val = document.createElement('span');
      renderJson(item, val, depth + 1);
      row.appendChild(val);
      details.appendChild(row);
    }});
    container.appendChild(details);
  }} else if (typeof data === 'object') {{
    const keys = Object.keys(data);
    const details = document.createElement('details');
    if (depth < 2) details.open = true;
    const summary = document.createElement('summary');
    summary.textContent = '{{' + keys.length + ' keys}}';
    details.appendChild(summary);
    keys.forEach(k => {{
      const row = document.createElement('div');
      row.style.marginLeft = '20px';
      const key = document.createElement('span');
      key.className = 'key';
      key.textContent = k + ': ';
      row.appendChild(key);
      const val = document.createElement('span');
      renderJson(data[k], val, depth + 1);
      row.appendChild(val);
      details.appendChild(row);
    }});
    container.appendChild(details);
  }} else {{
    container.textContent = String(data);
  }}
}}

function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function truncate(s, n) {{ return s.length > n ? s.slice(0, n) + '...' : s; }}

renderJson(docData, document.getElementById('json-root'), 0);
</script>
</body>
</html>""")

    print(f"\nReport saved to {report_path}")
    return report_path


if __name__ == '__main__':
    generate_report()
