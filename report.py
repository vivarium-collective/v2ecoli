"""
Comprehensive v2ecoli workflow report.

Examines the full pipeline: ParCa → sim_data → ecoli_wcm → simulation.
Includes data inspection, bigraph visualization, long simulation with
mass fractions, and a brief v1 comparison.
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

from v2ecoli.generate import build_document, DEFAULT_FLOW
from v2ecoli.composite import make_composite, _build_core, save_cache
from v2ecoli.cache import NumpyJSONEncoder, save_initial_state, load_initial_state

from bigraph_viz import plot_bigraph


SIM_DATA_PATH = os.path.join(ROOT_PATH, 'out', 'kb', 'simData.cPickle')
OUT_DIR = 'out/report'
CACHE_DIR = 'out/cache'
V2_DURATION = 60.0    # Main simulation
V1_DURATION = 60.0    # v1 comparison (same duration)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def state_to_json_str(state, max_depth=5, max_items=30):
    """Convert state to truncated JSON string for display."""
    def truncate(obj, depth):
        if depth <= 0: return "..."
        if isinstance(obj, np.ndarray):
            if obj.size <= 5: return obj.tolist()
            return {"_type": "ndarray", "dtype": str(obj.dtype), "shape": list(obj.shape)}
        if isinstance(obj, dict):
            result = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_items:
                    result['...'] = f'{len(obj) - max_items} more'
                    break
                result[str(k)] = truncate(v, depth - 1) if k != 'instance' else f"<{type(v).__name__}>"
            return result
        if isinstance(obj, (list, tuple)):
            if len(obj) <= 5: return [truncate(v, depth - 1) for v in obj]
            return {"_type": "list", "length": len(obj)}
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (int, float, str, bool, type(None))): return obj
        return f"<{type(obj).__name__}>"
    return json.dumps(truncate(state, max_depth), indent=2, cls=NumpyJSONEncoder)


# ---------------------------------------------------------------------------
# Mass fraction plots
# ---------------------------------------------------------------------------

MASS_COMPONENTS = {
    'Protein': 'protein_mass', 'tRNA': 'tRna_mass', 'rRNA': 'rRna_mass',
    'mRNA': 'mRna_mass', 'DNA': 'dna_mass', 'Small Mol': 'smallMolecule_mass',
    'Dry Mass': 'dry_mass',
}
MASS_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']


def plot_mass_timeseries(history, title='Mass Components'):
    """Plot mass component fold changes and absolute mass over time."""
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
        return fig_to_base64(fig)

    times = np.array([s.get('global_time', 0) for s in history])
    time_min = times / 60.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Fold change plot
    ax1 = axes[0]
    for i, (label, key) in enumerate(MASS_COMPONENTS.items()):
        values = np.array([s.get('listeners', {}).get('mass', {}).get(key, 0) for s in history])
        if len(values) > 0 and values[0] > 0:
            ax1.plot(time_min, values / values[0], color=MASS_COLORS[i], linewidth=1.5, label=label)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Fold change')
    ax1.set_title('Fold Change (normalized to t=0)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.15)
    # Auto-zoom: set y-axis to show actual range
    ax1.set_ylim(bottom=0.99)

    # Absolute mass — all components stacked
    ax2 = axes[1]
    for i, (label, key) in enumerate(MASS_COMPONENTS.items()):
        values = np.array([s.get('listeners', {}).get('mass', {}).get(key, 0) for s in history])
        if len(values) > 0 and values[0] > 0:
            ax2.plot(time_min, values, color=MASS_COLORS[i], linewidth=1.5, label=f'{label}')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Mass (fg)')
    ax2.set_title('Absolute Mass')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.15)

    # Percent change from initial
    ax3 = axes[2]
    for i, (label, key) in enumerate(MASS_COMPONENTS.items()):
        values = np.array([s.get('listeners', {}).get('mass', {}).get(key, 0) for s in history])
        if len(values) > 0 and values[0] > 0:
            pct = (values - values[0]) / values[0] * 100
            ax3.plot(time_min, pct, color=MASS_COLORS[i], linewidth=1.5, label=label)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Change from initial (%)')
    ax3.set_title('Percent Change')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.15)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_comparison_mass(v1_history, v2_history):
    """Side-by-side mass comparison between v1 and v2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, (label, history) in enumerate([('vEcoli (v1)', v1_history), ('v2ecoli (v2)', v2_history)]):
        ax = axes[col]
        if not history or len(history) < 2:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(label)
            continue

        times = np.array([s.get('global_time', 0) for s in history])
        time_min = times / 60.0

        for i, (comp_label, key) in enumerate(MASS_COMPONENTS.items()):
            values = np.array([s.get('listeners', {}).get('mass', {}).get(key, 0) for s in history])
            if len(values) > 0 and values[0] > 0:
                frac = np.mean(values / np.maximum(
                    np.array([s.get('listeners', {}).get('mass', {}).get('dry_mass', 1) for s in history]), 1))
                ax.plot(time_min, values / values[0], color=MASS_COLORS[i], linewidth=1.5,
                        label=f'{comp_label} ({frac:.3f})')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Fold change')
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(bottom=0.99)

    fig.suptitle('Mass Fraction Comparison', fontsize=13)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_growth_metrics(history):
    """Plot growth rate, volume, and mass fractions over time."""
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig_to_base64(fig)

    times = np.array([s.get('global_time', 0) for s in history])
    time_min = times / 60.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Growth rate
    growth = np.array([s.get('listeners', {}).get('mass', {}).get('instantaneous_growth_rate', 0) for s in history])
    axes[0].plot(time_min, growth * 3600, color='#2563eb', linewidth=1)
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Growth rate (1/h)')
    axes[0].set_title('Instantaneous Growth Rate')
    axes[0].grid(True, alpha=0.15)

    # Volume
    volume = np.array([s.get('listeners', {}).get('mass', {}).get('volume', 0) for s in history])
    axes[1].plot(time_min, volume, color='#16a34a', linewidth=1)
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Volume (fL)')
    axes[1].set_title('Cell Volume')
    axes[1].grid(True, alpha=0.15)

    # Mass fractions
    protein_frac = np.array([s.get('listeners', {}).get('mass', {}).get('protein_mass_fraction', 0) for s in history])
    rna_frac = np.array([s.get('listeners', {}).get('mass', {}).get('rna_mass_fraction', 0) for s in history])
    axes[2].plot(time_min, protein_frac, color='#e41a1c', linewidth=1, label='Protein')
    axes[2].plot(time_min, rna_frac, color='#377eb8', linewidth=1, label='RNA')
    axes[2].set_xlabel('Time (min)')
    axes[2].set_ylabel('Mass fraction')
    axes[2].set_title('Mass Fractions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.15)

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Bigraph visualization
# ---------------------------------------------------------------------------

ALWAYS_SKIP = {'unique_update', 'global_clock', 'bulk-timeline',
               'mark_d_period', 'division', 'exchange_data', 'media_update',
               'post-division-mass-listener'}
SKIP_PORTS = {'timestep', 'global_time', 'next_update_time', 'process'}
COLOR_GROUPS = {
    'dna': ('#FFB6C1', lambda n: 'chromosome' in n),
    'rna': ('#ADD8E6', lambda n: any(s in n for s in ('transcript', 'rna-', 'RNA', 'rnap'))),
    'protein': ('#90EE90', lambda n: any(s in n for s in ('polypeptide', 'protein', 'ribosome'))),
    'meta': ('#FFD700', lambda n: any(s in n for s in ('metabolism', 'equilibrium', 'complexation', 'two-component'))),
    'reg': ('#DDA0DD', lambda n: any(s in n for s in ('tf-', 'tf_'))),
    'listen': ('#D3D3D3', lambda n: 'listener' in n),
}


def _build_viz_cell(cell, show_partitioning=False):
    """Build filtered visualization dict from cell state."""
    skip = set(ALWAYS_SKIP)
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict): continue
        if '_type' in edge:
            if any(s in name for s in skip): continue
            if not show_partitioning and '_requester' in name: continue

            inputs = {}
            for port, wire in edge.get('inputs', {}).items():
                if port.startswith('_flow') or port in SKIP_PORTS: continue
                if not show_partitioning and isinstance(wire, list) and wire and wire[0] in ('request', 'allocate'): continue
                if isinstance(wire, list): inputs[port] = wire
                elif isinstance(wire, dict) and '_path' in wire: inputs[port] = wire['_path']

            if show_partitioning:
                clean = name.replace('ecoli-', '')
            else:
                clean = name.replace('ecoli-', '').replace('_evolver', '')
            viz[clean] = {'_type': edge['_type'], 'inputs': inputs}

        elif name == 'unique' and isinstance(edge, dict):
            viz[name] = {k: {} for k in edge.keys()}
        elif name in ('bulk', 'listeners', 'environment', 'boundary'):
            viz[name] = {}
        elif show_partitioning and name in ('request', 'allocate'):
            if isinstance(edge, dict):
                viz[name] = {k: {} for k in list(edge.keys())[:5]}
            else:
                viz[name] = {}
    return viz


def generate_bigraph_svg(state, filename='bigraph', show_partitioning=False):
    cell = state.get('agents', {}).get('0', state)
    viz_cell = _build_viz_cell(cell, show_partitioning=show_partitioning)
    viz_state = {'agents': {'0': viz_cell}}
    prefix = ('agents', '0')

    colors, groups = {}, {k: [] for k in COLOR_GROUPS}
    allocator_color = '#FFA07A'
    for name in viz_cell:
        if '_type' not in viz_cell.get(name, {}): continue
        path = prefix + (name,)
        matched = False
        if show_partitioning and 'allocator' in name:
            colors[path] = allocator_color
            groups.setdefault('allocate', []).append(path)
            matched = True
        if not matched:
            for gk, (color, matcher) in COLOR_GROUPS.items():
                if matcher(name):
                    colors[path] = color
                    groups[gk].append(path)
                    break
    groups_list = [g for g in groups.values() if g]

    try:
        plot_bigraph(viz_state, remove_process_place_edges=True,
                     node_groups=groups_list, node_fill_colors=colors,
                     size='22,18' if show_partitioning else '20,16',
                     rankdir='TB', dpi='150',
                     port_labels=show_partitioning,
                     node_label_size='20pt' if show_partitioning else '24pt',
                     label_margin='0.08', out_dir=OUT_DIR,
                     filename=filename, file_format='svg')
        svg_path = os.path.join(OUT_DIR, f'{filename}.svg')
        if os.path.exists(svg_path):
            with open(svg_path) as f:
                return f.read()
    except Exception as e:
        return f'<p>Visualization failed: {html_lib.escape(str(e))}</p>'
    return '<p>No SVG</p>'


# ---------------------------------------------------------------------------
# Scatter comparison
# ---------------------------------------------------------------------------

def plot_bulk_scatter(v1_initial, v1_final, v2_initial, v2_final):
    v1_d = v1_final.astype(float) - v1_initial.astype(float)
    v2_d = v2_final.astype(float) - v2_initial.astype(float)
    both = (v1_d != 0) & (v2_d != 0)
    corr = np.corrcoef(v1_d[both], v2_d[both])[0, 1] if both.sum() > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(v1_d[both], v2_d[both], alpha=0.5, s=8, c='#2563eb')
    lim = max(abs(v1_d[both]).max(), abs(v2_d[both]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('v1 count change'); ax.set_ylabel('v2 count change')
    ax.set_title(f'Bulk Count Changes (r={corr:.4f}, n={both.sum()})')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    return fig_to_base64(fig), corr


# ---------------------------------------------------------------------------
# Raw data tables
# ---------------------------------------------------------------------------

def get_raw_data_tables():
    """Read a few key TSV files and return HTML tables."""
    import csv
    tables = {}
    flat_dir = os.path.join(os.path.dirname(__file__), 'v2ecoli', 'reconstruction', 'ecoli', 'flat')
    if not os.path.isdir(flat_dir):
        flat_dir = 'v2ecoli/reconstruction/ecoli/flat'

    for name in ['genes', 'proteins', 'metabolites', 'compartments']:
        path = os.path.join(flat_dir, f'{name}.tsv')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
        if len(rows) < 2:
            continue
        headers = rows[0]
        data = rows[1:min(11, len(rows))]  # First 10 rows
        total = len(rows) - 1

        html = f'<table><caption>{name}.tsv ({total} rows, showing first 10)</caption>'
        html += '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers[:8]) + '</tr></thead>'
        html += '<tbody>'
        for row in data:
            html += '<tr>' + ''.join(f'<td>{c[:30]}</td>' for c in row[:8]) + '</tr>'
        html += '</tbody></table>'
        tables[name] = html
    return tables


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def generate_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    timings = {}

    # === Section 1: ParCa / sim_data ===
    print("=" * 50)
    print("Section 1: ParCa / sim_data")

    # Always time the cache generation for the report
    print("  Generating cache from simData...")
    t0 = time.time()
    save_cache(SIM_DATA_PATH, CACHE_DIR)
    timings['parca_cache'] = time.time() - t0
    print(f"  Cache generated in {timings['parca_cache']:.1f}s")

    # Load raw data tables
    raw_tables = get_raw_data_tables()

    # Load initial state for inspection
    initial_state = load_initial_state(os.path.join(CACHE_DIR, 'initial_state.json'))
    initial_state_json = state_to_json_str(initial_state)

    # === Section 2: Build ecoli_wcm document ===
    print("Section 2: Build ecoli_wcm document")
    t0 = time.time()
    composite = make_composite(cache_dir=CACHE_DIR)
    timings['build_document'] = time.time() - t0
    print(f"  Built in {timings['build_document']:.2f}s, {len(composite.step_paths)} steps")

    # Bigraph viz — clean view + detailed view
    print("  Generating bigraph (clean)...")
    bigraph_svg = generate_bigraph_svg(composite.state, filename='bigraph_clean')
    print("  Generating bigraph (detailed)...")
    bigraph_detail_svg = generate_bigraph_svg(composite.state, filename='bigraph_detail', show_partitioning=True)

    # Document JSON for inspection
    ecoli_wcm_json = state_to_json_str(composite.state)

    # === Section 3: Long v2 simulation ===
    print(f"Section 3: v2ecoli simulation ({V2_DURATION}s = {V2_DURATION/60:.0f} min)")
    v2_initial = np.array(composite.state['agents']['0']['bulk']['count'], copy=True)
    t0 = time.time()
    composite.run(V2_DURATION)
    timings['v2_simulation'] = time.time() - t0
    v2_final = composite.state['agents']['0']['bulk']['count'].copy()
    v2_changed = (v2_initial != v2_final).sum()
    print(f"  Done in {timings['v2_simulation']:.1f}s, {v2_changed} molecules changed")

    # Get emitter history
    cell = composite.state.get('agents', {}).get('0', {})
    emitter_edge = cell.get('emitter')
    v2_history = emitter_edge['instance'].history if isinstance(emitter_edge, dict) and 'instance' in emitter_edge else []
    print(f"  Emitter history: {len(v2_history)} snapshots")

    # Mass plots
    mass_b64 = plot_mass_timeseries(v2_history, 'v2ecoli Mass Components')
    growth_b64 = plot_growth_metrics(v2_history)

    # Final mass
    final_mass = cell.get('listeners', {}).get('mass', {})

    # === Section 4: v1 comparison (same duration) ===
    print(f"Section 4: v1 comparison ({V1_DURATION}s)")
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file()
        sim.max_duration = int(V1_DURATION)
        sim.emitter = 'timeseries'
        sim.divide = False; sim.build_ecoli()
        v1_initial_short = sim.generated_initial_state['bulk']['count'].copy()
        t0 = time.time()
        sim.run()
        timings['v1_simulation'] = time.time() - t0
    v1_state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
    v1_final_short = v1_state['bulk']['count'].copy()
    v1_timeseries = sim.query()

    # v2 at same duration for comparison
    composite_short = make_composite(cache_dir=CACHE_DIR)
    v2_initial_short = np.array(composite_short.state['agents']['0']['bulk']['count'], copy=True)
    t0 = time.time()
    composite_short.run(V1_DURATION)
    timings['v2_comparison'] = time.time() - t0
    v2_final_short = composite_short.state['agents']['0']['bulk']['count'].copy()

    # Get v2 emitter history for comparison
    v2_comp_cell = composite_short.state.get('agents', {}).get('0', {})
    v2_comp_em = v2_comp_cell.get('emitter', {}).get('instance')
    v2_comp_history = v2_comp_em.history if v2_comp_em else []

    scatter_b64, corr = plot_bulk_scatter(v1_initial_short, v1_final_short,
                                           v2_initial_short, v2_final_short)
    v1_changed_short = (v1_initial_short != v1_final_short).sum()
    v2_changed_short = (v2_initial_short != v2_final_short).sum()

    # Side-by-side mass comparison
    def v1_query_to_history(ts):
        times = sorted(t for t in ts.keys() if isinstance(t, (int, float)) and t > 0)
        return [{'global_time': t, 'listeners': {'mass': ts[t].get('listeners', {}).get('mass', {})}} for t in times]

    v1_history = v1_query_to_history(v1_timeseries)
    comparison_mass_b64 = plot_comparison_mass(v1_history, v2_comp_history)

    # === Build HTML ===
    print("Building HTML report...")

    raw_data_html = ''
    for name, table_html in raw_tables.items():
        raw_data_html += f'<h3>{name}</h3>{table_html}'

    report_path = os.path.join(OUT_DIR, 'comparison_report.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli Comprehensive Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 2em; margin: 20px 0; color: #0f172a; }}
  h2 {{ font-size: 1.4em; margin: 30px 0 15px; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  h3 {{ font-size: 1.1em; margin: 20px 0 10px; color: #475569; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 10px; margin: 15px 0; }}
  .stat {{ background: white; border-radius: 10px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat .label {{ font-size: 0.75em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 1.4em; font-weight: 700; margin-top: 3px; }}
  .green {{ color: #16a34a; }} .blue {{ color: #2563eb; }} .red {{ color: #dc2626; }} .purple {{ color: #7c3aed; }}
  .plot {{ background: white; border-radius: 10px; padding: 15px; margin: 12px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .plot img {{ max-width: 100%; height: auto; }}
  .section {{ background: white; border-radius: 10px; padding: 20px; margin: 12px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .bigraph {{ background: white; border-radius: 10px; padding: 15px; margin: 12px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: auto; max-height: 800px; }}
  .bigraph svg {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  caption {{ font-weight: 600; margin-bottom: 5px; text-align: left; color: #475569; }}
  .json-viewer {{ background: white; border-radius: 10px; padding: 15px; margin: 12px 0;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-height: 500px; overflow: auto; }}
  .json-tree {{ font-family: 'SF Mono', monospace; font-size: 12px; line-height: 1.4; }}
  .json-tree details {{ margin-left: 16px; }}
  .json-tree summary {{ cursor: pointer; color: #475569; padding: 1px 0; }}
  .json-tree summary:hover {{ background: #f1f5f9; }}
  .json-tree .key {{ color: #7c3aed; font-weight: 500; }}
  .json-tree .string {{ color: #16a34a; }}
  .json-tree .number {{ color: #2563eb; }}
  .json-tree .type-tag {{ color: #94a3b8; font-style: italic; font-size: 0.85em; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; padding: 8px;
             background: #f8fafc; border-radius: 6px; font-size: 0.8em; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #ccc; }}
  footer {{ margin-top: 30px; padding: 15px 0; border-top: 1px solid #e2e8f0;
            color: #94a3b8; font-size: 0.8em; text-align: center; }}
  .timing {{ display: inline-block; background: #dbeafe; color: #1e40af; padding: 2px 8px;
             border-radius: 4px; font-size: 0.85em; font-weight: 500; }}
</style>
</head>
<body>

<h1>v2ecoli Comprehensive Report</h1>
<p style="color: #64748b; margin-bottom: 20px;">Generated {time.strftime('%Y-%m-%d %H:%M')}</p>

<!-- ===== Section 1: ParCa ===== -->
<h2>1. Parameter Calculator (ParCa)</h2>
<div class="section">
  <p>The ParCa pipeline reads raw E. coli data (133 TSV files) and produces fitted simulation parameters.
     Currently uses pre-computed simData from vEcoli; the ParCa code is in the v2ecoli repo and will be
     modularized into process-bigraph steps.</p>
  <p>Cache generation (LoadSimData + serialize): <span class="timing">{timings['parca_cache']:.1f}s</span></p>
  <p>Output: <code>{CACHE_DIR}/initial_state.json</code> (10MB) + <code>sim_data_cache.dill</code> (190MB)</p>
</div>

<h3>Raw Input Data (samples)</h3>
<div class="section" style="overflow-x: auto;">
  {raw_data_html}
</div>

<h3>Initial State (from ParCa → LoadSimData)</h3>
<div class="json-viewer">
  <div class="json-tree" id="initial-state-json"></div>
</div>

<!-- ===== Section 2: ecoli_wcm Document ===== -->
<h2>2. E. coli WCM Document</h2>
<div class="summary">
  <div class="stat"><div class="label">Build Time</div><div class="value blue">{timings['build_document']:.2f}s</div></div>
  <div class="stat"><div class="label">Steps</div><div class="value">{len(composite.step_paths)}</div></div>
  <div class="stat"><div class="label">Processes</div><div class="value">{len(composite.process_paths)}</div></div>
  <div class="stat"><div class="label">Flow Steps</div><div class="value">{len(DEFAULT_FLOW)}</div></div>
</div>

<h3>Bigraph Visualization</h3>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#FFB6C1"></div> DNA</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ADD8E6"></div> RNA</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#90EE90"></div> Protein</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#FFD700"></div> Metabolism</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#DDA0DD"></div> Regulation</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#D3D3D3"></div> Listeners</div>
</div>
<div class="bigraph">{bigraph_svg}</div>

<h3>Detailed View (with partitioning)</h3>
<div class="bigraph">{bigraph_detail_svg}</div>

<h3>Document State (JSON)</h3>
<div class="json-viewer">
  <div class="json-tree" id="ecoli-wcm-json"></div>
</div>

<!-- ===== Section 3: Simulation ===== -->
<h2>3. Simulation ({V2_DURATION/60:.0f} minutes)</h2>
<div class="summary">
  <div class="stat"><div class="label">Sim Duration</div><div class="value">{V2_DURATION:.0f}s</div></div>
  <div class="stat"><div class="label">Wall Time</div><div class="value blue">{timings['v2_simulation']:.1f}s</div></div>
  <div class="stat"><div class="label">Molecules Changed</div><div class="value purple">{v2_changed}</div></div>
  <div class="stat"><div class="label">Final Dry Mass</div><div class="value">{final_mass.get('dry_mass', 0):.1f} fg</div></div>
  <div class="stat"><div class="label">Final Volume</div><div class="value">{final_mass.get('volume', 0):.3f} fL</div></div>
  <div class="stat"><div class="label">Emitter Snapshots</div><div class="value">{len(v2_history)}</div></div>
</div>

<h3>Mass Components</h3>
<div class="plot"><img src="data:image/png;base64,{mass_b64}" alt="Mass"></div>

<h3>Growth Metrics</h3>
<div class="plot"><img src="data:image/png;base64,{growth_b64}" alt="Growth"></div>

<!-- ===== Section 4: v1 Comparison ===== -->
<h2>4. v1 vs v2 Comparison ({V1_DURATION:.0f}s)</h2>
<div class="summary">
  <div class="stat"><div class="label">v1 Runtime</div><div class="value red">{timings['v1_simulation']:.2f}s</div></div>
  <div class="stat"><div class="label">v2 Runtime</div><div class="value blue">{timings.get('v2_comparison', 0):.2f}s</div></div>
  <div class="stat"><div class="label">v1 Changed</div><div class="value red">{v1_changed_short}</div></div>
  <div class="stat"><div class="label">v2 Changed</div><div class="value blue">{v2_changed_short}</div></div>
  <div class="stat"><div class="label">Correlation</div><div class="value green">{corr:.4f}</div></div>
</div>

<h3>Mass Fraction Comparison</h3>
<div class="plot"><img src="data:image/png;base64,{comparison_mass_b64}" alt="Mass comparison"></div>

<h3>Bulk Molecule Count Changes</h3>
<div class="plot"><img src="data:image/png;base64,{scatter_b64}" alt="Scatter"></div>

<!-- ===== Timing Summary ===== -->
<h2>5. Timing Summary</h2>
<div class="section">
  <table>
    <tr><th>Stage</th><th>Wall Time</th><th>Sim Time</th></tr>
    <tr><td>sim_data → cache (LoadSimData + serialize)</td><td>{timings.get('parca_cache', 0):.1f}s</td><td>—</td></tr>
    <tr><td>Build document from cache</td><td>{timings['build_document']:.2f}s</td><td>—</td></tr>
    <tr><td>v2 simulation (main)</td><td>{timings['v2_simulation']:.1f}s</td><td>{V2_DURATION:.0f}s ({V2_DURATION/60:.0f} min)</td></tr>
    <tr><td>v1 simulation (comparison)</td><td>{timings['v1_simulation']:.2f}s</td><td>{V1_DURATION:.0f}s</td></tr>
    <tr><td>v2 simulation (comparison)</td><td>{timings.get('v2_comparison', 0):.2f}s</td><td>{V1_DURATION:.0f}s</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>{sum(timings.values()):.1f}s</strong></td><td>—</td></tr>
  </table>
</div>

<footer>
  v2ecoli &middot; <a href="https://github.com/vivarium-collective/v2ecoli">github.com/vivarium-collective/v2ecoli</a>
  &middot; All processes run through process-bigraph Composite.run()
</footer>

<script>
const initialStateData = {initial_state_json};
const ecoliWcmData = {ecoli_wcm_json};

function renderJson(data, container, depth) {{
  if (depth > 6) {{ container.textContent = '...'; return; }}
  if (data === null) {{ container.innerHTML = '<span class="type-tag">null</span>'; }}
  else if (typeof data === 'string') {{
    if (data.startsWith('<') && data.endsWith('>'))
      container.innerHTML = '<span class="type-tag">' + esc(data) + '</span>';
    else container.innerHTML = '<span class="string">"' + esc(trunc(data, 60)) + '"</span>';
  }}
  else if (typeof data === 'number') {{ container.innerHTML = '<span class="number">' + data + '</span>'; }}
  else if (typeof data === 'boolean') {{ container.innerHTML = '<span class="number">' + data + '</span>'; }}
  else if (Array.isArray(data)) {{
    const d = document.createElement('details');
    if (depth < 1) d.open = true;
    d.innerHTML = '<summary>[' + data.length + ' items]</summary>';
    data.forEach((item, i) => {{
      const r = document.createElement('div'); r.style.marginLeft = '16px';
      r.innerHTML = '<span class="key">' + i + '</span>: ';
      const v = document.createElement('span'); renderJson(item, v, depth + 1);
      r.appendChild(v); d.appendChild(r);
    }});
    container.appendChild(d);
  }}
  else if (typeof data === 'object') {{
    const keys = Object.keys(data); const d = document.createElement('details');
    if (depth < 2) d.open = true;
    d.innerHTML = '<summary>{{' + keys.length + ' keys}}</summary>';
    keys.forEach(k => {{
      const r = document.createElement('div'); r.style.marginLeft = '16px';
      r.innerHTML = '<span class="key">' + esc(k) + '</span>: ';
      const v = document.createElement('span'); renderJson(data[k], v, depth + 1);
      r.appendChild(v); d.appendChild(r);
    }});
    container.appendChild(d);
  }} else {{ container.textContent = String(data); }}
}}
function esc(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function trunc(s, n) {{ return s.length > n ? s.slice(0, n) + '...' : s; }}

renderJson(initialStateData, document.getElementById('initial-state-json'), 0);
renderJson(ecoliWcmData, document.getElementById('ecoli-wcm-json'), 0);
</script>
</body>
</html>""")

    print(f"\nReport saved to {report_path}")
    return report_path


if __name__ == '__main__':
    generate_report()
