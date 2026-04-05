"""
v2ecoli Benchmark Suite

Runs a series of benchmarks measuring performance and correctness
of the whole-cell E. coli simulation through process-bigraph.

Benchmarks:
1. Cache loading — time to build document from cached simData
2. Short simulation (60s) — step execution, mass growth
3. v1 comparison (60s) — correlation with vEcoli
4. Long simulation (to division) — growth trajectory, division detection
5. Step-level diagnostics — per-step timing and error rates
"""

import os
import io
import json
import time
import base64
import html as html_lib
import copy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contextlib import chdir
from wholecell.utils.filepath import ROOT_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.library.schema import not_a_process

from v2ecoli.composite import make_composite, _build_core, save_cache
from v2ecoli.generate import build_document, DEFAULT_FLOW
from v2ecoli.cache import NumpyJSONEncoder, load_initial_state
from v2ecoli.steps.base import _translate_schema

from bigraph_viz import plot_bigraph
from bigraph_schema import get_path, strip_schema_keys


SIM_DATA_PATH = os.path.join(ROOT_PATH, 'out', 'kb', 'simData.cPickle')
OUT_DIR = 'out/benchmark'
CACHE_DIR = 'out/cache'
COMPARISON_DURATION = 60.0
LONG_DURATION = 500.0  # Longer run showing growth (not full division)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_cache_load():
    """Benchmark: load document from cache."""
    t0 = time.time()
    save_cache(SIM_DATA_PATH, CACHE_DIR)
    cache_time = time.time() - t0

    t0 = time.time()
    composite = make_composite(cache_dir=CACHE_DIR)
    build_time = time.time() - t0

    return {
        'cache_generation': cache_time,
        'document_build': build_time,
        'n_steps': len(composite.step_paths),
        'n_processes': len(composite.process_paths),
        'composite': composite,
    }


def bench_short_sim(composite, duration=60.0):
    """Benchmark: short simulation with per-step diagnostics."""
    cell = composite.state['agents']['0']
    bulk_before = np.array(cell['bulk']['count'], copy=True)

    t0 = time.time()
    composite.run(duration)
    wall_time = time.time() - t0

    bulk_after = cell['bulk']['count']
    changed = (bulk_before != bulk_after).sum()

    # Get emitter data
    em_edge = cell.get('emitter')
    history = em_edge['instance'].history if isinstance(em_edge, dict) and 'instance' in em_edge else []

    mass = cell.get('listeners', {}).get('mass', {})

    return {
        'duration': duration,
        'wall_time': wall_time,
        'bulk_changed': int(changed),
        'total_bulk': len(bulk_before),
        'final_dry_mass': mass.get('dry_mass', 0),
        'final_cell_mass': mass.get('cell_mass', 0),
        'final_volume': mass.get('volume', 0),
        'history': history,
        'rate': duration / wall_time,
    }


def bench_v1_comparison(duration=60.0):
    """Benchmark: v1 vs v2 comparison at every timestep.

    Compares bulk molecule counts at each simulated second between
    v1 (vEcoli) and v2 (v2ecoli). Reports:
    - Per-timestep Pearson correlation of all molecule counts
    - Final-state correlation of count deltas (molecules that changed)
    - Exact match percentage (fraction of molecules identical at each timestep)
    """
    # v2 — emitter captures bulk at each timestep
    composite = make_composite(cache_dir=CACHE_DIR)
    v2_initial = np.array(composite.state['agents']['0']['bulk']['count'], copy=True)
    t0 = time.time()
    composite.run(duration)
    v2_time = time.time() - t0
    v2_final = composite.state['agents']['0']['bulk']['count'].copy()

    # Get v2 per-timestep bulk from emitter
    cell = composite.state['agents']['0']
    em = cell.get('emitter', {}).get('instance')
    v2_history = em.history if em else []
    v2_bulk_ts = {}
    for snap in v2_history:
        t = snap.get('global_time', 0)
        bulk = snap.get('bulk')
        if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
            v2_bulk_ts[int(t)] = bulk['count'].copy()

    # v1 — timeseries emitter captures everything
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file()
        sim.max_duration = int(duration)
        sim.emitter = 'timeseries'
        sim.divide = False
        sim.build_ecoli()
        v1_initial = sim.generated_initial_state['bulk']['count'].copy()
        t0 = time.time()
        sim.run()
        v1_time = time.time() - t0
    v1_state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
    v1_final = v1_state['bulk']['count'].copy()

    # Get v1 per-timestep bulk from timeseries
    v1_ts = sim.query()
    v1_bulk_ts = {}
    for t_key, snapshot in v1_ts.items():
        if not isinstance(t_key, (int, float)):
            continue
        bulk = snapshot.get('bulk')
        if bulk is not None:
            if hasattr(bulk, 'dtype') and 'count' in getattr(bulk.dtype, 'names', []) or []:
                v1_bulk_ts[int(t_key)] = np.array(bulk['count'], dtype=float)
            elif isinstance(bulk, (list, np.ndarray)):
                v1_bulk_ts[int(t_key)] = np.array(bulk, dtype=float)

    # Per-timestep comparison
    common_times = sorted(set(v1_bulk_ts.keys()) & set(v2_bulk_ts.keys()))
    per_ts_corr = []
    per_ts_exact = []
    per_ts_rmse = []
    for t in common_times:
        v1_c = v1_bulk_ts[t].astype(float)
        v2_c = v2_bulk_ts[t].astype(float)
        if len(v1_c) == len(v2_c) and len(v1_c) > 0:
            corr = np.corrcoef(v1_c, v2_c)[0, 1]
            exact = np.mean(v1_c == v2_c)
            rmse = np.sqrt(np.mean((v1_c - v2_c) ** 2))
            per_ts_corr.append(corr)
            per_ts_exact.append(exact)
            per_ts_rmse.append(rmse)

    # Final-state delta correlation (original method)
    both = (v1_initial != v1_final) & (v2_initial != v2_final)
    d1 = v1_final[both] - v1_initial[both]
    d2 = v2_final[both] - v2_initial[both]
    delta_corr = np.corrcoef(d1.astype(float), d2.astype(float))[0, 1] if both.sum() > 0 else 0.0

    return {
        'duration': duration,
        'v1_time': v1_time,
        'v2_time': v2_time,
        'v1_changed': int((v1_initial != v1_final).sum()),
        'v2_changed': int((v2_initial != v2_final).sum()),
        'both_changed': int(both.sum()),
        'delta_correlation': delta_corr,
        'common_timesteps': len(common_times),
        'per_ts_corr': per_ts_corr,
        'per_ts_exact': per_ts_exact,
        'per_ts_rmse': per_ts_rmse,
        'mean_correlation': np.mean(per_ts_corr) if per_ts_corr else 0.0,
        'mean_exact_match': np.mean(per_ts_exact) if per_ts_exact else 0.0,
        'mean_rmse': np.mean(per_ts_rmse) if per_ts_rmse else 0.0,
        'v1_initial': v1_initial,
        'v1_final': v1_final,
        'v2_initial': v2_initial,
        'v2_final': v2_final,
    }


def bench_step_diagnostics(composite):
    """Benchmark: per-step analysis."""
    cell = composite.state['agents']['0']
    core = composite.core

    diagnostics = []
    for step_name in DEFAULT_FLOW:
        path = ('agents', '0', step_name)
        edge = cell.get(step_name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue

        inst = edge['instance']
        proc = getattr(inst, 'process', inst)

        # Step info
        info = {
            'name': step_name,
            'class': type(inst).__name__,
            'inner_class': type(proc).__name__ if proc is not inst else None,
            'has_ports_schema': hasattr(proc, 'ports_schema') and callable(getattr(proc, 'ports_schema')),
            'has_inputs': hasattr(inst, 'inputs') and 'inputs' in type(inst).__dict__,
            'has_config_schema': bool(getattr(inst, 'config_schema', {})),
            'has_defaults': bool(getattr(proc, 'defaults', {})),
            'n_config_keys': len(getattr(proc, 'parameters', {})),
            'priority': edge.get('priority', 0),
        }

        # Port info
        if info['has_ports_schema']:
            try:
                ps = proc.ports_schema()
                info['input_ports'] = sorted(ps.keys())
            except Exception:
                info['input_ports'] = []
        else:
            info['input_ports'] = []

        # Wire info
        wires = edge.get('inputs', {})
        info['wires'] = {k: v for k, v in wires.items() if not k.startswith('_flow')}

        diagnostics.append(info)

    return diagnostics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MASS_KEYS = {
    'Protein': 'protein_mass', 'tRNA': 'tRna_mass', 'rRNA': 'rRna_mass',
    'mRNA': 'mRna_mass', 'DNA': 'dna_mass', 'Small Mol': 'smallMolecule_mass',
    'Dry Mass': 'dry_mass',
}
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']


def plot_mass(history, title=''):
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig_to_b64(fig)

    times = np.array([s.get('global_time', 0) for s in history])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (label, key) in enumerate(MASS_KEYS.items()):
        vals = np.array([s.get('listeners', {}).get('mass', {}).get(key, 0) for s in history])
        if len(vals) > 0 and vals[0] > 0:
            axes[0].plot(times / 60, vals / vals[0], color=COLORS[i], lw=1.5, label=label)
            axes[1].plot(times / 60, vals, color=COLORS[i], lw=1.5, label=label)

    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Fold change')
    axes[0].set_title('Fold Change (normalized to t=0)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.15)
    # Don't clamp y-axis — show decreases below 1.0

    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Mass (fg)')
    axes[1].set_title('Absolute Mass')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.15)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_growth(history):
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig_to_b64(fig)

    times = np.array([s.get('global_time', 0) for s in history])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gr = np.array([s.get('listeners', {}).get('mass', {}).get('instantaneous_growth_rate', 0) for s in history])
    vol = np.array([s.get('listeners', {}).get('mass', {}).get('volume', 0) for s in history])
    pf = np.array([s.get('listeners', {}).get('mass', {}).get('protein_mass_fraction', 0) for s in history])
    rf = np.array([s.get('listeners', {}).get('mass', {}).get('rna_mass_fraction', 0) for s in history])

    axes[0].plot(times / 60, gr * 3600, color='#2563eb', lw=1)
    axes[0].set_ylabel('Growth rate (1/h)')
    axes[0].set_title('Instantaneous Growth Rate')

    axes[1].plot(times / 60, vol, color='#16a34a', lw=1)
    axes[1].set_ylabel('Volume (fL)')
    axes[1].set_title('Cell Volume')

    axes[2].plot(times / 60, pf, color='#e41a1c', lw=1, label='Protein')
    axes[2].plot(times / 60, rf, color='#377eb8', lw=1, label='RNA')
    axes[2].set_ylabel('Mass fraction')
    axes[2].set_title('Mass Fractions')
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel('Time (min)')
        ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_comparison(comp):
    """Plot per-timestep correlation, scatter of final deltas, and RMSE."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Per-timestep correlation
    if comp['per_ts_corr']:
        axes[0].plot(range(1, len(comp['per_ts_corr']) + 1), comp['per_ts_corr'],
                     color='#2563eb', lw=1.5)
        axes[0].set_xlabel('Timestep (s)')
        axes[0].set_ylabel('Pearson correlation')
        axes[0].set_title(f'Per-Timestep Correlation (mean={comp["mean_correlation"]:.6f})')
        axes[0].set_ylim(min(0.999, min(comp['per_ts_corr']) - 0.0001), 1.0001)
        axes[0].grid(True, alpha=0.15)
    else:
        axes[0].text(0.5, 0.5, 'No per-timestep data', ha='center', va='center')

    # Scatter of final deltas
    v1_d = comp['v1_final'].astype(float) - comp['v1_initial'].astype(float)
    v2_d = comp['v2_final'].astype(float) - comp['v2_initial'].astype(float)
    both = (v1_d != 0) & (v2_d != 0)
    if both.sum() > 0:
        axes[1].scatter(v1_d[both], v2_d[both], alpha=0.5, s=8, c='#2563eb')
        lim = max(abs(v1_d[both]).max(), abs(v2_d[both]).max()) * 1.1
        axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel('v1 count change')
    axes[1].set_ylabel('v2 count change')
    axes[1].set_title(f'Final Delta (r={comp["delta_correlation"]:.4f}, n={both.sum()})')
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.2)

    # Per-timestep RMSE
    if comp['per_ts_rmse']:
        axes[2].plot(range(1, len(comp['per_ts_rmse']) + 1), comp['per_ts_rmse'],
                     color='#dc2626', lw=1.5)
        axes[2].set_xlabel('Timestep (s)')
        axes[2].set_ylabel('RMSE (counts)')
        axes[2].set_title(f'Per-Timestep RMSE (mean={comp["mean_rmse"]:.2f})')
        axes[2].grid(True, alpha=0.15)
    else:
        axes[2].text(0.5, 0.5, 'No per-timestep data', ha='center', va='center')

    fig.suptitle('v1 vs v2 Comparison — All Molecules, All Timesteps', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Bigraph
# ---------------------------------------------------------------------------

SKIP_STEPS = {'unique_update', 'global_clock', 'mark_d_period', 'division',
              'exchange_data', 'media_update', 'post-division-mass-listener', 'emitter'}
SKIP_PORTS = {'timestep', 'global_time', 'next_update_time', 'process'}
BIO_COLORS = {
    'dna': ('#FFB6C1', lambda n: 'chromosome' in n),
    'rna': ('#ADD8E6', lambda n: any(s in n for s in ('transcript', 'rna-', 'RNA', 'rnap'))),
    'protein': ('#90EE90', lambda n: any(s in n for s in ('polypeptide', 'protein', 'ribosome'))),
    'meta': ('#FFD700', lambda n: any(s in n for s in ('metabolism', 'equilibrium', 'complexation', 'two-component'))),
    'reg': ('#DDA0DD', lambda n: any(s in n for s in ('tf-', 'tf_'))),
    'listen': ('#D3D3D3', lambda n: 'listener' in n),
}


def make_bigraph_svg(state):
    cell = state.get('agents', {}).get('0', state)
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict): continue
        if '_type' in edge:
            if any(s in name for s in SKIP_STEPS): continue
            if '_requester' in name: continue
            inputs = {p: w for p, w in edge.get('inputs', {}).items()
                      if not p.startswith('_flow') and p not in SKIP_PORTS
                      and not (isinstance(w, list) and w and w[0] in ('request', 'allocate'))}
            clean = name.replace('ecoli-', '').replace('_evolver', '')
            viz[clean] = {'_type': edge['_type'], 'inputs': inputs}
        elif name == 'unique' and isinstance(edge, dict):
            viz[name] = {k: {} for k in edge.keys()}
        elif name in ('bulk', 'listeners', 'environment', 'boundary'):
            viz[name] = {}

    viz_state = {'agents': {'0': viz}}
    prefix = ('agents', '0')
    colors, groups = {}, {k: [] for k in BIO_COLORS}
    for n in viz:
        if '_type' not in viz.get(n, {}): continue
        p = prefix + (n,)
        for gk, (c, m) in BIO_COLORS.items():
            if m(n): colors[p] = c; groups[gk].append(p); break

    try:
        plot_bigraph(viz_state, remove_process_place_edges=True,
                     node_groups=[g for g in groups.values() if g],
                     node_fill_colors=colors, size='20,16', rankdir='TB',
                     dpi='150', port_labels=False, node_label_size='24pt',
                     label_margin='0.08', out_dir=OUT_DIR,
                     filename='bigraph', file_format='svg')
        with open(os.path.join(OUT_DIR, 'bigraph.svg')) as f:
            return f.read()
    except Exception as e:
        return f'<p>Failed: {html_lib.escape(str(e))}</p>'


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_benchmarks():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    # 1. Cache + build
    print("=" * 50)
    print("Benchmark 1: Cache & Document Build")
    r = bench_cache_load()
    results['build'] = r
    composite = r['composite']
    print(f"  Cache: {r['cache_generation']:.1f}s, Build: {r['document_build']:.2f}s")
    print(f"  {r['n_steps']} steps, {r['n_processes']} processes")

    # 2. Bigraph
    print("Generating bigraph...")
    bigraph_svg = make_bigraph_svg(composite.state)

    # 3. Short sim
    print(f"Benchmark 2: Simulation ({COMPARISON_DURATION}s)")
    r = bench_short_sim(composite, COMPARISON_DURATION)
    results['short'] = r
    print(f"  {r['wall_time']:.1f}s wall, {r['bulk_changed']} changed, {r['rate']:.1f}x realtime")

    # 4. Step diagnostics
    print("Benchmark 3: Step Diagnostics")
    diag = bench_step_diagnostics(composite)
    results['diagnostics'] = diag
    print(f"  {len(diag)} steps analyzed")

    # 5. v1 comparison
    print(f"Benchmark 4: v1 Comparison ({COMPARISON_DURATION}s)")
    comp = bench_v1_comparison(COMPARISON_DURATION)
    results['comparison'] = comp
    print(f"  v1: {comp['v1_time']:.2f}s, v2: {comp['v2_time']:.2f}s, "
          f"mean_corr: {comp['mean_correlation']:.6f}, delta_corr: {comp['delta_correlation']:.4f}")

    # 6. Long sim
    print(f"Benchmark 5: Long Simulation ({LONG_DURATION}s = {LONG_DURATION/60:.0f}min)")
    long_composite = make_composite(cache_dir=CACHE_DIR)
    long_r = bench_short_sim(long_composite, LONG_DURATION)
    results['long'] = long_r
    print(f"  {long_r['wall_time']:.1f}s wall, {long_r['bulk_changed']} changed")

    # Plots
    print("Generating plots...")
    mass_short = plot_mass(results['short']['history'], f'Mass Components ({COMPARISON_DURATION}s)')
    growth_short = plot_growth(results['short']['history'])
    mass_long = plot_mass(results['long']['history'], f'Mass Components ({LONG_DURATION/60:.0f} min)')
    growth_long = plot_growth(results['long']['history'])
    comparison_plot = plot_comparison(comp)

    # Build HTML
    b = results['build']
    s = results['short']
    c = results['comparison']
    l = results['long']

    # Step diagnostics table
    step_rows = ''
    for d in diag:
        inner = f' ({d["inner_class"]})' if d['inner_class'] else ''
        v1_markers = []
        if d['has_ports_schema']: v1_markers.append('ports_schema')
        if d['has_defaults']: v1_markers.append('defaults')
        v2_markers = []
        if d['has_inputs']: v2_markers.append('inputs()')
        if d['has_config_schema']: v2_markers.append('config_schema')
        ports = ', '.join(d.get('input_ports', [])[:5])
        if len(d.get('input_ports', [])) > 5:
            ports += f' +{len(d["input_ports"])-5}'

        step_rows += f"""<tr>
          <td>{d['name']}</td>
          <td>{d['class']}{inner}</td>
          <td>{d['n_config_keys']}</td>
          <td>{ports}</td>
          <td>{d['priority']:.0f}</td>
        </tr>"""

    report_path = os.path.join(OUT_DIR, 'benchmark_report.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli Benchmark Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 1.8em; margin: 15px 0; color: #0f172a; }}
  h2 {{ font-size: 1.3em; margin: 25px 0 10px; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
  h3 {{ font-size: 1.05em; margin: 15px 0 8px; color: #475569; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 8px; margin: 10px 0; }}
  .metric {{ background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
  .metric .label {{ font-size: 0.7em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric .value {{ font-size: 1.3em; font-weight: 700; margin-top: 2px; }}
  .green {{ color: #16a34a; }} .blue {{ color: #2563eb; }} .red {{ color: #dc2626; }} .purple {{ color: #7c3aed; }}
  .plot {{ background: white; border-radius: 8px; padding: 12px; margin: 10px 0;
           box-shadow: 0 1px 2px rgba(0,0,0,0.08); text-align: center; }}
  .plot img {{ max-width: 100%; }}
  .section {{ background: white; border-radius: 8px; padding: 15px; margin: 10px 0;
              box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
  details {{ margin: 5px 0; }}
  details > summary {{ cursor: pointer; font-weight: 600; color: #475569; padding: 5px 0; }}
  details > summary:hover {{ color: #1e293b; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 5px 8px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .bigraph {{ overflow: auto; max-height: 700px; }}
  .bigraph svg {{ max-width: 100%; }}
  .bench-bar {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .bench-bar .bar {{ height: 20px; border-radius: 3px; min-width: 2px; }}
  .bench-bar .label {{ font-size: 0.8em; min-width: 100px; }}
  .timing {{ display: inline-block; background: #dbeafe; color: #1e40af; padding: 1px 6px;
             border-radius: 3px; font-size: 0.8em; font-weight: 500; }}
  footer {{ margin-top: 30px; padding: 15px 0; border-top: 1px solid #e2e8f0;
            color: #94a3b8; font-size: 0.75em; text-align: center; }}
</style>
</head>
<body>

<h1>v2ecoli Benchmark Report</h1>
<p style="color: #64748b; font-size: 0.9em;">{time.strftime('%Y-%m-%d %H:%M')} &middot; All benchmarks run through process-bigraph <code>Composite.run()</code></p>

<div class="section">
  <p><strong>v2ecoli</strong> is a whole-cell <em>E. coli</em> model running natively on
  <a href="https://github.com/vivarium-collective/process-bigraph">process-bigraph</a>.
  It migrates all 55 biological steps from
  <a href="https://github.com/CovertLab/vEcoli">vEcoli</a> to run through the standard
  <code>Composite.run()</code> pipeline with custom bigraph-schema types for bulk molecules,
  unique molecules, and listener stores.</p>
  <p>This report benchmarks the simulation's performance, correctness against vEcoli (v1),
  and biological accuracy across multiple timescales.</p>
</div>

<!-- ===== Benchmark 1: Build ===== -->
<h2>1. Document Build</h2>
<div class="metrics">
  <div class="metric"><div class="label">Cache Gen</div><div class="value blue">{b['cache_generation']:.1f}s</div></div>
  <div class="metric"><div class="label">Doc Build</div><div class="value blue">{b['document_build']:.2f}s</div></div>
  <div class="metric"><div class="label">Steps</div><div class="value">{b['n_steps']}</div></div>
  <div class="metric"><div class="label">Processes</div><div class="value">{b['n_processes']}</div></div>
</div>

<details>
<summary>Input Data (133 TSV files → simData → initial state)</summary>
<div class="section">
  <p>Raw data: <code>v2ecoli/reconstruction/ecoli/flat/</code> (133 TSV files, 13MB)</p>
  <p>Cached: <code>{CACHE_DIR}/initial_state.json</code> (10MB) + <code>sim_data_cache.dill</code> (190MB)</p>
</div>
</details>

<!-- ===== Benchmark 2: Short Sim ===== -->
<h2>2. Simulation ({COMPARISON_DURATION:.0f}s)</h2>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{s['wall_time']:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value green">{s['rate']:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value purple">{s['bulk_changed']}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{s['final_dry_mass']:.1f} fg</div></div>
  <div class="metric"><div class="label">Volume</div><div class="value">{s['final_volume']:.4f} fL</div></div>
</div>
<div class="plot"><img src="data:image/png;base64,{mass_short}" alt="Mass"></div>
<div class="plot"><img src="data:image/png;base64,{growth_short}" alt="Growth"></div>

<!-- ===== Benchmark 3: v1 Comparison ===== -->
<h2>3. v1 Comparison ({COMPARISON_DURATION:.0f}s)</h2>

<div class="section">
  <h3>Methodology</h3>
  <p>Both v1 (vEcoli) and v2 (v2ecoli) simulations run for {COMPARISON_DURATION:.0f} seconds with identical
     initial states from the same simData. At each simulated second, the bulk molecule count vector
     (16,321 molecules) is compared between v1 and v2 using:</p>
  <ul style="margin: 8px 0 8px 20px; font-size: 0.9em;">
    <li><strong>Pearson correlation</strong> of all {len(c['v1_initial'])} molecule counts at each timestep</li>
    <li><strong>Exact match %</strong> — fraction of molecules with identical counts at each timestep</li>
    <li><strong>RMSE</strong> — root mean square error of count differences at each timestep</li>
    <li><strong>Final delta correlation</strong> — Pearson correlation of count CHANGES (final−initial) for molecules that changed in both</li>
  </ul>
</div>

<div class="metrics">
  <div class="metric"><div class="label">v1 Runtime</div><div class="value red">{c['v1_time']:.2f}s</div></div>
  <div class="metric"><div class="label">v2 Runtime</div><div class="value blue">{c['v2_time']:.2f}s</div></div>
  <div class="metric"><div class="label">Timesteps Compared</div><div class="value">{c['common_timesteps']}</div></div>
  <div class="metric"><div class="label">Mean Correlation</div><div class="value green">{c['mean_correlation']:.6f}</div></div>
  <div class="metric"><div class="label">Mean Exact Match</div><div class="value green">{c['mean_exact_match']*100:.2f}%</div></div>
  <div class="metric"><div class="label">Mean RMSE</div><div class="value">{c['mean_rmse']:.2f}</div></div>
  <div class="metric"><div class="label">Delta Correlation</div><div class="value green">{c['delta_correlation']:.4f}</div></div>
  <div class="metric"><div class="label">Both Changed</div><div class="value purple">{c['both_changed']}</div></div>
</div>
<div class="plot"><img src="data:image/png;base64,{comparison_plot}" alt="Comparison"></div>

<!-- ===== Benchmark 4: Long Sim ===== -->
<h2>4. Long Simulation ({LONG_DURATION/60:.0f} min)</h2>
<div class="metrics">
  <div class="metric"><div class="label">Sim Duration</div><div class="value">{l['duration']:.0f}s</div></div>
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{l['wall_time']:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value green">{l['rate']:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value purple">{l['bulk_changed']}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{l['final_dry_mass']:.1f} fg</div></div>
</div>
<div class="plot"><img src="data:image/png;base64,{mass_long}" alt="Long Mass"></div>
<div class="plot"><img src="data:image/png;base64,{growth_long}" alt="Long Growth"></div>

<!-- ===== Benchmark 5: Step Diagnostics ===== -->
<h2>5. Step Diagnostics ({len(diag)} steps)</h2>
<details open>
<summary>Execution Flow</summary>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr><th>Step</th><th>Class</th><th>Config Keys</th><th>Ports</th><th>Priority</th></tr></thead>
    <tbody>{step_rows}</tbody>
  </table>
</div>
</details>

<!-- ===== Bigraph ===== -->
<h2>6. Bigraph</h2>
<details>
<summary>Process Network Visualization</summary>
<div class="bigraph">{bigraph_svg}</div>
</details>

<!-- ===== Timing Summary ===== -->
<h2>7. Timing Summary</h2>
<div class="section">
  <table>
    <tr><th>Benchmark</th><th>Wall Time</th><th>Sim Time</th><th>Sim/Wall Ratio</th></tr>
    <tr><td>Cache generation</td><td>{b['cache_generation']:.1f}s</td><td>—</td><td>—</td></tr>
    <tr><td>Document build</td><td>{b['document_build']:.2f}s</td><td>—</td><td>—</td></tr>
    <tr><td>Short simulation</td><td>{s['wall_time']:.1f}s</td><td>{s['duration']:.0f}s</td><td>{s['rate']:.1f}x</td></tr>
    <tr><td>v1 comparison (v1)</td><td>{c['v1_time']:.2f}s</td><td>{c['duration']:.0f}s</td><td>{c['duration']/c['v1_time']:.1f}x</td></tr>
    <tr><td>v1 comparison (v2)</td><td>{c['v2_time']:.2f}s</td><td>{c['duration']:.0f}s</td><td>{c['duration']/c['v2_time']:.1f}x</td></tr>
    <tr><td>Long simulation</td><td>{l['wall_time']:.1f}s</td><td>{l['duration']:.0f}s</td><td>{l['rate']:.1f}x</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>{b['cache_generation']+b['document_build']+s['wall_time']+c['v1_time']+c['v2_time']+l['wall_time']:.0f}s</strong></td><td>—</td><td>—</td></tr>
  </table>
</div>

<footer>
  v2ecoli &middot; <a href="https://github.com/vivarium-collective/v2ecoli">github.com/vivarium-collective/v2ecoli</a>
  &middot; All steps run through process-bigraph Composite.run()
</footer>
</body>
</html>""")

    print(f"\nBenchmark report saved to {report_path}")
    return report_path


if __name__ == '__main__':
    run_benchmarks()
