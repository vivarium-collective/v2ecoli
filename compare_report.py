"""
Three-way architecture comparison: Baseline vs Departitioned vs Reconciled.

Runs all three simulations in parallel using multiprocessing, then generates
a single-file HTML report with bigraph-viz composition figures, interactive
PBG JSON state viewers, mass trajectories, divergence analysis, and an
n-way molecule divergence table comparing all architectures to baseline.

Usage:
    python compare_report.py                        # default 2520s sim
    python compare_report.py --duration 600         # 10-min sim
    python compare_report.py --seed 42 --output out/my_report.html
    python compare_report.py --no-parallel          # sequential fallback
"""

import os
import io
import re
import time
import json
import base64
import argparse
import html as html_lib
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DURATION = 2520  # 42 minutes
SNAPSHOT_INTERVAL = 10

MASS_FIELDS = [
    'cell_mass', 'dry_mass', 'protein_mass', 'rna_mass',
    'dna_mass', 'smallMolecule_mass', 'water_mass',
]
MASS_LABELS = {
    'cell_mass': 'Cell Mass', 'dry_mass': 'Dry Mass',
    'protein_mass': 'Protein', 'rna_mass': 'RNA', 'dna_mass': 'DNA',
    'smallMolecule_mass': 'Small Molecule', 'water_mass': 'Water',
}
ERROR_THRESHOLD = 5.0

MODELS = {
    'baseline': {'label': 'Baseline (Partitioned)', 'color': '#2563eb', 'ls': '-',
                 'short': 'Baseline'},
    'departitioned': {'label': 'Departitioned', 'color': '#dc2626', 'ls': '--',
                      'short': 'Departitioned'},
    'reconciled': {'label': 'Reconciled', 'color': '#16a34a', 'ls': '-.',
                   'short': 'Reconciled'},
}

ARCH_DESCRIPTIONS = {
    'baseline': {
        'title': 'Baseline (Partitioned)',
        'strategy': (
            'The partitioned architecture splits each biological process into '
            'three coordinated steps per timestep: <strong>Requester</strong>, '
            '<strong>Allocator</strong>, and <strong>Evolver</strong>. '
            'Requesters read the current state and compute how many molecules '
            'each process needs. The Allocator collects all requests and distributes '
            'available molecules using <em>priority-based proportional scaling</em> — '
            'high-priority processes (RNA Degradation +10, Protein Degradation +10) '
            'get resources first, then default-priority processes share the remainder, '
            'and low-priority processes (Two-Component System -5, Metabolism -10) get '
            'what is left. When total requests exceed supply, molecules are distributed '
            'proportionally within each priority level, with integer remainders '
            'allocated randomly. Evolvers then execute with their reconciled allocation.'
        ),
        'layers': (
            '<strong>3 allocator layers</strong> execute sequentially: '
            'Layer 1 (Equilibrium, RNA Maturation, Two-Component System), '
            'Layer 2 (Complexation, Protein Degradation, RNA Degradation, '
            'Transcript Initiation, Polypeptide Initiation, Chromosome Replication), '
            'Layer 3 (Transcript Elongation, Polypeptide Elongation). '
            'Processes within a layer request simultaneously and compete for '
            'the same molecular pool.'
        ),
        'tradeoffs': (
            'Highest fidelity to the original wcEcoli model. Guarantees fair resource '
            'distribution under scarcity. Cost: 55 steps per timestep (3 per process '
            '+ allocators + listeners + infrastructure).'
        ),
    },
    'departitioned': {
        'title': 'Departitioned',
        'strategy': (
            'The departitioned architecture wraps each biological process in a single '
            '<strong>DepartitionedStep</strong> that calls <code>_do_update()</code> — '
            'which runs <code>calculate_request()</code> followed by '
            '<code>evolve_state()</code> in sequence, with the requested molecules '
            'immediately applied to a local bulk copy. There is <em>no allocator</em> '
            'and <em>no fairness mechanism</em>. Each process gets exactly what it asks '
            'for, limited only by what remains after earlier processes have run.'
        ),
        'layers': (
            'Processes execute <strong>sequentially</strong> in the same order as '
            'the partitioned model, but without the request-allocate-evolve split. '
            'Earlier processes effectively have higher priority by virtue of running '
            'first and seeing the full molecular pool. '
            '<strong>Evolve-only optimization:</strong> RNA Maturation and Complexation '
            'skip <code>calculate_request()</code> entirely after the first timestep, '
            'since their <code>evolve_state()</code> recomputes everything independently.'
        ),
        'tradeoffs': (
            'Simpler execution graph (41 steps). No allocation overhead. '
            'But no scarcity management — diverges from baseline by ~27% in water mass '
            'over a full cell cycle because different metabolite allocations cascade '
            'through metabolism (FBA) every timestep. The cell never reaches division '
            'threshold within 42 minutes.'
        ),
    },
    'reconciled': {
        'title': 'Reconciled',
        'strategy': (
            'The reconciled architecture groups processes by allocator layer into '
            '<strong>ReconciledStep</strong> instances — one step per layer. '
            'For example, Layer 2 in the baseline needs 13 steps for 6 processes '
            '(6 Requesters + 1 Allocator + 6 Evolvers); here it is a single '
            'ReconciledStep. Each ReconciledStep implements the reconcile pattern from '
            'bigraph-schema: it collects <code>calculate_request()</code> outputs from '
            'all processes in the layer, then <strong>reconciles</strong> the requests '
            'against available supply using proportional scaling (same algorithm as the '
            'Allocator but without priority levels). Finally, it runs '
            '<code>evolve_state()</code> for each process with its reconciled allocation.'
        ),
        'layers': (
            'Same 3-layer grouping as baseline: Layer 1 (Equilibrium, Two-Component System), '
            'Layer 2 (Protein Degradation, RNA Degradation, Transcript Initiation, '
            'Polypeptide Initiation, Chromosome Replication), '
            'Layer 3 (Transcript Elongation, Polypeptide Elongation). '
            '<strong>Evolve-only:</strong> RNA Maturation and Complexation skip the '
            'request phase and run via <code>_do_update()</code> with full bulk state. '
            'A <strong>bulk delta clamping</strong> step prevents negative molecule counts '
            'when combined deltas from multiple evolvers overdraw a molecule.'
        ),
        'tradeoffs': (
            'Fewest steps (33). Proportional allocation preserves fairness — '
            'diverges only ~4.5% from baseline (vs 27% departitioned). '
            'The cell reaches division within the same cell cycle. '
            'Remaining gap comes from: (a) no priority levels (all processes equal), '
            '(b) evolve-only processes not participating in reconciliation, '
            '(c) different stochastic remainder distribution.'
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def _style_ax(ax):
    ax.grid(True, alpha=0.15); ax.tick_params(labelsize=8)

def _get_emitter(composite):
    cell = composite.state.get('agents', {}).get('0', {})
    em = cell.get('emitter', {})
    return em.get('instance') if isinstance(em, dict) and 'instance' in em else None

def _extract_snapshots(emitter):
    if emitter is None or not hasattr(emitter, 'history'):
        return []
    snaps = []
    for snap in emitter.history:
        t = snap.get('global_time', 0)
        mass = (snap.get('listeners', {}).get('mass', {})
                if isinstance(snap.get('listeners'), dict) else {})
        row = {'time': float(t)}
        for f in MASS_FIELDS:
            row[f] = float(mass.get(f, 0))
        row['growth_rate'] = float(mass.get('instantaneous_growth_rate', 0))
        row['volume'] = float(mass.get('volume', 0))
        snaps.append(row)
    return snaps

def _extract_bulk(composite):
    cell = composite.state.get('agents', {}).get('0', {})
    bulk = cell.get('bulk')
    if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
        return bulk['count'].astype(float)
    return np.array([])

def _extract_bulk_ids(composite):
    cell = composite.state.get('agents', {}).get('0', {})
    bulk = cell.get('bulk')
    if bulk is not None and hasattr(bulk, 'dtype') and 'id' in bulk.dtype.names:
        return [str(x) for x in bulk['id']]
    return []


# ---------------------------------------------------------------------------
# Bigraph-viz & PBG JSON
# ---------------------------------------------------------------------------

BIO_COLORS = {
    'dna': ('#FFB6C1', lambda n: 'chromosome' in n),
    'rna': ('#ADD8E6', lambda n: any(s in n for s in ('transcript', 'rna-', 'rna_', 'RNA', 'rnap'))),
    'protein': ('#90EE90', lambda n: any(s in n for s in ('polypeptide', 'protein', 'ribosome'))),
    'meta': ('#FFD700', lambda n: any(s in n for s in ('metabolism', 'equilibrium', 'complexation', 'two-component'))),
    'reg': ('#DDA0DD', lambda n: any(s in n for s in ('tf-', 'tf_'))),
    'alloc': ('#FFA07A', lambda n: any(s in n for s in ('allocator', 'reconciled'))),
    'infra': ('#E0E0E0', lambda n: any(s in n for s in ('unique_update', 'global_clock', 'emitter',
                                                          'mark_d_period', 'division', 'exchange',
                                                          'media_update', 'post-division'))),
    'listen': ('#D3D3D3', lambda n: 'listener' in n),
}


def _make_bigraph_svg(composite):
    """Generate a bigraph-viz SVG string from a composite."""
    try:
        from bigraph_viz import plot_bigraph
    except ImportError:
        return None

    cell = composite.state.get('agents', {}).get('0', composite.state)
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict):
            continue
        if '_type' in edge:
            inputs = {p: w for p, w in edge.get('inputs', {}).items()
                      if not p.startswith('_layer') and not p.startswith('_flow')}
            outputs = {p: w for p, w in edge.get('outputs', {}).items()
                       if not p.startswith('_layer') and not p.startswith('_flow')}
            clean = name.replace('ecoli-', '')
            viz[clean] = {'_type': edge['_type'], 'inputs': inputs, 'outputs': outputs}
        elif name == 'unique' and isinstance(edge, dict):
            viz[name] = {k: {} for k in edge.keys()}
        elif name in ('bulk', 'listeners', 'environment', 'boundary',
                       'request', 'allocate', 'process_state'):
            viz[name] = {}

    viz_state = {'agents': {'0': viz}}
    prefix = ('agents', '0')
    colors, groups = {}, {k: [] for k in BIO_COLORS}
    for n in viz:
        if '_type' not in viz.get(n, {}):
            continue
        p = prefix + (n,)
        for gk, (c, m) in BIO_COLORS.items():
            if m(n):
                colors[p] = c
                groups[gk].append(p)
                break

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            plot_bigraph(viz_state, remove_process_place_edges=True,
                         node_groups=[g for g in groups.values() if g],
                         node_fill_colors=colors, rankdir='LR',
                         dpi='72', port_labels=False, node_label_size='14pt',
                         label_margin='0.05', out_dir=td,
                         filename='bg', file_format='svg')
            with open(os.path.join(td, 'bg.svg')) as f:
                svg = f.read()
            svg = re.sub(r'width="[^"]*pt"', '', svg, count=1)
            svg = re.sub(r'height="[^"]*pt"', '', svg, count=1)
            return svg
        except Exception as e:
            return f'<p style="color:#999">bigraph-viz: {html_lib.escape(str(e))}</p>'


def _serialize_state_json(composite):
    """Serialize composite state to JSON-safe dict for the interactive viewer."""
    try:
        cell = composite.state.get('agents', {}).get('0', {})

        # Build a clean state dict excluding unserializable objects
        clean = {}
        for key, val in cell.items():
            if isinstance(val, dict) and '_type' in val:
                # Process/step edge — extract config, inputs, outputs
                edge = {
                    '_type': val.get('_type'),
                    'address': val.get('address', ''),
                    'config': val.get('config', {}),
                    'inputs': val.get('inputs', {}),
                    'outputs': val.get('outputs', {}),
                    'priority': val.get('priority'),
                }
                clean[key] = edge
            elif key == 'bulk' and hasattr(val, 'dtype'):
                # Structured numpy array — show first 20 molecules
                ids = [str(x) for x in val['id'][:20]]
                cnts = [int(x) for x in val['count'][:20]]
                clean[key] = {
                    '_note': f'{len(val)} bulk molecules (showing first 20)',
                    'sample': {i: c for i, c in zip(ids, cnts)},
                }
            elif key == 'unique' and isinstance(val, dict):
                clean[key] = {
                    k: {'n_entries': int(v['_entryState'].sum()) if hasattr(v, 'dtype') and '_entryState' in v.dtype.names else len(v) if hasattr(v, '__len__') else '?'}
                    for k, v in val.items()
                }
            elif key == 'listeners' and isinstance(val, dict):
                clean[key] = {}
                for lk, lv in val.items():
                    if isinstance(lv, dict):
                        clean[key][lk] = {
                            k: float(v) if isinstance(v, (int, float)) else
                            f'array[{len(v)}]' if hasattr(v, '__len__') else str(v)
                            for k, v in lv.items()
                        }
                    else:
                        clean[key][lk] = str(type(lv).__name__)
            elif key in ('global_time', 'timestep', 'divide',
                         'division_threshold'):
                clean[key] = val if isinstance(val, (int, float, str, bool)) else str(val)
            elif key == 'next_update_time' and isinstance(val, dict):
                clean[key] = {k: float(v) for k, v in val.items()}
            elif key in ('environment', 'boundary') and isinstance(val, dict):
                clean[key] = {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in val.items()
                }

        result = {'agents': {'0': clean}}

        # JSON-safe pass
        result = json.loads(
            json.dumps(result, default=lambda x: str(x))
            .replace('Infinity', 'null')
            .replace('-Infinity', 'null')
            .replace('NaN', 'null')
        )
        return result
    except Exception as e:
        return {'_error': str(e)}


# ---------------------------------------------------------------------------
# Parallel simulation runner
# ---------------------------------------------------------------------------

def _run_one_model(args):
    """Run a single model in a subprocess."""
    model_key, cache_dir, seed, duration, snapshot_interval = args
    import warnings; warnings.filterwarnings('ignore')

    if model_key == 'baseline':
        from v2ecoli.composite import make_composite
        composite = make_composite(cache_dir=cache_dir, seed=seed)
    elif model_key == 'departitioned':
        from v2ecoli.composite_departitioned import make_departitioned_composite
        composite = make_departitioned_composite(cache_dir=cache_dir, seed=seed)
    elif model_key == 'reconciled':
        from v2ecoli.composite_reconciled import make_reconciled_composite
        composite = make_reconciled_composite(cache_dir=cache_dir, seed=seed)
    else:
        raise ValueError(f'Unknown model: {model_key}')

    n_steps = len(composite.step_paths)
    label = MODELS[model_key]['label']

    # Generate bigraph SVG and state JSON before running
    svg = _make_bigraph_svg(composite)
    state_json = _serialize_state_json(composite)

    emitter = _get_emitter(composite)

    t0 = time.time()
    total_run = 0.0
    while total_run < duration:
        chunk = min(snapshot_interval, duration - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            print(f'    [{label}] Error at ~t={total_run + chunk:.0f}s: '
                  f'{type(e).__name__}: {e}')
            break
        total_run += chunk
        cell = composite.state.get('agents', {}).get('0')
        if cell is None:
            print(f'    [{label}] Division at t={total_run}s')
            break
        mass = cell.get('listeners', {}).get('mass', {})
        dm = float(mass.get('dry_mass', 0))
        if total_run % 100 < snapshot_interval:
            print(f'    [{label}] t={total_run:.0f}s: dry_mass={dm:.0f}fg')

    wall_time = time.time() - t0
    print(f'  {label}: {wall_time:.1f}s wall, {total_run:.0f}s sim, {n_steps} steps')

    snaps = _extract_snapshots(emitter)
    bulk = _extract_bulk(composite)
    bulk_ids = _extract_bulk_ids(composite)

    return {
        'key': model_key,
        'wall_time': wall_time,
        'sim_time': total_run,
        'n_steps': n_steps,
        'snaps': snaps,
        'bulk': bulk,
        'bulk_ids': bulk_ids,
        'svg': svg,
        'state_json': state_json,
    }


def run_all_parallel(cache_dir, seed, duration, snapshot_interval):
    print(f'\nStep 2: Running 3 simulations in parallel ({duration}s each)...')
    t0 = time.time()
    args_list = [(k, cache_dir, seed, duration, snapshot_interval) for k in MODELS]
    ctx = mp.get_context('spawn')
    with ctx.Pool(3) as pool:
        results = pool.map(_run_one_model, args_list)
    total_wall = time.time() - t0
    print(f'  All 3 finished in {total_wall:.1f}s wall (parallel)')
    return {r['key']: r for r in results}, total_wall


def run_all_sequential(cache_dir, seed, duration, snapshot_interval):
    print(f'\nStep 2: Running 3 simulations sequentially ({duration}s each)...')
    t0 = time.time()
    sim_data = {}
    for key in MODELS:
        r = _run_one_model((key, cache_dir, seed, duration, snapshot_interval))
        sim_data[r['key']] = r
    total_wall = time.time() - t0
    print(f'  All 3 finished in {total_wall:.1f}s wall (sequential)')
    return sim_data, total_wall


# ---------------------------------------------------------------------------
# Metrics: n-way molecule divergence
# ---------------------------------------------------------------------------

def compute_metrics(sim_data):
    print('\nStep 3: Computing metrics...')
    base_snaps = sim_data['baseline']['snaps']
    base_by_t = {int(s['time']): s for s in base_snaps}
    base_bulk = sim_data['baseline']['bulk']
    base_ids = sim_data['baseline']['bulk_ids']

    per_model = {}
    for key in ['departitioned', 'reconciled']:
        snaps = sim_data[key]['snaps']
        by_t = {int(s['time']): s for s in snaps}
        common = sorted(set(base_by_t.keys()) & set(by_t.keys()))
        pct_diff = {f: [] for f in MASS_FIELDS}
        for t in common:
            p, d = base_by_t[t], by_t[t]
            for f in MASS_FIELDS:
                pv, dv = p.get(f, 0), d.get(f, 0)
                ref = max(abs(pv), abs(dv), 1e-12)
                pct_diff[f].append(abs(pv - dv) / ref * 100)
        max_errors = {f: max(v) if v else 0 for f, v in pct_diff.items()}
        overall = max(max_errors.values()) if max_errors else 0
        per_model[key] = {
            'pct_times': common, 'pct_diff': pct_diff,
            'max_errors': max_errors, 'overall_max_error': overall,
        }
        label = MODELS[key]['label']
        print(f'  {label}: max_err={overall:.4f}%')

    # N-way molecule divergence: one table comparing all to baseline
    n_mols = len(base_bulk)
    mol_table = []
    for i in range(n_mols):
        name = base_ids[i] if i < len(base_ids) else str(i)
        bv = base_bulk[i]
        row = {'idx': i, 'name': name, 'baseline': bv}
        max_diff = 0
        for key in ['departitioned', 'reconciled']:
            other_bulk = sim_data[key]['bulk']
            ov = other_bulk[i] if i < len(other_bulk) else 0
            diff = ov - bv
            row[key] = ov
            row[f'{key}_diff'] = diff
            max_diff = max(max_diff, abs(diff))
        row['max_abs_diff'] = max_diff
        mol_table.append(row)

    mol_table.sort(key=lambda x: x['max_abs_diff'], reverse=True)
    n_div = sum(1 for m in mol_table if m['max_abs_diff'] > 0)
    print(f'  Molecules with any divergence: {n_div}/{n_mols}')

    return {'per_model': per_model, 'mol_table': mol_table, 'n_mols': n_mols}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mass_trajectories(sim_data):
    fields = ['dry_mass', 'protein_mass', 'rna_mass',
              'dna_mass', 'smallMolecule_mass', 'cell_mass']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, f in enumerate(fields):
        ax = axes.ravel()[i]
        for key, m in MODELS.items():
            snaps = sim_data[key]['snaps']
            if snaps:
                ax.plot([s['time']/60 for s in snaps], [s.get(f, 0) for s in snaps],
                        m['ls'], color=m['color'], lw=1.5, label=m['short'], alpha=0.8)
        ax.set_title(MASS_LABELS.get(f, f), fontsize=10)
        ax.set_xlabel('Time (min)', fontsize=8); ax.set_ylabel('Mass (fg)', fontsize=8)
        ax.legend(fontsize=7); _style_ax(ax)
    fig.suptitle('Mass Trajectories — Three Architectures', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mass_divergence(metrics):
    fields = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']
    fig, axes = plt.subplots(1, len(fields), figsize=(3.6*len(fields), 3.5))
    for i, f in enumerate(fields):
        ax = axes[i]
        for key in ['departitioned', 'reconciled']:
            m = MODELS[key]; pm = metrics['per_model'][key]
            times = np.array(pm['pct_times']) / 60
            vals = pm['pct_diff'].get(f, [])
            if vals and len(times) == len(vals):
                ax.plot(times, vals, m['ls'], color=m['color'], lw=1.2, label=m['short'])
        ax.set_title(MASS_LABELS.get(f, f), fontsize=9)
        ax.set_xlabel('Time (min)', fontsize=7); ax.set_ylabel('% Diff vs Baseline', fontsize=7)
        ax.legend(fontsize=6); _style_ax(ax)
    fig.suptitle('Mass Divergence vs Baseline', fontsize=11, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_growth(sim_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, m in MODELS.items():
        snaps = sim_data[key]['snaps']
        if not snaps: continue
        t = [s['time']/60 for s in snaps]
        axes[0].plot(t, [s.get('growth_rate', 0)*3600 for s in snaps],
                     m['ls'], color=m['color'], lw=1.2, label=m['short'])
        axes[1].plot(t, [s.get('volume', 0) for s in snaps],
                     m['ls'], color=m['color'], lw=1.2, label=m['short'])
    axes[0].set_ylabel('Growth rate (1/h)'); axes[0].set_title('Growth Rate')
    axes[1].set_ylabel('Volume (fL)'); axes[1].set_title('Cell Volume')
    for ax in axes:
        ax.set_xlabel('Time (min)'); ax.legend(fontsize=8); _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_timing(sim_data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    keys = list(MODELS.keys())
    colors = [MODELS[k]['color'] for k in keys]
    labels = [MODELS[k]['short'] for k in keys]
    ax = axes[0]
    ax.bar(labels, [sim_data[k]['wall_time'] for k in keys], color=colors, alpha=0.8)
    ax.set_ylabel('Wall Time (s)'); ax.set_title('Simulation Wall Time'); _style_ax(ax)
    ax = axes[1]
    ax.bar(labels, [sim_data[k]['n_steps'] for k in keys], color=colors, alpha=0.8)
    ax.set_ylabel('# Steps'); ax.set_title('Steps per Timestep'); _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_nway_divergence(metrics, top_n=30):
    """Single figure: top diverging molecules with distance from baseline for each architecture."""
    mol_table = metrics['mol_table']
    top = [m for m in mol_table[:top_n] if m['max_abs_diff'] > 0]
    if not top:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, 'No divergence', ha='center', va='center', transform=ax.transAxes)
        return fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(16, max(6, len(top) * 0.4)))
    y = np.arange(len(top))
    bar_h = 0.35
    names = [m['name'][:45] for m in top]

    for j, key in enumerate(['departitioned', 'reconciled']):
        diffs = [m[f'{key}_diff'] for m in top]
        offset = -bar_h/2 + j * bar_h
        color = MODELS[key]['color']
        ax.barh(y + offset, diffs, height=bar_h, color=color, alpha=0.75,
                label=MODELS[key]['short'])

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Count Difference from Baseline (signed)')
    ax.set_title(f'Top {len(top)} Diverging Molecules: Distance from Baseline')
    ax.axvline(x=0, color='black', lw=0.8, alpha=0.5)
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _json_viewer_html(state_json, viewer_id):
    """Build the interactive JSON viewer block."""
    if not state_json:
        return ''
    blob = json.dumps(state_json, ensure_ascii=False, default=str)
    return f"""
      <div class="json-viewer" data-test="{viewer_id}">
        <div class="json-toolbar">
          <input class="json-search" placeholder="Search keys..." />
          <button type="button" class="json-reset">Top-level</button>
          <span class="json-status"></span>
        </div>
        <div class="json-layout">
          <div class="json-nav"></div>
          <div class="json-main">
            <div class="json-path"></div>
            <div class="json-value"></div>
          </div>
        </div>
        <script type="application/json" id="json-data-{viewer_id}">
{blob}
        </script>
      </div>"""


def _json_viewer_js():
    """Inline JS for interactive JSON navigation (from pymunk-process pattern)."""
    return r"""<script>
(function(){
  function isObj(x){ return x && typeof x === "object" && !Array.isArray(x); }
  function getAt(root, path){
    let c = root;
    for (const p of path){ if (c==null) return undefined; c = Array.isArray(c) ? c[Number(p)] : c[p]; }
    return c;
  }
  function renderVal(el, v){
    el.innerHTML = "";
    const pre = document.createElement("pre");
    if (v === null || typeof v !== "object"){ pre.textContent = JSON.stringify(v, null, 2); }
    else if (Array.isArray(v)){
      el.innerHTML = `<span class="json-pill">array [${v.length}]</span>`;
      pre.textContent = JSON.stringify(v, null, 2);
    } else {
      const keys = Object.keys(v);
      el.innerHTML = `<span class="json-pill">object {${keys.length} keys}</span>`;
      pre.textContent = JSON.stringify(v, null, 2);
    }
    el.appendChild(pre);
  }
  document.querySelectorAll(".json-viewer").forEach(viewer => {
    const testId = viewer.dataset.test;
    const dataEl = document.getElementById("json-data-" + testId);
    if (!dataEl) return;
    const root = JSON.parse(dataEl.textContent);
    const nav = viewer.querySelector(".json-nav");
    const pathEl = viewer.querySelector(".json-path");
    const valEl = viewer.querySelector(".json-value");
    const search = viewer.querySelector(".json-search");
    const resetBtn = viewer.querySelector(".json-reset");
    const status = viewer.querySelector(".json-status");
    function showKeys(obj, basePath){
      nav.innerHTML = "";
      const keys = Object.keys(obj);
      status.textContent = keys.length + " keys";
      keys.forEach(k => {
        const item = document.createElement("div");
        item.className = "json-item";
        const v = obj[k];
        const hint = v === null ? "null" : Array.isArray(v) ? `[${v.length}]` : typeof v === "object" ? "{...}" : JSON.stringify(v).slice(0,30);
        item.textContent = k + "  " + hint;
        item.onclick = () => {
          nav.querySelectorAll(".json-item").forEach(x => x.classList.remove("active"));
          item.classList.add("active");
          const path = basePath.concat([k]);
          pathEl.textContent = path.join(".");
          const val = getAt(root, path);
          if (isObj(val) && Object.keys(val).length > 0){ showKeys(val, path); }
          else { renderVal(valEl, val); }
        };
        nav.appendChild(item);
      });
    }
    function reset(){ pathEl.textContent = ""; valEl.innerHTML = ""; showKeys(root, []); }
    resetBtn.onclick = reset;
    search.oninput = () => {
      const q = search.value.toLowerCase().trim();
      if (!q){ reset(); return; }
      nav.querySelectorAll(".json-item").forEach(el => {
        el.style.display = el.textContent.toLowerCase().includes(q) ? "" : "none";
      });
    };
    reset();
  });
})();
</script>"""


def _nway_mol_table_html(mol_table, top_n=40):
    """Build n-way molecule divergence table rows."""
    rows = ''
    for i, m in enumerate(mol_table[:top_n]):
        if m['max_abs_diff'] == 0: continue
        dep_d = m['departitioned_diff']
        rec_d = m['reconciled_diff']
        dep_c = '#dc2626' if dep_d > 0 else '#2563eb' if dep_d < 0 else '#999'
        rec_c = '#16a34a' if abs(rec_d) < abs(dep_d) else '#ea580c'
        rows += (
            f'<tr><td>{i+1}</td>'
            f'<td style="font-family:monospace;font-size:0.8em">{m["name"]}</td>'
            f'<td style="text-align:right">{m["baseline"]:.0f}</td>'
            f'<td style="text-align:right;color:{dep_c}">{dep_d:+.0f}</td>'
            f'<td style="text-align:right;color:{rec_c}">{rec_d:+.0f}</td>'
            f'<td style="text-align:right">{m["max_abs_diff"]:.0f}</td>'
            f'</tr>'
        )
    return rows


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html(sim_data, metrics, plots, seed, duration, output, parallel_wall):
    print(f'\nStep 5: Generating HTML report -> {output}')
    date_str = time.strftime('%Y-%m-%d %H:%M')

    dep_err = metrics['per_model']['departitioned']['overall_max_error']
    rec_err = metrics['per_model']['reconciled']['overall_max_error']
    n_div = sum(1 for m in metrics['mol_table'] if m['max_abs_diff'] > 0)

    # Per-field error table
    field_rows = ''
    for f in MASS_FIELDS:
        de = metrics['per_model']['departitioned']['max_errors'].get(f, 0)
        re_ = metrics['per_model']['reconciled']['max_errors'].get(f, 0)
        dc = '#16a34a' if de < ERROR_THRESHOLD else '#dc2626'
        rc = '#16a34a' if re_ < ERROR_THRESHOLD else '#dc2626'
        field_rows += (f'<tr><td>{MASS_LABELS.get(f, f)}</td>'
                       f'<td style="color:{dc};text-align:right">{de:.4f}%</td>'
                       f'<td style="color:{rc};text-align:right">{re_:.4f}%</td></tr>')

    # Architecture sections with SVG + JSON
    arch_sections = ''
    for key in MODELS:
        ad = ARCH_DESCRIPTIONS[key]
        m = MODELS[key]
        svg_html = ''
        if sim_data[key].get('svg'):
            svg_html = (
                f'<details open style="margin:10px 0"><summary style="cursor:pointer;font-size:0.85em;color:#2563eb">'
                f'Composition Diagram</summary>'
                f'<div class="viz-wrap"><div class="viz-zoom">'
                f'<button onclick="vizZoom(this,-1)" title="Zoom out">&minus;</button>'
                f'<button onclick="vizZoom(this,1)" title="Zoom in">+</button>'
                f'<button onclick="vizZoom(this,0)" title="Reset">&#8634;</button>'
                f'</div><div class="viz-box">{sim_data[key]["svg"]}</div></div></details>'
            )
        json_html = _json_viewer_html(sim_data[key].get('state_json'), f'state_{key}')
        arch_sections += f"""
<div class="section" style="border-top:4px solid {m['color']}">
  <h3 style="color:{m['color']};margin-bottom:6px">{ad['title']}
    <span style="font-size:0.7em;color:#64748b;font-weight:normal;margin-left:8px">
    {sim_data[key]['n_steps']} steps &middot; {sim_data[key]['wall_time']:.1f}s wall</span></h3>
  <p style="font-size:0.88em;margin-bottom:10px"><strong>Strategy:</strong> {ad['strategy']}</p>
  <p style="font-size:0.85em;color:#475569;margin-bottom:8px"><strong>Layers:</strong> {ad['layers']}</p>
  <p style="font-size:0.85em;color:#64748b;margin-bottom:10px"><strong>Trade-offs:</strong> {ad['tradeoffs']}</p>
  {svg_html}
  <details style="margin-top:8px"><summary style="cursor:pointer;font-size:0.85em;color:#2563eb">
    Initial State (interactive JSON viewer)</summary>{json_html}</details>
</div>"""

    # N-way molecule table
    mol_rows = _nway_mol_table_html(metrics['mol_table'])

    improvement = dep_err / max(rec_err, 0.001)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>v2ecoli Architecture Comparison</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
max-width:1500px;margin:0 auto;padding:20px;background:#f8fafc;color:#1e293b}}
h1{{font-size:1.8em;margin:15px 0;color:#0f172a}}
h2{{font-size:1.3em;margin:25px 0 10px;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
.header{{background:#0f172a;color:white;padding:20px 25px;border-radius:10px;margin-bottom:15px}}
.header h1{{color:white;margin:0 0 6px}}.header p{{color:#94a3b8;font-size:0.9em}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:15px 0}}
.card{{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
.card .label{{font-size:0.72em;color:#64748b;text-transform:uppercase;letter-spacing:0.05em}}
.card .value{{font-size:1.4em;font-weight:700;margin-top:4px}}
.blue{{color:#2563eb}}.red{{color:#dc2626}}.green{{color:#16a34a}}.purple{{color:#7c3aed}}.orange{{color:#ea580c}}
.plot{{background:white;border-radius:8px;padding:14px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center}}
.plot img{{max-width:100%}}
.section{{background:white;border-radius:8px;padding:18px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
table{{border-collapse:collapse;width:100%;font-size:0.85em}}
th,td{{border:1px solid #e2e8f0;padding:6px 10px;text-align:left}}
th{{background:#f1f5f9;font-weight:600}}
.verdict{{border-radius:10px;padding:20px 25px;margin:15px 0;font-size:1.05em}}
.json-viewer{{border:1px solid #ddd;background:#fff;border-radius:8px;padding:10px;margin:10px 0}}
.json-toolbar{{display:flex;gap:8px;align-items:center;margin-bottom:8px}}
.json-toolbar input{{flex:1;padding:6px 10px;border:1px solid #ccc;border-radius:6px}}
.json-toolbar button{{padding:6px 10px;border:1px solid #ccc;border-radius:6px;background:#f5f5f5;cursor:pointer}}
.json-toolbar button:hover{{background:#eee}}
.json-status{{font-size:12px;color:#555}}
.json-layout{{display:grid;grid-template-columns:280px 1fr;gap:10px;height:400px}}
.json-nav{{overflow:auto;border-right:1px solid #eee;padding-right:8px}}
.json-main{{overflow:auto;padding-left:6px}}
.json-item{{padding:5px 8px;border-radius:6px;cursor:pointer;font-family:ui-monospace,monospace;font-size:12px}}
.json-item:hover{{background:#f3f3f3}}
.json-item.active{{background:#e9eefc}}
.json-path{{font-family:ui-monospace,monospace;font-size:12px;margin-bottom:8px;color:#333}}
.json-value pre{{background:#f8f8f8;border:1px solid #eee;border-radius:8px;padding:10px;overflow:auto;font-size:12px}}
.json-pill{{display:inline-block;padding:2px 8px;border:1px solid #ddd;border-radius:999px;font-size:12px;margin-right:6px;background:#fafafa}}
.viz-wrap{{position:relative}}
.viz-box{{max-height:350px;overflow:auto;border:1px solid #e2e8f0;border-radius:8px;
padding:8px;margin:6px 0;resize:vertical;background:#fafafa;text-align:center}}
.viz-box svg{{height:auto;transform-origin:top left;transition:transform 0.15s}}
.viz-zoom{{position:absolute;top:8px;right:12px;display:flex;gap:4px;z-index:2}}
.viz-zoom button{{width:28px;height:28px;border:1px solid #ccc;border-radius:6px;
background:#fff;cursor:pointer;font-size:16px;line-height:1}}
footer{{margin-top:30px;padding:15px 0;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.75em;text-align:center}}
</style></head><body>

<div class="header">
<h1>v2ecoli Architecture Comparison</h1>
<p>{date_str} &middot; Duration: {duration}s ({duration/60:.1f} min) &middot; Seed: {seed}
 &middot; 3 simulations in parallel &middot; Total wall: {parallel_wall:.0f}s</p>
</div>

<h2>Summary</h2>
<div class="cards">
<div class="card"><div class="label">Baseline Steps</div>
<div class="value blue">{sim_data['baseline']['n_steps']}</div></div>
<div class="card"><div class="label">Departitioned Steps</div>
<div class="value red">{sim_data['departitioned']['n_steps']}</div></div>
<div class="card"><div class="label">Reconciled Steps</div>
<div class="value green">{sim_data['reconciled']['n_steps']}</div></div>
<div class="card"><div class="label">Baseline Wall</div>
<div class="value blue">{sim_data['baseline']['wall_time']:.1f}s</div></div>
<div class="card"><div class="label">Departitioned Wall</div>
<div class="value red">{sim_data['departitioned']['wall_time']:.1f}s</div></div>
<div class="card"><div class="label">Reconciled Wall</div>
<div class="value green">{sim_data['reconciled']['wall_time']:.1f}s</div></div>
<div class="card"><div class="label">Departitioned Max Error</div>
<div class="value {'green' if dep_err < ERROR_THRESHOLD else 'red'}">{dep_err:.2f}%</div></div>
<div class="card"><div class="label">Reconciled Max Error</div>
<div class="value {'green' if rec_err < ERROR_THRESHOLD else 'red'}">{rec_err:.2f}%</div></div>
<div class="card"><div class="label">Molecules Diverged</div>
<div class="value orange">{n_div} / {metrics['n_mols']}</div></div>
<div class="card"><div class="label">Reconciled Improvement</div>
<div class="value green">{improvement:.1f}x</div></div>
</div>

<h2>Architecture Descriptions</h2>
{arch_sections}

<h2>Mass Trajectories</h2>
<div class="plot"><img src="data:image/png;base64,{plots['mass_traj']}" alt="Mass"></div>

<h2>Mass Divergence vs Baseline</h2>
<div class="section"><table>
<thead><tr><th>Component</th><th style="text-align:right">Departitioned Max %</th>
<th style="text-align:right">Reconciled Max %</th></tr></thead>
<tbody>{field_rows}</tbody></table></div>
<div class="plot"><img src="data:image/png;base64,{plots['mass_div']}" alt="Divergence"></div>

<h2>Growth &amp; Volume</h2>
<div class="plot"><img src="data:image/png;base64,{plots['growth']}" alt="Growth"></div>

<h2>Top Diverging Molecules (N-Way vs Baseline)</h2>
<p style="font-size:0.85em;color:#64748b;margin:8px 0">
Signed count differences from the baseline for each architecture. Red/green bars show
how far each model deviates. When the reconciled bar is shorter than the departitioned bar,
reconciliation reduced the divergence for that molecule.</p>
<div class="plot"><img src="data:image/png;base64,{plots['nway_div']}" alt="N-way"></div>
<div class="section" style="max-height:500px;overflow-y:auto"><table>
<thead><tr><th>#</th><th>Molecule</th><th style="text-align:right">Baseline</th>
<th style="text-align:right">Departitioned &Delta;</th>
<th style="text-align:right">Reconciled &Delta;</th>
<th style="text-align:right">Max |&Delta;|</th></tr></thead>
<tbody>{mol_rows}</tbody></table></div>

<h2>Timing</h2>
<div class="plot"><img src="data:image/png;base64,{plots['timing']}" alt="Timing"></div>

<div class="verdict" style="background:#f0f9ff;border:2px solid #2563eb">
<strong>Reconciled closes the gap:</strong> {rec_err:.2f}% max error vs
departitioned {dep_err:.2f}% ({improvement:.1f}x improvement).
{sim_data['reconciled']['n_steps']} steps vs {sim_data['baseline']['n_steps']} baseline
({100*(1-sim_data['reconciled']['n_steps']/sim_data['baseline']['n_steps']):.0f}% fewer).
</div>

<footer>v2ecoli Architecture Comparison &middot; {date_str} &middot;
Duration: {duration}s &middot; Seed {seed}</footer>

<script>
function vizZoom(btn, dir) {{
  const box = btn.closest('.viz-wrap').querySelector('.viz-box');
  const svg = box.querySelector('svg');
  if (!svg) return;
  let scale = parseFloat(svg.dataset.scale || '1');
  if (dir === 0) scale = 1;
  else scale = Math.max(0.3, Math.min(4, scale + dir * 0.3));
  svg.dataset.scale = scale;
  svg.style.transform = 'scale(' + scale + ')';
  svg.style.maxWidth = dir === 0 ? '100%' : 'none';
}}
</script>
{_json_viewer_js()}
</body></html>""")

    print(f'  Report written to {output}')
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(duration=DEFAULT_DURATION, seed=0, cache_dir='out/cache',
                   output='out/comparison_report.html',
                   snapshot_interval=SNAPSHOT_INTERVAL, parallel=True):
    print(f'=== v2ecoli Architecture Comparison ===')
    print(f'    Duration: {duration}s, Seed: {seed}, Cache: {cache_dir}')
    print(f'    Parallel: {parallel}\n')

    if parallel:
        sim_data, par_wall = run_all_parallel(cache_dir, seed, duration, snapshot_interval)
    else:
        sim_data, par_wall = run_all_sequential(cache_dir, seed, duration, snapshot_interval)

    metrics = compute_metrics(sim_data)

    print('\nStep 4: Generating plots...')
    plots = {
        'mass_traj': plot_mass_trajectories(sim_data),
        'mass_div': plot_mass_divergence(metrics),
        'growth': plot_growth(sim_data),
        'timing': plot_timing(sim_data),
        'nway_div': plot_nway_divergence(metrics),
    }

    report = generate_html(sim_data, metrics, plots, seed, duration, output, par_wall)

    # Save PBG state JSON files to docs/ for GitHub Pages
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    for key in MODELS:
        state_json = sim_data[key].get('state_json')
        if state_json:
            pbg_path = os.path.join(docs_dir, f'{key}_state.json')
            with open(pbg_path, 'w') as f:
                json.dump(state_json, f, indent=2, default=str)
            print(f'  PBG state: {pbg_path}')

    # Copy report to docs/
    import shutil
    docs_report = os.path.join(docs_dir, 'comparison_report.html')
    shutil.copy2(output, docs_report)
    print(f'  Docs copy: {docs_report}')

    print(f'\n=== Done. Report: {report} ===')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='v2ecoli architecture comparison')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-dir', default='out/cache')
    parser.add_argument('--output', default='out/comparison_report.html')
    parser.add_argument('--snapshot-interval', type=int, default=SNAPSHOT_INTERVAL)
    parser.add_argument('--no-parallel', action='store_true')
    args = parser.parse_args()
    run_comparison(duration=args.duration, seed=args.seed, cache_dir=args.cache_dir,
                   output=args.output, snapshot_interval=args.snapshot_interval,
                   parallel=not args.no_parallel)
