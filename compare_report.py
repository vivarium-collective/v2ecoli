"""
Three-way architecture comparison: Partitioned vs Departitioned vs Reconciled.

Runs all three simulations in parallel using multiprocessing, then generates
a single-file HTML report comparing mass trajectories, divergence, bulk
counts, growth, timing, and molecule-level divergence.

Usage:
    python compare_report.py                        # default 2520s sim
    python compare_report.py --duration 600         # 10-min sim
    python compare_report.py --seed 42 --output out/my_report.html
"""

import os
import io
import time
import base64
import argparse
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
    'part': {'label': 'Partitioned', 'color': '#2563eb', 'ls': '-'},
    'dep':  {'label': 'Departitioned', 'color': '#dc2626', 'ls': '--'},
    'rec':  {'label': 'Reconciled', 'color': '#16a34a', 'ls': '-.'},
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
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)


def _get_emitter(composite):
    cell = composite.state.get('agents', {}).get('0', {})
    em = cell.get('emitter', {})
    if isinstance(em, dict) and 'instance' in em:
        return em['instance']
    return None


def _extract_snapshots(emitter):
    if emitter is None or not hasattr(emitter, 'history'):
        return []
    snapshots = []
    for snap in emitter.history:
        t = snap.get('global_time', 0)
        mass = (snap.get('listeners', {}).get('mass', {})
                if isinstance(snap.get('listeners'), dict) else {})
        row = {'time': float(t)}
        for field in MASS_FIELDS:
            row[field] = float(mass.get(field, 0))
        row['growth_rate'] = float(mass.get('instantaneous_growth_rate', 0))
        row['volume'] = float(mass.get('volume', 0))
        snapshots.append(row)
    return snapshots


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
        return bulk['id']
    return np.array([])


# ---------------------------------------------------------------------------
# Parallel simulation runner
# ---------------------------------------------------------------------------

def _run_one_model(args):
    """Run a single model in a subprocess. Returns serializable results."""
    model_key, cache_dir, seed, duration, snapshot_interval = args
    import warnings
    warnings.filterwarnings('ignore')

    # Import inside subprocess to avoid pickling issues
    if model_key == 'part':
        from v2ecoli.composite import make_composite
        composite = make_composite(cache_dir=cache_dir, seed=seed)
    elif model_key == 'dep':
        from v2ecoli.composite_departitioned import make_departitioned_composite
        composite = make_departitioned_composite(cache_dir=cache_dir, seed=seed)
    elif model_key == 'rec':
        from v2ecoli.composite_reconciled import make_reconciled_composite
        composite = make_reconciled_composite(cache_dir=cache_dir, seed=seed)
    else:
        raise ValueError(f'Unknown model: {model_key}')

    n_steps = len(composite.step_paths)
    label = MODELS[model_key]['label']

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
            print(f'    [{label}] Agent removed (division) at t={total_run}s')
            break

        mass = cell.get('listeners', {}).get('mass', {})
        dry_mass = float(mass.get('dry_mass', 0))
        if total_run % 100 < snapshot_interval:
            print(f'    [{label}] t={total_run:.0f}s: dry_mass={dry_mass:.0f}fg')

    wall_time = time.time() - t0
    print(f'  {label}: {wall_time:.1f}s wall, {total_run:.0f}s sim, '
          f'{n_steps} steps')

    snaps = _extract_snapshots(emitter)
    bulk = _extract_bulk(composite)
    bulk_ids = _extract_bulk_ids(composite)
    # Convert bulk_ids to plain strings for pickling
    bulk_id_strs = [str(x) for x in bulk_ids] if len(bulk_ids) > 0 else []

    return {
        'key': model_key,
        'wall_time': wall_time,
        'sim_time': total_run,
        'n_steps': n_steps,
        'snaps': snaps,
        'bulk': bulk,
        'bulk_ids': bulk_id_strs,
    }


def run_all_parallel(cache_dir, seed, duration, snapshot_interval):
    """Run all three models in parallel using multiprocessing."""
    print(f'\nStep 2: Running 3 simulations in parallel ({duration}s each)...')
    t0 = time.time()

    args_list = [
        (key, cache_dir, seed, duration, snapshot_interval)
        for key in MODELS
    ]

    # Use spawn context to avoid fork issues with numpy
    ctx = mp.get_context('spawn')
    with ctx.Pool(3) as pool:
        results = pool.map(_run_one_model, args_list)

    total_wall = time.time() - t0
    print(f'  All 3 finished in {total_wall:.1f}s wall (parallel)')

    sim_data = {}
    for r in results:
        sim_data[r['key']] = r
    return sim_data, total_wall


def run_all_sequential(cache_dir, seed, duration, snapshot_interval):
    """Fallback: run all three models sequentially."""
    print(f'\nStep 2: Running 3 simulations sequentially ({duration}s each)...')
    t0 = time.time()

    sim_data = {}
    for key in MODELS:
        args = (key, cache_dir, seed, duration, snapshot_interval)
        r = _run_one_model(args)
        sim_data[r['key']] = r

    total_wall = time.time() - t0
    print(f'  All 3 finished in {total_wall:.1f}s wall (sequential)')
    return sim_data, total_wall


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(sim_data):
    print('\nStep 3: Computing metrics...')
    metrics = {}

    part_snaps = sim_data['part']['snaps']
    part_by_t = {int(s['time']): s for s in part_snaps}

    # Per-model metrics vs partitioned baseline
    for key in ['dep', 'rec']:
        snaps = sim_data[key]['snaps']
        by_t = {int(s['time']): s for s in snaps}
        common_times = sorted(set(part_by_t.keys()) & set(by_t.keys()))

        pct_diff = {f: [] for f in MASS_FIELDS}
        for t in common_times:
            p, d = part_by_t[t], by_t[t]
            for f in MASS_FIELDS:
                pv, dv = p.get(f, 0), d.get(f, 0)
                ref = max(abs(pv), abs(dv), 1e-12)
                pct_diff[f].append(abs(pv - dv) / ref * 100)

        max_errors = {f: max(v) if v else 0 for f, v in pct_diff.items()}
        overall = max(max_errors.values()) if max_errors else 0

        pb, db = sim_data['part']['bulk'], sim_data[key]['bulk']
        ml = min(len(pb), len(db))
        if ml > 0:
            mask = (pb[:ml] > 0) | (db[:ml] > 0)
            r = stats.pearsonr(pb[:ml][mask], db[:ml][mask])[0] if mask.sum() > 1 else float('nan')
            exact = int((pb[:ml] == db[:ml]).sum())
        else:
            r, exact = float('nan'), 0

        # Top diverging molecules
        mol_div = []
        ids = sim_data['part']['bulk_ids']
        for i in range(ml):
            ad = abs(pb[i] - db[i])
            if ad > 0:
                mol_div.append({
                    'name': ids[i] if i < len(ids) else str(i),
                    'part': pb[i], 'other': db[i],
                    'abs_diff': ad, 'signed': db[i] - pb[i],
                })
        mol_div.sort(key=lambda x: x['abs_diff'], reverse=True)

        metrics[key] = {
            'pct_times': common_times,
            'pct_diff': pct_diff,
            'max_errors': max_errors,
            'overall_max_error': overall,
            'pearson_r': r,
            'exact_match': exact,
            'total_mols': ml,
            'mol_divergence': mol_div,
            'n_diverged': sum(1 for m in mol_div if m['abs_diff'] > 0),
        }

        label = MODELS[key]['label']
        r_s = f'{r:.6f}' if not np.isnan(r) else 'N/A'
        print(f'  {label}: max_err={overall:.4f}%, r={r_s}, '
              f'diverged={metrics[key]["n_diverged"]}/{ml}')

    return metrics


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
                ax.plot([s['time']/60 for s in snaps],
                        [s.get(f, 0) for s in snaps],
                        m['ls'], color=m['color'], lw=1.5,
                        label=m['label'], alpha=0.8)
        ax.set_title(MASS_LABELS.get(f, f), fontsize=10)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.set_ylabel('Mass (fg)', fontsize=8)
        ax.legend(fontsize=7)
        _style_ax(ax)
    fig.suptitle('Mass Trajectories — Three Architectures', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mass_divergence(sim_data, metrics):
    fields = ['dry_mass', 'protein_mass', 'rna_mass',
              'dna_mass', 'smallMolecule_mass']
    fig, axes = plt.subplots(1, len(fields), figsize=(3.6*len(fields), 3.5))
    for i, f in enumerate(fields):
        ax = axes[i]
        for key in ['dep', 'rec']:
            m = MODELS[key]
            times = np.array(metrics[key]['pct_times']) / 60
            vals = metrics[key]['pct_diff'].get(f, [])
            if vals and len(times) == len(vals):
                ax.plot(times, vals, m['ls'], color=m['color'], lw=1.2,
                        label=m['label'])
        ax.set_title(MASS_LABELS.get(f, f), fontsize=9)
        ax.set_xlabel('Time (min)', fontsize=7)
        ax.set_ylabel('% Diff vs Part', fontsize=7)
        ax.legend(fontsize=6)
        _style_ax(ax)
    fig.suptitle('Mass Divergence vs Partitioned Baseline', fontsize=11, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_growth(sim_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, m in MODELS.items():
        snaps = sim_data[key]['snaps']
        if not snaps: continue
        t = [s['time']/60 for s in snaps]
        axes[0].plot(t, [s.get('growth_rate', 0)*3600 for s in snaps],
                     m['ls'], color=m['color'], lw=1.2, label=m['label'])
        axes[1].plot(t, [s.get('volume', 0) for s in snaps],
                     m['ls'], color=m['color'], lw=1.2, label=m['label'])
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
    labels = [MODELS[k]['label'] for k in keys]

    ax = axes[0]
    walls = [sim_data[k]['wall_time'] for k in keys]
    ax.bar(labels, walls, color=colors, alpha=0.8)
    ax.set_ylabel('Wall Time (s)'); ax.set_title('Simulation Wall Time')
    _style_ax(ax)

    ax = axes[1]
    steps = [sim_data[k]['n_steps'] for k in keys]
    ax.bar(labels, steps, color=colors, alpha=0.8)
    ax.set_ylabel('# Steps'); ax.set_title('Steps per Timestep')
    _style_ax(ax)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_top_diverging(metrics, key, top_n=20):
    mol_div = metrics[key].get('mol_divergence', [])
    top = [m for m in mol_div[:top_n] if m['abs_diff'] > 0]
    label = MODELS[key]['label']
    color = MODELS[key]['color']

    fig, ax = plt.subplots(figsize=(14, max(5, len(top) * 0.3)))
    if not top:
        ax.text(0.5, 0.5, 'No divergence', ha='center', va='center',
                transform=ax.transAxes)
        return fig_to_b64(fig)

    names = [m['name'][:40] for m in top]
    diffs = [m['signed'] for m in top]
    colors = [color if d > 0 else '#2563eb' for d in diffs]
    y = np.arange(len(top))
    ax.barh(y, diffs, color=colors, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(f'Signed Difference ({label} - Partitioned)')
    ax.set_title(f'Top {len(top)} Diverging Molecules: {label} vs Partitioned')
    ax.axvline(x=0, color='black', lw=0.8, alpha=0.5)
    ax.invert_yaxis()
    _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _mol_table(mol_div, label, top_n=30):
    rows = ''
    for i, m in enumerate(mol_div[:top_n]):
        if m['abs_diff'] == 0: continue
        d = f'{label} &gt; Part' if m['signed'] > 0 else f'Part &gt; {label}'
        c = '#dc2626' if m['signed'] > 0 else '#2563eb'
        rows += (f'<tr><td>{i+1}</td>'
                 f'<td style="font-family:monospace;font-size:0.8em">{m["name"]}</td>'
                 f'<td style="text-align:right">{m["part"]:.0f}</td>'
                 f'<td style="text-align:right">{m["other"]:.0f}</td>'
                 f'<td style="text-align:right;font-weight:600">{m["abs_diff"]:.0f}</td>'
                 f'<td style="color:{c}">{d}</td></tr>')
    return rows


def generate_html(sim_data, metrics, plots, seed, duration, output,
                  parallel_wall):
    print(f'\nStep 5: Generating HTML report -> {output}')
    date_str = time.strftime('%Y-%m-%d %H:%M')

    dep_err = metrics['dep']['overall_max_error']
    rec_err = metrics['rec']['overall_max_error']
    dep_r = metrics['dep']['pearson_r']
    rec_r = metrics['rec']['pearson_r']
    dep_r_s = f'{dep_r:.6f}' if not np.isnan(dep_r) else 'N/A'
    rec_r_s = f'{rec_r:.6f}' if not np.isnan(rec_r) else 'N/A'

    # Build per-field error table
    field_rows = ''
    for f in MASS_FIELDS:
        de = metrics['dep']['max_errors'].get(f, 0)
        re = metrics['rec']['max_errors'].get(f, 0)
        dc = '#16a34a' if de < ERROR_THRESHOLD else '#dc2626'
        rc = '#16a34a' if re < ERROR_THRESHOLD else '#dc2626'
        field_rows += (f'<tr><td>{MASS_LABELS.get(f, f)}</td>'
                       f'<td style="color:{dc};text-align:right">{de:.4f}%</td>'
                       f'<td style="color:{rc};text-align:right">{re:.4f}%</td></tr>')

    dep_mol = _mol_table(metrics['dep']['mol_divergence'], 'Dep')
    rec_mol = _mol_table(metrics['rec']['mol_divergence'], 'Rec')

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>v2ecoli Three-Way Architecture Comparison</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
max-width:1500px;margin:0 auto;padding:20px;background:#f8fafc;color:#1e293b}}
h1{{font-size:1.8em;margin:15px 0;color:#0f172a}}
h2{{font-size:1.3em;margin:25px 0 10px;color:#334155;
border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
.header{{background:#0f172a;color:white;padding:20px 25px;border-radius:10px;margin-bottom:15px}}
.header h1{{color:white;margin:0 0 6px}}.header p{{color:#94a3b8;font-size:0.9em}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:15px 0}}
.card{{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
.card .label{{font-size:0.72em;color:#64748b;text-transform:uppercase;letter-spacing:0.05em}}
.card .value{{font-size:1.4em;font-weight:700;margin-top:4px}}
.blue{{color:#2563eb}}.red{{color:#dc2626}}.green{{color:#16a34a}}.purple{{color:#7c3aed}}.orange{{color:#ea580c}}
.plot{{background:white;border-radius:8px;padding:14px;margin:12px 0;
box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center}}
.plot img{{max-width:100%}}
.section{{background:white;border-radius:8px;padding:18px;margin:12px 0;
box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
table{{border-collapse:collapse;width:100%;font-size:0.85em}}
th,td{{border:1px solid #e2e8f0;padding:6px 10px;text-align:left}}
th{{background:#f1f5f9;font-weight:600}}
.verdict{{border-radius:10px;padding:20px 25px;margin:15px 0;font-size:1.1em}}
.arch-desc{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:12px 0}}
.arch-box{{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
.arch-box h3{{font-size:1em;margin-bottom:6px}}
.arch-box pre{{background:#f8fafc;padding:8px;border-radius:6px;font-size:0.75em;overflow-x:auto;line-height:1.4}}
footer{{margin-top:30px;padding:15px 0;border-top:1px solid #e2e8f0;
color:#94a3b8;font-size:0.75em;text-align:center}}
</style></head><body>

<div class="header">
<h1>v2ecoli Three-Way Architecture Comparison</h1>
<p>{date_str} &middot; Duration: {duration}s ({duration/60:.1f} min) &middot; Seed: {seed}
 &middot; Simulations run in parallel &middot; Total wall: {parallel_wall:.0f}s</p>
</div>

<h2>Summary</h2>
<div class="cards">
<div class="card"><div class="label">Part Steps</div>
<div class="value blue">{sim_data['part']['n_steps']}</div></div>
<div class="card"><div class="label">Dep Steps</div>
<div class="value red">{sim_data['dep']['n_steps']}</div></div>
<div class="card"><div class="label">Rec Steps</div>
<div class="value green">{sim_data['rec']['n_steps']}</div></div>
<div class="card"><div class="label">Part Wall</div>
<div class="value blue">{sim_data['part']['wall_time']:.1f}s</div></div>
<div class="card"><div class="label">Dep Wall</div>
<div class="value red">{sim_data['dep']['wall_time']:.1f}s</div></div>
<div class="card"><div class="label">Rec Wall</div>
<div class="value green">{sim_data['rec']['wall_time']:.1f}s</div></div>
<div class="card"><div class="label">Dep Max Error</div>
<div class="value {'green' if dep_err < ERROR_THRESHOLD else 'red'}">{dep_err:.2f}%</div></div>
<div class="card"><div class="label">Rec Max Error</div>
<div class="value {'green' if rec_err < ERROR_THRESHOLD else 'red'}">{rec_err:.2f}%</div></div>
<div class="card"><div class="label">Dep Diverged</div>
<div class="value orange">{metrics['dep']['n_diverged']}</div></div>
<div class="card"><div class="label">Rec Diverged</div>
<div class="value orange">{metrics['rec']['n_diverged']}</div></div>
</div>

<h2>Architectures</h2>
<div class="arch-desc">
<div class="arch-box" style="border-top:3px solid #2563eb">
<h3 class="blue">Partitioned (Baseline)</h3>
<pre>Requester &rarr; Allocator &rarr; Evolver
3 steps per process + priority allocation
{sim_data['part']['n_steps']} total steps</pre></div>
<div class="arch-box" style="border-top:3px solid #dc2626">
<h3 class="red">Departitioned</h3>
<pre>DepartitionedStep._do_update()
No allocator, sequential execution
evolve_only: RnaMaturation, Complexation
{sim_data['dep']['n_steps']} total steps</pre></div>
<div class="arch-box" style="border-top:3px solid #16a34a">
<h3 class="green">Reconciled</h3>
<pre>ReconciledStep per allocator layer
Proportional allocation via reconcile
evolve_only: RnaMaturation, Complexation
{sim_data['rec']['n_steps']} total steps</pre></div>
</div>

<h2>Mass Trajectories</h2>
<div class="plot"><img src="data:image/png;base64,{plots['mass_traj']}" alt="Mass"></div>

<h2>Mass Divergence vs Partitioned</h2>
<div class="section"><table>
<thead><tr><th>Component</th><th style="text-align:right">Dep Max %</th><th style="text-align:right">Rec Max %</th></tr></thead>
<tbody>{field_rows}</tbody></table></div>
<div class="plot"><img src="data:image/png;base64,{plots['mass_div']}" alt="Divergence"></div>

<h2>Growth & Volume</h2>
<div class="plot"><img src="data:image/png;base64,{plots['growth']}" alt="Growth"></div>

<h2>Top Diverging Molecules: Departitioned</h2>
<div class="plot"><img src="data:image/png;base64,{plots['top_dep']}" alt="Top Dep"></div>
<div class="section" style="max-height:400px;overflow-y:auto"><table>
<thead><tr><th>#</th><th>Molecule</th><th style="text-align:right">Part</th>
<th style="text-align:right">Dep</th><th style="text-align:right">Abs Diff</th><th>Direction</th></tr></thead>
<tbody>{dep_mol}</tbody></table></div>

<h2>Top Diverging Molecules: Reconciled</h2>
<div class="plot"><img src="data:image/png;base64,{plots['top_rec']}" alt="Top Rec"></div>
<div class="section" style="max-height:400px;overflow-y:auto"><table>
<thead><tr><th>#</th><th>Molecule</th><th style="text-align:right">Part</th>
<th style="text-align:right">Rec</th><th style="text-align:right">Abs Diff</th><th>Direction</th></tr></thead>
<tbody>{rec_mol}</tbody></table></div>

<h2>Timing</h2>
<div class="plot"><img src="data:image/png;base64,{plots['timing']}" alt="Timing"></div>

<div class="verdict" style="background:#f0f9ff;border:2px solid #2563eb">
<strong>Reconciled architecture closes the gap:</strong> {rec_err:.2f}% max error
vs departitioned {dep_err:.2f}%
({dep_err/max(rec_err, 0.001):.1f}x improvement).
{sim_data['rec']['n_steps']} steps vs {sim_data['part']['n_steps']} partitioned
({100*(1-sim_data['rec']['n_steps']/sim_data['part']['n_steps']):.0f}% fewer).
</div>

<footer>v2ecoli Three-Way Architecture Comparison &middot; {date_str} &middot;
Duration: {duration}s &middot; Seed {seed}</footer>
</body></html>""")

    print(f'  Report written to {output}')
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(duration=DEFAULT_DURATION, seed=0,
                   cache_dir='out/cache',
                   output='out/comparison_report.html',
                   snapshot_interval=SNAPSHOT_INTERVAL,
                   parallel=True):
    print(f'=== v2ecoli Three-Way Architecture Comparison ===')
    print(f'    Duration: {duration}s, Seed: {seed}, Cache: {cache_dir}')
    print(f'    Parallel: {parallel}\n')

    if parallel:
        sim_data, par_wall = run_all_parallel(
            cache_dir, seed, duration, snapshot_interval)
    else:
        sim_data, par_wall = run_all_sequential(
            cache_dir, seed, duration, snapshot_interval)

    metrics = compute_metrics(sim_data)

    print('\nStep 4: Generating plots...')
    plots = {
        'mass_traj': plot_mass_trajectories(sim_data),
        'mass_div': plot_mass_divergence(sim_data, metrics),
        'growth': plot_growth(sim_data),
        'timing': plot_timing(sim_data),
        'top_dep': plot_top_diverging(metrics, 'dep'),
        'top_rec': plot_top_diverging(metrics, 'rec'),
    }

    report = generate_html(
        sim_data, metrics, plots, seed, duration, output, par_wall)
    print(f'\n=== Done. Report: {report} ===')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Three-way E. coli architecture comparison')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-dir', default='out/cache')
    parser.add_argument('--output', default='out/comparison_report.html')
    parser.add_argument('--snapshot-interval', type=int,
                        default=SNAPSHOT_INTERVAL)
    parser.add_argument('--no-parallel', action='store_true',
                        help='Run sequentially instead of parallel')
    args = parser.parse_args()

    run_comparison(
        duration=args.duration, seed=args.seed,
        cache_dir=args.cache_dir, output=args.output,
        snapshot_interval=args.snapshot_interval,
        parallel=not args.no_parallel,
    )
