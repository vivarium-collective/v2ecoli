"""
Compare partitioned vs departitioned E. coli simulation architectures.

Uses the existing partitioned model (Requester/Allocator/Evolver) as
baseline and compares it with the departitioned model (standalone
DepartitionedStep wrappers, no allocator).

Generates a single-file HTML report comparing mass trajectories,
divergence, bulk counts, growth, and timing.

Usage:
    python compare_report.py                        # default 120s sim
    python compare_report.py --duration 600         # 10-min sim
    python compare_report.py --seed 42 --output out/my_report.html
"""

import os
import io
import time
import base64
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from v2ecoli.composite import make_composite
from v2ecoli.composite_departitioned import make_departitioned_composite


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
    'cell_mass': 'Cell Mass',
    'dry_mass': 'Dry Mass',
    'protein_mass': 'Protein',
    'rna_mass': 'RNA',
    'dna_mass': 'DNA',
    'smallMolecule_mass': 'Small Molecule',
    'water_mass': 'Water',
}
ERROR_THRESHOLD = 5.0


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
        row['growth_rate'] = float(
            mass.get('instantaneous_growth_rate', 0))
        row['volume'] = float(mass.get('volume', 0))
        snapshots.append(row)
    return snapshots


def _extract_bulk(composite):
    cell = composite.state.get('agents', {}).get('0', {})
    bulk = cell.get('bulk')
    if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
        return bulk['count'].astype(float)
    if isinstance(bulk, dict) and 'count' in bulk:
        return np.array(bulk['count'], dtype=float)
    return np.array([])


def _style_ax(ax):
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def load_models(cache_dir, seed):
    print('Step 1: Loading models...')

    t0 = time.time()
    part = make_composite(cache_dir=cache_dir, seed=seed)
    part_time = time.time() - t0
    print(f'  Partitioned:    {len(part.step_paths)} steps, '
          f'built in {part_time:.1f}s')

    t0 = time.time()
    dep = make_departitioned_composite(cache_dir=cache_dir, seed=seed)
    dep_time = time.time() - t0
    print(f'  Departitioned:  {len(dep.step_paths)} steps, '
          f'built in {dep_time:.1f}s')

    return part, dep, {
        'part_build_time': part_time,
        'dep_build_time': dep_time,
        'part_steps': len(part.step_paths),
        'dep_steps': len(dep.step_paths),
    }


def _run_simulation(composite, label, duration, snapshot_interval):
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
            print(f'    [{label}] Agent removed (division) at '
                  f't={total_run}s')
            break

        mass = cell.get('listeners', {}).get('mass', {})
        dry_mass = float(mass.get('dry_mass', 0))
        if total_run % 50 < snapshot_interval:
            print(f'    [{label}] t={total_run:.0f}s: '
                  f'dry_mass={dry_mass:.0f}fg')

    wall_time = time.time() - t0
    print(f'  {label}: {wall_time:.1f}s wall, {total_run:.0f}s sim')
    return wall_time, total_run


def run_simulations(part, dep, duration, snapshot_interval):
    print(f'\nStep 2: Running simulations ({duration}s each)...')

    part_emitter = _get_emitter(part)
    dep_emitter = _get_emitter(dep)

    part_wall, part_sim = _run_simulation(
        part, 'Partitioned', duration, snapshot_interval)
    dep_wall, dep_sim = _run_simulation(
        dep, 'Departitioned', duration, snapshot_interval)

    part_snaps = _extract_snapshots(part_emitter)
    dep_snaps = _extract_snapshots(dep_emitter)

    print(f'  Emitter snapshots: {len(part_snaps)} part, '
          f'{len(dep_snaps)} dep')

    return {
        'part_wall': part_wall,
        'dep_wall': dep_wall,
        'part_sim_time': part_sim,
        'dep_sim_time': dep_sim,
        'part_snaps': part_snaps,
        'dep_snaps': dep_snaps,
        'part_bulk': _extract_bulk(part),
        'dep_bulk': _extract_bulk(dep),
    }


def compute_metrics(sim_data):
    print('\nStep 3: Computing metrics...')

    part_by_t = {int(s['time']): s for s in sim_data['part_snaps']}
    dep_by_t = {int(s['time']): s for s in sim_data['dep_snaps']}
    common_times = sorted(set(part_by_t.keys()) & set(dep_by_t.keys()))

    pct_diff = {field: [] for field in MASS_FIELDS}
    pct_times = []
    for t in common_times:
        p, d = part_by_t[t], dep_by_t[t]
        pct_times.append(t)
        for field in MASS_FIELDS:
            pv = p.get(field, 0)
            dv = d.get(field, 0)
            ref = max(abs(pv), abs(dv), 1e-12)
            pct_diff[field].append(abs(pv - dv) / ref * 100)

    max_errors = {f: max(vals) if vals else 0.0
                  for f, vals in pct_diff.items()}
    overall_max_error = max(max_errors.values()) if max_errors else 0.0

    pb = sim_data['part_bulk']
    db = sim_data['dep_bulk']
    min_len = min(len(pb), len(db))
    if min_len > 0:
        pb, db = pb[:min_len], db[:min_len]
        mask = (pb > 0) | (db > 0)
        r = stats.pearsonr(pb[mask], db[mask])[0] if mask.sum() > 1 else float('nan')
        exact_match = int((pb == db).sum())
    else:
        r, exact_match = float('nan'), 0

    wall_ratio = sim_data['part_wall'] / max(sim_data['dep_wall'], 1e-9)

    metrics = {
        'pct_times': pct_times, 'pct_diff': pct_diff,
        'max_errors': max_errors, 'overall_max_error': overall_max_error,
        'bulk_pearson_r': r, 'bulk_exact_match': exact_match,
        'bulk_total': min_len, 'wall_ratio': wall_ratio,
    }

    r_str = f'{r:.6f}' if not np.isnan(r) else 'N/A'
    print(f'  Max mass error:  {overall_max_error:.4f}%')
    print(f'  Bulk Pearson r:  {r_str}')
    print(f'  Bulk exact match: {exact_match}/{min_len}')
    print(f'  Wall time ratio (part/dep): {wall_ratio:.2f}x')

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mass_trajectories(sim_data):
    fields = ['dry_mass', 'protein_mass', 'rna_mass',
              'dna_mass', 'smallMolecule_mass', 'cell_mass']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, field in enumerate(fields):
        ax = axes.ravel()[i]
        for snaps, color, ls, label in [
            (sim_data['part_snaps'], '#2563eb', '-', 'Partitioned'),
            (sim_data['dep_snaps'], '#dc2626', '--', 'Departitioned'),
        ]:
            if snaps:
                ax.plot([s['time']/60 for s in snaps],
                        [s.get(field, 0) for s in snaps],
                        ls, color=color, lw=1.5, label=label, alpha=0.8)
        ax.set_title(MASS_LABELS.get(field, field), fontsize=10)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.set_ylabel('Mass (fg)', fontsize=8)
        ax.legend(fontsize=7)
        _style_ax(ax)
    fig.suptitle('Mass Trajectories', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mass_divergence(metrics):
    fields = ['dry_mass', 'protein_mass', 'rna_mass',
              'dna_mass', 'smallMolecule_mass']
    fig, axes = plt.subplots(1, len(fields), figsize=(3.6*len(fields), 3.5))
    times = np.array(metrics['pct_times']) / 60
    for i, field in enumerate(fields):
        ax = axes[i]
        vals = metrics['pct_diff'].get(field, [])
        if vals and len(times) == len(vals):
            ax.plot(times, vals, '-', color='#7c3aed', lw=1.2)
            ax.fill_between(times, vals, alpha=0.15, color='#7c3aed')
        ax.set_title(MASS_LABELS.get(field, field), fontsize=9)
        ax.set_xlabel('Time (min)', fontsize=7)
        ax.set_ylabel('% Difference', fontsize=7)
        _style_ax(ax)
    fig.suptitle('Mass % Difference', fontsize=11, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_bulk_scatter(sim_data):
    pb, db = sim_data['part_bulk'], sim_data['dep_bulk']
    min_len = min(len(pb), len(db))
    fig, ax = plt.subplots(figsize=(6, 6))
    if min_len > 0:
        pb, db = pb[:min_len], db[:min_len]
        mask = (pb > 0) & (db > 0)
        if mask.sum() > 0:
            ax.scatter(pb[mask], db[mask], s=4, alpha=0.3,
                       color='#2563eb', edgecolors='none')
            lo = min(pb[mask].min(), db[mask].min()) * 0.5
            hi = max(pb[mask].max(), db[mask].max()) * 2
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5,
                    label='y = x')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel('Partitioned', fontsize=9)
    ax.set_ylabel('Departitioned', fontsize=9)
    ax.set_title('Final Bulk Molecule Counts', fontsize=11)
    ax.set_aspect('equal')
    _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_growth_comparison(sim_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for snaps, color, ls, label in [
        (sim_data['part_snaps'], '#2563eb', '-', 'Partitioned'),
        (sim_data['dep_snaps'], '#dc2626', '--', 'Departitioned'),
    ]:
        if not snaps: continue
        t = [s['time']/60 for s in snaps]
        axes[0].plot(t, [s.get('growth_rate',0)*3600 for s in snaps],
                     ls, color=color, lw=1.2, label=label)
        axes[1].plot(t, [s.get('volume',0) for s in snaps],
                     ls, color=color, lw=1.2, label=label)
    axes[0].set_ylabel('Growth rate (1/h)'); axes[0].set_title('Growth Rate')
    axes[1].set_ylabel('Volume (fL)'); axes[1].set_title('Cell Volume')
    for ax in axes:
        ax.set_xlabel('Time (min)'); ax.legend(fontsize=8); _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_timing(model_info, sim_data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    labels = ['Build', 'Simulation']
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x-w/2, [model_info['part_build_time'], sim_data['part_wall']],
           w, color='#2563eb', alpha=0.8, label='Partitioned')
    ax.bar(x+w/2, [model_info['dep_build_time'], sim_data['dep_wall']],
           w, color='#dc2626', alpha=0.8, label='Departitioned')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Wall Time (s)'); ax.set_title('Wall Time')
    ax.legend(fontsize=8); _style_ax(ax)

    ax = axes[1]
    dw = max(sim_data['dep_wall'], 1e-9)
    pw = max(sim_data['part_wall'], 1e-9)
    ax.bar(['Partitioned', 'Departitioned'],
           [sim_data['part_sim_time']/pw, sim_data['dep_sim_time']/dw],
           color=['#2563eb', '#dc2626'], alpha=0.8)
    ax.set_ylabel('Sim s / Wall s'); ax.set_title('Speed (higher=faster)')
    _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html(model_info, sim_data, metrics, plots, seed, duration,
                  output):
    print(f'\nStep 5: Generating HTML report -> {output}')

    max_err = metrics['overall_max_error']
    ok = max_err < ERROR_THRESHOLD
    vc = '#16a34a' if ok else '#dc2626'
    vt = 'PASS' if ok else 'FAIL'
    vd = (f'Max mass divergence {max_err:.4f}% '
          f'{"below" if ok else "exceeds"} {ERROR_THRESHOLD}% threshold.')
    r_val = metrics['bulk_pearson_r']
    r_str = f'{r_val:.6f}' if not np.isnan(r_val) else 'N/A'

    field_rows = ''
    for field in MASS_FIELDS:
        err = metrics['max_errors'].get(field, 0)
        c = '#16a34a' if err < ERROR_THRESHOLD else '#dc2626'
        field_rows += (f'<tr><td>{MASS_LABELS.get(field, field)}</td>'
                       f'<td style="color:{c}">{err:.4f}%</td></tr>')

    date_str = time.strftime('%Y-%m-%d %H:%M')
    wr = metrics['wall_ratio']
    mi = model_info; sd = sim_data

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>v2ecoli Architecture Comparison</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
max-width:1400px;margin:0 auto;padding:20px;background:#f8fafc;color:#1e293b}}
h1{{font-size:1.8em;margin:15px 0;color:#0f172a}}
h2{{font-size:1.3em;margin:25px 0 10px;color:#334155;
border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
.header{{background:#0f172a;color:white;padding:20px 25px;border-radius:10px;margin-bottom:15px}}
.header h1{{color:white;margin:0 0 6px}}.header p{{color:#94a3b8;font-size:0.9em}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:15px 0}}
.card{{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
.card .label{{font-size:0.72em;color:#64748b;text-transform:uppercase;letter-spacing:0.05em}}
.card .value{{font-size:1.5em;font-weight:700;margin-top:4px}}
.blue{{color:#2563eb}}.red{{color:#dc2626}}.green{{color:#16a34a}}.purple{{color:#7c3aed}}
.plot{{background:white;border-radius:8px;padding:14px;margin:12px 0;
box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center}}
.plot img{{max-width:100%}}
.section{{background:white;border-radius:8px;padding:18px;margin:12px 0;
box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
table{{border-collapse:collapse;width:100%;font-size:0.85em}}
th,td{{border:1px solid #e2e8f0;padding:6px 10px;text-align:left}}
th{{background:#f1f5f9;font-weight:600}}
.verdict{{border-radius:10px;padding:20px 25px;margin:15px 0;font-size:1.1em}}
.arch-desc{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0}}
.arch-box{{background:white;border-radius:8px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
.arch-box h3{{font-size:1.05em;margin-bottom:8px}}
.arch-box pre{{background:#f8fafc;padding:10px;border-radius:6px;font-size:0.78em;
overflow-x:auto;line-height:1.5}}
footer{{margin-top:30px;padding:15px 0;border-top:1px solid #e2e8f0;
color:#94a3b8;font-size:0.75em;text-align:center}}
</style></head><body>

<div class="header">
<h1>v2ecoli Architecture Comparison</h1>
<p>{date_str} &middot; Duration: {duration}s &middot; Seed: {seed}
 &middot; Partitioned vs Departitioned</p>
</div>

<div class="cards">
<div class="card"><div class="label">Steps (Part / Dep)</div>
<div class="value blue">{mi['part_steps']} / {mi['dep_steps']}</div></div>
<div class="card"><div class="label">Wall Time (Part / Dep)</div>
<div class="value blue">{sd['part_wall']:.1f}s / {sd['dep_wall']:.1f}s</div></div>
<div class="card"><div class="label">Speed Ratio (Part/Dep)</div>
<div class="value {'green' if wr > 1 else 'red'}">{wr:.2f}x</div></div>
<div class="card"><div class="label">Max Mass Error</div>
<div class="value" style="color:{vc}">{max_err:.4f}%</div></div>
<div class="card"><div class="label">Bulk Correlation (r)</div>
<div class="value purple">{r_str}</div></div>
</div>

<h2>Architecture Comparison</h2>
<div class="arch-desc">
<div class="arch-box"><h3 class="blue">Partitioned (Baseline)</h3>
<p style="font-size:0.85em;color:#64748b;margin-bottom:10px">
Each biological process is split into three coordinated steps.
The allocator mediates resource sharing via priority-based distribution.</p>
<pre>Requester  --request-->  Allocator  --allocate-->  Evolver
  (reads state,             (priority-based           (applies allocated
   computes need)            distribution)              resources)</pre>
<p style="font-size:0.82em;margin-top:10px">
<strong>{mi['part_steps']} steps</strong> total
(11 requesters + 3 allocators + 11 evolvers + listeners + infra).
Build: {mi['part_build_time']:.1f}s.</p></div>
<div class="arch-box"><h3 class="red">Departitioned</h3>
<p style="font-size:0.85em;color:#64748b;margin-bottom:10px">
Each biological process runs as a single step. Requests are computed
and immediately applied &mdash; no allocator, no shared resource pool.</p>
<pre>DepartitionedStep
  calculate_request() -> apply directly -> evolve_state()
  - No allocation overhead
  - Each process gets exactly what it requests
  - Simpler execution graph</pre>
<p style="font-size:0.82em;margin-top:10px">
<strong>{mi['dep_steps']} steps</strong> total
(11 standalone + listeners + infra).
Build: {mi['dep_build_time']:.1f}s.</p></div></div>

<div class="section"><table>
<thead><tr><th>Metric</th><th>Partitioned</th><th>Departitioned</th></tr></thead>
<tbody>
<tr><td>Steps</td><td>{mi['part_steps']}</td><td>{mi['dep_steps']}</td></tr>
<tr><td>Build Time</td><td>{mi['part_build_time']:.1f}s</td><td>{mi['dep_build_time']:.1f}s</td></tr>
<tr><td>Sim Wall Time</td><td>{sd['part_wall']:.1f}s</td><td>{sd['dep_wall']:.1f}s</td></tr>
<tr><td>Sim Duration</td><td>{sd['part_sim_time']:.0f}s</td><td>{sd['dep_sim_time']:.0f}s</td></tr>
</tbody></table></div>

<h2>Mass Trajectories</h2>
<div class="plot"><img src="data:image/png;base64,{plots['mass_traj']}" alt="Mass"></div>

<h2>Mass Divergence</h2>
<div class="section"><table>
<thead><tr><th>Component</th><th>Max % Difference</th></tr></thead>
<tbody>{field_rows}</tbody></table></div>
<div class="plot"><img src="data:image/png;base64,{plots['mass_div']}" alt="Divergence"></div>

<h2>Bulk Count Scatter</h2>
<div class="section"><p>Final bulk counts: <strong>{metrics['bulk_exact_match']}</strong> /
{metrics['bulk_total']} exact matches. Pearson r = <strong>{r_str}</strong>.</p></div>
<div class="plot"><img src="data:image/png;base64,{plots['bulk_scatter']}" alt="Bulk"></div>

<h2>Growth Comparison</h2>
<div class="plot"><img src="data:image/png;base64,{plots['growth']}" alt="Growth"></div>

<h2>Timing Comparison</h2>
<div class="plot"><img src="data:image/png;base64,{plots['timing']}" alt="Timing"></div>

<div class="verdict" style="background:{'#dcfce7' if ok else '#fee2e2'};
border:2px solid {vc}">
<strong style="font-size:1.3em;color:{vc}">{vt}</strong>
<span style="margin-left:12px">{vd}</span></div>

<footer>v2ecoli Architecture Comparison &middot; {date_str} &middot;
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
                   snapshot_interval=SNAPSHOT_INTERVAL):
    print(f'=== v2ecoli Architecture Comparison ===')
    print(f'    Duration: {duration}s, Seed: {seed}, Cache: {cache_dir}\n')

    part, dep, model_info = load_models(cache_dir, seed)
    sim_data = run_simulations(part, dep, duration, snapshot_interval)
    metrics = compute_metrics(sim_data)

    print('\nStep 4: Generating plots...')
    plots = {
        'mass_traj': plot_mass_trajectories(sim_data),
        'mass_div': plot_mass_divergence(metrics),
        'bulk_scatter': plot_bulk_scatter(sim_data),
        'growth': plot_growth_comparison(sim_data),
        'timing': plot_timing(model_info, sim_data),
    }

    report = generate_html(
        model_info, sim_data, metrics, plots, seed, duration, output)
    print(f'\n=== Done. Report: {report} ===')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare partitioned vs departitioned E. coli')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-dir', default='out/cache')
    parser.add_argument('--output', default='out/comparison_report.html')
    parser.add_argument('--snapshot-interval', type=int,
                        default=SNAPSHOT_INTERVAL)
    args = parser.parse_args()

    run_comparison(
        duration=args.duration, seed=args.seed,
        cache_dir=args.cache_dir, output=args.output,
        snapshot_interval=args.snapshot_interval,
    )
