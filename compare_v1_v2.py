"""
Three-Way E. coli Simulation Comparison

Runs three engines in parallel:
  - vEcoli 1.0 (vivarium engine, master branch)
  - vEcoli 2.0 (composite migration, composite branch)
  - v2ecoli   (pure process-bigraph, this repo)

Generates an HTML report with overlaid plots, side-by-side mass components,
and a detailed comparison table.

Usage:
    python compare_v1_v2.py                    # 2500s default
    python compare_v1_v2.py --duration 300     # short comparison
"""

import os
import sys
import time
import json
import base64
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SNAPSHOT_INTERVAL = 50  # seconds between snapshots
REPORT_DIR = 'out/comparison'


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _extract_mass(cell):
    """Extract mass metrics from a cell state."""
    mass = cell.get('listeners', {}).get('mass', {})
    return {k: float(mass.get(k, 0)) for k in [
        'dry_mass', 'cell_mass', 'protein_mass', 'rna_mass',
        'rRna_mass', 'tRna_mass', 'mRna_mass', 'dna_mass',
        'smallMolecule_mass', 'water_mass', 'volume',
        'instantaneous_growth_rate',
    ]}


def _extract_chromosome(cell):
    """Extract chromosome metrics from a cell state."""
    unique = cell.get('unique', {})
    fc = unique.get('full_chromosome')
    n_chrom = 0
    if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
        n_chrom = int(fc['_entryState'].sum())
    rep = unique.get('active_replisome')
    n_forks = 0
    if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
        n_forks = int(rep['_entryState'].sum())
    return {'n_chromosomes': n_chrom, 'n_forks': n_forks}


def fig_to_b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


ENGINES = [
    ('vecoli_v1', 'vEcoli 1.0 (vivarium)', '#3b82f6', '--'),
    ('vecoli_composite', 'vEcoli 2.0 (composite)', '#8b5cf6', '-.'),
    ('v2ecoli', 'v2ecoli (pure PBG)', '#ef4444', '-'),
]


def plot_comparison(datasets, metric, ylabel, title):
    """Single metric overlay plot for all available engines."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, color, ls in ENGINES:
        snaps = datasets.get(key, {}).get('snapshots', [])
        if not snaps:
            continue
        t = [s['time']/60 for s in snaps]
        y = [s.get(metric, 0) for s in snaps]
        ax.plot(t, y, color=color, linestyle=ls, label=label, linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_b64(fig)


def plot_side_by_side(datasets, metrics, ylabel, title):
    """Side-by-side panels for each engine showing multiple metrics."""
    active = [(k, l, c, ls) for k, l, c, ls in ENGINES if datasets.get(k, {}).get('snapshots')]
    n = max(len(active), 1)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=13)

    for i, (key, label, _, _) in enumerate(active):
        ax = axes[0][i]
        snaps = datasets[key]['snapshots']
        t = [s['time']/60 for s in snaps]
        for mkey, name, color in metrics:
            y = [s.get(mkey, 0) for s in snaps]
            ax.plot(t, y, color=color, label=name, linewidth=1.2)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Fill empty panels
    for i in range(len(active), n):
        axes[0][i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0][i].transAxes)

    plt.tight_layout()
    return fig_to_b64(fig)


def plot_mass_components(datasets, title='Mass Components'):
    """Side-by-side mass component plots for each engine."""
    components = [
        ('protein_mass', 'Protein', '#22c55e'),
        ('dna_mass', 'DNA', '#8b5cf6'),
        ('rRna_mass', 'rRNA', '#3b82f6'),
        ('tRna_mass', 'tRNA', '#06b6d4'),
        ('mRna_mass', 'mRNA', '#f97316'),
        ('smallMolecule_mass', 'Small mol', '#f59e0b'),
    ]
    return plot_side_by_side(datasets, components, 'Mass (fg)', title)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_report(datasets, duration):
    """Generate three-way HTML comparison report."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Mass comparison at regular intervals
    intervals = [0, 60, 300, 600, 1200, 1800, 2400]
    intervals = [t for t in intervals if t <= duration]

    def snap_at(snaps, t):
        """Find snapshot closest to time t."""
        best = {}
        for s in snaps:
            if abs(s['time'] - t) < abs(best.get('time', 1e9) - t):
                best = s
        return best

    # Generate plots
    plots = {}
    has_data = any(datasets.get(k, {}).get('snapshots') for k, _, _, _ in ENGINES)
    if has_data:
        plots['dry_mass'] = plot_comparison(datasets, 'dry_mass', 'Dry Mass (fg)', 'Dry Mass Over Time')
        plots['cell_mass'] = plot_comparison(datasets, 'cell_mass', 'Cell Mass (fg)', 'Cell Mass (wet)')
        plots['growth_rate'] = plot_comparison(datasets, 'instantaneous_growth_rate', 'Growth Rate (1/s)', 'Instantaneous Growth Rate')
        plots['volume'] = plot_comparison(datasets, 'volume', 'Volume (fL)', 'Cell Volume')
        plots['chromosomes'] = plot_comparison(datasets, 'n_chromosomes', 'Chromosomes', 'Chromosome Count')
        plots['forks'] = plot_comparison(datasets, 'n_forks', 'Forks', 'Replication Forks')
        plots['mass_components'] = plot_mass_components(datasets, 'Mass Components')

        rna_metrics = [
            ('rRna_mass', 'rRNA', '#3b82f6'),
            ('tRna_mass', 'tRNA', '#06b6d4'),
            ('mRna_mass', 'mRNA', '#f97316'),
        ]
        plots['rna_breakdown'] = plot_side_by_side(datasets, rna_metrics, 'Mass (fg)', 'RNA Mass Breakdown')

        struct_metrics = [
            ('protein_mass', 'Protein', '#22c55e'),
            ('dna_mass', 'DNA', '#8b5cf6'),
            ('smallMolecule_mass', 'Small molecules', '#f59e0b'),
        ]
        plots['structural'] = plot_side_by_side(datasets, struct_metrics, 'Mass (fg)', 'Structural Components')

    # Build mass comparison table — one row per time, columns per engine
    mass_table_rows = ''
    for t in intervals:
        row = f'<tr><td style="font-weight:bold">{t/60:.0f} min</td>'
        for key, _, _, _ in ENGINES:
            snaps = datasets.get(key, {}).get('snapshots', [])
            st = snap_at(snaps, t) if snaps else {}
            dm = st.get('dry_mass', 0)
            row += f'<td>{dm:.1f}</td>' if dm > 0 else '<td>—</td>'
        row += '</tr>'
        mass_table_rows += row

    # Build performance cards for each engine
    perf_cards = ''
    for key, label, color, _ in ENGINES:
        d = datasets.get(key, {})
        speed = d.get('speed', 0)
        wall = d.get('wall_time', 0)
        sim = d.get('sim_time', 0)
        if speed > 0:
            perf_cards += f'''<div class="perf-card">
    <div class="label">{label}</div>
    <div class="value" style="color:{color}">{speed:.1f}x</div>
    <div class="label">{wall:.0f}s wall for {sim:.0f}s sim</div>
  </div>\n'''
        else:
            perf_cards += f'''<div class="perf-card">
    <div class="label">{label}</div>
    <div class="value" style="color:#ccc">N/A</div>
    <div class="label">not available</div>
  </div>\n'''

    # Build overview table rows
    overview_rows = ''
    metrics_list = [
        ('Engine', lambda d: d.get('engine', 'N/A')),
        ('Load time', lambda d: f"{d.get('load_time',0):.2f}s" if d.get('load_time') else 'N/A'),
        ('Sim duration', lambda d: f"{d.get('sim_time',0):.0f}s ({d.get('sim_time',0)/60:.1f} min)" if d.get('sim_time') else 'N/A'),
        ('Wall time', lambda d: f"{d.get('wall_time',0):.1f}s" if d.get('wall_time') else 'N/A'),
        ('Speed', lambda d: f"{d.get('speed',0):.1f}x" if d.get('speed') else 'N/A'),
    ]
    for name, fn in metrics_list:
        row = f'<tr><td>{name}</td>'
        for key, _, _, _ in ENGINES:
            d = datasets.get(key, {})
            row += f'<td>{fn(d)}</td>'
        row += '</tr>'
        overview_rows += row

    # Final snapshot metrics
    for name, metric in [('Final dry mass', 'dry_mass'), ('Chromosomes', 'n_chromosomes'), ('Forks', 'n_forks')]:
        row = f'<tr><td>{name}</td>'
        for key, _, _, _ in ENGINES:
            snaps = datasets.get(key, {}).get('snapshots', [])
            val = snaps[-1].get(metric, 0) if snaps else 0
            fmt = f'{val:.1f} fg' if 'mass' in name.lower() else str(int(val))
            row += f'<td>{fmt}</td>'
        row += '</tr>'
        overview_rows += row

    report_path = os.path.join(REPORT_DIR, 'comparison.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>E. coli Simulation Comparison</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }}
h2 {{ color: #1e40af; margin-top: 2em; }}
h3 {{ color: #334155; }}
table {{ border-collapse: collapse; margin: 1em 0; width: 100%; }}
th, td {{ padding: 6px 12px; border: 1px solid #e2e8f0; text-align: center; }}
th {{ background: #f1f5f9; font-weight: 600; }}
.plot {{ margin: 1em 0; text-align: center; }}
.plot img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }}
.perf {{ display: flex; gap: 2em; justify-content: center; margin: 1em 0; }}
.perf-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em 2em; text-align: center; }}
.perf-card .value {{ font-size: 2em; font-weight: bold; }}
.perf-card .label {{ color: #64748b; font-size: 0.9em; }}
.green {{ color: #16a34a; }}
.blue {{ color: #2563eb; }}
.red {{ color: #ef4444; }}
</style>
</head><body>
<h1>E. coli Whole-Cell Simulation Comparison</h1>
<p>Three-way comparison of vEcoli 1.0 (vivarium engine), vEcoli 2.0 (composite migration),
and v2ecoli (pure process-bigraph). All use the same ParCa parameters, initial state,
and biological process logic.</p>

<h2>Performance</h2>
<div class="perf">
  {perf_cards}
</div>

<table>
<tr><th>Metric</th><th>vEcoli 1.0</th><th>vEcoli 2.0</th><th>v2ecoli</th></tr>
{overview_rows}
</table>

<h2>Growth Comparison</h2>
""")

        for key, title in [('dry_mass', 'Dry Mass'), ('cell_mass', 'Cell Mass'),
                           ('growth_rate', 'Growth Rate'), ('volume', 'Volume')]:
            if plots.get(key):
                f.write(f'<div class="plot"><img src="data:image/png;base64,{plots[key]}" alt="{title}"></div>\n')

        f.write('<h2>Chromosome Replication</h2>\n')
        for key, title in [('chromosomes', 'Chromosomes'), ('forks', 'Replication Forks')]:
            if plots.get(key):
                f.write(f'<div class="plot"><img src="data:image/png;base64,{plots[key]}" alt="{title}"></div>\n')

        f.write('<h2>Mass Components (side-by-side)</h2>\n')
        if plots.get('mass_components'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["mass_components"]}" alt="Mass Components"></div>\n')
        if plots.get('rna_breakdown'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["rna_breakdown"]}" alt="RNA Breakdown"></div>\n')
        if plots.get('structural'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["structural"]}" alt="Structural Components"></div>\n')

        f.write(f"""
<h2>Dry Mass Comparison Table (fg)</h2>
<table>
<tr><th>Time</th><th>vEcoli 1.0</th><th>vEcoli 2.0</th><th>v2ecoli</th></tr>
{mass_table_rows}
</table>

<footer>
  Generated by compare_v1_v2.py &middot; v2ecoli (pure process-bigraph)
</footer>
</body></html>""")

    print(f"\nReport: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='v1 vs v2 comparison')
    parser.add_argument('--duration', type=int, default=2500,
                        help='Simulation duration in seconds (default: 2500)')
    args = parser.parse_args()

    # Ensure we're in the v2ecoli directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Three-Way Comparison ({args.duration}s)")
    print("=" * 60)

    import subprocess as sp
    base = os.path.dirname(os.path.abspath(__file__))

    runners = {
        'vecoli_v1': ('scripts/run_vecoli_v1.py', '_vecoli_v1_result.json'),
        'vecoli_composite': ('scripts/run_vecoli_composite.py', '_vecoli_composite_result.json'),
        'v2ecoli': ('scripts/run_v2.py', '_v2ecoli_result.json'),
    }

    t0 = time.time()
    datasets = {}
    empty = {'snapshots': [], 'wall_time': 0, 'sim_time': 0, 'speed': 0, 'load_time': 0}
    result_paths = {}

    def _launch(key):
        script, result_file = runners[key]
        rpath = os.path.join(base, REPORT_DIR, result_file)
        spath = os.path.join(base, script)
        result_paths[key] = rpath
        if os.path.exists(spath):
            return sp.Popen([sys.executable, spath, str(args.duration), str(SNAPSHOT_INTERVAL), rpath])
        print(f"  {key}: script not found ({spath})")
        return None

    def _collect(key, proc):
        rpath = result_paths[key]
        label = next(l for k, l, _, _ in ENGINES if k == key)
        if proc is None:
            datasets[key] = {**empty, 'engine': f'{label} (skipped)'}
            return
        proc.wait()
        if os.path.exists(rpath):
            with open(rpath) as f:
                data = json.load(f)
            os.unlink(rpath)
            print(f"  {label}: {data['sim_time']}s in {data['wall_time']:.1f}s ({data['speed']:.1f}x)")
            datasets[key] = data
        else:
            print(f"  {label}: FAILED (rc={proc.returncode})")
            datasets[key] = {**empty, 'engine': f'{label} (FAILED)'}

    # Phase 1: composite + v2ecoli in parallel (both need vEcoli on composite branch)
    print(f"  Launching composite + v2ecoli in parallel...")
    p_comp = _launch('vecoli_composite')
    p_v2 = _launch('v2ecoli')
    _collect('vecoli_composite', p_comp)
    _collect('v2ecoli', p_v2)

    # Phase 2: v1 sequentially (switches vEcoli to master branch)
    print(f"  Launching v1 (vivarium engine)...")
    p_v1 = _launch('vecoli_v1')
    _collect('vecoli_v1', p_v1)

    total = time.time() - t0

    report = generate_report(datasets, args.duration)
    print(f"Total: {total:.0f}s")

    # Open in browser
    import subprocess as sp
    sp.run(['open', report], capture_output=True)


if __name__ == '__main__':
    main()
