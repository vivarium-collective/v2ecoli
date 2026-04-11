"""
v1 vs v2 Side-by-Side Comparison Report

Runs both vEcoli (v1, composite branch) and v2ecoli (v2, pure process-bigraph)
through the same lifecycle, collecting snapshots at regular intervals.
Generates an HTML report with side-by-side metrics, plots, and timing.

Usage:
    python compare_v1_v2.py                    # full lifecycle to division
    python compare_v1_v2.py --duration 300     # short comparison (300s)
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


def plot_comparison(v1, v2, metric, ylabel, title):
    """Single metric comparison plot. Works with either or both datasets."""
    fig, ax = plt.subplots(figsize=(10, 4))
    s1 = v1.get('snapshots', [])
    s2 = v2.get('snapshots', [])
    if s1:
        t1 = [s['time']/60 for s in s1]
        y1 = [s.get(metric, 0) for s in s1]
        ax.plot(t1, y1, 'b-', label='v1 (vEcoli)', linewidth=1.5)
    if s2:
        t2 = [s['time']/60 for s in s2]
        y2 = [s.get(metric, 0) for s in s2]
        ax.plot(t2, y2, 'r-', label='v2 (v2ecoli)', linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_b64(fig)


def plot_side_by_side(v1, v2, metrics, ylabel, title):
    """Side-by-side plots of multiple metrics for v1 and v2."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle(title, fontsize=13)

    for ax, data, label in [(axes[0], v1, 'v1 (vEcoli)'), (axes[1], v2, 'v2 (v2ecoli)')]:
        snaps = data.get('snapshots', [])
        if not snaps:
            ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        t = [s['time']/60 for s in snaps]
        for key, name, color in metrics:
            y = [s.get(key, 0) for s in snaps]
            ax.plot(t, y, color=color, label=name, linewidth=1.2)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_mass_components(v1, v2, title='Mass Components'):
    """Side-by-side mass component plots."""
    components = [
        ('protein_mass', 'Protein', '#22c55e'),
        ('dna_mass', 'DNA', '#8b5cf6'),
        ('rRna_mass', 'rRNA', '#3b82f6'),
        ('tRna_mass', 'tRNA', '#06b6d4'),
        ('mRna_mass', 'mRNA', '#f97316'),
        ('smallMolecule_mass', 'Small mol', '#f59e0b'),
    ]
    return plot_side_by_side(v1, v2, components, 'Mass (fg)', title)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_report(v1, v2, duration):
    """Generate side-by-side HTML comparison report."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Compute comparison metrics
    s1 = v1['snapshots']
    s2 = v2['snapshots']
    final1 = s1[-1] if s1 else {}
    final2 = s2[-1] if s2 else {}

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
    # Generate plots with whatever data is available
    if s1 or s2:
        plots['dry_mass'] = plot_comparison(v1, v2, 'dry_mass', 'Dry Mass (fg)', 'Dry Mass Over Time')
        plots['cell_mass'] = plot_comparison(v1, v2, 'cell_mass', 'Cell Mass (fg)', 'Cell Mass (wet)')
        plots['growth_rate'] = plot_comparison(v1, v2, 'instantaneous_growth_rate', 'Growth Rate (1/s)', 'Instantaneous Growth Rate')
        plots['volume'] = plot_comparison(v1, v2, 'volume', 'Volume (fL)', 'Cell Volume')
        plots['chromosomes'] = plot_comparison(v1, v2, 'n_chromosomes', 'Chromosomes', 'Chromosome Count')
        plots['forks'] = plot_comparison(v1, v2, 'n_forks', 'Forks', 'Replication Forks')
        plots['mass_components'] = plot_mass_components(v1, v2, 'Mass Components')

        rna_metrics = [
            ('rRna_mass', 'rRNA', '#3b82f6'),
            ('tRna_mass', 'tRNA', '#06b6d4'),
            ('mRna_mass', 'mRNA', '#f97316'),
        ]
        plots['rna_breakdown'] = plot_side_by_side(v1, v2, rna_metrics, 'Mass (fg)', 'RNA Mass Breakdown')

        struct_metrics = [
            ('protein_mass', 'Protein', '#22c55e'),
            ('dna_mass', 'DNA', '#8b5cf6'),
            ('smallMolecule_mass', 'Small molecules', '#f59e0b'),
        ]
        plots['structural'] = plot_side_by_side(v1, v2, struct_metrics, 'Mass (fg)', 'Structural Components')

    # Build mass comparison table
    mass_keys = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']
    mass_table_rows = ''
    for t in intervals:
        s1t = snap_at(s1, t)
        s2t = snap_at(s2, t)
        if not s1t and not s2t:
            continue
        row = f'<tr><td style="font-weight:bold">{t/60:.0f} min</td>'
        for key in mass_keys:
            v1v = s1t.get(key, 0)
            v2v = s2t.get(key, 0)
            diff = abs(v2v - v1v) / v1v * 100 if v1v > 0 else 0
            color = '#16a34a' if diff < 2 else '#f59e0b' if diff < 5 else '#ef4444'
            row += f'<td>{v1v:.1f}</td><td>{v2v:.1f}</td><td style="color:{color}">{diff:.1f}%</td>'
        row += '</tr>'
        mass_table_rows += row

    # Performance comparison
    v1_speed = v1.get('speed', 0)
    v2_speed = v2.get('speed', 0)
    speedup = v2_speed / v1_speed if v1_speed > 0 else 0

    report_path = os.path.join(REPORT_DIR, 'v1_v2_comparison.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>v1 vs v2 E. coli Comparison</title>
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
<h1>v1 vs v2 E. coli Whole-Cell Comparison</h1>
<p>Side-by-side comparison of vEcoli (v1, vivarium/composite) and v2ecoli (v2, pure process-bigraph).
Both use the same ParCa parameters, initial state, and biological process logic.</p>

<h2>Performance</h2>
<div class="perf">
  <div class="perf-card">
    <div class="label">v1 (vEcoli)</div>
    <div class="value blue">{v1_speed:.1f}x</div>
    <div class="label">{v1.get('wall_time',0):.0f}s wall for {v1.get('sim_time',0):.0f}s sim</div>
  </div>
  <div class="perf-card">
    <div class="label">v2 (v2ecoli)</div>
    <div class="value green">{v2_speed:.1f}x</div>
    <div class="label">{v2.get('wall_time',0):.0f}s wall for {v2.get('sim_time',0):.0f}s sim</div>
  </div>
  <div class="perf-card">
    <div class="label">v2 Speedup</div>
    <div class="value {'green' if speedup >= 1 else 'red'}">{speedup:.2f}x</div>
    <div class="label">{'faster' if speedup >= 1 else 'slower'}</div>
  </div>
</div>

<table>
<tr><th>Metric</th><th>v1 (vEcoli)</th><th>v2 (v2ecoli)</th></tr>
<tr><td>Engine</td><td>vivarium-core + process-bigraph</td><td>pure process-bigraph</td></tr>
<tr><td>Load time</td><td>{v1.get('load_time',0):.2f}s</td><td>{v2.get('load_time',0):.2f}s</td></tr>
<tr><td>Sim duration</td><td>{v1.get('sim_time',0):.0f}s ({v1.get('sim_time',0)/60:.1f} min)</td><td>{v2.get('sim_time',0):.0f}s ({v2.get('sim_time',0)/60:.1f} min)</td></tr>
<tr><td>Wall time</td><td>{v1.get('wall_time',0):.1f}s</td><td>{v2.get('wall_time',0):.1f}s</td></tr>
<tr><td>Speed</td><td>{v1_speed:.1f}x realtime</td><td style="font-weight:bold">{v2_speed:.1f}x realtime</td></tr>
<tr><td>Final dry mass</td><td>{final1.get('dry_mass',0):.1f} fg</td><td>{final2.get('dry_mass',0):.1f} fg</td></tr>
<tr><td>Final chromosomes</td><td>{final1.get('n_chromosomes',0)}</td><td>{final2.get('n_chromosomes',0)}</td></tr>
<tr><td>Final forks</td><td>{final1.get('n_forks',0)}</td><td>{final2.get('n_forks',0)}</td></tr>
<tr><td>Snapshots</td><td>{len(s1)}</td><td>{len(s2)}</td></tr>
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
<h2>Mass Comparison Table</h2>
<p>Values in femtograms (fg). Diff is |v2-v1|/v1.</p>
<table>
<tr><th rowspan="2">Time</th>
    <th colspan="3">Dry Mass</th>
    <th colspan="3">Protein</th>
    <th colspan="3">RNA</th>
    <th colspan="3">DNA</th>
    <th colspan="3">Small Mol</th></tr>
<tr>{''.join('<th>v1</th><th>v2</th><th>Diff</th>' for _ in mass_keys)}</tr>
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
    print(f"v1 vs v2 Comparison ({args.duration}s)")
    print("=" * 60)

    # Run both in parallel as subprocesses
    import subprocess as sp
    base = os.path.dirname(os.path.abspath(__file__))
    v1_result = os.path.join(base, REPORT_DIR, '_v1_result.json')
    v2_result = os.path.join(base, REPORT_DIR, '_v2_result.json')
    v1_script = os.path.join(base, 'scripts', 'run_v1.py')
    v2_script = os.path.join(base, 'scripts', 'run_v2.py')

    t0 = time.time()
    print(f"  Launching v1 and v2 in parallel...")
    v1_proc = sp.Popen([sys.executable, v1_script, str(args.duration), str(SNAPSHOT_INTERVAL), v1_result])
    v2_proc = sp.Popen([sys.executable, v2_script, str(args.duration), str(SNAPSHOT_INTERVAL), v2_result])

    v2_proc.wait()
    if os.path.exists(v2_result):
        with open(v2_result) as f:
            v2_data = json.load(f)
        os.unlink(v2_result)
        print(f"  v2: {v2_data['sim_time']}s in {v2_data['wall_time']:.1f}s ({v2_data['speed']:.1f}x)")
    else:
        print(f"  v2: FAILED")
        v2_data = {'engine': 'v2ecoli (FAILED)', 'snapshots': [], 'wall_time': 0, 'sim_time': 0, 'speed': 0, 'load_time': 0}

    v1_proc.wait()
    if os.path.exists(v1_result):
        with open(v1_result) as f:
            v1_data = json.load(f)
        os.unlink(v1_result)
        print(f"  v1: {v1_data['sim_time']}s in {v1_data['wall_time']:.1f}s ({v1_data['speed']:.1f}x)")
    else:
        print(f"  v1: FAILED (rc={v1_proc.returncode})")
        v1_data = {'engine': 'vEcoli (FAILED)', 'snapshots': [], 'wall_time': 0, 'sim_time': 0, 'speed': 0, 'load_time': 0}

    total = time.time() - t0

    report = generate_report(v1_data, v2_data, args.duration)
    print(f"Total: {total:.0f}s")

    # Open in browser
    import subprocess as sp
    sp.run(['open', report], capture_output=True)


if __name__ == '__main__':
    main()
