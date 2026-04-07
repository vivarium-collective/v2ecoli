"""
Compare partitioned vs departitioned E. coli simulation architectures.

Builds both composites, runs them to division (or MAX_DURATION), then
generates an HTML report with bigraph visualizations, mass trajectory
overlays, divergence plots, bulk count scatter, growth comparison, and
timing analysis.

Usage:
    python compare_architectures.py                        # default max 3600s
    python compare_architectures.py --max-duration 1800    # 30-min cap
    python compare_architectures.py --seed 42 --output out/my_report.html
"""

import os
import io
import re
import time
import base64
import json
import html as html_lib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from bigraph_viz import plot_bigraph

from v2ecoli.composite import make_composite
from v2ecoli.partitioned import make_partitioned_composite


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DURATION = 3600        # max seconds before giving up on division
SNAPSHOT_INTERVAL = 50     # capture snapshot every N seconds

MASS_FIELDS = [
    'cell_mass', 'dry_mass', 'protein_mass', 'rna_mass',
    'dna_mass', 'smallmolecule_mass', 'water_mass',
]
MASS_LABELS = {
    'cell_mass': 'Cell Mass',
    'dry_mass': 'Dry Mass',
    'protein_mass': 'Protein',
    'rna_mass': 'RNA',
    'dna_mass': 'DNA',
    'smallmolecule_mass': 'Small Molecule',
    'water_mass': 'Water',
}
ERROR_THRESHOLD = 5.0  # percent — max acceptable mass divergence

BIO_COLORS = {
    'dna': ('#FFB6C1', lambda n: 'chromosome' in n),
    'rna': ('#ADD8E6', lambda n: any(s in n for s in ('transcript', 'rna-', 'rna_', 'RNA', 'rnap'))),
    'protein': ('#90EE90', lambda n: any(s in n for s in ('polypeptide', 'protein', 'ribosome'))),
    'meta': ('#FFD700', lambda n: any(s in n for s in ('metabolism', 'equilibrium', 'complexation', 'two-component'))),
    'reg': ('#DDA0DD', lambda n: any(s in n for s in ('tf-', 'tf_'))),
    'alloc': ('#FFA07A', lambda n: 'allocator' in n),
    'infra': ('#E0E0E0', lambda n: any(s in n for s in ('unique_update', 'global_clock', 'emitter',
                                                          'mark_d_period', 'division', 'exchange',
                                                          'media_update', 'post-division'))),
    'listen': ('#D3D3D3', lambda n: 'listener' in n),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _get_emitter(composite):
    """Return the emitter instance from the composite, or None."""
    cell = composite.state.get('agents', {}).get('0', {})
    em = cell.get('emitter', {})
    if isinstance(em, dict) and 'instance' in em:
        return em['instance']
    return None


def _extract_snapshots(emitter):
    """Extract per-second mass snapshots from emitter history."""
    if emitter is None or not hasattr(emitter, 'history'):
        return []
    snapshots = []
    for snap in emitter.history:
        t = snap.get('global_time', 0)
        mass = snap.get('listeners', {}).get('mass', {}) if isinstance(snap.get('listeners'), dict) else {}
        row = {'time': float(t)}
        for field in MASS_FIELDS:
            row[field] = float(mass.get(field, 0))
        row['growth'] = float(mass.get('instantaneous_growth_rate', 0))
        snapshots.append(row)
    return snapshots


def _extract_bulk(composite):
    """Return the final bulk count array."""
    cell = composite.state.get('agents', {}).get('0', {})
    bulk = cell.get('bulk')
    if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
        return bulk['count'].astype(float)
    if isinstance(bulk, dict) and 'count' in bulk:
        return np.array(bulk['count'], dtype=float)
    return np.array([])


# ---------------------------------------------------------------------------
# Bigraph Visualization
# ---------------------------------------------------------------------------

def make_bigraph_svg(composite, out_dir, filename):
    """Generate an SVG visualization of the composite's process-bigraph network.

    Returns the SVG string with width/height stripped for responsive embedding.
    """
    state = composite.state
    cell = state.get('agents', {}).get('0', state)
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

    os.makedirs(out_dir, exist_ok=True)
    try:
        plot_bigraph(viz_state, remove_process_place_edges=True,
                     node_groups=[g for g in groups.values() if g],
                     node_fill_colors=colors, rankdir='LR',
                     dpi='72', port_labels=False, node_label_size='16pt',
                     label_margin='0.05', out_dir=out_dir,
                     filename=filename, file_format='svg')
        with open(os.path.join(out_dir, f'{filename}.svg')) as f:
            svg = f.read()
        svg = re.sub(r'width="[^"]*pt"', '', svg, count=1)
        svg = re.sub(r'height="[^"]*pt"', '', svg, count=1)
        return svg
    except Exception as e:
        return f'<p>Bigraph visualization failed: {html_lib.escape(str(e))}</p>'


# ---------------------------------------------------------------------------
# Step 1: Load Models
# ---------------------------------------------------------------------------

def load_models(cache_dir, seed):
    """Build both composites and return timing/structure info."""
    print("Step 1: Loading models...")

    t0 = time.time()
    dep = make_composite(cache_dir=cache_dir, seed=seed)
    dep_time = time.time() - t0
    print(f"  Departitioned composite built in {dep_time:.1f}s")

    t0 = time.time()
    part = make_partitioned_composite(cache_dir=cache_dir, seed=seed)
    part_time = time.time() - t0
    print(f"  Partitioned composite built in {part_time:.1f}s")

    dep_steps = len(dep.step_paths)
    dep_procs = len(dep.process_paths)
    part_steps = len(part.step_paths)
    part_procs = len(part.process_paths)

    info = {
        'dep_build_time': dep_time,
        'part_build_time': part_time,
        'dep_steps': dep_steps,
        'dep_processes': dep_procs,
        'part_steps': part_steps,
        'part_processes': part_procs,
    }
    print(f"  Departitioned: {dep_steps} steps, {dep_procs} processes")
    print(f"  Partitioned:   {part_steps} steps, {part_procs} processes")

    return dep, part, info


# ---------------------------------------------------------------------------
# Step 2: Run Simulations (to division)
# ---------------------------------------------------------------------------

def _run_to_division(composite, label):
    """Run a composite in chunks until division or MAX_DURATION.

    Returns (wall_time, total_sim_time, divided, chunk_snapshots).
    chunk_snapshots is a list of dicts recorded at each SNAPSHOT_INTERVAL.
    """
    cell = composite.state['agents']['0']
    initial_dry_mass = float(
        cell.get('listeners', {}).get('mass', {}).get('dry_mass', 380))

    t0 = time.time()
    total_run = 0
    divided = False
    chunk_snapshots = []

    while total_run < MAX_DURATION:
        chunk = min(SNAPSHOT_INTERVAL, MAX_DURATION - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            print(f"    [{label}] Simulation stopped at ~t={total_run + chunk}s: "
                  f"{type(e).__name__}")
            divided = True
            break
        total_run += chunk

        cell = composite.state.get('agents', {}).get('0')
        if cell is None:
            print(f"    [{label}] Agent removed (division) at t={total_run}s")
            divided = True
            break

        # Collect snapshot
        mass = cell.get('listeners', {}).get('mass', {})
        dry_mass = float(mass.get('dry_mass', 0))

        unique = cell.get('unique', {})
        fc = unique.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        chunk_snapshots.append({
            'time': total_run,
            'dry_mass': dry_mass,
            'protein_mass': float(mass.get('protein_mass', 0)),
            'rna_mass': float(mass.get('rna_mass', 0)),
            'dna_mass': float(mass.get('dna_mass', 0)),
            'n_chromosomes': n_chrom,
        })

        # Check division condition
        if n_chrom >= 2 and dry_mass >= initial_dry_mass * 2:
            print(f"    [{label}] Division ready at t={total_run}s: "
                  f"{n_chrom} chromosomes, dry_mass={dry_mass:.0f}fg "
                  f"(>= {initial_dry_mass * 2:.0f}fg)")
            divided = True
            break

        if total_run % 500 == 0:
            print(f"    [{label}] t={total_run}s: {n_chrom} chroms, "
                  f"dry_mass={dry_mass:.0f}fg")

    wall_time = time.time() - t0
    if not divided:
        print(f"    [{label}] Reached MAX_DURATION ({MAX_DURATION}s) without division")

    return wall_time, total_run, divided, chunk_snapshots


def run_simulations(dep, part):
    """Run both composites to division and capture snapshots."""
    print(f"\nStep 2: Running simulations (to division, max {MAX_DURATION}s each)...")

    dep_emitter = _get_emitter(dep)
    part_emitter = _get_emitter(part)

    dep_wall, dep_sim_time, dep_divided, dep_chunks = _run_to_division(dep, 'Dep')
    print(f"  Departitioned: {dep_wall:.1f}s wall, {dep_sim_time}s sim, "
          f"divided={dep_divided}")

    part_wall, part_sim_time, part_divided, part_chunks = _run_to_division(part, 'Part')
    print(f"  Partitioned:   {part_wall:.1f}s wall, {part_sim_time}s sim, "
          f"divided={part_divided}")

    dep_snaps = _extract_snapshots(dep_emitter)
    part_snaps = _extract_snapshots(part_emitter)
    dep_bulk = _extract_bulk(dep)
    part_bulk = _extract_bulk(part)

    print(f"  Emitter snapshots: {len(dep_snaps)} dep, {len(part_snaps)} part")

    return {
        'dep_wall': dep_wall,
        'part_wall': part_wall,
        'dep_sim_time': dep_sim_time,
        'part_sim_time': part_sim_time,
        'dep_divided': dep_divided,
        'part_divided': part_divided,
        'dep_snaps': dep_snaps,
        'part_snaps': part_snaps,
        'dep_chunks': dep_chunks,
        'part_chunks': part_chunks,
        'dep_bulk': dep_bulk,
        'part_bulk': part_bulk,
    }


# ---------------------------------------------------------------------------
# Step 3: Compute Metrics
# ---------------------------------------------------------------------------

def compute_metrics(sim_data):
    """Compute per-timestep mass divergence, bulk correlation, etc."""
    print("\nStep 3: Computing metrics...")

    dep_snaps = sim_data['dep_snaps']
    part_snaps = sim_data['part_snaps']

    # Build time-indexed dicts
    dep_by_t = {int(s['time']): s for s in dep_snaps}
    part_by_t = {int(s['time']): s for s in part_snaps}
    common_times = sorted(set(dep_by_t.keys()) & set(part_by_t.keys()))

    # Per-timestep percent difference for each mass field
    pct_diff = {field: [] for field in MASS_FIELDS}
    pct_times = []
    for t in common_times:
        d, p = dep_by_t[t], part_by_t[t]
        pct_times.append(t)
        for field in MASS_FIELDS:
            dv = d[field]
            pv = p[field]
            ref = max(abs(dv), abs(pv), 1e-12)
            pct_diff[field].append(abs(dv - pv) / ref * 100)

    # Max error across all fields and times
    max_errors = {f: max(vals) if vals else 0.0 for f, vals in pct_diff.items()}
    overall_max_error = max(max_errors.values()) if max_errors else 0.0

    # Bulk correlation
    dep_bulk = sim_data['dep_bulk']
    part_bulk = sim_data['part_bulk']
    min_len = min(len(dep_bulk), len(part_bulk))
    if min_len > 0:
        db = dep_bulk[:min_len]
        pb = part_bulk[:min_len]
        mask = (db > 0) | (pb > 0)
        if mask.sum() > 1:
            r, _ = stats.pearsonr(db[mask], pb[mask])
        else:
            r = float('nan')
        exact_match = int((db == pb).sum())
    else:
        r = float('nan')
        exact_match = 0

    # Total growth comparison
    dep_growth = [s['growth'] for s in dep_snaps]
    part_growth = [s['growth'] for s in part_snaps]
    dep_total_growth = sum(dep_growth)
    part_total_growth = sum(part_growth)

    # Wall time ratio
    wall_ratio = sim_data['part_wall'] / max(sim_data['dep_wall'], 1e-9)

    metrics = {
        'pct_times': pct_times,
        'pct_diff': pct_diff,
        'max_errors': max_errors,
        'overall_max_error': overall_max_error,
        'bulk_pearson_r': r,
        'bulk_exact_match': exact_match,
        'bulk_total': min_len,
        'dep_total_growth': dep_total_growth,
        'part_total_growth': part_total_growth,
        'wall_ratio': wall_ratio,
    }

    print(f"  Max mass error:  {overall_max_error:.4f}%")
    print(f"  Bulk Pearson r:  {r:.6f}")
    print(f"  Bulk exact match: {exact_match}/{min_len}")
    print(f"  Wall time ratio (part/dep): {wall_ratio:.2f}x")

    return metrics


# ---------------------------------------------------------------------------
# Step 4: Generate Plots
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)


def plot_mass_trajectories(sim_data):
    """2x3 grid: dry, protein, RNA, DNA, small molecule, cell mass."""
    fields = ['dry_mass', 'protein_mass', 'rna_mass',
              'dna_mass', 'smallmolecule_mass', 'cell_mass']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    dep_snaps = sim_data['dep_snaps']
    part_snaps = sim_data['part_snaps']

    for i, field in enumerate(fields):
        ax = axes[i]
        dep_t = [s['time'] / 60 for s in dep_snaps]
        dep_v = [s[field] for s in dep_snaps]
        part_t = [s['time'] / 60 for s in part_snaps]
        part_v = [s[field] for s in part_snaps]

        ax.plot(dep_t, dep_v, '-', color='#dc2626', lw=1.5, label='Departitioned')
        ax.plot(part_t, part_v, '--', color='#2563eb', lw=1.5, label='Partitioned')
        ax.set_title(MASS_LABELS.get(field, field), fontsize=10)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.set_ylabel('Mass (fg)', fontsize=8)
        ax.legend(fontsize=7)
        _style_ax(ax)

    fig.suptitle('Mass Trajectories', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mass_divergence(metrics):
    """1x5 subplots: percent difference over time for key mass components."""
    fields = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallmolecule_mass']
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    times = np.array(metrics['pct_times']) / 60

    for i, field in enumerate(fields):
        ax = axes[i]
        vals = metrics['pct_diff'][field]
        ax.plot(times, vals, '-', color='#7c3aed', lw=1.2)
        ax.fill_between(times, vals, alpha=0.15, color='#7c3aed')
        ax.set_title(MASS_LABELS.get(field, field), fontsize=9)
        ax.set_xlabel('Time (min)', fontsize=7)
        ax.set_ylabel('% Difference', fontsize=7)
        _style_ax(ax)

    fig.suptitle('Mass Percent Difference (partitioned vs departitioned)', fontsize=11, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_bulk_scatter(sim_data):
    """Scatter plot of final bulk counts on log scale."""
    dep_bulk = sim_data['dep_bulk']
    part_bulk = sim_data['part_bulk']
    min_len = min(len(dep_bulk), len(part_bulk))

    fig, ax = plt.subplots(figsize=(6, 6))
    if min_len > 0:
        db = dep_bulk[:min_len]
        pb = part_bulk[:min_len]
        mask = (db > 0) & (pb > 0)
        ax.scatter(db[mask], pb[mask], s=4, alpha=0.3, color='#2563eb', edgecolors='none')

        lim_min = min(db[mask].min(), pb[mask].min()) * 0.5 if mask.sum() > 0 else 0.1
        lim_max = max(db[mask].max(), pb[mask].max()) * 2 if mask.sum() > 0 else 10
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=0.8, alpha=0.5, label='y = x')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No bulk data', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Departitioned (count)', fontsize=9)
    ax.set_ylabel('Partitioned (count)', fontsize=9)
    ax.set_title('Final Bulk Molecule Counts', fontsize=11)
    ax.set_aspect('equal')
    _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_growth_comparison(sim_data):
    """Cumulative growth over time for both architectures."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for snaps, color, ls, label in [
        (sim_data['dep_snaps'], '#dc2626', '-', 'Departitioned'),
        (sim_data['part_snaps'], '#2563eb', '--', 'Partitioned'),
    ]:
        if not snaps:
            continue
        times = [s['time'] / 60 for s in snaps]
        growth = np.cumsum([s['growth'] for s in snaps])
        ax.plot(times, growth, ls, color=color, lw=1.5, label=label)

    ax.set_xlabel('Time (min)', fontsize=9)
    ax.set_ylabel('Cumulative Growth Rate', fontsize=9)
    ax.set_title('Growth Comparison', fontsize=11)
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_timing(model_info, sim_data):
    """Bar chart comparing wall time and sim/wall ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Wall time
    ax = axes[0]
    labels = ['Build', 'Simulation']
    dep_vals = [model_info['dep_build_time'], sim_data['dep_wall']]
    part_vals = [model_info['part_build_time'], sim_data['part_wall']]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, dep_vals, w, color='#dc2626', alpha=0.8, label='Departitioned')
    ax.bar(x + w / 2, part_vals, w, color='#2563eb', alpha=0.8, label='Partitioned')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Wall Time (s)')
    ax.set_title('Wall Time Comparison')
    ax.legend(fontsize=8)
    _style_ax(ax)

    # Sim/wall ratio
    ax = axes[1]
    part_ratio = sim_data['part_wall'] / max(sim_data['dep_wall'], 1e-9)
    ax.bar(['Departitioned', 'Partitioned'], [1.0, part_ratio],
           color=['#dc2626', '#2563eb'], alpha=0.8)
    ax.set_ylabel('Relative Wall Time')
    ax.set_title('Simulation Speed (lower = faster)')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    _style_ax(ax)

    fig.tight_layout()
    return fig_to_b64(fig)


def generate_plots(model_info, sim_data, metrics):
    """Generate all plots and return dict of base64 PNGs."""
    print("\nStep 4: Generating plots...")
    plots = {}
    plots['mass_traj'] = plot_mass_trajectories(sim_data)
    plots['mass_div'] = plot_mass_divergence(metrics)
    plots['bulk_scatter'] = plot_bulk_scatter(sim_data)
    plots['growth'] = plot_growth_comparison(sim_data)
    plots['timing'] = plot_timing(model_info, sim_data)
    print(f"  Generated {len(plots)} plots")
    return plots


# ---------------------------------------------------------------------------
# Step 5: Generate HTML Report
# ---------------------------------------------------------------------------

def generate_html(model_info, sim_data, metrics, plots, bigraph_svgs,
                  seed, output):
    """Write a single-file HTML report with embedded plots and bigraphs."""
    print(f"\nStep 5: Generating HTML report -> {output}")

    dep_sim_time = sim_data['dep_sim_time']
    part_sim_time = sim_data['part_sim_time']
    dep_divided = sim_data['dep_divided']
    part_divided = sim_data['part_divided']

    max_err = metrics['overall_max_error']
    verdict_ok = max_err < ERROR_THRESHOLD
    verdict_color = '#16a34a' if verdict_ok else '#dc2626'
    verdict_text = 'PASS' if verdict_ok else 'FAIL'
    verdict_detail = (
        f'Max mass divergence {max_err:.4f}% is below {ERROR_THRESHOLD}% threshold.'
        if verdict_ok else
        f'Max mass divergence {max_err:.4f}% exceeds {ERROR_THRESHOLD}% threshold.'
    )

    r_val = metrics['bulk_pearson_r']
    r_str = f'{r_val:.6f}' if not np.isnan(r_val) else 'N/A'

    # Division status strings
    dep_div_str = (f'Yes (t={dep_sim_time}s)' if dep_divided
                   else f'No (ran {dep_sim_time}s)')
    part_div_str = (f'Yes (t={part_sim_time}s)' if part_divided
                    else f'No (ran {part_sim_time}s)')

    # Architecture comparison table rows
    arch_rows = f"""
    <tr><td>Steps</td><td>{model_info['dep_steps']}</td><td>{model_info['part_steps']}</td></tr>
    <tr><td>Processes</td><td>{model_info['dep_processes']}</td><td>{model_info['part_processes']}</td></tr>
    <tr><td>Build Time</td><td>{model_info['dep_build_time']:.1f}s</td><td>{model_info['part_build_time']:.1f}s</td></tr>
    <tr><td>Sim Wall Time</td><td>{sim_data['dep_wall']:.1f}s</td><td>{sim_data['part_wall']:.1f}s</td></tr>
    <tr><td>Sim Duration</td><td>{dep_sim_time}s</td><td>{part_sim_time}s</td></tr>
    <tr><td>Divided</td><td>{dep_div_str}</td><td>{part_div_str}</td></tr>
    """

    # Per-field max error rows
    field_rows = ''
    for field in MASS_FIELDS:
        err = metrics['max_errors'].get(field, 0)
        color = '#16a34a' if err < ERROR_THRESHOLD else '#dc2626'
        field_rows += (
            f'<tr><td>{MASS_LABELS.get(field, field)}</td>'
            f'<td style="color:{color}">{err:.4f}%</td></tr>'
        )

    date_str = time.strftime('%Y-%m-%d %H:%M')
    dep_svg = bigraph_svgs.get('dep', '<p>Not available</p>')
    part_svg = bigraph_svgs.get('part', '<p>Not available</p>')

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli Architecture Comparison Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 1.8em; margin: 15px 0; color: #0f172a; }}
  h2 {{ font-size: 1.3em; margin: 25px 0 10px; color: #334155;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
  .header {{ background: #0f172a; color: white; padding: 20px 25px; border-radius: 10px;
             margin-bottom: 15px; }}
  .header h1 {{ color: white; margin: 0 0 6px; }}
  .header p {{ color: #94a3b8; font-size: 0.9em; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px; margin: 15px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 0.72em; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.5em; font-weight: 700; margin-top: 4px; }}
  .green {{ color: #16a34a; }} .blue {{ color: #2563eb; }}
  .red {{ color: #dc2626; }} .purple {{ color: #7c3aed; }}
  .plot {{ background: white; border-radius: 8px; padding: 14px; margin: 12px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
  .plot img {{ max-width: 100%; }}
  .section {{ background: white; border-radius: 8px; padding: 18px; margin: 12px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .verdict {{ border-radius: 10px; padding: 20px 25px; margin: 15px 0;
              font-size: 1.1em; }}
  footer {{ margin-top: 30px; padding: 15px 0; border-top: 1px solid #e2e8f0;
            color: #94a3b8; font-size: 0.75em; text-align: center; }}

  /* Bigraph visualization */
  .bigraph-row {{ display: flex; gap: 16px; margin: 12px 0; }}
  .bigraph-col {{ flex: 1; min-width: 0; }}
  .bigraph-col h3 {{ font-size: 1em; color: #334155; margin-bottom: 6px; text-align: center; }}
  .bigraph-container {{ position: relative; border: 1px solid #e2e8f0; border-radius: 8px;
                        background: #fafafa; overflow: hidden; height: 700px; cursor: grab; }}
  .bigraph-container.grabbing {{ cursor: grabbing; }}
  .bigraph-container svg {{ position: absolute; transform-origin: 0 0; }}
  .bigraph-controls {{ display: flex; gap: 6px; margin: 8px 0; justify-content: center; }}
  .bigraph-controls button {{ padding: 4px 12px; border: 1px solid #cbd5e1; border-radius: 4px;
                              background: white; cursor: pointer; font-size: 0.85em; }}
  .bigraph-controls button:hover {{ background: #f1f5f9; }}
</style>
</head>
<body>

<div class="header">
  <h1>v2ecoli Architecture Comparison Report</h1>
  <p>{date_str} &middot; Run to division (max {MAX_DURATION}s) &middot; Seed: {seed} &middot;
     Partitioned vs Departitioned</p>
</div>

<!-- Summary Cards -->
<div class="cards">
  <div class="card">
    <div class="label">Steps (Dep / Part)</div>
    <div class="value blue">{model_info['dep_steps']} / {model_info['part_steps']}</div>
  </div>
  <div class="card">
    <div class="label">Sim Duration (Dep / Part)</div>
    <div class="value blue">{dep_sim_time}s / {part_sim_time}s</div>
  </div>
  <div class="card">
    <div class="label">Wall Time (Dep / Part)</div>
    <div class="value blue">{sim_data['dep_wall']:.1f}s / {sim_data['part_wall']:.1f}s</div>
  </div>
  <div class="card">
    <div class="label">Max Mass Error</div>
    <div class="value" style="color:{verdict_color}">{max_err:.4f}%</div>
  </div>
  <div class="card">
    <div class="label">Bulk Correlation (r)</div>
    <div class="value purple">{r_str}</div>
  </div>
</div>

<!-- Architecture Comparison -->
<h2>Architecture Comparison</h2>
<div class="section">
  <table>
    <thead><tr><th>Metric</th><th>Departitioned</th><th>Partitioned</th></tr></thead>
    <tbody>{arch_rows}</tbody>
  </table>
</div>

<!-- Network Architecture (Bigraph Visualization) -->
<h2>Network Architecture</h2>
<div class="section">
  <p>Interactive visualization of each architecture's process-bigraph network.
  Scroll to zoom, drag to pan.</p>
</div>
<div class="bigraph-row">
  <div class="bigraph-col">
    <h3>Departitioned</h3>
    <div class="bigraph-controls">
      <button onclick="bgZoom('dep', 1.3)">Zoom In</button>
      <button onclick="bgZoom('dep', 0.77)">Zoom Out</button>
      <button onclick="bgReset('dep')">Fit</button>
      <span id="bg-zoom-dep" style="font-size:0.8em;color:#64748b;padding:4px;"></span>
    </div>
    <div class="bigraph-container" id="bg-ctr-dep">{dep_svg}</div>
  </div>
  <div class="bigraph-col">
    <h3>Partitioned</h3>
    <div class="bigraph-controls">
      <button onclick="bgZoom('part', 1.3)">Zoom In</button>
      <button onclick="bgZoom('part', 0.77)">Zoom Out</button>
      <button onclick="bgReset('part')">Fit</button>
      <span id="bg-zoom-part" style="font-size:0.8em;color:#64748b;padding:4px;"></span>
    </div>
    <div class="bigraph-container" id="bg-ctr-part">{part_svg}</div>
  </div>
</div>
<script>
(function() {{
  function initBigraph(id) {{
    const ctr = document.getElementById('bg-ctr-' + id);
    const svg = ctr.querySelector('svg');
    if (!svg) return;
    let scale = 1, tx = 0, ty = 0, dragging = false, sx, sy;
    const zoomLabel = document.getElementById('bg-zoom-' + id);
    function apply() {{
      svg.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
      zoomLabel.textContent = Math.round(scale * 100) + '%';
    }}
    function fit() {{
      const bb = svg.getBBox();
      const cw = ctr.clientWidth, ch = ctr.clientHeight;
      scale = Math.min(cw / (bb.width + 40), ch / (bb.height + 40), 2);
      tx = (cw - bb.width * scale) / 2 - bb.x * scale;
      ty = (ch - bb.height * scale) / 2 - bb.y * scale;
      apply();
    }}
    svg.style.width = svg.getAttribute('viewBox') ? '' : (svg.getBBox().width + 'px');
    svg.style.height = svg.getAttribute('viewBox') ? '' : (svg.getBBox().height + 'px');
    svg.removeAttribute('width'); svg.removeAttribute('height');
    setTimeout(fit, 50);
    // Store fit/zoom functions keyed by id
    if (!window._bgFns) window._bgFns = {{}};
    window._bgFns[id] = {{ fit: fit }};
    ctr.addEventListener('wheel', function(e) {{
      e.preventDefault();
      const f = e.deltaY < 0 ? 1.12 : 0.89;
      const r = ctr.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      tx = mx - f * (mx - tx); ty = my - f * (my - ty);
      scale *= f; apply();
    }}, {{passive: false}});
    ctr.addEventListener('mousedown', function(e) {{
      dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
      ctr.classList.add('grabbing');
    }});
    window.addEventListener('mousemove', function(e) {{
      if (!dragging) return;
      tx = e.clientX - sx; ty = e.clientY - sy; apply();
    }});
    window.addEventListener('mouseup', function() {{
      if (dragging) {{ dragging = false; ctr.classList.remove('grabbing'); }}
    }});
    // Expose zoom for this instance
    if (!window._bgZoomFns) window._bgZoomFns = {{}};
    window._bgZoomFns[id] = function(f) {{
      const cx = ctr.clientWidth / 2, cy = ctr.clientHeight / 2;
      tx = cx - f * (cx - tx); ty = cy - f * (cy - ty);
      scale *= f; apply();
    }};
  }}
  initBigraph('dep');
  initBigraph('part');
  window.bgZoom = function(id, f) {{
    if (window._bgZoomFns && window._bgZoomFns[id]) window._bgZoomFns[id](f);
  }};
  window.bgReset = function(id) {{
    if (window._bgFns && window._bgFns[id]) window._bgFns[id].fit();
  }};
}})();
</script>

<!-- Mass Trajectories -->
<h2>Mass Trajectories</h2>
<div class="plot">
  <img src="data:image/png;base64,{plots['mass_traj']}" alt="Mass Trajectories">
</div>

<!-- Mass Divergence -->
<h2>Mass Divergence</h2>
<div class="section">
  <table>
    <thead><tr><th>Component</th><th>Max % Difference</th></tr></thead>
    <tbody>{field_rows}</tbody>
  </table>
</div>
<div class="plot">
  <img src="data:image/png;base64,{plots['mass_div']}" alt="Mass Divergence">
</div>

<!-- Bulk Count Scatter -->
<h2>Bulk Count Scatter</h2>
<div class="section">
  <p>Final bulk molecule counts: <strong>{metrics['bulk_exact_match']}</strong> /
     {metrics['bulk_total']} exact matches.
     Pearson r = <strong>{r_str}</strong>.</p>
</div>
<div class="plot">
  <img src="data:image/png;base64,{plots['bulk_scatter']}" alt="Bulk Count Scatter">
</div>

<!-- Growth Comparison -->
<h2>Growth Comparison</h2>
<div class="plot">
  <img src="data:image/png;base64,{plots['growth']}" alt="Growth Comparison">
</div>

<!-- Timing Comparison -->
<h2>Timing Comparison</h2>
<div class="plot">
  <img src="data:image/png;base64,{plots['timing']}" alt="Timing Comparison">
</div>

<!-- Verdict -->
<div class="verdict" style="background:{'#dcfce7' if verdict_ok else '#fee2e2'};
     border: 2px solid {verdict_color};">
  <strong style="font-size: 1.3em; color: {verdict_color};">{verdict_text}</strong>
  <span style="margin-left: 12px;">{verdict_detail}</span>
</div>

<footer>
  v2ecoli Architecture Comparison &middot; Generated {date_str} &middot;
  Dep: {dep_sim_time}s / Part: {part_sim_time}s &middot; Seed {seed}
</footer>

</body>
</html>""")

    print(f"  Report written to {output}")
    return output


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_comparison(max_duration=3600, seed=0, cache_dir='out/cache',
                   output='out/comparison_report.html'):
    """Execute the full comparison pipeline."""
    global MAX_DURATION
    MAX_DURATION = max_duration

    print(f"=== v2ecoli Architecture Comparison ===")
    print(f"    Max Duration: {MAX_DURATION}s, Seed: {seed}, Cache: {cache_dir}\n")

    # Step 1
    dep, part, model_info = load_models(cache_dir, seed)

    # Generate bigraph SVGs (before simulation, while state is clean)
    print("\n  Generating bigraph visualizations...")
    out_dir = os.path.join(os.path.dirname(output) or '.', 'plots')
    bigraph_svgs = {
        'dep': make_bigraph_svg(dep, out_dir, 'bigraph_dep'),
        'part': make_bigraph_svg(part, out_dir, 'bigraph_part'),
    }
    print("  Bigraph SVGs generated.")

    # Step 2
    sim_data = run_simulations(dep, part)

    # Step 3
    metrics = compute_metrics(sim_data)

    # Step 4
    plots = generate_plots(model_info, sim_data, metrics)

    # Step 5
    report_path = generate_html(
        model_info, sim_data, metrics, plots, bigraph_svgs, seed, output)

    print(f"\n=== Done. Report: {report_path} ===")
    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare partitioned vs departitioned E. coli architectures')
    parser.add_argument('--max-duration', type=int, default=3600,
                        help='Max simulation duration in seconds (default: 3600)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--cache-dir', default='out/cache',
                        help='Cache directory for initial state (default: out/cache)')
    parser.add_argument('--output', default='out/comparison_report.html',
                        help='Output HTML report path (default: out/comparison_report.html)')
    args = parser.parse_args()

    run_comparison(
        max_duration=args.max_duration,
        seed=args.seed,
        cache_dir=args.cache_dir,
        output=args.output,
    )
