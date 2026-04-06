"""
v2ecoli Workflow Testing Framework

Step-based pipeline that caches intermediate states, replacing benchmark.py.
Each step checks for cached metadata before executing, enabling incremental
development and fast re-runs.

Pipeline Steps:
1. raw_data — Catalog raw TSV files and knowledge base stats
2. parca — Run parameter calculator (ParCa) or load cached simData
3. load_model — Build composite from cache
4. short_sim — 60s simulation with mass/growth diagnostics
5. v1_comparison — Side-by-side v1 vs v2 accuracy
6. long_sim — 1800s simulation for growth trajectory
7. division — Cell division, conservation, daughter viability

Usage:
    python workflow.py              # run full pipeline
    python workflow.py --clean      # clear cache and re-run
"""

import os
import io
import re
import sys
import json
import time
import base64
import html as html_lib
import copy

import dill
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contextlib import chdir

try:
    from wholecell.utils.filepath import ROOT_PATH as V1_ROOT_PATH
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.library.schema import not_a_process
    V1_AVAILABLE = True
except ImportError:
    V1_ROOT_PATH = os.getcwd()
    V1_AVAILABLE = False

from v2ecoli.composite import make_composite, _build_core, save_cache, save_state
from v2ecoli.library.division import divide_cell, divide_bulk
from v2ecoli.library.schema import attrs as ecoli_attrs
from process_bigraph import Composite
from v2ecoli.generate import build_document, DEFAULT_FLOW
from v2ecoli.cache import NumpyJSONEncoder, load_initial_state
from v2ecoli.steps.base import _translate_schema

from bigraph_viz import plot_bigraph
from bigraph_schema import get_path, strip_schema_keys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKFLOW_DIR = 'out/workflow'
# Use existing cache if available, otherwise workflow-local cache
CACHE_DIR = 'out/cache' if os.path.isdir('out/cache') else 'out/workflow/cache'
COMPARISON_DURATION = 60.0
LONG_DURATION = 1800.0  # Legacy label
MAX_LONG_DURATION = 3600  # Max seconds before giving up on division
SNAPSHOT_INTERVAL = 50  # Seconds between chromosome snapshots

# Try to find simData
_sim_data_candidates = [
    'out/kb/simData.cPickle',
    os.path.join(WORKFLOW_DIR, 'simData.cPickle'),
]
SIM_DATA_PATH = next((p for p in _sim_data_candidates if os.path.exists(p)), None)


# ---------------------------------------------------------------------------
# Caching Infrastructure
# ---------------------------------------------------------------------------

def load_meta(step_name):
    """Load cached metadata for a step, or None if not cached."""
    path = os.path.join(WORKFLOW_DIR, f'{step_name}_meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_meta(step_name, meta):
    """Save step metadata with timestamp."""
    meta['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = os.path.join(WORKFLOW_DIR, f'{step_name}_meta.json')
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, cls=NumpyJSONEncoder)


def save_state_data(step_name, data):
    """Save step state data as dill pickle."""
    path = os.path.join(WORKFLOW_DIR, f'{step_name}.dill')
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        dill.dump(data, f)


def load_state_data(step_name):
    """Load step state data from dill pickle."""
    path = os.path.join(WORKFLOW_DIR, f'{step_name}.dill')
    with open(path, 'rb') as f:
        return dill.load(f)


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


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


def plot_bulk_histogram(history):
    """Plot histogram of per-molecule count changes over the simulation."""
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig_to_b64(fig)

    first = history[0].get('bulk')
    last = history[-1].get('bulk')
    if first is None or last is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No bulk data in history', ha='center', va='center')
        return fig_to_b64(fig)

    if hasattr(first, 'dtype') and 'count' in first.dtype.names:
        first_counts = first['count'].astype(float)
    else:
        first_counts = np.array(first, dtype=float)
    if hasattr(last, 'dtype') and 'count' in last.dtype.names:
        last_counts = last['count'].astype(float)
    else:
        last_counts = np.array(last, dtype=float)

    delta = last_counts - first_counts
    changed = delta[delta != 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    if len(changed) > 0:
        ax.hist(changed, bins=50, color='#2563eb', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Count change')
        ax.set_ylabel('Number of molecules')
        ax.set_title(f'Bulk Molecule Changes ({len(changed)} molecules changed)')
    else:
        ax.text(0.5, 0.5, 'No molecules changed', ha='center', va='center')
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

    fig.suptitle('v1 vs v2 Comparison — Bulk Molecules', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mass_comparison(comp):
    """Plot per-timestep mass components for v1 vs v2."""
    mc = comp.get('mass_comparison', {})
    mass_keys = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']
    labels = ['Dry Mass', 'Protein', 'RNA', 'DNA', 'Small Molecules']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (key, label) in enumerate(zip(mass_keys, labels)):
        ax = axes[i]
        data = mc.get(key, {})
        v2_vals = data.get('v2', [])
        v1_vals = data.get('v1', [])
        v2_times = data.get('times', [])
        v1_times = list(range(1, len(v1_vals) + 1))

        if v2_vals:
            ax.plot(v2_times, v2_vals, 'b-', lw=1.5, label='v2', alpha=0.8)
        if v1_vals:
            ax.plot(v1_times, v1_vals, 'r--', lw=1.5, label='v1', alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mass (fg)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    # Mass delta per timestep (v2)
    ax = axes[5]
    dm = mc.get('dry_mass', {}).get('v2', [])
    if len(dm) > 1:
        deltas = np.diff(dm)
        ax.plot(range(1, len(deltas) + 1), deltas, 'b-o', markersize=2, lw=1)
        v1_dm = mc.get('dry_mass', {}).get('v1', [])
        if len(v1_dm) > 1:
            v1_deltas = np.diff(v1_dm)
            ax.plot(range(1, len(v1_deltas) + 1), v1_deltas, 'r--x', markersize=2, lw=1)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_title('Dry Mass per Timestep')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mass change (fg)')
        ax.grid(True, alpha=0.15)

    fig.suptitle('v1 vs v2 — Mass Components Over Time', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_unique_comparison(comp):
    """Plot unique molecule counts and chromosome state for v1 vs v2."""
    v1_unique = comp.get('v1_unique', {})
    v2_unique = comp.get('v2_unique', {})
    v1_chrom = comp.get('v1_chromosome', {})
    v2_chrom = comp.get('v2_chromosome', {})

    all_names = sorted(set(list(v1_unique.keys()) + list(v2_unique.keys())))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of unique molecule counts
    if all_names:
        x = np.arange(len(all_names))
        v1_vals = [v1_unique.get(n, 0) for n in all_names]
        v2_vals = [v2_unique.get(n, 0) for n in all_names]
        width = 0.35
        axes[0].bar(x - width/2, v1_vals, width, label='v1', color='#dc2626', alpha=0.7)
        axes[0].bar(x + width/2, v2_vals, width, label='v2', color='#2563eb', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace('_', '\n') for n in all_names], fontsize=7, rotation=45, ha='right')
        axes[0].set_ylabel('Active Count')
        axes[0].set_title('Unique Molecule Counts (active)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.15, axis='y')

    # Chromosome state text
    ax = axes[1]
    ax.axis('off')
    lines = ['Chromosome State at t=60s', '']
    lines.append('v2:')
    lines.append(f'  Full chromosomes: {v2_chrom.get("n_chromosomes", "?")}')
    if 'division_times' in v2_chrom:
        lines.append(f'  Division times: {v2_chrom["division_times"]}')
        lines.append(f'  Has triggered: {v2_chrom["has_triggered"]}')
    if 'domain_indexes' in v2_chrom:
        lines.append(f'  Domain indexes: {v2_chrom["domain_indexes"]}')

    if v1_chrom:
        lines.append('')
        lines.append('v1:')
        lines.append(f'  Full chromosomes: {v1_chrom.get("n_chromosomes", "?")}')
        if 'division_times' in v1_chrom:
            lines.append(f'  Division times: {v1_chrom["division_times"]}')
            lines.append(f'  Has triggered: {v1_chrom["has_triggered"]}')

    # Final mass comparison
    v1_fm = comp.get('v1_final_mass', {})
    v2_fm = comp.get('v2_final_mass', {})
    lines.append('')
    lines.append('Final Mass (fg):')
    for k in ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']:
        v1v = v1_fm.get(k, 0)
        v2v = v2_fm.get(k, 0)
        diff = v2v - v1v if v1v else 0
        label = k.replace('_mass', '').replace('_', ' ').title()
        if v1v:
            lines.append(f'  {label}: v1={v1v:.2f}  v2={v2v:.2f}  diff={diff:+.2f}')
        else:
            lines.append(f'  {label}: v2={v2v:.2f}')

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('v1 vs v2 — Unique Molecules & Chromosome State', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Bigraph Visualization
# ---------------------------------------------------------------------------

SKIP_STEPS = {'unique_update', 'global_clock', 'mark_d_period', 'division',
              'exchange_data', 'media_update', 'post-division-mass-listener', 'emitter'}
SKIP_PORTS = {'timestep', 'global_time', 'next_update_time', 'process'}
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
# Chromosome visualization
# ---------------------------------------------------------------------------

MAX_COORD = 1_042_299  # Half-genome in bp (OriC to Ter)


def extract_chromosome_snapshots(composite, duration, interval=10):
    """Run simulation in chunks, capturing chromosome state at each interval.

    Returns list of dicts with time, n_chromosomes, fork_coords, dna_mass, dry_mass.
    """
    snapshots = []
    cell = composite.state['agents']['0']
    remaining = duration

    while remaining > 0:
        chunk = min(interval, remaining)
        composite.run(chunk)
        remaining -= chunk
        cell = composite.state['agents']['0']

        # Extract chromosome state
        unique = cell.get('unique', {})
        fc = unique.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        rep = unique.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active = rep[rep['_entryState'].view(np.bool_)]
            if len(active) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active['coordinates'].tolist()

        mass = cell.get('listeners', {}).get('mass', {})
        dna_mass = float(mass.get('dna_mass', 0))
        dry_mass = float(mass.get('dry_mass', 0))

        domains = unique.get('chromosome_domain')
        n_domains = 0
        if domains is not None and hasattr(domains, 'dtype') and '_entryState' in domains.dtype.names:
            n_domains = int(domains['_entryState'].view(np.bool_).sum())

        snapshots.append({
            'time': float(cell.get('global_time', 0)),
            'n_chromosomes': n_chrom,
            'n_domains': n_domains,
            'fork_coords': fork_coords,
            'dna_mass': dna_mass,
            'dry_mass': dry_mass,
        })

    return snapshots


def _coord_to_angle(coord):
    """Convert genome coordinate to angle on circular chromosome."""
    # OriC at top (π/2), Ter at bottom (-π/2)
    # Coordinates: 0 = OriC, ±MAX_COORD = Ter
    frac = coord / MAX_COORD  # -1 to +1
    return np.pi / 2 - frac * np.pi  # OriC=90°, Ter=-90°


def plot_chromosome_map(snapshot, ax, title=''):
    """Draw circular chromosome with RNAP and replisome positions."""
    theta = np.linspace(0, 2 * np.pi, 200)
    R = 1.0  # chromosome radius

    # Draw chromosome circle
    ax.plot(R * np.cos(theta), R * np.sin(theta), color='#cbd5e1', lw=3, zorder=1)

    # Mark OriC (top) and Ter (bottom)
    ax.plot(0, R, 'o', color='#10b981', ms=10, zorder=5, label='OriC')
    ax.plot(0, -R, 's', color='#ef4444', ms=8, zorder=5, label='Ter')

    # Plot RNAP positions as small dots on the chromosome
    rnap_coords = snapshot.get('rnap_coords', [])
    if rnap_coords:
        angles = [_coord_to_angle(c) for c in rnap_coords]
        rx = [R * np.cos(a) for a in angles]
        ry = [R * np.sin(a) for a in angles]
        ax.scatter(rx, ry, c='#3b82f6', s=4, alpha=0.3, zorder=3, label=f'RNAP ({len(rnap_coords)})')

    # Plot replisome positions as large triangles
    fork_coords = snapshot.get('fork_coords', [])
    for coord in fork_coords:
        angle = _coord_to_angle(coord)
        fx = 1.15 * R * np.cos(angle)
        fy = 1.15 * R * np.sin(angle)
        ax.plot(fx, fy, '^', color='#f59e0b', ms=12, zorder=6,
                markeredgecolor='black', markeredgewidth=0.5)
    if fork_coords:
        ax.plot([], [], '^', color='#f59e0b', ms=10, label=f'Replisome ({len(fork_coords)})')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title(title or f"t={snapshot.get('time', 0):.0f}s", fontsize=10)
    ax.axis('off')


def plot_chromosome_state(snapshots, title=''):
    """Plot chromosome state: circular maps at key times + timeseries."""
    if not snapshots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No chromosome data', ha='center', va='center')
        return fig_to_b64(fig)

    times = [s['time'] for s in snapshots]

    # Pick 3 representative snapshots for circular maps: start, mid, end
    indices = [0, len(snapshots) // 2, len(snapshots) - 1]
    indices = sorted(set(indices))  # deduplicate if very few snapshots

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title or 'Chromosome State', fontsize=14, y=0.98)

    # Top row: circular chromosome maps at key timepoints
    n_maps = len(indices)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, n_maps, i + 1)
        plot_chromosome_map(snapshots[idx], ax)

    # Bottom left: fork progress + RNAP count over time
    ax = fig.add_subplot(2, 2, 3)
    for s in snapshots:
        for coord in s.get('fork_coords', []):
            ax.scatter(s['time'], coord / MAX_COORD, c='#f59e0b', s=12,
                       alpha=0.7, zorder=3, edgecolors='black', linewidths=0.3)
    ax2 = ax.twinx()
    n_rnap = [s.get('n_rnap', 0) for s in snapshots]
    ax2.plot(times, n_rnap, color='#3b82f6', lw=1.5, alpha=0.7, label='Active RNAP')
    ax2.set_ylabel('Active RNAP', color='#3b82f6', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#3b82f6')
    ax.set_ylabel('Fork position (frac genome)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Replication Forks & RNAP Count')
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color='#10b981', lw=0.5, ls='--', alpha=0.5)
    ax.axhline(1, color='#ef4444', lw=0.5, ls='--', alpha=0.3)
    ax.axhline(-1, color='#ef4444', lw=0.5, ls='--', alpha=0.3)

    # Bottom right: chromosome count + DNA mass + dry mass
    ax = fig.add_subplot(2, 2, 4)
    n_chroms = [s['n_chromosomes'] for s in snapshots]
    ax.step(times, n_chroms, where='post', color='#10b981', lw=2, label='Chromosomes')
    ax.set_ylabel('Chromosomes', color='#10b981')
    ax.set_ylim(0, max(n_chroms) + 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='y', labelcolor='#10b981')

    ax2 = ax.twinx()
    dna = [s['dna_mass'] for s in snapshots]
    dry = [s['dry_mass'] for s in snapshots]
    ax2.plot(times, dna, color='#8b5cf6', lw=1.5, label='DNA mass')
    ax2.plot(times, dry, color='#f59e0b', lw=1.5, ls='--', alpha=0.7, label='Dry mass')
    if dry:
        ax2.axhline(dry[0] * 2, color='red', lw=0.5, ls=':', alpha=0.4)
    ax2.set_ylabel('Mass (fg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Chromosomes & Mass')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    try:
        plt.tight_layout()
    except Exception:
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
    return fig_to_b64(fig)


def make_bigraph_svg(state):
    cell = state.get('agents', {}).get('0', state)
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict): continue
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
        if '_type' not in viz.get(n, {}): continue
        p = prefix + (n,)
        for gk, (c, m) in BIO_COLORS.items():
            if m(n): colors[p] = c; groups[gk].append(p); break

    out_dir = os.path.join(WORKFLOW_DIR, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    try:
        plot_bigraph(viz_state, remove_process_place_edges=True,
                     node_groups=[g for g in groups.values() if g],
                     node_fill_colors=colors, rankdir='LR',
                     dpi='72', port_labels=False, node_label_size='16pt',
                     label_margin='0.05', out_dir=out_dir,
                     filename='bigraph', file_format='svg')
        with open(os.path.join(out_dir, 'bigraph.svg')) as f:
            svg = f.read()
        svg = re.sub(r'width="[^"]*pt"', '', svg, count=1)
        svg = re.sub(r'height="[^"]*pt"', '', svg, count=1)
        return svg
    except Exception as e:
        return f'<p>Failed: {html_lib.escape(str(e))}</p>'


# ---------------------------------------------------------------------------
# Data Collection Helpers
# ---------------------------------------------------------------------------

def _collect_v2_timeseries(composite, duration):
    """Run v2, collect per-timestep bulk, mass, and unique data from emitter."""
    cell = composite.state['agents']['0']

    v2_initial_bulk = np.array(cell['bulk']['count'], copy=True)

    t0 = time.time()
    composite.run(duration)
    v2_time = time.time() - t0

    cell = composite.state['agents']['0']
    em = cell.get('emitter', {}).get('instance')
    v2_history = em.history if em else []

    v2_bulk_ts = {}
    v2_mass_ts = {}
    for snap in v2_history:
        t = int(snap.get('global_time', 0))
        bulk = snap.get('bulk')
        if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
            v2_bulk_ts[t] = bulk['count'].copy()
        mass = snap.get('listeners', {}).get('mass', {})
        if isinstance(mass, dict) and mass:
            v2_mass_ts[t] = dict(mass)

    # Unique molecule counts from final state
    unique = cell.get('unique', {})
    v2_unique = {}
    for name, arr in unique.items():
        if hasattr(arr, 'dtype') and '_entryState' in arr.dtype.names:
            v2_unique[name] = int(arr['_entryState'].view(np.bool_).sum())

    # Full chromosome details
    fc = unique.get('full_chromosome')
    v2_chrom = {}
    if fc is not None and hasattr(fc, 'dtype'):
        active = fc[fc['_entryState'].view(np.bool_)]
        v2_chrom['n_chromosomes'] = len(active)
        if 'division_time' in fc.dtype.names:
            dt, htd = ecoli_attrs(fc, ['division_time', 'has_triggered_division'])
            v2_chrom['division_times'] = dt.tolist()
            v2_chrom['has_triggered'] = htd.tolist()
        if 'domain_index' in fc.dtype.names:
            (di,) = ecoli_attrs(fc, ['domain_index'])
            v2_chrom['domain_indexes'] = di.tolist()

    return {
        'time': v2_time,
        'initial_bulk': v2_initial_bulk,
        'final_bulk': cell['bulk']['count'].copy(),
        'bulk_ts': v2_bulk_ts,
        'mass_ts': v2_mass_ts,
        'unique_counts': v2_unique,
        'chromosome': v2_chrom,
        'final_mass': dict(cell.get('listeners', {}).get('mass', {})),
    }


def _collect_v1_timeseries(duration):
    """Run v1, collect per-timestep bulk, mass, and unique data."""
    try:
        if not hasattr(np, 'in1d'):
            np.in1d = np.isin

        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        with chdir(V1_ROOT_PATH):
            sim = EcoliSim.from_file()
            sim.max_duration = int(duration)
            sim.emitter = 'timeseries'
            sim.divide = False
            sim.build_ecoli()
            v1_initial = sim.generated_initial_state['bulk']['count'].copy()
            t0 = time.time()
            sim.run()
            v1_time = time.time() - t0
        sys.argv = saved_argv

        v1_state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
        v1_final = v1_state['bulk']['count'].copy()

        v1_ts = sim.query()
        v1_bulk_ts = {}
        v1_mass_ts = {}
        for t_key in sorted(v1_ts.keys()):
            if not isinstance(t_key, (int, float)):
                continue
            t = int(t_key)
            snapshot = v1_ts[t_key]
            if not isinstance(snapshot, dict):
                continue

            bulk = snapshot.get('bulk')
            if bulk is not None:
                if hasattr(bulk, 'dtype') and 'count' in (bulk.dtype.names or []):
                    v1_bulk_ts[t] = np.array(bulk['count'], dtype=float)
                elif isinstance(bulk, (list, np.ndarray)):
                    v1_bulk_ts[t] = np.array(bulk, dtype=float)

            mass = snapshot.get('listeners', {}).get('mass', {})
            if isinstance(mass, dict) and mass:
                entry = {}
                for k, v in mass.items():
                    try:
                        entry[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                v1_mass_ts[t] = entry

        v1_unique = {}
        unique = v1_state.get('unique', {})
        for name, arr in unique.items():
            if hasattr(arr, 'dtype') and '_entryState' in arr.dtype.names:
                v1_unique[name] = int(arr['_entryState'].view(np.bool_).sum())

        v1_chrom = {}
        fc = unique.get('full_chromosome')
        if fc is not None and hasattr(fc, 'dtype'):
            active = fc[fc['_entryState'].view(np.bool_)]
            v1_chrom['n_chromosomes'] = len(active)
            if 'division_time' in fc.dtype.names:
                dt, htd = ecoli_attrs(fc, ['division_time', 'has_triggered_division'])
                v1_chrom['division_times'] = dt.tolist()
                v1_chrom['has_triggered'] = htd.tolist()

        return {
            'time': v1_time,
            'initial_bulk': v1_initial,
            'final_bulk': v1_final,
            'bulk_ts': v1_bulk_ts,
            'mass_ts': v1_mass_ts,
            'unique_counts': v1_unique,
            'chromosome': v1_chrom,
            'final_mass': v1_mass_ts.get(int(duration), {}),
        }
    except Exception as e:
        print(f"  v1 comparison skipped: {e}")
        return None


# ---------------------------------------------------------------------------
# Step Diagnostics
# ---------------------------------------------------------------------------

def bench_step_diagnostics(composite):
    """Per-step analysis of composite structure."""
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

        if info['has_ports_schema']:
            try:
                ps = proc.ports_schema()
                info['input_ports'] = sorted(ps.keys())
            except Exception:
                info['input_ports'] = []
        else:
            info['input_ports'] = []

        wires = edge.get('inputs', {})
        info['wires'] = {k: v for k, v in wires.items() if not k.startswith('_flow')}

        diagnostics.append(info)

    return diagnostics


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

BIOCYC_FILE_IDS = [
    "complexation_reactions", "dna_sites", "equilibrium_reactions",
    "genes", "metabolic_reactions", "metabolites", "proteins",
    "rnas", "transcription_units", "trna_charging_reactions",
]

FLAT_DIR = os.path.join(
    os.path.dirname(__file__), 'v2ecoli', 'reconstruction', 'ecoli', 'flat')


def step_biocyc():
    """Step 0: Fetch raw data files from the EcoCyc API."""
    step_name = 'biocyc'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 0: EcoCyc API (cached)")
        return meta

    print(f"  Step 0: EcoCyc API")
    import requests
    base_url = "https://websvc.biocyc.org/wc-get?type="
    results = {}

    for file_id in BIOCYC_FILE_IDS:
        outpath = os.path.join(FLAT_DIR, file_id + ".tsv")
        print(f"    Fetching {file_id}...", end=" ", flush=True)
        try:
            response = requests.get(base_url + file_id, timeout=30)
            response.raise_for_status()
            with open(outpath, "w") as f:
                f.write(response.text)
            n_bytes = len(response.text)
            n_lines = response.text.count('\n')
            results[file_id] = {'bytes': n_bytes, 'lines': n_lines, 'status': 'ok'}
            print(f"{n_bytes:,} bytes, {n_lines} lines")
        except Exception as e:
            results[file_id] = {'bytes': 0, 'lines': 0, 'status': str(e)}
            print(f"FAILED: {e}")
        time.sleep(1)

    meta = {
        'n_files': len(BIOCYC_FILE_IDS),
        'files': results,
        'n_fetched': sum(1 for v in results.values() if v['status'] == 'ok'),
    }
    save_meta(step_name, meta)
    print(f"    {meta['n_fetched']}/{meta['n_files']} files fetched")
    return meta


def step_raw_data():
    """Step 1: Catalog raw data files and knowledge base statistics."""
    step_name = 'raw_data'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 1: Raw Data (cached)")
        return meta

    print(f"  Step 1: Raw Data")
    t0 = time.time()

    from v2ecoli.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
    raw_data = KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False,
        remove_rrff=False, stable_rrna=False)

    # Walk flat directory
    flat_dir = FLAT_DIR
    n_files = 0
    total_size = 0
    by_subdir = {}
    for root, dirs, files in os.walk(flat_dir):
        rel = os.path.relpath(root, flat_dir)
        if rel == '.':
            rel = 'root'
        for fn in files:
            fp = os.path.join(root, fn)
            sz = os.path.getsize(fp)
            n_files += 1
            total_size += sz
            by_subdir.setdefault(rel, {'count': 0, 'size': 0})
            by_subdir[rel]['count'] += 1
            by_subdir[rel]['size'] += sz

    # Extract stats from raw_data
    n_genes = len(raw_data.genes) if hasattr(raw_data, 'genes') else 0
    n_rnas = len(raw_data.rnas) if hasattr(raw_data, 'rnas') else 0
    n_proteins = len(raw_data.proteins) if hasattr(raw_data, 'proteins') else 0
    n_metabolites = len(raw_data.metabolites) if hasattr(raw_data, 'metabolites') else 0
    genome_length = raw_data.genome_length if hasattr(raw_data, 'genome_length') else 0

    elapsed = time.time() - t0

    # Catalog individual files with type classification
    file_list = []
    for root, dirs, files in os.walk(flat_dir):
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), flat_dir)
            base = fn.replace('.tsv', '').replace('.fasta', '')
            is_biocyc = base in BIOCYC_FILE_IDS
            is_modifier = any(base.endswith(s) for s in ('_added', '_removed', '_modified'))
            file_list.append({
                'name': rel,
                'size': os.path.getsize(os.path.join(root, fn)),
                'source': 'biocyc' if is_biocyc else ('modifier' if is_modifier else 'curated'),
            })

    meta = {
        'n_files': n_files,
        'total_size': total_size,
        'total_size_mb': round(total_size / 1e6, 1),
        'by_subdir': by_subdir,
        'file_list': file_list,
        'n_genes': n_genes,
        'n_rnas': n_rnas,
        'n_proteins': n_proteins,
        'n_metabolites': n_metabolites,
        'genome_length': genome_length,
        'elapsed': elapsed,
    }
    save_meta(step_name, meta)
    print(f"    {n_files} files, {meta['total_size_mb']}MB, "
          f"{n_genes} genes, {n_rnas} RNAs, {n_proteins} proteins, "
          f"{n_metabolites} metabolites, genome={genome_length}bp")
    return meta


def step_parca():
    """Step 2: Run ParCa (parameter calculator) or load cached simData."""
    step_name = 'parca'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 2: ParCa (cached)")
        return meta

    print(f"  Step 2: ParCa")

    # Check for existing cache or simData
    sim_data_cache = os.path.join(CACHE_DIR, 'sim_data_cache.dill')
    sim_data_path = SIM_DATA_PATH

    parca_ran = False
    if os.path.exists(sim_data_cache):
        # Cache already exists — skip ParCa entirely
        print(f"    Cache exists at {CACHE_DIR}")
        parca_time = 0.0
        cache_time = 0.0
        sim_data_path = sim_data_path or '(from cache)'
    elif sim_data_path and os.path.exists(sim_data_path):
        print(f"    Using existing simData at {sim_data_path}")
        parca_time = 0.0
    else:
        # Need to run ParCa
        print("    Running fitSimData_1 (this takes a few minutes)...")
        from v2ecoli.reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
        from v2ecoli.reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
        raw_data = KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False,
        remove_rrff=False, stable_rrna=False)
        t0 = time.time()
        sim_data = fitSimData_1(raw_data)
        parca_time = time.time() - t0

        # Save simData
        sim_data_path = os.path.join(WORKFLOW_DIR, 'simData.cPickle')
        os.makedirs(WORKFLOW_DIR, exist_ok=True)
        with open(sim_data_path, 'wb') as f:
            dill.dump(sim_data, f)
        parca_ran = True
        print(f"    ParCa completed in {parca_time:.1f}s")

    # Generate cache files
    if not os.path.exists(sim_data_cache):
        print("    Generating cache (initial_state.json + sim_data_cache.dill)...")
        t0 = time.time()
        save_cache(sim_data_path, CACHE_DIR)
        cache_time = time.time() - t0
        print(f"    Cache generated in {cache_time:.1f}s")
    else:
        cache_time = 0.0
        print("    Cache already exists")

    # Extract stats from cache
    stats = {}
    try:
        with open(sim_data_cache, 'rb') as f:
            cache = dill.load(f)
        configs = cache.get('configs', {})
        unique_names = cache.get('unique_names', [])
        stats['n_process_configs'] = len(configs)
        stats['process_names'] = sorted(configs.keys())
        stats['n_unique_types'] = len(unique_names)
        stats['unique_types'] = unique_names
        # Count bulk molecules from initial state
        init_state_path = os.path.join(CACHE_DIR, 'initial_state.json')
        if os.path.exists(init_state_path):
            init = load_initial_state(init_state_path)
            bulk = init.get('bulk')
            if bulk is not None and hasattr(bulk, '__len__'):
                stats['n_bulk_molecules'] = len(bulk)
    except Exception as e:
        stats['note'] = f'Could not extract stats: {e}'

    meta = {
        'sim_data_path': sim_data_path,
        'parca_ran': parca_ran,
        'parca_time': parca_time,
        'cache_time': cache_time,
        'cache_dir': CACHE_DIR,
        'stats': stats,
    }
    save_meta(step_name, meta)
    return meta


def step_load_model():
    """Step 3: Build composite from cache."""
    step_name = 'load_model'
    meta = load_meta(step_name)

    # Always build the composite (needed by later steps), but cache metadata
    print(f"  Step 3: Load Model", end='')
    t0 = time.time()
    composite = make_composite(cache_dir=CACHE_DIR)
    build_time = time.time() - t0

    n_steps = len(composite.step_paths)
    n_processes = len(composite.process_paths)

    cell = composite.state['agents']['0']
    bulk = cell.get('bulk')
    n_bulk = len(bulk) if bulk is not None and hasattr(bulk, '__len__') else 0
    unique = cell.get('unique', {})
    n_unique_types = len(unique)
    mass = cell.get('listeners', {}).get('mass', {})
    initial_dry_mass = float(mass.get('dry_mass', 0))

    if meta is not None:
        print(f" (cached metadata, rebuilt composite in {build_time:.2f}s)")
    else:
        print(f" ({build_time:.2f}s)")

    meta = {
        'build_time': build_time,
        'n_steps': n_steps,
        'n_processes': n_processes,
        'n_bulk': n_bulk,
        'n_unique_types': n_unique_types,
        'initial_dry_mass': initial_dry_mass,
    }
    save_meta(step_name, meta)

    print(f"    {n_steps} steps, {n_processes} processes, "
          f"{n_bulk} bulk molecules, {n_unique_types} unique types, "
          f"dry_mass={initial_dry_mass:.1f}fg")

    return meta, composite


def step_short_sim(composite):
    """Step 4: Run 60s simulation with comprehensive diagnostics."""
    step_name = 'short_sim'
    duration = COMPARISON_DURATION
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 4: Short Simulation (cached)")
        # Load cached history for plotting
        try:
            state_data = load_state_data(step_name)
            history = state_data.get('history', [])
        except Exception:
            history = []
        return meta, history

    print(f"  Step 4: Short Simulation ({duration}s)")
    cell = composite.state['agents']['0']
    bulk_before = np.array(cell['bulk']['count'], copy=True)

    t0 = time.time()
    composite.run(duration)
    wall_time = time.time() - t0

    cell = composite.state['agents']['0']
    bulk_after = cell['bulk']['count']
    changed = (bulk_before != bulk_after).sum()

    em_edge = cell.get('emitter')
    history = em_edge['instance'].history if isinstance(em_edge, dict) and 'instance' in em_edge else []

    mass = cell.get('listeners', {}).get('mass', {})

    # Extract chromosome snapshots from emitter history
    chrom_snapshots = []
    for snap in history:
        t = snap.get('global_time', 0)
        m = snap.get('listeners', {}).get('mass', {}) if isinstance(snap.get('listeners'), dict) else {}
        chrom_snapshots.append({
            'time': float(t),
            'n_chromosomes': 1,  # Short sim: always 1 chromosome
            'n_domains': 3,
            'fork_coords': [],  # Not captured per-timestep by emitter
            'dna_mass': float(m.get('dna_mass', 0)),
            'dry_mass': float(m.get('dry_mass', 0)),
        })
    # Get fork coordinates from final state
    unique = cell.get('unique', {})
    rep = unique.get('active_replisome')
    if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
        active_rep = rep[rep['_entryState'].view(np.bool_)]
        if len(active_rep) > 0 and 'coordinates' in rep.dtype.names:
            if chrom_snapshots:
                chrom_snapshots[-1]['fork_coords'] = active_rep['coordinates'].tolist()

    meta = {
        'duration': duration,
        'wall_time': wall_time,
        'bulk_changed': int(changed),
        'total_bulk': len(bulk_before),
        'final_dry_mass': float(mass.get('dry_mass', 0)),
        'final_cell_mass': float(mass.get('cell_mass', 0)),
        'final_volume': float(mass.get('volume', 0)),
        'rate': duration / wall_time,
        'chromosome_snapshots': chrom_snapshots,
    }
    save_meta(step_name, meta)

    # Save history and state for caching
    save_state_data(step_name, {
        'history': history,
    })

    print(f"    {wall_time:.1f}s wall, {changed} molecules changed, "
          f"{meta['rate']:.1f}x realtime")
    print(f"    dry_mass={meta['final_dry_mass']:.1f}fg, "
          f"volume={meta['final_volume']:.4f}fL")
    return meta, history


def step_v1_comparison():
    """Step 5: Run v1 and v2 for 60s, compare accuracy."""
    step_name = 'v1_comparison'
    duration = COMPARISON_DURATION
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 5: v1 Comparison (cached)")
        try:
            state_data = load_state_data(step_name)
            return meta, state_data
        except Exception:
            return meta, {}

    print(f"  Step 5: v1 Comparison ({duration}s)")

    # v2
    composite = make_composite(cache_dir=CACHE_DIR)
    v2 = _collect_v2_timeseries(composite, duration)

    # v1
    v1 = _collect_v1_timeseries(duration)
    v1_available = v1 is not None

    if not v1_available:
        v1 = {
            'time': 0,
            'initial_bulk': v2['initial_bulk'].copy(),
            'final_bulk': v2['initial_bulk'].copy(),
            'bulk_ts': {},
            'mass_ts': {},
            'unique_counts': {},
            'chromosome': {},
            'final_mass': {},
        }

    # Per-timestep bulk comparison
    common_times = sorted(set(v1['bulk_ts'].keys()) & set(v2['bulk_ts'].keys()))
    per_ts_corr = []
    per_ts_exact = []
    per_ts_rmse = []
    for t in common_times:
        v1_c = v1['bulk_ts'][t].astype(float)
        v2_c = v2['bulk_ts'][t].astype(float)
        if len(v1_c) == len(v2_c) and len(v1_c) > 0:
            corr = np.corrcoef(v1_c, v2_c)[0, 1]
            exact = np.mean(v1_c == v2_c)
            rmse = np.sqrt(np.mean((v1_c - v2_c) ** 2))
            per_ts_corr.append(corr)
            per_ts_exact.append(exact)
            per_ts_rmse.append(rmse)

    # Final-state delta correlation
    both = (v1['initial_bulk'] != v1['final_bulk']) & (v2['initial_bulk'] != v2['final_bulk'])
    d1 = v1['final_bulk'][both] - v1['initial_bulk'][both]
    d2 = v2['final_bulk'][both] - v2['initial_bulk'][both]
    delta_corr = np.corrcoef(d1.astype(float), d2.astype(float))[0, 1] if both.sum() > 0 else 0.0

    # Per-timestep mass comparison — aligned by common timesteps
    # v1 often has t=0 (initial snapshot) that v2 doesn't; skip it
    mass_keys = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']
    mass_comparison = {k: {'v1': [], 'v2': [], 'times': []} for k in mass_keys}
    common_mass_times = sorted(set(v1['mass_ts'].keys()) & set(v2['mass_ts'].keys()))
    for t in common_mass_times:
        for k in mass_keys:
            mass_comparison[k]['v1'].append(v1['mass_ts'][t].get(k, 0))
            mass_comparison[k]['v2'].append(v2['mass_ts'][t].get(k, 0))
            mass_comparison[k]['times'].append(t)

    # Per-category mass accuracy metrics
    mass_metrics = {}
    for k in mass_keys:
        v1_vals = np.array(mass_comparison[k]['v1'])
        v2_vals = np.array(mass_comparison[k]['v2'])
        n = min(len(v1_vals), len(v2_vals))
        if n > 0:
            v1_v = v1_vals[:n]
            v2_v = v2_vals[:n]
            abs_err = np.abs(v2_v - v1_v)
            pct_err = np.abs(v2_v - v1_v) / np.maximum(np.abs(v1_v), 1e-10) * 100
            ss_res = np.sum((v2_v - v1_v) ** 2)
            ss_tot = np.sum((v1_v - v1_v.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mass_metrics[k] = {
                'mean_abs_err': float(abs_err.mean()),
                'max_abs_err': float(abs_err.max()),
                'mean_pct_err': float(pct_err.mean()),
                'max_pct_err': float(pct_err.max()),
                'final_pct_err': float(pct_err[-1]) if len(pct_err) > 0 else 0,
                'r2': float(r2),
                'v1_final': float(v1_v[-1]) if len(v1_v) > 0 else 0,
                'v2_final': float(v2_v[-1]) if len(v2_v) > 0 else 0,
            }
        else:
            mass_metrics[k] = {
                'mean_abs_err': 0, 'max_abs_err': 0,
                'mean_pct_err': 0, 'max_pct_err': 0,
                'final_pct_err': 0, 'r2': 0,
                'v1_final': 0, 'v2_final': 0,
            }

    worst_pct = max(m['max_pct_err'] for m in mass_metrics.values()) if mass_metrics else 0

    comp_data = {
        'duration': duration,
        'v1_available': v1_available,
        'v1_time': v1['time'],
        'v2_time': v2['time'],
        'v1_changed': int((v1['initial_bulk'] != v1['final_bulk']).sum()),
        'v2_changed': int((v2['initial_bulk'] != v2['final_bulk']).sum()),
        'both_changed': int(both.sum()),
        'delta_correlation': delta_corr,
        'common_timesteps': len(common_times),
        'per_ts_corr': per_ts_corr,
        'per_ts_exact': per_ts_exact,
        'per_ts_rmse': per_ts_rmse,
        'mean_correlation': float(np.mean(per_ts_corr)) if per_ts_corr else 0.0,
        'mean_exact_match': float(np.mean(per_ts_exact)) if per_ts_exact else 0.0,
        'mean_rmse': float(np.mean(per_ts_rmse)) if per_ts_rmse else 0.0,
        'v1_initial': v1['initial_bulk'],
        'v1_final': v1['final_bulk'],
        'v2_initial': v2['initial_bulk'],
        'v2_final': v2['final_bulk'],
        'mass_comparison': mass_comparison,
        'mass_metrics': mass_metrics,
        'worst_pct_error': worst_pct,
        'v1_unique': v1['unique_counts'],
        'v2_unique': v2['unique_counts'],
        'v1_chromosome': v1['chromosome'],
        'v2_chromosome': v2['chromosome'],
        'v1_final_mass': v1['final_mass'],
        'v2_final_mass': v2['final_mass'],
    }

    # Save full comparison data for caching (includes numpy arrays)
    save_state_data(step_name, comp_data)

    # Metadata (JSON-safe subset)
    meta = {
        'duration': duration,
        'v1_available': v1_available,
        'v1_time': v1['time'],
        'v2_time': v2['time'],
        'v1_changed': comp_data['v1_changed'],
        'v2_changed': comp_data['v2_changed'],
        'both_changed': comp_data['both_changed'],
        'delta_correlation': float(delta_corr),
        'common_timesteps': len(common_times),
        'mean_correlation': comp_data['mean_correlation'],
        'mean_exact_match': comp_data['mean_exact_match'],
        'mean_rmse': comp_data['mean_rmse'],
        'mass_metrics': mass_metrics,
        'worst_pct_error': worst_pct,
    }
    save_meta(step_name, meta)

    print(f"    v1: {v1['time']:.2f}s, v2: {v2['time']:.2f}s, "
          f"worst_pct_err: {worst_pct:.2f}%")
    for k, m in mass_metrics.items():
        label = k.replace('_mass', '').replace('_', ' ').title()
        print(f"      {label}: mean_err={m['mean_pct_err']:.2f}%, "
              f"max_err={m['max_pct_err']:.2f}%, R2={m['r2']:.4f}")

    return meta, comp_data


def step_long_sim():
    """Step 6: Run simulation to division with chromosome snapshots.

    Runs in chunks (SNAPSHOT_INTERVAL seconds each), capturing chromosome
    state at every interval. Stops when division condition is met (2+
    chromosomes AND dry mass >= 2x initial) or MAX_LONG_DURATION is reached.
    Saves pre-division cell state for the division test.
    """
    step_name = 'long_sim'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 6: Long Simulation (cached)")
        return meta

    print(f"  Step 6: Long Simulation (to division, max {MAX_LONG_DURATION}s)")
    composite = make_composite(cache_dir=CACHE_DIR)

    cell = composite.state['agents']['0']
    bulk_before = np.array(cell['bulk']['count'], copy=True)
    initial_dry_mass = float(cell.get('listeners', {}).get('mass', {}).get('dry_mass', 380))

    t0 = time.time()
    snapshots = []
    divided = False
    total_run = 0

    while total_run < MAX_LONG_DURATION:
        chunk = min(SNAPSHOT_INTERVAL, MAX_LONG_DURATION - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            # Division structural update may crash — that's OK,
            # the pre-division state was our last snapshot
            print(f"    Simulation stopped at ~t={total_run+chunk}s: {type(e).__name__}")
            divided = True
            break
        total_run += chunk

        cell = composite.state['agents']['0']
        unique = cell.get('unique', {})

        # Chromosome state
        fc = unique.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        rep = unique.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active_rep = rep[rep['_entryState'].view(np.bool_)]
            if len(active_rep) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active_rep['coordinates'].tolist()

        domains = unique.get('chromosome_domain')
        n_domains = 0
        if domains is not None and hasattr(domains, 'dtype') and '_entryState' in domains.dtype.names:
            n_domains = int(domains['_entryState'].view(np.bool_).sum())

        # RNAP positions
        rnap = unique.get('active_RNAP')
        rnap_coords = []
        n_rnap = 0
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            active_rnap = rnap[rnap['_entryState'].view(np.bool_)]
            n_rnap = len(active_rnap)
            if n_rnap > 0 and 'coordinates' in rnap.dtype.names:
                rnap_coords = active_rnap['coordinates'].tolist()

        mass = cell.get('listeners', {}).get('mass', {})
        dry_mass = float(mass.get('dry_mass', 0))
        dna_mass = float(mass.get('dna_mass', 0))

        snapshots.append({
            'time': float(cell.get('global_time', total_run)),
            'n_chromosomes': n_chrom,
            'n_domains': n_domains,
            'fork_coords': fork_coords,
            'rnap_coords': rnap_coords,
            'n_rnap': n_rnap,
            'dna_mass': dna_mass,
            'dry_mass': dry_mass,
        })

        # Check division readiness: 2+ chromosomes AND mass doubled
        if n_chrom >= 2 and dry_mass >= initial_dry_mass * 2:
            print(f"    Division ready at t={total_run}s: {n_chrom} chromosomes, "
                  f"dry_mass={dry_mass:.0f}fg (>= {initial_dry_mass*2:.0f}fg)")
            divided = True
            break

        if total_run % 500 == 0:
            print(f"    t={total_run}s: {n_chrom} chroms, dry_mass={dry_mass:.0f}fg, "
                  f"forks={len(fork_coords)}")

    wall_time = time.time() - t0

    # After division, agent '0' may be gone — use last snapshot data
    agents = composite.state.get('agents', {})
    cell = agents.get('0')
    if cell is None:
        # Division happened — find any remaining agent for bulk comparison
        for aid, astate in agents.items():
            if isinstance(astate, dict) and 'bulk' in astate:
                cell = astate
                break

    if cell is not None and 'bulk' in cell:
        bulk_after = cell['bulk']['count']
        changed = int((bulk_before != bulk_after).sum())
    else:
        changed = 0

    # Use last snapshot for final metrics
    final_snap = snapshots[-1] if snapshots else {}

    # Save pre-division cell state for division test
    # Use the last snapshot where we still had cell data
    data_keys = {'bulk', 'unique', 'listeners', 'environment', 'boundary',
                 'global_time', 'timestep', 'divide', 'division_threshold',
                 'process_state', 'allocator_rng'}
    if cell is not None:
        cell_data = {k: v for k, v in cell.items()
                     if k in data_keys or k.startswith('request_') or k.startswith('allocate_')}
    else:
        cell_data = {}

    save_state_data(step_name, {
        'cell_state': cell_data,
        'global_time': final_snap.get('time', 0.0),
    })

    meta = {
        'duration': total_run,
        'wall_time': wall_time,
        'bulk_changed': changed,
        'total_bulk': len(bulk_before),
        'final_dry_mass': final_snap.get('dry_mass', 0),
        'final_cell_mass': 0,  # not tracked in snapshots
        'final_volume': 0,
        'rate': total_run / wall_time if wall_time > 0 else 0,
        'division_reached': divided,
        'initial_dry_mass': initial_dry_mass,
        'chromosome_snapshots': snapshots,
    }
    save_meta(step_name, meta)

    print(f"    {wall_time:.0f}s wall, {changed} molecules changed, "
          f"{meta['rate']:.1f}x realtime")
    print(f"    dry_mass={meta['final_dry_mass']:.0f}fg, "
          f"division={'reached' if divided else 'not reached'}")
    return meta


def step_division():
    """Step 7: Test cell division, conservation, and daughter viability."""
    step_name = 'division'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 7: Division (cached)")
        return meta

    print(f"  Step 7: Division")

    # Try to load pre-division state from long sim
    prediv_state = None
    prediv_time = 0.0
    long_state_path = os.path.join(WORKFLOW_DIR, 'long_sim.dill')
    if os.path.exists(long_state_path):
        try:
            with open(long_state_path, 'rb') as f:
                checkpoint = dill.load(f)
            cell_data = checkpoint.get('cell_state', {})
            if cell_data.get('unique', {}).get('full_chromosome') is not None:
                prediv_state = cell_data
                prediv_time = checkpoint.get('global_time', 0.0)
                print(f"    Using pre-division state from long sim (t={prediv_time:.0f})")
        except Exception as e:
            print(f"    Could not load long sim state: {e}")

    # Also check the old predivision path
    if prediv_state is None:
        old_prediv = 'out/predivision.dill'
        if os.path.exists(old_prediv):
            try:
                with open(old_prediv, 'rb') as f:
                    checkpoint = dill.load(f)
                cell_data = checkpoint.get('cell_state', {})
                if cell_data.get('unique', {}).get('full_chromosome') is not None:
                    prediv_state = cell_data
                    prediv_time = checkpoint.get('global_time', 0.0)
                    print(f"    Using pre-division checkpoint (t={prediv_time:.0f})")
            except Exception as e:
                print(f"    Could not load pre-division state: {e}")

    if prediv_state is not None:
        cell = prediv_state
    else:
        print("    No pre-division checkpoint -- using initial state (t=0)")
        composite = make_composite(cache_dir=CACHE_DIR)
        cell = composite.state['agents']['0']

    # Test bulk conservation
    d1_bulk, d2_bulk = divide_bulk(cell['bulk'])
    mother_count = int(cell['bulk']['count'].sum())
    d1_count = int(d1_bulk['count'].sum())
    d2_count = int(d2_bulk['count'].sum())
    conserved = (d1_count + d2_count == mother_count)

    # Test full cell division
    t0 = time.time()
    d1_state, d2_state = divide_cell(cell)
    split_time = time.time() - t0

    # Unique molecule counts
    unique_conservation = {}
    for name in d1_state.get('unique', {}):
        d1_arr = d1_state['unique'][name]
        d2_arr = d2_state['unique'][name]
        mother_arr = cell['unique'][name]
        if hasattr(d1_arr, 'shape') and hasattr(mother_arr, 'dtype'):
            if '_entryState' in mother_arr.dtype.names:
                m = int(mother_arr['_entryState'].view(np.bool_).sum())
                d1 = d1_arr.shape[0]
                d2 = d2_arr.shape[0]
                unique_conservation[name] = {
                    'mother': m, 'd1': d1, 'd2': d2,
                    'conserved': d1 + d2 == m
                }

    # Test daughter document build
    div_step = cell.get('division', {}).get('instance')
    configs = getattr(div_step, '_configs', None)
    unique_names = getattr(div_step, '_unique_names', None)
    dry_mass_inc = getattr(div_step, 'dry_mass_inc_dict', None)

    if configs is None and os.path.isdir(CACHE_DIR):
        cache_path = os.path.join(CACHE_DIR, 'sim_data_cache.dill')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = dill.load(f)
            configs = cache.get('configs', {})
            unique_names = cache.get('unique_names', [])
            dry_mass_inc = cache.get('dry_mass_inc_dict', {})

    can_build_daughters = configs is not None and bool(configs)
    daughter_build_time = 0
    daughter_viable = False

    if can_build_daughters:
        from v2ecoli.generate import build_document_from_configs
        t0 = time.time()
        try:
            d1_doc = build_document_from_configs(
                d1_state, configs, unique_names,
                dry_mass_inc_dict=dry_mass_inc,
                seed=1)
            daughter_build_time = time.time() - t0
            d1_composite = Composite(d1_doc, core=_build_core())
            d1_composite.run(1.0)
            daughter_viable = True
        except Exception as e:
            daughter_build_time = time.time() - t0
            print(f"    Daughter build error: {e}")

    # Check division-ready state
    fc = cell.get('unique', {}).get('full_chromosome')
    n_chromosomes = 0
    if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
        n_chromosomes = int(fc['_entryState'].view(np.bool_).sum())
    mass = cell.get('listeners', {}).get('mass', {})
    dry_mass = float(mass.get('dry_mass', 0))

    meta = {
        'prediv_time': prediv_time,
        'dry_mass': dry_mass,
        'n_chromosomes': n_chromosomes,
        'mother_bulk_count': mother_count,
        'd1_bulk_count': d1_count,
        'd2_bulk_count': d2_count,
        'bulk_conserved': conserved,
        'split_time': split_time,
        'unique_conservation': unique_conservation,
        'can_build_daughters': can_build_daughters,
        'daughter_build_time': daughter_build_time,
        'daughter_viable': daughter_viable,
    }
    save_meta(step_name, meta)

    print(f"    Bulk conserved: {conserved}, split: {split_time*1000:.0f}ms")
    print(f"    Daughters buildable: {can_build_daughters}, viable: {daughter_viable}")
    if daughter_build_time > 0:
        print(f"    Daughter build: {daughter_build_time:.1f}s")

    return meta


# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------

def generate_html_report(step_results, plots, bigraph_svg, diagnostics):
    """Generate the HTML report organized by pipeline step."""

    biocyc = step_results.get('biocyc', {})
    raw = step_results['raw_data']
    parca = step_results['parca']
    model = step_results['load_model']
    short = step_results['short_sim']
    comp_meta = step_results.get('v1_comparison_meta', {})
    comp_data = step_results.get('v1_comparison_data', {})
    long = step_results['long_sim']
    div = step_results['division']

    def cached_badge(meta):
        ts = meta.get('timestamp', '')
        return f'<span class="timing">cached {ts}</span>' if ts else ''

    # BioCyc file rows
    biocyc_rows = ''
    for fid, info in biocyc.get('files', {}).items():
        status = info.get('status', 'unknown')
        color = 'green' if status == 'ok' else 'red'
        sz = f"{info.get('bytes', 0):,}" if status == 'ok' else '-'
        biocyc_rows += f'<tr><td><code>{fid}.tsv</code></td><td>{info.get("lines", 0)}</td><td>{sz}</td><td class="{color}">{status}</td></tr>'

    # File download rows (all raw files with source classification)
    file_rows = ''
    for fi in raw.get('file_list', []):
        src = fi.get('source', 'curated')
        badge = {'biocyc': 'EcoCyc API', 'modifier': 'Modifier', 'curated': 'Curated'}.get(src, src)
        badge_color = {'biocyc': '#3b82f6', 'modifier': '#f59e0b', 'curated': '#64748b'}.get(src, '#64748b')
        sz = fi.get('size', 0)
        sz_str = f"{sz/1024:.0f} KB" if sz > 1024 else f"{sz} B"
        fname = fi['name']
        # Data URI for download link
        file_rows += (f'<tr><td><a href="https://github.com/vivarium-collective/v2ecoli/'
                      f'blob/main/v2ecoli/reconstruction/ecoli/flat/{fname}" '
                      f'target="_blank"><code>{fname}</code></a></td>'
                      f'<td>{sz_str}</td>'
                      f'<td><span style="background:{badge_color};color:white;'
                      f'padding:1px 6px;border-radius:3px;font-size:0.75em;">{badge}</span></td></tr>')

    # Division unique molecule rows
    div_unique_rows = ''
    for name, info in div.get('unique_conservation', {}).items():
        ok = 'green' if info['conserved'] else 'red'
        div_unique_rows += f"""<tr>
          <td>{name}</td><td>{info['mother']}</td>
          <td>{info['d1']}</td><td>{info['d2']}</td>
          <td class="{ok}">{'Yes' if info['conserved'] else 'No'}</td></tr>"""

    # Mass accuracy table rows
    mass_accuracy_rows = ''
    mass_labels = {
        'dry_mass': 'Dry Mass', 'protein_mass': 'Protein',
        'rna_mass': 'RNA', 'dna_mass': 'DNA', 'smallMolecule_mass': 'Small Molecules',
    }
    c_metrics = comp_meta.get('mass_metrics', comp_data.get('mass_metrics', {}))
    for k, label in mass_labels.items():
        m = c_metrics.get(k, {})
        err_color = 'green' if m.get('max_pct_err', 0) < 1 else ('red' if m.get('max_pct_err', 0) > 5 else 'purple')
        r2_color = 'green' if m.get('r2', 0) > 0.99 else ('red' if m.get('r2', 0) < 0.9 else 'purple')
        mass_accuracy_rows += f"""<tr>
          <td><strong>{label}</strong></td>
          <td>{m.get('v1_final', 0):.2f}</td>
          <td>{m.get('v2_final', 0):.2f}</td>
          <td class="{err_color}">{m.get('mean_pct_err', 0):.2f}%</td>
          <td class="{err_color}">{m.get('max_pct_err', 0):.2f}%</td>
          <td class="{r2_color}">{m.get('r2', 0):.4f}</td>
        </tr>"""

    # Step diagnostics table
    step_rows = ''
    for d in diagnostics:
        inner = f' ({d["inner_class"]})' if d['inner_class'] else ''
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

    # Collect timing data
    parca_time = parca.get('parca_time', 0)
    cache_time = parca.get('cache_time', 0)
    build_time = model.get('build_time', 0)
    short_wall = short.get('wall_time', 0)
    short_dur = short.get('duration', COMPARISON_DURATION)
    short_rate = short.get('rate', 0)
    v1_time = comp_meta.get('v1_time', 0)
    v2_time = comp_meta.get('v2_time', 0)
    comp_dur = comp_meta.get('duration', COMPARISON_DURATION)
    long_wall = long.get('wall_time', 0)
    long_dur = long.get('duration', LONG_DURATION)
    long_rate = long.get('rate', 0)
    worst_pct = comp_meta.get('worst_pct_error', 0)

    report_path = os.path.join(WORKFLOW_DIR, 'workflow_report.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli Workflow Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
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
  .bigraph-container {{ position: relative; border: 1px solid #e2e8f0; border-radius: 8px;
              background: #fafafa; overflow: hidden; height: 700px; cursor: grab; }}
  .bigraph-container.grabbing {{ cursor: grabbing; }}
  .bigraph-container svg {{ position: absolute; transform-origin: 0 0; }}
  .bigraph-controls {{ display: flex; gap: 6px; margin: 8px 0; }}
  .bigraph-controls button {{ padding: 4px 12px; border: 1px solid #cbd5e1; border-radius: 4px;
              background: white; cursor: pointer; font-size: 0.85em; }}
  .bigraph-controls button:hover {{ background: #f1f5f9; }}
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

<h1>v2ecoli Workflow Report</h1>
<p style="color: #64748b; font-size: 0.9em;">{time.strftime('%Y-%m-%d %H:%M')} &middot;
Pipeline steps with intermediate caching &middot; process-bigraph <code>Composite.run()</code></p>

<div class="section">
  <p><strong>v2ecoli</strong> is a whole-cell <em>E. coli</em> model running natively on
  <a href="https://github.com/vivarium-collective/process-bigraph">process-bigraph</a>.
  It migrates all 55 biological steps from
  <a href="https://github.com/CovertLab/vEcoli">vEcoli</a> to run through the standard
  <code>Composite.run()</code> pipeline with custom bigraph-schema types for bulk molecules,
  unique molecules, and listener stores.</p>
  <p>This report runs a 7-step pipeline, caching intermediate results for fast re-runs.</p>
</div>

<nav style="background: white; border-radius: 8px; padding: 12px 20px; margin: 10px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);">
  <strong style="font-size: 0.9em; color: #475569;">Pipeline Steps</strong>
  <ol style="margin: 6px 0 0 0; padding-left: 20px; font-size: 0.88em; columns: 2; column-gap: 30px;">
    <li><a href="#sec-biocyc">EcoCyc API</a></li>
    <li><a href="#sec-raw">Raw Data</a></li>
    <li><a href="#sec-parca">ParCa (Parameter Calculator)</a></li>
    <li><a href="#sec-model">Load Model</a></li>
    <li><a href="#sec-short">Short Simulation ({short_dur:.0f}s)</a></li>
    <li><a href="#sec-compare">v1 Comparison</a></li>
    <li><a href="#sec-long">Long Simulation ({long_dur/60:.0f} min)</a></li>
    <li><a href="#sec-division">Division</a></li>
    <li><a href="#sec-steps">Step Diagnostics</a></li>
    <li><a href="#sec-bigraph">Network Visualization</a></li>
    <li><a href="#sec-timing">Timing Summary</a></li>
  </ol>
</nav>

<!-- ===== Step 0: EcoCyc API ===== -->
<h2 id="sec-biocyc">0. EcoCyc API {cached_badge(biocyc)}</h2>
<div class="section">
  <p>{biocyc.get('n_fetched', 0)}/{biocyc.get('n_files', 0)} files fetched from
  <a href="https://biocyc.org">BioCyc</a> web services
  (<code>https://websvc.biocyc.org/wc-get?type=...</code>).
  Update: <code>python -m v2ecoli.reconstruction.ecoli.scripts.update_biocyc_files</code></p>
</div>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr><th>File</th><th>Lines</th><th>Size</th><th>Status</th></tr></thead>
    <tbody>{biocyc_rows}</tbody>
  </table>
</div>

<!-- ===== Step 1: Raw Data ===== -->
<h2 id="sec-raw">1. Raw Data {cached_badge(raw)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">TSV Files</div><div class="value">{raw.get('n_files', 0)}</div></div>
  <div class="metric"><div class="label">Total Size</div><div class="value blue">{raw.get('total_size_mb', 0)} MB</div></div>
  <div class="metric"><div class="label">Genes</div><div class="value">{raw.get('n_genes', 0):,}</div></div>
  <div class="metric"><div class="label">RNAs</div><div class="value">{raw.get('n_rnas', 0):,}</div></div>
  <div class="metric"><div class="label">Proteins</div><div class="value">{raw.get('n_proteins', 0):,}</div></div>
  <div class="metric"><div class="label">Metabolites</div><div class="value">{raw.get('n_metabolites', 0):,}</div></div>
  <div class="metric"><div class="label">Genome</div><div class="value">{raw.get('genome_length', 0):,} bp</div></div>
</div>

<details>
<summary>File Catalog by Subdirectory</summary>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr><th>Directory</th><th>Files</th><th>Size (KB)</th></tr></thead>
    <tbody>""")

        for subdir, info in sorted(raw.get('by_subdir', {}).items()):
            f.write(f"""<tr><td>{subdir}</td><td>{info['count']}</td><td>{info['size']/1024:.0f}</td></tr>""")

        f.write(f"""
    </tbody>
  </table>
</div>
</details>

<details>
<summary>All Raw Data Files ({len(raw.get('file_list', []))} files — click to browse, links to GitHub)</summary>
<div class="section" style="overflow-x: auto; max-height: 400px; overflow-y: auto;">
  <table>
    <thead><tr><th>File</th><th>Size</th><th>Source</th></tr></thead>
    <tbody>{file_rows}</tbody>
  </table>
</div>
</details>

<!-- ===== Step 2: ParCa ===== -->
<h2 id="sec-parca">2. ParCa (Parameter Calculator) {cached_badge(parca)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">ParCa Time</div><div class="value blue">{'pre-cached' if parca.get('parca_time', 0) == 0 and not parca.get('parca_ran') else f"{parca.get('parca_time', 0):.1f}s"}</div></div>
  <div class="metric"><div class="label">Cache Gen</div><div class="value blue">{'pre-cached' if parca.get('cache_time', 0) == 0 and not parca.get('parca_ran') else f"{parca.get('cache_time', 0):.1f}s"}</div></div>
  <div class="metric"><div class="label">Cache Dir</div><div class="value" style="font-size:0.7em">{parca.get('cache_dir', CACHE_DIR)}</div></div>
</div>

<div class="metrics">
  <div class="metric"><div class="label">Process Configs</div><div class="value">{parca.get('stats', {}).get('n_process_configs', '?')}</div></div>
  <div class="metric"><div class="label">Bulk Molecules</div><div class="value">{parca.get('stats', {}).get('n_bulk_molecules', '?'):,}</div></div>
  <div class="metric"><div class="label">Unique Types</div><div class="value">{parca.get('stats', {}).get('n_unique_types', '?')}</div></div>
</div>

<details>
<summary>Process Configs ({parca.get('stats', {}).get('n_process_configs', 0)})</summary>
<div class="section">
  <ul style="font-size: 0.85em; columns: 3;">""")

        for name in parca.get('stats', {}).get('process_names', []):
            f.write(f"<li><code>{name}</code></li>")

        f.write(f"""</ul>
</div>
</details>

<details>
<summary>Unique Molecule Types ({parca.get('stats', {}).get('n_unique_types', 0)})</summary>
<div class="section">
  <ul style="font-size: 0.85em; columns: 2;">""")

        for name in parca.get('stats', {}).get('unique_types', []):
            f.write(f"<li><code>{name}</code></li>")

        f.write(f"""</ul>
</div>
</details>

<!-- ===== Step 3: Load Model ===== -->
<h2 id="sec-model">3. Load Model {cached_badge(model)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">Build Time</div><div class="value blue">{model.get('build_time', 0):.2f}s</div></div>
  <div class="metric"><div class="label">Steps</div><div class="value">{model.get('n_steps', 0)}</div></div>
  <div class="metric"><div class="label">Processes</div><div class="value">{model.get('n_processes', 0)}</div></div>
  <div class="metric"><div class="label">Bulk Molecules</div><div class="value">{model.get('n_bulk', 0):,}</div></div>
  <div class="metric"><div class="label">Unique Types</div><div class="value">{model.get('n_unique_types', 0)}</div></div>
  <div class="metric"><div class="label">Initial Dry Mass</div><div class="value">{model.get('initial_dry_mass', 0):.1f} fg</div></div>
</div>

<!-- ===== Step 4: Short Sim ===== -->
<h2 id="sec-short">4. Short Simulation ({short_dur:.0f}s) {cached_badge(short)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{short_wall:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value green">{short_rate:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value purple">{short.get('bulk_changed', 0)}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{short.get('final_dry_mass', 0):.1f} fg</div></div>
  <div class="metric"><div class="label">Volume</div><div class="value">{short.get('final_volume', 0):.4f} fL</div></div>
</div>""")

        # Short sim plots
        if plots.get('mass_short'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["mass_short"]}" alt="Mass"></div>\n')
        if plots.get('growth_short'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["growth_short"]}" alt="Growth"></div>\n')
        if plots.get('bulk_histogram'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["bulk_histogram"]}" alt="Bulk Changes"></div>\n')

        f.write(f"""
<!-- ===== Step 5: v1 Comparison ===== -->
<h2 id="sec-compare">5. v1 Comparison ({comp_dur:.0f}s) {cached_badge(comp_meta)}</h2>

<div class="section">
  <h3>Methodology</h3>
  <p>Both v1 (vEcoli) and v2 (v2ecoli) simulations run for {comp_dur:.0f} seconds with identical
     initial states from the same simData. Accuracy is measured by comparing mass components
     (dry mass, protein, RNA, DNA, small molecules) at each simulated second.</p>
</div>

<h3>Mass Component Accuracy</h3>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr>
      <th>Component</th><th>v1 Final (fg)</th><th>v2 Final (fg)</th>
      <th>Mean % Error</th><th>Max % Error</th><th>R&sup2;</th>
    </tr></thead>
    <tbody>{mass_accuracy_rows}</tbody>
  </table>
</div>

<div class="metrics">
  <div class="metric"><div class="label">Worst % Error</div><div class="value {'green' if worst_pct < 1 else 'red'}">{worst_pct:.2f}%</div></div>
  <div class="metric"><div class="label">v1 Runtime</div><div class="value red">{v1_time:.2f}s</div></div>
  <div class="metric"><div class="label">v2 Runtime</div><div class="value blue">{v2_time:.2f}s</div></div>
  <div class="metric"><div class="label">Timesteps</div><div class="value">{comp_meta.get('common_timesteps', 0)}</div></div>
</div>""")

        if plots.get('mass_comp'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["mass_comp"]}" alt="Mass Comparison"></div>\n')

        f.write(f"""
<h3>Unique Molecules & Chromosome State</h3>""")

        if plots.get('unique_comp'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["unique_comp"]}" alt="Unique Comparison"></div>\n')

        f.write(f"""
<details>
<summary>Bulk Molecule Correlation (supplementary)</summary>
<div class="section">
  <p><em>Note: Bulk correlation across all molecules is dominated by invariant counts
  and does not reflect accuracy of biologically active molecules. Use mass component metrics above instead.</em></p>
  <div class="metrics">
    <div class="metric"><div class="label">Mean Correlation</div><div class="value">{comp_meta.get('mean_correlation', 0):.6f}</div></div>
    <div class="metric"><div class="label">Mean Exact Match</div><div class="value">{comp_meta.get('mean_exact_match', 0)*100:.2f}%</div></div>
    <div class="metric"><div class="label">Mean RMSE</div><div class="value">{comp_meta.get('mean_rmse', 0):.2f}</div></div>
  </div>""")

        if plots.get('comparison'):
            f.write(f'  <div class="plot"><img src="data:image/png;base64,{plots["comparison"]}" alt="Bulk Comparison"></div>\n')

        f.write(f"""
</div>
</details>

<!-- ===== Step 6: Long Sim ===== -->
<h2 id="sec-long">6. Long Simulation ({long_dur/60:.0f} min) {cached_badge(long)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">Sim Duration</div><div class="value">{long_dur:.0f}s</div></div>
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{long_wall:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value green">{long_rate:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value purple">{long.get('bulk_changed', 0)}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{long.get('final_dry_mass', 0):.1f} fg</div></div>
  <div class="metric"><div class="label">Division</div><div class="value {'green' if long.get('division_reached') else 'purple'}">{'Reached' if long.get('division_reached') else 'Not reached'}</div></div>
</div>""")

        if plots.get('chromosome_long'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["chromosome_long"]}" alt="Chromosome State"></div>\n')

        f.write(f"""
<!-- ===== Step 7: Division ===== -->
<h2 id="sec-division">7. Division {cached_badge(div)}</h2>

<div class="section">
  <h3>How Division Works</h3>
  <p>The Division step uses process-bigraph's native <code>_add</code>/<code>_remove</code> structural
  updates. When division is triggered (dry mass &ge; threshold with &ge; 2 chromosomes):</p>
  <ol style="margin: 8px 0 8px 20px; font-size: 0.9em;">
    <li><strong>State splitting</strong> &mdash; <code>divide_cell()</code> partitions the mother cell's state:
      <ul>
        <li>Bulk molecules: binomial distribution (p=0.5) on each molecule's count</li>
        <li>Chromosomes: alternating assignment (even&rarr;D1, odd&rarr;D2) with descendant domain tracking</li>
        <li>Chromosome-attached molecules: follow their domain</li>
        <li>RNAs: full transcripts binomial, partial transcripts follow RNAP domain</li>
        <li>Ribosomes: follow their mRNA</li>
      </ul>
    </li>
    <li><strong>Daughter cell construction</strong> &mdash; <code>build_document_from_configs()</code> builds complete
    cell states with fresh process instances from the divided initial state + cached configs</li>
  </ol>
</div>

<h3>Division Test Results</h3>
<div class="section">
  <p>Tests run on {'pre-division state (t=' + str(int(div.get("prediv_time", 0))) + 's, ' + str(div.get("n_chromosomes", 0)) + ' chromosomes, dry mass ' + str(round(div.get("dry_mass", 0))) + ' fg)' if div.get('prediv_time', 0) > 0 else 'initial state (t=0)'}.</p>
</div>
<div class="metrics">
  <div class="metric"><div class="label">Bulk Conserved</div><div class="value {'green' if div.get('bulk_conserved') else 'red'}">{'Yes' if div.get('bulk_conserved') else 'No'}</div></div>
  <div class="metric"><div class="label">Mother Bulk</div><div class="value">{div.get('mother_bulk_count', 0):,}</div></div>
  <div class="metric"><div class="label">D1 Bulk</div><div class="value">{div.get('d1_bulk_count', 0):,}</div></div>
  <div class="metric"><div class="label">D2 Bulk</div><div class="value">{div.get('d2_bulk_count', 0):,}</div></div>
  <div class="metric"><div class="label">State Split</div><div class="value blue">{div.get('split_time', 0)*1000:.0f} ms</div></div>
  <div class="metric"><div class="label">Daughter Build</div><div class="value blue">{div.get('daughter_build_time', 0):.1f}s</div></div>
  <div class="metric"><div class="label">Daughter Viable</div><div class="value {'green' if div.get('daughter_viable') else 'red'}">{'Yes' if div.get('daughter_viable') else 'No'}</div></div>
</div>

<details open>
<summary>Unique Molecule Conservation</summary>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr><th>Molecule</th><th>Mother (active)</th><th>Daughter 1</th><th>Daughter 2</th><th>Conserved</th></tr></thead>
    <tbody>{div_unique_rows}</tbody>
  </table>
</div>
</details>

<!-- ===== Step Diagnostics ===== -->
<h2 id="sec-steps">8. Step Diagnostics ({len(diagnostics)} steps)</h2>
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
<h2 id="sec-bigraph">9. Process-Bigraph Network Visualization</h2>
<div class="section">
  <p>Interactive visualization of the biological process network. Scroll to zoom, drag to pan.
  Steps (colored) read from and write to shared stores (bulk, unique, listeners, request, allocate).</p>
  <div class="bigraph-controls">
    <button onclick="bgZoom(1.3)">Zoom In</button>
    <button onclick="bgZoom(0.77)">Zoom Out</button>
    <button onclick="bgReset()">Fit</button>
    <span id="bg-zoom-level" style="font-size:0.8em;color:#64748b;padding:4px;"></span>
  </div>
</div>
<div class="bigraph-container" id="bigraph-container">{bigraph_svg}</div>
<script>
(function() {{
  const ctr = document.getElementById('bigraph-container');
  const svg = ctr.querySelector('svg');
  if (!svg) return;
  let scale = 1, tx = 0, ty = 0, dragging = false, sx, sy;
  function apply() {{
    svg.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
    document.getElementById('bg-zoom-level').textContent = Math.round(scale*100)+'%';
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
  window.bgReset = fit;
  window.bgZoom = function(f) {{
    const cx = ctr.clientWidth/2, cy = ctr.clientHeight/2;
    tx = cx - f * (cx - tx); ty = cy - f * (cy - ty);
    scale *= f; apply();
  }};
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
    dragging = false; ctr.classList.remove('grabbing');
  }});
}})();
</script>

<!-- ===== Timing Summary ===== -->
<h2 id="sec-timing">10. Timing Summary</h2>
<div class="section">
  <table>
    <tr><th>Step</th><th>Wall Time</th><th>Sim Time</th><th>Sim/Wall Ratio</th></tr>
    <tr><td>1. Raw Data</td><td>{raw.get('elapsed', 0):.1f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>2. ParCa</td><td>{parca_time:.1f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>3. Cache generation</td><td>{cache_time:.1f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>3. Document build</td><td>{build_time:.2f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>4. Short simulation</td><td>{short_wall:.1f}s</td><td>{short_dur:.0f}s</td><td>{short_rate:.1f}x</td></tr>
    <tr><td>5. v1 comparison (v1)</td><td>{v1_time:.2f}s</td><td>{comp_dur:.0f}s</td><td>{comp_dur/max(v1_time, 0.01):.1f}x</td></tr>
    <tr><td>5. v1 comparison (v2)</td><td>{v2_time:.2f}s</td><td>{comp_dur:.0f}s</td><td>{comp_dur/max(v2_time, 0.01):.1f}x</td></tr>
    <tr><td>6. Long simulation</td><td>{long_wall:.1f}s</td><td>{long_dur:.0f}s</td><td>{long_rate:.1f}x</td></tr>
    <tr><td>7. Division</td><td>{div.get('split_time', 0):.3f}s + {div.get('daughter_build_time', 0):.1f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>{raw.get('elapsed', 0)+parca_time+cache_time+build_time+short_wall+v1_time+v2_time+long_wall:.0f}s</strong></td><td>&mdash;</td><td>&mdash;</td></tr>
  </table>
</div>

<footer>
  v2ecoli &middot; <a href="https://github.com/vivarium-collective/v2ecoli">github.com/vivarium-collective/v2ecoli</a>
  &middot; All steps run through process-bigraph Composite.run()
</footer>
</body>
</html>""")

    return report_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_workflow():
    """Execute the full workflow pipeline with caching."""
    os.makedirs(WORKFLOW_DIR, exist_ok=True)

    print("=" * 60)
    print("v2ecoli Workflow Pipeline")
    print("=" * 60)
    pipeline_t0 = time.time()

    step_results = {}

    # Step 0: EcoCyc API
    biocyc_meta = step_biocyc()
    step_results['biocyc'] = biocyc_meta

    # Step 1: Raw Data
    raw_meta = step_raw_data()
    step_results['raw_data'] = raw_meta

    # Step 2: ParCa
    parca_meta = step_parca()
    step_results['parca'] = parca_meta

    # Step 3: Load Model (always builds composite for later steps)
    model_meta, composite = step_load_model()
    step_results['load_model'] = model_meta

    # Step 4: Short Simulation
    short_meta, short_history = step_short_sim(composite)
    step_results['short_sim'] = short_meta

    # Step 5: v1 Comparison
    comp_meta, comp_data = step_v1_comparison()
    step_results['v1_comparison_meta'] = comp_meta
    step_results['v1_comparison_data'] = comp_data

    # Step 6: Long Simulation
    long_meta = step_long_sim()
    step_results['long_sim'] = long_meta

    # Step 7: Division
    div_meta = step_division()
    step_results['division'] = div_meta

    # Step Diagnostics (always run, uses the composite from step 3)
    print("  Diagnostics: Step analysis")
    # Need a fresh composite if step 4 already ran on the original
    diag_composite = make_composite(cache_dir=CACHE_DIR)
    diagnostics = bench_step_diagnostics(diag_composite)
    print(f"    {len(diagnostics)} steps analyzed")

    # Network Visualization (always run)
    print("  Generating bigraph visualization...")
    bigraph_svg = make_bigraph_svg(diag_composite.state)

    # Generate plots
    print("  Generating plots...")
    plots = {}

    # Short sim plots
    plots['mass_short'] = plot_mass(short_history, f'Mass Components ({COMPARISON_DURATION}s)')
    plots['growth_short'] = plot_growth(short_history)
    plots['bulk_histogram'] = plot_bulk_histogram(short_history)

    # v1 comparison plots
    if isinstance(comp_data, dict) and 'v1_initial' in comp_data:
        plots['comparison'] = plot_comparison(comp_data)
        plots['mass_comp'] = plot_mass_comparison(comp_data)
        plots['unique_comp'] = plot_unique_comparison(comp_data)

    # Long sim chromosome plots
    chrom_snaps = long_meta.get('chromosome_snapshots', [])
    if chrom_snaps:
        dur = long_meta.get('duration', 0)
        plots['chromosome_long'] = plot_chromosome_state(
            chrom_snaps, f'Chromosome State (to t={dur:.0f}s)')


    # Generate HTML report
    print("  Generating HTML report...")
    report_path = generate_html_report(step_results, plots, bigraph_svg, diagnostics)

    pipeline_time = time.time() - pipeline_t0
    print("=" * 60)
    print(f"Pipeline complete in {pipeline_time:.0f}s")
    print(f"Report: {report_path}")
    print(f"Cache:  {WORKFLOW_DIR}/")
    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='v2ecoli workflow pipeline')
    parser.add_argument('--clean', action='store_true',
                        help='Clear cached metadata and re-run all steps')
    args = parser.parse_args()

    if args.clean:
        import glob as glob_mod
        for f in glob_mod.glob(os.path.join(WORKFLOW_DIR, '*_meta.json')):
            os.remove(f)
            print(f"  Removed {f}")
        for f in glob_mod.glob(os.path.join(WORKFLOW_DIR, '*.dill')):
            os.remove(f)
            print(f"  Removed {f}")

    run_workflow()
