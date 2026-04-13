"""
v2ecoli Workflow Testing Framework

Step-based pipeline that caches intermediate states, replacing benchmark.py.
Each step checks for cached metadata before executing, enabling incremental
development and fast re-runs.

Pipeline Steps:
0. biocyc — Fetch raw data files from the EcoCyc API
1. raw_data — Catalog raw TSV files and knowledge base stats
2. parca — Run parameter calculator (ParCa) or load cached simData
3. load_model — Build composite from cache
4. single_cell — Run single cell to division
5. division — Cell division, conservation, daughter viability
6. daughters — Divide and run both daughters

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

from v2ecoli.composite import make_composite, _build_core, save_cache
from v2ecoli.library.schema import attrs as ecoli_attrs
from process_bigraph import Composite
from v2ecoli.generate import build_document, FLOW_ORDER, build_execution_layers, DEFAULT_FEATURES
from v2ecoli.viz import build_graph, write_outputs
from v2ecoli.cache import NumpyJSONEncoder, load_initial_state

try:
    from v2ecoli.library.division import divide_cell, divide_bulk
except ImportError:
    divide_cell = divide_bulk = None

try:
    from bigraph_viz import plot_bigraph
except ImportError:
    plot_bigraph = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKFLOW_DIR = 'out/workflow'
# Use existing cache if available, otherwise workflow-local cache
CACHE_DIR = 'out/cache' if os.path.isdir('out/cache') else 'out/workflow/cache'
LONG_DURATION = 1800.0  # Legacy label
MAX_LONG_DURATION = 3600  # Max seconds before giving up on division
SNAPSHOT_INTERVAL = 50  # Seconds between chromosome snapshots
DAUGHTER_DURATION = None  # Set to half the single-cell division time at runtime

# Runtime options (overridden by CLI args)
_OPTIONS = {
    'make_composite': make_composite,
    'fetch_biocyc': False,
    'max_duration': MAX_LONG_DURATION,
}

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

MAX_COORD = 2_320_826  # Half-genome in bp (OriC to Ter = 4,641,652 / 2)


def _coord_to_angle(coord):
    """Convert genome coordinate to angle on circular chromosome."""
    frac = coord / MAX_COORD  # -1 to +1
    return np.pi / 2 - frac * np.pi  # OriC=90deg, Ter=-90deg


def _draw_chromosome(ax, cx, cy, R, rnap_coords, fork_coords):
    """Draw one circular chromosome at (cx, cy) with RNAP and forks."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + R * np.cos(theta), cy + R * np.sin(theta),
            color='#cbd5e1', lw=3, zorder=1)
    ax.plot(cx, cy + R, 'o', color='#10b981', ms=7, zorder=5)
    ax.plot(cx, cy - R, 's', color='#ef4444', ms=5, zorder=5)

    if rnap_coords:
        angles = [_coord_to_angle(c) for c in rnap_coords]
        rx = [cx + R * np.cos(a) for a in angles]
        ry = [cy + R * np.sin(a) for a in angles]
        ax.scatter(rx, ry, c='#3b82f6', s=3, alpha=0.3, zorder=3)

    for coord in fork_coords:
        angle = _coord_to_angle(coord)
        fx = cx + (R + 0.08) * np.cos(angle)
        fy = cy + (R + 0.08) * np.sin(angle)
        ax.plot(fx, fy, '^', color='#f59e0b', ms=9, zorder=6,
                markeredgecolor='black', markeredgewidth=0.5)


def plot_chromosome_map(snapshot, ax, title=''):
    """Draw chromosomes stacked vertically, each with RNAP and forks."""
    n_chrom = max(1, snapshot.get('n_chromosomes', 1))
    rnap_coords = snapshot.get('rnap_coords', [])
    fork_coords = snapshot.get('fork_coords', [])

    rnap_per = len(rnap_coords) // max(n_chrom, 1)
    forks_per = 2

    if n_chrom == 1:
        R = 0.9
        _draw_chromosome(ax, 0, 0, R, rnap_coords, fork_coords)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
    else:
        R = 0.7
        spacing = 2.0 * R + 0.3
        total_h = (n_chrom - 1) * spacing
        for ci in range(n_chrom):
            cy = total_h / 2 - ci * spacing

            r_start = ci * rnap_per
            r_end = r_start + rnap_per if ci < n_chrom - 1 else len(rnap_coords)
            f_start = ci * forks_per
            f_end = min(f_start + forks_per, len(fork_coords))

            _draw_chromosome(ax, 0, cy, R,
                             rnap_coords[r_start:r_end],
                             fork_coords[f_start:f_end])

        margin = R + 0.3
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-total_h / 2 - margin, total_h / 2 + margin)

    # Legend
    ax.plot([], [], 'o', color='#10b981', ms=7, label='OriC')
    ax.plot([], [], 's', color='#ef4444', ms=5, label='Ter')
    if rnap_coords:
        ax.scatter([], [], c='#3b82f6', s=12, label=f'RNAP ({len(rnap_coords)})')
    if fork_coords:
        ax.plot([], [], '^', color='#f59e0b', ms=9, label=f'Replisome ({len(fork_coords)})')

    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=5, framealpha=0.9)
    t = snapshot.get('time', 0)
    chrom_label = f'{n_chrom} chr' if n_chrom > 1 else '1 chr'
    ax.set_title(title or f"t={t:.0f}s ({chrom_label})", fontsize=9)
    ax.axis('off')


def plot_chromosome_state(snapshots, title=''):
    """Plot chromosome state: circular maps at key times + timeseries."""
    if not snapshots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No chromosome data', ha='center', va='center')
        return fig_to_b64(fig)

    times = [s['time'] for s in snapshots]

    # Pick 5 representative snapshots for circular maps
    n = len(snapshots)
    if n >= 5:
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    elif n >= 3:
        indices = [0, n // 2, n - 1]
    else:
        indices = list(range(n))
    indices = sorted(set(indices))

    n_maps = len(indices)
    fig = plt.figure(figsize=(max(14, n_maps * 3.2), 12))
    fig.suptitle(title or 'Chromosome State', fontsize=14, y=0.98)

    # Top row: circular chromosome maps at key timepoints
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, n_maps, i + 1)
        plot_chromosome_map(snapshots[idx], ax)

    # Bottom left: fork progress + chromosome count over time
    ax = fig.add_subplot(2, 2, 3)
    for s in snapshots:
        for coord in s.get('fork_coords', []):
            ax.scatter(s['time'], coord / MAX_COORD, c='#f59e0b', s=12,
                       alpha=0.7, zorder=3, edgecolors='black', linewidths=0.3)
    ax.set_ylabel('Fork position (fraction of half-genome)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Replication Fork Progress')
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color='#10b981', lw=0.8, ls='--', alpha=0.5, label='OriC')
    ax.axhline(1, color='#ef4444', lw=0.8, ls='--', alpha=0.4, label='Ter')
    ax.axhline(-1, color='#ef4444', lw=0.8, ls='--', alpha=0.4)
    # Overlay chromosome count as step
    ax2 = ax.twinx()
    n_chroms = [s['n_chromosomes'] for s in snapshots]
    ax2.step(times, n_chroms, where='post', color='#10b981', lw=2, alpha=0.5, label='Chromosomes')
    ax2.set_ylabel('Chromosomes', color='#10b981', fontsize=9)
    ax2.set_ylim(0, max(n_chroms) + 2)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.tick_params(axis='y', labelcolor='#10b981')
    ax.legend(fontsize=7, loc='lower left')

    # Bottom right: DNA mass + dry mass + division threshold
    ax = fig.add_subplot(2, 2, 4)
    dna = [s['dna_mass'] for s in snapshots]
    dry = [s['dry_mass'] for s in snapshots]
    ax.plot(times, dry, color='#f59e0b', lw=2, label='Dry mass')
    ax.plot(times, dna, color='#8b5cf6', lw=1.5, label='DNA mass')
    if dry:
        ax.axhline(dry[0] * 2, color='red', lw=1, ls=':', alpha=0.4, label='~2x initial (division)')
    ax.set_ylabel('Mass (fg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Mass Growth')
    ax.legend(fontsize=7)

    try:
        plt.tight_layout()
    except Exception:
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
    return fig_to_b64(fig)


def plot_single_cell_growth(snapshots, title=''):
    """Plot growth metrics from long sim snapshots: growth rate, volume, absolute mass, fold change."""
    if not snapshots or len(snapshots) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig_to_b64(fig)

    times = np.array([s['time'] for s in snapshots]) / 60  # minutes

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title or 'Growth Metrics', fontsize=13)

    # 1. Instantaneous growth rate
    ax = axes[0, 0]
    gr = np.array([s.get('instantaneous_growth_rate', 0) for s in snapshots])
    ax.plot(times, gr * 3600, color='#2563eb', lw=1)
    ax.set_ylabel('Growth rate (1/h)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Instantaneous Growth Rate')
    ax.grid(True, alpha=0.15)

    # 2. Cell volume
    ax = axes[0, 1]
    vol = np.array([s.get('volume', 0) for s in snapshots])
    ax.plot(times, vol, color='#16a34a', lw=1.5)
    ax.set_ylabel('Volume (fL)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Cell Volume')
    ax.grid(True, alpha=0.15)

    # 3. Absolute mass
    ax = axes[1, 0]
    for (label, key), color in zip(MASS_KEYS.items(), COLORS):
        vals = np.array([s.get(key, 0) for s in snapshots])
        ax.plot(times, vals, color=color, lw=1.5, label=label)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Mass (fg)')
    ax.set_title('Absolute Mass')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # 4. Fold change
    ax = axes[1, 1]
    for (label, key), color in zip(MASS_KEYS.items(), COLORS):
        vals = np.array([s.get(key, 0) for s in snapshots])
        if len(vals) > 0 and vals[0] > 0:
            ax.plot(times, vals / vals[0], color=color, lw=1.5, label=label)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fold change')
    ax.set_title('Fold Change (normalized to t=0)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_ppgpp_dynamics(snapshots, title=''):
    """Plot ppGpp, amino acid pools, RNA fractions, and NTP pools."""
    if not snapshots or len(snapshots) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No ppGpp data', ha='center', va='center')
        return fig_to_b64(fig)

    times = np.array([s['time'] for s in snapshots]) / 60  # minutes

    # ppGpp concentration: count / (volume_L * N_A) → uM
    N_A = 6.022e23
    ppgpp_counts = np.array([s.get('ppgpp_count', 0) for s in snapshots], dtype=float)
    volumes = np.array([s.get('volume', 0) for s in snapshots], dtype=float)
    volumes_L = np.where(volumes > 0, volumes * 1e-15, 1e-15)
    ppgpp_uM = ppgpp_counts / (N_A * volumes_L) * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title or 'ppGpp & Metabolic Dynamics', fontsize=13)

    # 1. ppGpp concentration
    ax = axes[0, 0]
    ax.plot(times, ppgpp_uM, color='#dc2626', lw=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('ppGpp (μM)')
    ax.set_title('ppGpp Concentration')
    ax.grid(True, alpha=0.15)

    # 2. Amino acid pool (individual AA counts)
    ax = axes[0, 1]
    # Collect all AA IDs from the first snapshot that has data
    aa_ids = []
    for s in snapshots:
        ac = s.get('aa_counts', {})
        if ac:
            aa_ids = sorted(ac.keys())
            break
    if aa_ids:
        cmap = plt.cm.get_cmap('tab20', len(aa_ids))
        has_data = False
        for i, aa_id in enumerate(aa_ids):
            counts = np.array([s.get('aa_counts', {}).get(aa_id, 0) for s in snapshots], dtype=float)
            if counts.max() > 0:
                has_data = True
            label = aa_id.replace('[c]', '')
            ax.plot(times, counts / 1e3, color=cmap(i), lw=0.8, label=label)
        if has_data:
            ax.set_ylabel('Counts (thousands)')
            ax.legend(fontsize=5, ncol=3, loc='upper left',
                      bbox_to_anchor=(0, 1), framealpha=0.7)
        else:
            ax.text(0.5, 0.5, 'No AA data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No AA data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time (min)')
    ax.set_title('Amino Acid Pool')
    ax.grid(True, alpha=0.15)

    # 3. RNA mass fractions (rRNA, tRNA, mRNA as fraction of total RNA)
    ax = axes[1, 0]
    rRNA = np.array([s.get('rRna_mass', 0) for s in snapshots])
    tRNA = np.array([s.get('tRna_mass', 0) for s in snapshots])
    mRNA = np.array([s.get('mRna_mass', 0) for s in snapshots])
    total_rna = rRNA + tRNA + mRNA
    total_rna = np.where(total_rna > 0, total_rna, 1)
    ax.stackplot(times, rRNA / total_rna, tRNA / total_rna, mRNA / total_rna,
                 labels=['rRNA', 'tRNA', 'mRNA'],
                 colors=['#2563eb', '#16a34a', '#f59e0b'], alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fraction of total RNA')
    ax.set_title('RNA Composition')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.15)

    # 4. NTP pools
    ax = axes[1, 1]
    ntp_colors = {'ATP[c]': '#dc2626', 'GTP[c]': '#2563eb',
                  'CTP[c]': '#16a34a', 'UTP[c]': '#f59e0b'}
    has_ntp = False
    for ntp_name, color in ntp_colors.items():
        counts = np.array([s.get('ntp_counts', {}).get(ntp_name, 0)
                          for s in snapshots], dtype=float)
        if counts.max() > 0:
            label = ntp_name.replace('[c]', '')
            ax.plot(times, counts / 1e6, color=color, lw=1.5, label=label)
            has_ntp = True
    if has_ntp:
        ax.legend(fontsize=8)
        ax.set_ylabel('Counts (millions)')
    else:
        ax.text(0.5, 0.5, 'No NTP data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time (min)')
    ax.set_title('NTP Pools')
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_lifecycle_comparison(lifecycle_meta):
    """Plot v1 vs v2 lifecycle comparison: mass, chromosomes, forks, RNAP."""
    v1_snaps = lifecycle_meta.get('v1_snapshots', [])
    v2_snaps = lifecycle_meta.get('v2_snapshots', [])

    if not v1_snaps and not v2_snaps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No lifecycle data', ha='center', va='center')
        return fig_to_b64(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('v1 vs v2 Lifecycle Comparison', fontsize=14, y=0.98)

    def get_ts(snaps, key):
        return [s['time'] for s in snaps], [s.get(key, 0) for s in snaps]

    # 1. Dry mass
    ax = axes[0, 0]
    if v1_snaps:
        t, v = get_ts(v1_snaps, 'dry_mass')
        ax.plot(t, v, 'b-', lw=1.5, alpha=0.8, label='v1')
    if v2_snaps:
        t, v = get_ts(v2_snaps, 'dry_mass')
        ax.plot(t, v, 'r--', lw=1.5, alpha=0.8, label='v2')
    ax.set_ylabel('Dry mass (fg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Dry Mass')
    ax.legend(fontsize=8)

    # 2. DNA mass
    ax = axes[0, 1]
    if v1_snaps:
        t, v = get_ts(v1_snaps, 'dna_mass')
        ax.plot(t, v, 'b-', lw=1.5, alpha=0.8, label='v1')
    if v2_snaps:
        t, v = get_ts(v2_snaps, 'dna_mass')
        ax.plot(t, v, 'r--', lw=1.5, alpha=0.8, label='v2')
    ax.set_ylabel('DNA mass (fg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('DNA Mass')
    ax.legend(fontsize=8)

    # 3. Chromosome count
    ax = axes[1, 0]
    if v1_snaps:
        t, v = get_ts(v1_snaps, 'n_chromosomes')
        ax.step(t, v, 'b-', where='post', lw=2, alpha=0.8, label='v1')
    if v2_snaps:
        t, v = get_ts(v2_snaps, 'n_chromosomes')
        ax.step(t, v, 'r--', where='post', lw=2, alpha=0.8, label='v2')
    ax.set_ylabel('Chromosomes')
    ax.set_xlabel('Time (s)')
    ax.set_title('Chromosome Count')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)

    # 4. Replication fork coordinates
    ax = axes[1, 1]
    for snaps, color, label in [(v1_snaps, '#3b82f6', 'v1'), (v2_snaps, '#ef4444', 'v2')]:
        for s in snaps:
            for coord in s.get('fork_coords', []):
                ax.scatter(s['time'], coord / MAX_COORD, c=color, s=6, alpha=0.4)
        if snaps:
            ax.scatter([], [], c=color, s=20, label=label)  # legend entry
    ax.set_ylabel('Fork position (frac genome)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Replication Forks')
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.3)
    ax.legend(fontsize=8)

    # 5. Active RNAP count
    ax = axes[2, 0]
    if v1_snaps:
        t, v = get_ts(v1_snaps, 'n_rnap')
        ax.plot(t, v, 'b-', lw=1.5, alpha=0.8, label='v1')
    if v2_snaps:
        t, v = get_ts(v2_snaps, 'n_rnap')
        ax.plot(t, v, 'r--', lw=1.5, alpha=0.8, label='v2')
    ax.set_ylabel('Active RNAP')
    ax.set_xlabel('Time (s)')
    ax.set_title('Active RNA Polymerases')
    ax.legend(fontsize=8)

    # 6. Number of active forks
    ax = axes[2, 1]
    if v1_snaps:
        t = [s['time'] for s in v1_snaps]
        n = [len(s.get('fork_coords', [])) for s in v1_snaps]
        ax.step(t, n, 'b-', where='post', lw=2, alpha=0.8, label='v1')
    if v2_snaps:
        t = [s['time'] for s in v2_snaps]
        n = [len(s.get('fork_coords', [])) for s in v2_snaps]
        ax.step(t, n, 'r--', where='post', lw=2, alpha=0.8, label='v2')
    ax.set_ylabel('Active forks')
    ax.set_xlabel('Time (s)')
    ax.set_title('Replication Forks (count)')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)

    plt.tight_layout()
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
# Step Diagnostics
# ---------------------------------------------------------------------------

def bench_step_diagnostics(composite):
    """Per-step analysis of composite structure."""
    cell = composite.state['agents']['0']
    core = composite.core

    diagnostics = []
    for step_name in FLOW_ORDER:
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
# v1 Lifecycle Data Collection
# ---------------------------------------------------------------------------

def _collect_v1_lifecycle(duration):
    """Run v1 for the full lifecycle, extracting data from emitted listeners.

    The v1 timeseries emitter captures:
    - listeners.mass (dry_mass, dna_mass, etc.)
    - listeners.replication_data (fork_coordinates, number_of_oric)
    - listeners.unique_molecule_counts (full_chromosome, active_RNAP, etc.)
    - listeners.rnap_data (active_rnap_coordinates)
    """
    try:
        if not hasattr(np, 'in1d'):
            np.in1d = np.isin

        # Monkey-patch git metadata (fails outside git repos)
        import ecoli.experiments.ecoli_master_sim as _ems
        os.environ.setdefault('IMAGE_GIT_HASH', 'v2ecoli')
        if not hasattr(_ems, '_orig_get_git_diff'):
            _ems._orig_get_git_diff = _ems.get_git_diff
            _ems.get_git_diff = lambda: ''

        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]

        # Find simData: check multiple locations
        sim_data_candidates = [
            'out/kb/simData.cPickle',
            os.path.join(WORKFLOW_DIR, 'simData.cPickle'),
            os.path.join(V1_ROOT_PATH, 'out', 'kb', 'simData.cPickle'),
        ]
        sim_data_path = None
        for p in sim_data_candidates:
            if os.path.exists(p):
                sim_data_path = os.path.abspath(p)
                break
        if sim_data_path is None:
            raise FileNotFoundError(
                f"v1 simData not found. Tried: {sim_data_candidates}")

        with chdir(V1_ROOT_PATH):
            sim = EcoliSim.from_file()
            sim.sim_data_path = sim_data_path
            sim.max_duration = int(duration)
            sim.emitter = 'timeseries'
            sim.divide = False
            sim.build_ecoli()

            t0 = time.time()
            sim.run()
            wall_time = time.time() - t0

        sys.argv = saved_argv
        print(f"    v1 completed in {wall_time:.0f}s")

        # Extract snapshots from emitted timeseries (every SNAPSHOT_INTERVAL)
        v1_ts = sim.query()
        snapshots = []
        for t_key in sorted(v1_ts.keys()):
            if not isinstance(t_key, (int, float)):
                continue
            t = int(t_key)
            if t % SNAPSHOT_INTERVAL != 0 and t != 1:
                continue
            snap = v1_ts[t_key]
            if not isinstance(snap, dict):
                continue

            listeners = snap.get('listeners', {})
            mass = listeners.get('mass', {})
            dry_mass = float(mass.get('dry_mass', 0)) if isinstance(mass, dict) else 0
            dna_mass = float(mass.get('dna_mass', 0)) if isinstance(mass, dict) else 0

            # Chromosome count from unique_molecule_counts listener
            umc = listeners.get('unique_molecule_counts', {})
            n_chrom = 0
            if isinstance(umc, dict):
                fc_count = umc.get('full_chromosome', 0)
                n_chrom = int(fc_count) if isinstance(fc_count, (int, float, np.integer)) else 0

            # Replication fork coordinates from replication_data listener
            rd = listeners.get('replication_data', {})
            fork_coords = []
            if isinstance(rd, dict):
                fc = rd.get('fork_coordinates')
                if fc is not None:
                    if isinstance(fc, (list, np.ndarray)) and len(fc) > 0:
                        fork_coords = [int(c) for c in fc]

            # Active RNAP count from rnap_data listener
            rnap_data = listeners.get('rnap_data', {})
            n_rnap = 0
            if isinstance(rnap_data, dict):
                rnap_coords = rnap_data.get('active_rnap_coordinates')
                if rnap_coords is not None and hasattr(rnap_coords, '__len__'):
                    n_rnap = len(rnap_coords)

            snapshots.append({
                'time': t,
                'n_chromosomes': n_chrom,
                'fork_coords': fork_coords,
                'n_rnap': n_rnap,
                'dna_mass': dna_mass,
                'dry_mass': dry_mass,
            })

        print(f"    {len(snapshots)} snapshots extracted")
        return {'snapshots': snapshots, 'wall_time': wall_time, 'duration': int(duration)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    v1 lifecycle failed: {e}")
        return {'snapshots': [], 'wall_time': 0, 'duration': int(duration)}


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

BIOCYC_FILE_IDS = [
    "complexation_reactions", "dna_sites", "equilibrium_reactions",
    "genes", "metabolic_reactions", "metabolites", "proteins",
    "rnas", "transcription_units", "trna_charging_reactions",
]

FLAT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'vEcoli', 'reconstruction', 'ecoli', 'flat')


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

    try:
        from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
    except ImportError:
        print(f"    Skipped (reconstruction module not available)")
        meta = {'skipped': True, 'reason': 'reconstruction module not in v2ecoli'}
        save_meta(step_name, meta)
        return meta
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
    genome_length = len(raw_data.genome_sequence) if hasattr(raw_data, 'genome_sequence') else 0

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
    simdata_source = 'unknown'
    if os.path.exists(sim_data_cache):
        print(f"    Cache exists at {CACHE_DIR}")
        parca_time = 0.0
        cache_time = 0.0
        # Classify by the source pickle the cache was built from, not "cache".
        if sim_data_path and os.path.normpath(sim_data_path) == os.path.normpath('out/kb/simData.cPickle'):
            simdata_source = 'vecoli_pickle'
        elif sim_data_path and os.path.exists(sim_data_path):
            simdata_source = 'workflow_pickle'
        else:
            simdata_source = 'cache'
        sim_data_path = sim_data_path or '(from cache)'
    elif sim_data_path and os.path.exists(sim_data_path):
        print(f"    Using existing simData at {sim_data_path}")
        parca_time = 0.0
        # out/kb/simData.cPickle is the vEcoli ParCa output location;
        # out/workflow/simData.cPickle is a prior in-workflow run.
        if os.path.normpath(sim_data_path) == os.path.normpath('out/kb/simData.cPickle'):
            simdata_source = 'vecoli_pickle'
        else:
            simdata_source = 'workflow_pickle'
    else:
        # Need to run ParCa
        print("    Running fitSimData_1 (this takes a few minutes)...")
        from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
        from reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
        raw_data = KnowledgeBaseEcoli(
        operons_on=True, remove_rrna_operons=False,
        remove_rrff=False, stable_rrna=False)
        t0 = time.time()
        cache_dir = os.path.join('out', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        sim_data = fitSimData_1(
            raw_data,
            basal_expression_condition="M9 Glucose minus AAs",
            cache_dir=cache_dir,
        )
        parca_time = time.time() - t0

        # Save simData
        sim_data_path = os.path.join(WORKFLOW_DIR, 'simData.cPickle')
        os.makedirs(WORKFLOW_DIR, exist_ok=True)
        with open(sim_data_path, 'wb') as f:
            dill.dump(sim_data, f)
        parca_ran = True
        simdata_source = 'computed'
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
        'simdata_source': simdata_source,
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
    composite = _OPTIONS['make_composite'](cache_dir=CACHE_DIR)
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


def step_single_cell():
    """Step 4: Run single-cell simulation to division.

    Runs in chunks (SNAPSHOT_INTERVAL seconds each), capturing chromosome
    state at every interval. Stops when division condition is met (2+
    chromosomes AND dry mass >= 2x initial) or MAX_LONG_DURATION is reached.
    Saves pre-division cell state as JSON and .pbg for downstream use.
    """
    step_name = 'single_cell'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 4: Single Cell Simulation (cached)")
        return meta

    max_dur = _OPTIONS['max_duration']
    print(f"  Step 4: Single Cell Simulation (to division, max {max_dur}s)")
    composite = _OPTIONS['make_composite'](cache_dir=CACHE_DIR)

    cell = composite.state['agents']['0']
    bulk_before = np.array(cell['bulk']['count'], copy=True)
    initial_dry_mass = float(cell.get('listeners', {}).get('mass', {}).get('dry_mass', 380))

    # Keep reference to emitter instance — survives division (agent removal)
    em_edge = cell.get('emitter', {})
    emitter_instance = em_edge.get('instance') if isinstance(em_edge, dict) else None

    t0 = time.time()
    divided = False
    last_cell_data = None
    total_run = 0

    while total_run < max_dur:
        chunk = min(SNAPSHOT_INTERVAL, max_dur - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            total_run += chunk
            # Check if this is from the Division step or an unrelated crash
            err_str = str(e)
            if 'divide' in err_str.lower() or '_add' in err_str or '_remove' in err_str:
                print(f"    Cell divided at ~t={total_run}s ({total_run/60:.0f}min)")
                divided = True
                break
            else:
                # Non-division error — log and continue
                import traceback
                print(f"    Warning at ~t={total_run}s: {type(e).__name__}: {err_str[:100]}")
                if total_run <= SNAPSHOT_INTERVAL:
                    traceback.print_exc()
                continue
        total_run += chunk

        cell = composite.state.get('agents', {}).get('0')
        if cell is None:
            divided = True
            break

        # Save cell state snapshot for pre-division backup
        _data_keys = {'bulk', 'unique', 'listeners', 'environment', 'boundary',
                      'global_time', 'timestep', 'divide', 'division_threshold',
                      'process_state', 'allocator_rng'}
        last_cell_data = {k: v for k, v in cell.items()
                          if k in _data_keys or k.startswith('request_') or k.startswith('allocate_')}

        unique = cell.get('unique', {})

        # Check division readiness from live state
        fc = unique.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        mass = cell.get('listeners', {}).get('mass', {})
        dry_mass = float(mass.get('dry_mass', 0))

        # Check mass threshold for division readiness reporting
        threshold = cell.get('division_threshold', float('inf'))
        if isinstance(threshold, str):
            threshold = float('inf')

        if total_run % 500 == 0:
            rep = unique.get('active_replisome')
            fork_count = 0
            if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
                fork_count = int(rep['_entryState'].view(np.bool_).sum())
            thresh_str = f'{threshold:.0f}fg' if threshold < float('inf') else '?'
            print(f"    t={total_run}s ({total_run/60:.0f}min): {n_chrom} chroms, "
                  f"dry_mass={dry_mass:.0f}/{thresh_str}, forks={fork_count}")

    # Extract snapshot data from emitter history (saved reference survives division)
    emitter_history = emitter_instance.history if emitter_instance else []

    # Build bulk molecule indexes once for efficient extraction
    ppgpp_idx = None
    aa_idxs = {}  # aa_id -> idx
    ntp_idxs = {}  # name -> idx
    if emitter_history:
        first_bulk = emitter_history[0].get('bulk')
        if first_bulk is not None and hasattr(first_bulk, 'dtype') and 'id' in first_bulk.dtype.names:
            bulk_ids = first_bulk['id']

            def _find_idx(mol_id):
                mask = np.where(bulk_ids == mol_id)[0]
                if len(mask) == 0:
                    mask = np.where(bulk_ids == mol_id.encode())[0]
                return mask[0] if len(mask) else None

            ppgpp_idx = _find_idx('GUANOSINE-5DP-3DP[c]')

            # Amino acid indexes (21 canonical)
            _AA_IDS = [
                'L-ALPHA-ALANINE[c]', 'ARG[c]', 'ASN[c]', 'L-ASPARTATE[c]',
                'CYS[c]', 'GLT[c]', 'GLN[c]', 'GLY[c]', 'HIS[c]', 'ILE[c]',
                'LEU[c]', 'LYS[c]', 'MET[c]', 'PHE[c]', 'PRO[c]', 'SER[c]',
                'THR[c]', 'TRP[c]', 'TYR[c]', 'VAL[c]', 'L-SELENOCYSTEINE[c]',
            ]
            for aa_id in _AA_IDS:
                idx = _find_idx(aa_id)
                if idx is not None:
                    aa_idxs[aa_id] = idx

            # NTP indexes
            for ntp in ['ATP[c]', 'GTP[c]', 'CTP[c]', 'UTP[c]']:
                idx = _find_idx(ntp)
                if idx is not None:
                    ntp_idxs[ntp] = idx

    snapshots = []
    for snap in emitter_history:
        t = snap.get('global_time', 0)
        if int(t) % SNAPSHOT_INTERVAL != 0 and t != 1:
            continue

        mass = snap.get('listeners', {}).get('mass', {}) if isinstance(snap.get('listeners'), dict) else {}

        # Chromosome state from emitter
        fc = snap.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        rep = snap.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active_rep = rep[rep['_entryState'].view(np.bool_)]
            if len(active_rep) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active_rep['coordinates'].tolist()

        domains = snap.get('chromosome_domain')
        n_domains = 0
        if domains is not None and hasattr(domains, 'dtype') and '_entryState' in domains.dtype.names:
            n_domains = int(domains['_entryState'].view(np.bool_).sum())

        rnap = snap.get('active_RNAP')
        rnap_coords = []
        n_rnap = 0
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            active_rnap = rnap[rnap['_entryState'].view(np.bool_)]
            n_rnap = len(active_rnap)
            if n_rnap > 0 and 'coordinates' in rnap.dtype.names:
                rnap_coords = active_rnap['coordinates'].tolist()

        # Bulk molecule counts
        ppgpp_count = 0
        aa_counts_dict = {}
        ntp_counts = {}
        bulk = snap.get('bulk')
        if bulk is not None and hasattr(bulk, 'dtype'):
            bc = bulk['count']
            if ppgpp_idx is not None:
                ppgpp_count = int(bc[ppgpp_idx])
            for aa_id, aa_idx in aa_idxs.items():
                aa_counts_dict[aa_id] = int(bc[aa_idx])
            for ntp_name, ntp_idx in ntp_idxs.items():
                ntp_counts[ntp_name] = int(bc[ntp_idx])

        snapshots.append({
            'time': float(t),
            'n_chromosomes': n_chrom,
            'n_domains': n_domains,
            'fork_coords': fork_coords,
            'rnap_coords': rnap_coords,
            'n_rnap': n_rnap,
            'dna_mass': float(mass.get('dna_mass', 0)),
            'dry_mass': float(mass.get('dry_mass', 0)),
            'protein_mass': float(mass.get('protein_mass', 0)),
            'rna_mass': float(mass.get('rRna_mass', 0)) + float(mass.get('tRna_mass', 0)) + float(mass.get('mRna_mass', 0)),
            'rRna_mass': float(mass.get('rRna_mass', 0)),
            'tRna_mass': float(mass.get('tRna_mass', 0)),
            'mRna_mass': float(mass.get('mRna_mass', 0)),
            'smallMolecule_mass': float(mass.get('smallMolecule_mass', 0)),
            'instantaneous_growth_rate': float(mass.get('instantaneous_growth_rate', 0)),
            'volume': float(mass.get('volume', 0)),
            'ppgpp_count': ppgpp_count,
            'aa_counts': aa_counts_dict,
            'ntp_counts': ntp_counts,
        })

    wall_time = time.time() - t0

    # After division, agent '0' may be gone — use last snapshot data
    agents = composite.state.get('agents', {})
    cell = agents.get('0')
    if cell is None:
        for aid, astate in agents.items():
            if isinstance(astate, dict) and 'bulk' in astate:
                cell = astate
                break

    if last_cell_data and 'bulk' in last_cell_data:
        bulk_after = last_cell_data['bulk']['count']
        changed = int((bulk_before != bulk_after).sum())
    else:
        changed = 0

    # Use last snapshot for final metrics
    final_snap = snapshots[-1] if snapshots else {}

    # Save pre-division cell state for division test
    save_state_data(step_name, {
        'cell_state': last_cell_data,
        'global_time': final_snap.get('time', 0.0),
    })

    # Save pre-division state as JSON and .pbg
    pre_div_dir = os.path.join(WORKFLOW_DIR, 'pre_division')
    os.makedirs(pre_div_dir, exist_ok=True)
    if last_cell_data and 'bulk' in last_cell_data:
        from v2ecoli.cache import save_initial_state
        save_initial_state(last_cell_data, os.path.join(pre_div_dir, 'pre_division_state.json'))
        print(f"    Saved pre-division state: {pre_div_dir}/pre_division_state.json")

    # Save .pbg snapshot of the full composite
    try:
        from v2ecoli.pbg import save_pbg
        pbg_path = os.path.join(pre_div_dir, 'pre_division.pbg')
        save_pbg(composite, pbg_path)
        print(f"    Saved pre-division .pbg: {pbg_path}")
    except Exception as e:
        print(f"    Warning: could not save .pbg: {e}")

    meta = {
        'duration': total_run,
        'wall_time': wall_time,
        'bulk_changed': changed,
        'total_bulk': len(bulk_before),
        'final_dry_mass': final_snap.get('dry_mass', 0),
        'final_cell_mass': 0,
        'final_volume': 0,
        'rate': total_run / wall_time if wall_time > 0 else 0,
        'division_reached': divided,
        'initial_dry_mass': initial_dry_mass,
        'chromosome_snapshots': snapshots,
        'pre_division_dir': pre_div_dir,
    }
    save_meta(step_name, meta)

    print(f"    {wall_time:.0f}s wall, {changed} molecules changed, "
          f"{meta['rate']:.1f}x realtime")
    print(f"    dry_mass={meta['final_dry_mass']:.0f}fg, "
          f"division={'reached' if divided else 'not reached'}")
    return meta


def step_v1_comparison():
    """Step 4b: Run v1 lifecycle comparison (cached independently).

    The v1 result is expensive (~40s for 2500s sim) and rarely changes.
    It's cached in its own JSON file that survives --clean runs.
    """
    step_name = 'v1_comparison'
    meta = load_meta(step_name)
    if meta is not None:
        n = len(meta.get('v1_snapshots', []))
        print(f"  Step 4b: v1 Comparison (cached, {n} snapshots)")
        return meta

    single_cell_meta = load_meta('single_cell')
    if single_cell_meta is None:
        print(f"  Step 4b: v1 Comparison (skipped, no long sim data)")
        meta = {'skipped': True, 'reason': 'no long sim data', 'v1_snapshots': []}
        save_meta(step_name, meta)
        return meta

    duration = single_cell_meta.get('duration', 0)

    # Check for legacy v1 cache file (list of snapshots)
    v1_cache_path = os.path.join(WORKFLOW_DIR, f'v1_lifecycle_{duration}s.json')
    v1_snapshots = []
    v1_wall_time = 0

    if os.path.exists(v1_cache_path):
        with open(v1_cache_path) as f:
            cached = json.load(f)
        # Handle both old (list) and new (dict with wall_time) formats
        if isinstance(cached, list):
            v1_snapshots = cached
        elif isinstance(cached, dict):
            v1_snapshots = cached.get('snapshots', [])
            v1_wall_time = cached.get('wall_time', 0)
        print(f"  Step 4b: v1 Comparison (loaded {len(v1_snapshots)} cached snapshots)")
    elif V1_AVAILABLE:
        print(f"  Step 4b: Running v1 for {duration}s...")
        try:
            result = _collect_v1_lifecycle(duration)
            if isinstance(result, dict):
                v1_snapshots = result.get('snapshots', [])
                v1_wall_time = result.get('wall_time', 0)
            else:
                v1_snapshots = result  # legacy list format
            if v1_snapshots:
                with open(v1_cache_path, 'w') as f:
                    json.dump({'snapshots': v1_snapshots, 'wall_time': v1_wall_time,
                               'duration': duration}, f)
        except Exception as e:
            print(f"    v1 failed: {e}")
    else:
        print(f"  Step 4b: v1 Comparison (v1 not available)")

    meta = {
        'duration': duration,
        'v1_available': len(v1_snapshots) > 0,
        'v1_snapshots': v1_snapshots,
        'v1_wall_time': v1_wall_time,
        'n_snapshots': len(v1_snapshots),
    }
    save_meta(step_name, meta)
    return meta


def step_division():
    """Step 5: Test cell division, conservation, and daughter viability."""
    step_name = 'division'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 5: Division (cached)")
        return meta

    print(f"  Step 5: Division")

    # Try to load pre-division state from long sim
    prediv_state = None
    prediv_time = 0.0
    long_state_path = os.path.join(WORKFLOW_DIR, 'single_cell.dill')
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
        composite = _OPTIONS['make_composite'](cache_dir=CACHE_DIR)
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
        from v2ecoli.generate import (
            build_document)
        t0 = time.time()
        try:
            d1_doc = build_document(
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


def _extract_snapshots_from_emitter(composite, label=''):
    """Extract snapshot data from a composite's emitter history."""
    cell = composite.state['agents']['0']
    em_edge = cell.get('emitter')
    history = em_edge['instance'].history if isinstance(em_edge, dict) and 'instance' in em_edge else []

    snapshots = []
    for snap in history:
        t = snap.get('global_time', 0)
        if int(t) % SNAPSHOT_INTERVAL != 0 and t != 1:
            continue

        mass = snap.get('listeners', {}).get('mass', {}) if isinstance(snap.get('listeners'), dict) else {}

        fc = snap.get('full_chromosome')
        n_chrom = 0
        if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
            n_chrom = int(fc['_entryState'].view(np.bool_).sum())

        rep = snap.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active_rep = rep[rep['_entryState'].view(np.bool_)]
            if len(active_rep) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active_rep['coordinates'].tolist()

        domains = snap.get('chromosome_domain')
        n_domains = 0
        if domains is not None and hasattr(domains, 'dtype') and '_entryState' in domains.dtype.names:
            n_domains = int(domains['_entryState'].view(np.bool_).sum())

        rnap = snap.get('active_RNAP')
        rnap_coords = []
        n_rnap = 0
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            active_rnap = rnap[rnap['_entryState'].view(np.bool_)]
            n_rnap = len(active_rnap)
            if n_rnap > 0 and 'coordinates' in rnap.dtype.names:
                rnap_coords = active_rnap['coordinates'].tolist()

        snapshots.append({
            'time': float(t),
            'n_chromosomes': n_chrom,
            'n_domains': n_domains,
            'fork_coords': fork_coords,
            'rnap_coords': rnap_coords,
            'n_rnap': n_rnap,
            'dna_mass': float(mass.get('dna_mass', 0)),
            'dry_mass': float(mass.get('dry_mass', 0)),
            'protein_mass': float(mass.get('protein_mass', 0)),
            'rna_mass': float(mass.get('rRna_mass', 0)) + float(mass.get('tRna_mass', 0)) + float(mass.get('mRna_mass', 0)),
            'rRna_mass': float(mass.get('rRna_mass', 0)),
            'tRna_mass': float(mass.get('tRna_mass', 0)),
            'mRna_mass': float(mass.get('mRna_mass', 0)),
            'smallMolecule_mass': float(mass.get('smallMolecule_mass', 0)),
        })

    return snapshots


def step_daughters():
    """Step 6: Divide pre-division cell into 2 daughters, run both for half a generation.

    Builds two daughter composites from the pre-division state, runs each for
    DAUGHTER_DURATION seconds, and extracts snapshot data from their emitters.
    """
    step_name = 'daughters'
    meta = load_meta(step_name)
    if meta is not None:
        print(f"  Step 6: Daughter Simulations (cached)")
        return meta

    # Default daughter duration: half the single-cell generation time
    single_cell_meta = load_meta('single_cell') or {}
    generation_time = single_cell_meta.get('duration', 2500)
    default_dur = int(generation_time / 2)
    daughter_dur = _OPTIONS.get('daughter_duration', default_dur)
    print(f"  Step 6: Daughter Simulations ({daughter_dur}s per daughter, "
          f"half generation = {default_dur}s)")

    # Load pre-division state from long sim
    long_state_path = os.path.join(WORKFLOW_DIR, 'single_cell.dill')
    if not os.path.exists(long_state_path):
        print("    No pre-division state available — skipping")
        meta = {'skipped': True, 'reason': 'no pre-division state'}
        save_meta(step_name, meta)
        return meta

    with open(long_state_path, 'rb') as f:
        checkpoint = dill.load(f)
    cell_data = checkpoint.get('cell_state', {})

    if not cell_data or 'bulk' not in cell_data:
        print("    No valid cell state (mother did not divide) — skipping")
        meta = {'skipped': True, 'reason': 'no valid pre-division cell state'}
        save_meta(step_name, meta)
        return meta

    # Divide the cell
    print("    Dividing mother cell...")
    d1_state, d2_state = divide_cell(cell_data)

    # Load configs for building daughter composites
    cache_path = os.path.join(CACHE_DIR, 'sim_data_cache.dill')
    with open(cache_path, 'rb') as f:
        cache = dill.load(f)
    configs = cache.get('configs', {})
    unique_names = cache.get('unique_names', [])
    dry_mass_inc = cache.get('dry_mass_inc_dict', {})

    from v2ecoli.generate import (
        build_document)

    def _run_daughter(label, dstate, seed):
        """Build and run a single daughter composite, return results dict."""
        t0 = time.time()
        doc = build_document(
            dstate, configs, unique_names,
            dry_mass_inc_dict=dry_mass_inc, seed=seed)
        comp = Composite(doc, core=_build_core())
        build_time = time.time() - t0

        d_cell = comp.state['agents']['0']
        d_mass = d_cell.get('listeners', {}).get('mass', {})
        initial_dry = float(d_mass.get('dry_mass', 0))

        t0 = time.time()
        try:
            comp.run(daughter_dur)
            run_ok = True
        except Exception as e:
            run_ok = False
        wall_time = time.time() - t0

        snaps = _extract_snapshots_from_emitter(comp, label)
        final_snap = snaps[-1] if snaps else {}
        final_dry = final_snap.get('dry_mass', 0)

        # Fallback: read final mass from composite state (emitter may be absent)
        if final_dry == 0:
            d_cell_post = comp.state.get('agents', {}).get('0', {})
            d_mass_post = d_cell_post.get('listeners', {}).get('mass', {})
            final_dry = float(d_mass_post.get('dry_mass', 0))

        return {
            'build_time': build_time,
            'wall_time': wall_time,
            'run_ok': run_ok,
            'initial_dry_mass': initial_dry,
            'final_dry_mass': final_dry,
            'fold_change': final_dry / initial_dry if initial_dry > 0 else 0,
            'n_snapshots': len(snaps),
            'snapshots': snaps,
        }

    daughters = {}
    for label, dstate, seed in [('daughter_1', d1_state, 1), ('daughter_2', d2_state, 2)]:
        print(f"    Building {label}...")
        d = _run_daughter(label, dstate, seed)
        daughters[label] = d
        if d['initial_dry_mass'] > 0:
            print(f"    {label}: {d['wall_time']:.0f}s wall, "
                  f"dry_mass {d['initial_dry_mass']:.0f} -> {d['final_dry_mass']:.0f}fg "
                  f"({d['fold_change']:.2f}x)")

    meta = {
        'duration': daughter_dur,
        'daughter_1': {k: v for k, v in daughters.get('daughter_1', {}).items() if k != 'snapshots'},
        'daughter_2': {k: v for k, v in daughters.get('daughter_2', {}).items() if k != 'snapshots'},
        'daughter_1_snapshots': daughters.get('daughter_1', {}).get('snapshots', []),
        'daughter_2_snapshots': daughters.get('daughter_2', {}).get('snapshots', []),
    }
    save_meta(step_name, meta)
    return meta


def plot_daughters_mass(daughters_meta, title=''):
    """Plot both daughters' mass fold change side by side (2 subplots)."""
    d1_snaps = daughters_meta.get('daughter_1_snapshots', [])
    d2_snaps = daughters_meta.get('daughter_2_snapshots', [])

    if not d1_snaps and not d2_snaps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No daughters data', ha='center', va='center')
        return fig_to_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title or 'Daughter Simulations — Daughter Mass', fontsize=13)

    mass_components = [
        ('dry_mass', 'Dry Mass', 'k'),
        ('protein_mass', 'Protein', '#22c55e'),
        ('dna_mass', 'DNA', '#8b5cf6'),
        ('rRna_mass', 'rRNA', '#3b82f6'),
        ('tRna_mass', 'tRNA', '#06b6d4'),
        ('mRna_mass', 'mRNA', '#f97316'),
        ('smallMolecule_mass', 'Small mol', '#f59e0b'),
    ]

    for ax, snaps, label in [(axes[0], d1_snaps, 'Daughter 1'),
                              (axes[1], d2_snaps, 'Daughter 2')]:
        if not snaps:
            ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center')
            ax.set_title(label)
            continue

        times = [s['time'] for s in snaps]
        for key, comp_label, color in mass_components:
            vals = [s.get(key, 0) for s in snaps]
            if vals and vals[0] > 0:
                fold = [v / vals[0] for v in vals]
                ax.plot(times, fold, color=color, lw=1.5, label=comp_label)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fold change')
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------

def generate_html_report(step_results, plots, network_html_rel, diagnostics):
    """Generate the HTML report organized by pipeline step."""

    try:
        from v2ecoli.library.repro_banner import banner_html
        banner = banner_html()
    except Exception:
        banner = ''

    biocyc = step_results.get('biocyc', {})
    raw = step_results['raw_data']
    parca = step_results['parca']
    model = step_results['load_model']
    long = step_results['single_cell']  # single cell sim results
    div = step_results['division']
    daughters = step_results.get('daughters', {})

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
    long_wall = long.get('wall_time', 0)
    long_dur = long.get('duration', LONG_DURATION)
    long_rate = long.get('rate', 0)
    d1_wall = daughters.get('daughter_1', {}).get('wall_time', 0)
    d2_wall = daughters.get('daughter_2', {}).get('wall_time', 0)
    daughters_wall = d1_wall + d2_wall
    daughters_dur = daughters.get('duration', 0)

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

{banner}
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
    <li><a href="#sec-long">Single Cell Simulation + v1 Comparison ({long_dur/60:.0f} min)</a></li>
    <li><a href="#sec-division">Division</a></li>
    <li><a href="#sec-daughters">Daughter Simulations</a></li>
    <li><a href="#sec-network">Process-Bigraph Network</a></li>
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
{'<div class="section" style="background:#fff7ed;border-left:4px solid #f59e0b;padding:0.5em 0.75em;margin:0.5em 0;"><strong>simData source:</strong> reused from <a href="https://github.com/CovertLab/vEcoli">vEcoli</a> (<code>' + parca.get('sim_data_path', '') + '</code>) — ParCa was not re-run in this workflow.</div>' if parca.get('simdata_source') == 'vecoli_pickle' else ''}
<div class="metrics">
  <div class="metric"><div class="label">simData Source</div><div class="value" style="font-size:0.8em">{ {'vecoli_pickle':'vEcoli pickle','workflow_pickle':'workflow pickle','cache':'cached','computed':'computed here'}.get(parca.get('simdata_source'), parca.get('simdata_source','?')) }</div></div>
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

<!-- ===== Step 4: Long Sim + v1 Comparison ===== -->
<h2 id="sec-long">4. Single Cell Simulation + v1 Comparison ({long_dur/60:.0f} min) {cached_badge(long)}</h2>
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
        if plots.get('growth_long'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["growth_long"]}" alt="Growth Metrics"></div>\n')
        if plots.get('ppgpp_dynamics'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["ppgpp_dynamics"]}" alt="ppGpp Dynamics"></div>\n')

        # Single cell simulation summary
        v2_wall = long.get('wall_time', 0)
        v2_rate = long_dur / v2_wall if v2_wall > 0 else 0
        v2_final_dry = long.get('final_dry_mass', 0)

        f.write(f"""
<h3>Single Cell Summary</h3>
<table style="margin: 1em auto; border-collapse: collapse; font-size: 0.9em;">
<tr style="background: #f3f4f6;"><th style="padding: 6px 16px; text-align: left;">Metric</th>
    <th style="padding: 6px 16px;">Value</th></tr>
<tr><td style="padding: 4px 16px;">Sim duration</td>
    <td style="padding: 4px 16px; text-align: center;">{long_dur:.0f}s</td></tr>
<tr><td style="padding: 4px 16px;">Wall time</td>
    <td style="padding: 4px 16px; text-align: center;">{v2_wall:.0f}s</td></tr>
<tr><td style="padding: 4px 16px;">Speed (sim/wall)</td>
    <td style="padding: 4px 16px; text-align: center; font-weight: bold;">{v2_rate:.1f}x</td></tr>
<tr><td style="padding: 4px 16px;">Final dry mass</td>
    <td style="padding: 4px 16px; text-align: center;">{v2_final_dry:.1f} fg</td></tr>
<tr><td style="padding: 4px 16px;">Division reached</td>
    <td style="padding: 4px 16px; text-align: center;">{'Yes' if long.get('division_reached') else 'No'}</td></tr>
<tr><td style="padding: 4px 16px;">Snapshots</td>
    <td style="padding: 4px 16px; text-align: center;">{len(long.get('chromosome_snapshots', []))}</td></tr>
</table>""")

        f.write(f"""
<!-- ===== Step 5: Division ===== -->
<h2 id="sec-division">5. Division {cached_badge(div)}</h2>

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

<!-- ===== Step 6: Daughter Simulations ===== -->
<h2 id="sec-daughters">6. Daughter Simulations {cached_badge(daughters)}</h2>""")

        if daughters.get('skipped'):
            f.write(f"""
<div class="section"><p>Skipped: {daughters.get('reason', 'unknown')}</p></div>""")
        else:
            d1 = daughters.get('daughter_1', {})
            d2 = daughters.get('daughter_2', {})
            f.write(f"""
<div class="section">
  <p>Two daughter cells from the pre-division state, each run for {daughters.get('duration', 0)}s
  (approximately half a generation).</p>
</div>
<h3>Daughter 1</h3>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{d1.get('wall_time', 0):.0f}s</div></div>
  <div class="metric"><div class="label">Initial Dry Mass</div><div class="value">{d1.get('initial_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Final Dry Mass</div><div class="value">{d1.get('final_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Fold Change</div><div class="value green">{d1.get('fold_change', 0):.2f}x</div></div>
</div>
<h3>Daughter 2</h3>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{d2.get('wall_time', 0):.0f}s</div></div>
  <div class="metric"><div class="label">Initial Dry Mass</div><div class="value">{d2.get('initial_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Final Dry Mass</div><div class="value">{d2.get('final_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Fold Change</div><div class="value green">{d2.get('fold_change', 0):.2f}x</div></div>
</div>""")

        if plots.get('daughters_mass'):
            f.write(f'<div class="plot"><img src="data:image/png;base64,{plots["daughters_mass"]}" alt="Daughters Mass"></div>\n')

        f.write(f"""
<!-- ===== Process-Bigraph Network (Cytoscape.js viewer) ===== -->
<h2 id="sec-network">7. Process-Bigraph Network</h2>
<div class="section">
  <p>Interactive Cytoscape.js viewer of the composite — stores on the left, processes on the right,
  sorted by execution layer. Click a node for math, ports, config; switch layouts from the dropdown.
  Full-screen viewer: <a href="{network_html_rel}" target="_blank"><code>{network_html_rel}</code></a>.</p>
</div>
<iframe src="{network_html_rel}" style="width:100%;height:900px;border:1px solid #e2e8f0;border-radius:6px;"></iframe>

<!-- ===== Timing Summary ===== -->
<h2 id="sec-timing">Timing Summary</h2>
<div class="section">
  <table>
    <tr><th>Step</th><th>Wall Time</th><th>Sim Time</th><th>Speed</th></tr>
    <tr><td>Model build</td><td>{build_time:.2f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>Single cell (to division)</td><td>{long_wall:.1f}s</td><td>{long_dur:.0f}s</td><td>{long_rate:.1f}x realtime</td></tr>
    <tr><td>Division split</td><td>{div.get('split_time', 0):.3f}s</td><td>&mdash;</td><td>&mdash;</td></tr>
    <tr><td>Daughter simulations</td><td>{daughters_wall:.1f}s</td><td>{daughters_dur*2:.0f}s (2x{daughters_dur:.0f}s)</td><td>{daughters_dur*2/max(daughters_wall, 0.1):.1f}x realtime</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>{build_time + long_wall + daughters_wall:.0f}s</strong></td><td><strong>{long_dur + daughters_dur*2:.0f}s</strong></td><td>&mdash;</td></tr>
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

    # Step 0: EcoCyc API (skipped by default, use --fetch-biocyc to enable)
    if _OPTIONS['fetch_biocyc']:
        biocyc_meta = step_biocyc()
    else:
        biocyc_meta = load_meta('biocyc') or {'skipped': True, 'files': {}}
        print(f"  Step 0: EcoCyc API (skipped, use --fetch-biocyc to refresh)")
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

    # Step 4: Single Cell Simulation (to division)
    single_cell_meta = step_single_cell()
    step_results['single_cell'] = single_cell_meta

    # Step 5: Division
    div_meta = step_division()
    step_results['division'] = div_meta

    # Step 6: Daughter Simulations (skip with --no-daughters)
    if _OPTIONS.get('skip_daughters'):
        daughters_meta = load_meta('daughters') or {'skipped': True, 'reason': '--no-daughters'}
        print(f"  Step 6: Daughter Simulations (skipped)")
    else:
        daughters_meta = step_daughters()
    step_results['daughters'] = daughters_meta

    # Step Diagnostics (always run, uses the composite from step 3)
    print("  Diagnostics: Step analysis")
    diag_composite = _OPTIONS['make_composite'](cache_dir=CACHE_DIR)
    diagnostics = bench_step_diagnostics(diag_composite)
    print(f"    {len(diagnostics)} steps analyzed")

    # Update .pbg model files
    print("  Updating .pbg model files...")
    from v2ecoli.pbg import save_pbg
    os.makedirs('models', exist_ok=True)
    save_pbg(diag_composite, 'models/partitioned.pbg')
    print(f"    models/partitioned.pbg updated")

    # Network Visualization (Cytoscape.js interactive viewer)
    print("  Generating interactive network visualization...")
    network_data = build_graph(diag_composite, build_execution_layers(DEFAULT_FEATURES))
    _, network_html_path = write_outputs(
        network_data,
        out_dir=WORKFLOW_DIR,
        name='network',
        title='v2ecoli Baseline Composite',
        subtitle='Interactive Cytoscape.js view (built from the workflow composite)',
    )
    network_html_rel = os.path.relpath(network_html_path, WORKFLOW_DIR)

    # Generate plots
    print("  Generating plots...")
    plots = {}

    # Long sim plots
    chrom_snaps = single_cell_meta.get('chromosome_snapshots', [])
    if chrom_snaps:
        dur = single_cell_meta.get('duration', 0)
        plots['chromosome_long'] = plot_chromosome_state(
            chrom_snaps, f'v2 Chromosome State (to t={dur:.0f}s)')
        plots['growth_long'] = plot_single_cell_growth(
            chrom_snaps, f'Growth Metrics ({dur/60:.0f} min)')
        if any(s.get('ppgpp_count', 0) > 0 for s in chrom_snaps):
            plots['ppgpp_dynamics'] = plot_ppgpp_dynamics(
                chrom_snaps, f'ppGpp Dynamics ({dur/60:.0f} min)')

    # Daughters plots
    daughters = step_results.get('daughters', {})
    if not daughters.get('skipped') and (daughters.get('daughter_1_snapshots') or daughters.get('daughter_2_snapshots')):
        plots['daughters_mass'] = plot_daughters_mass(daughters, 'Daughters — Daughter Mass Fold Change')

    # Generate HTML report
    print("  Generating HTML report...")
    report_path = generate_html_report(step_results, plots, network_html_rel, diagnostics)

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
    parser.add_argument('--fetch-biocyc', action='store_true',
                        help='Fetch fresh data from EcoCyc API (slow, skipped by default)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Override max sim duration in seconds (default: run to division)')
    parser.add_argument('--no-daughters', action='store_true',
                        help='Skip the daughters simulation step')
    parser.add_argument('--daughter-duration', type=int, default=None,
                        help='Override daughter sim duration in seconds')
    args = parser.parse_args()

    # Apply CLI overrides
    _OPTIONS['fetch_biocyc'] = args.fetch_biocyc
    if args.duration is not None:
        _OPTIONS['max_duration'] = args.duration
    _OPTIONS['skip_daughters'] = args.no_daughters
    if args.daughter_duration is not None:
        _OPTIONS['daughter_duration'] = args.daughter_duration

    if args.clean:
        import glob as glob_mod
        for f in glob_mod.glob(os.path.join(WORKFLOW_DIR, '*_meta.json')):
            os.remove(f)
            print(f"  Removed {f}")
        for f in glob_mod.glob(os.path.join(WORKFLOW_DIR, '*.dill')):
            os.remove(f)
            print(f"  Removed {f}")

    run_workflow()
