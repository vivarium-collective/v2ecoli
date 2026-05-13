"""WorkflowVisualization — full cell lifecycle report.

Migrated rendering from reports/workflow_report.py (the largest legacy
report, 2452 LOC). The Step takes a single trajectory (list of snapshot
dicts) and step-level metadata, and produces a multi-panel lifecycle HTML
(mass trajectory, growth rate, chromosome state, ppGpp dynamics, division
results, daughters, timing). The wrapper at reports/workflow_report.py
keeps all pipeline orchestration (parca → cache → composite → run →
division → daughters) and dispatches to this Step for rendering.

Matplotlib is lazy-imported inside the rendering functions.
"""

from __future__ import annotations

import base64
import io
import time
from html import escape
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_document

# ---------------------------------------------------------------------------
# Constants (replicated from legacy for standalone use)
# ---------------------------------------------------------------------------

MAX_COORD = 2_320_826  # Half-genome in bp (OriC to Ter = 4,641,652 / 2)

MASS_KEYS = {
    'Protein': 'protein_mass',
    'tRNA': 'tRna_mass',
    'rRNA': 'rRna_mass',
    'mRNA': 'mRna_mass',
    'DNA': 'dna_mass',
    'Small Mol': 'smallMolecule_mass',
    'Dry Mass': 'dry_mass',
}
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    """Save a matplotlib figure to a base64-encoded PNG string."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _plot_mass(history, title=''):
    import numpy as np
    import matplotlib.pyplot as plt

    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_b64(fig)

    times = _np_array([s.get('global_time', s.get('time', 0)) for s in history])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (label, key) in enumerate(MASS_KEYS.items()):
        vals = _np_array([
            s.get('listeners', {}).get('mass', {}).get(key, s.get(key, 0))
            for s in history
        ])
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
    return _fig_to_b64(fig)


def _plot_growth(history):
    import numpy as np
    import matplotlib.pyplot as plt

    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_b64(fig)

    times = _np_array([s.get('global_time', s.get('time', 0)) for s in history])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gr = _np_array([s.get('listeners', {}).get('mass', {}).get('instantaneous_growth_rate',
                          s.get('instantaneous_growth_rate', 0)) for s in history])
    vol = _np_array([s.get('listeners', {}).get('mass', {}).get('volume',
                           s.get('volume', 0)) for s in history])
    pf = _np_array([s.get('listeners', {}).get('mass', {}).get('protein_mass_fraction',
                           s.get('protein_mass_fraction', 0)) for s in history])
    rf = _np_array([s.get('listeners', {}).get('mass', {}).get('rna_mass_fraction',
                          s.get('rna_mass_fraction', 0)) for s in history])

    axes[0].plot(times / 60, gr * 3600, color='#2563eb', lw=1)
    axes[0].set_ylabel('Growth rate (1/h)')
    axes[0].set_title('Instantaneous Growth Rate')

    axes[1].plot(times / 60, vol, color='#16a34a', lw=1.5)
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
    return _fig_to_b64(fig)


def _coord_to_angle(coord):
    """Convert genome coordinate to angle on circular chromosome."""
    frac = coord / MAX_COORD
    import numpy as np
    return np.pi / 2 - frac * np.pi


def _draw_chromosome(ax, cx, cy, R, rnap_coords, fork_coords):
    """Draw one circular chromosome at (cx, cy) with RNAP and forks."""
    import numpy as np
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


def _plot_chromosome_map(snapshot, ax, title=''):
    """Draw chromosomes stacked vertically, each with RNAP and forks."""
    import numpy as np
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


def _plot_chromosome_state(snapshots, title=''):
    """Plot chromosome state: circular maps at key times + timeseries."""
    import numpy as np
    import matplotlib.pyplot as plt

    if not snapshots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No chromosome data', ha='center', va='center')
        return _fig_to_b64(fig)

    times = [s['time'] for s in snapshots]

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

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, n_maps, i + 1)
        _plot_chromosome_map(snapshots[idx], ax)

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
    ax2 = ax.twinx()
    n_chroms = [s['n_chromosomes'] for s in snapshots]
    ax2.step(times, n_chroms, where='post', color='#10b981', lw=2, alpha=0.5, label='Chromosomes')
    ax2.set_ylabel('Chromosomes', color='#10b981', fontsize=9)
    ax2.set_ylim(0, max(n_chroms) + 2)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.tick_params(axis='y', labelcolor='#10b981')
    ax.legend(fontsize=7, loc='lower left')

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
    return _fig_to_b64(fig)


def _plot_single_cell_growth(snapshots, title=''):
    """Plot growth metrics: growth rate, volume, absolute mass, fold change."""
    import numpy as np
    import matplotlib.pyplot as plt

    if not snapshots or len(snapshots) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_b64(fig)

    times = _np_array([s['time'] for s in snapshots]) / 60

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title or 'Growth Metrics', fontsize=13)

    ax = axes[0, 0]
    gr = _np_array([s.get('instantaneous_growth_rate', 0) for s in snapshots])
    ax.plot(times, gr * 3600, color='#2563eb', lw=1)
    ax.set_ylabel('Growth rate (1/h)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Instantaneous Growth Rate')
    ax.grid(True, alpha=0.15)

    ax = axes[0, 1]
    vol = _np_array([s.get('volume', 0) for s in snapshots])
    ax.plot(times, vol, color='#16a34a', lw=1.5)
    ax.set_ylabel('Volume (fL)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Cell Volume')
    ax.grid(True, alpha=0.15)

    ax = axes[1, 0]
    for (label, key), color in zip(MASS_KEYS.items(), COLORS):
        vals = _np_array([s.get(key, 0) for s in snapshots])
        ax.plot(times, vals, color=color, lw=1.5, label=label)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Mass (fg)')
    ax.set_title('Absolute Mass')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    ax = axes[1, 1]
    for (label, key), color in zip(MASS_KEYS.items(), COLORS):
        vals = _np_array([s.get(key, 0) for s in snapshots])
        if len(vals) > 0 and vals[0] > 0:
            ax.plot(times, vals / vals[0], color=color, lw=1.5, label=label)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fold change')
    ax.set_title('Fold Change (normalized to t=0)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_ppgpp_dynamics(snapshots, title=''):
    """Plot ppGpp, amino acid pools, RNA fractions, and NTP pools."""
    import numpy as np
    import matplotlib.pyplot as plt

    if not snapshots or len(snapshots) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No ppGpp data', ha='center', va='center')
        return _fig_to_b64(fig)

    times = _np_array([s['time'] for s in snapshots]) / 60

    N_A = 6.022e23
    ppgpp_counts = _np_array([s.get('ppgpp_count', 0) for s in snapshots], dtype=float)
    volumes = _np_array([s.get('volume', 0) for s in snapshots], dtype=float)
    volumes_L = np.where(volumes > 0, volumes * 1e-15, 1e-15)
    ppgpp_uM = ppgpp_counts / (N_A * volumes_L) * 1e6

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title or 'ppGpp & Metabolic Dynamics', fontsize=13)

    ax = axes[0, 0]
    ax.plot(times, ppgpp_uM, color='#dc2626', lw=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('ppGpp (μM)')
    ax.set_title('ppGpp Concentration')
    ax.grid(True, alpha=0.15)

    ax = axes[0, 1]
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
            counts = _np_array([s.get('aa_counts', {}).get(aa_id, 0) for s in snapshots], dtype=float)
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

    ax = axes[1, 0]
    rRNA = _np_array([s.get('rRna_mass', 0) for s in snapshots])
    tRNA = _np_array([s.get('tRna_mass', 0) for s in snapshots])
    mRNA = _np_array([s.get('mRna_mass', 0) for s in snapshots])
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

    ax = axes[1, 1]
    ntp_colors = {'ATP[c]': '#dc2626', 'GTP[c]': '#2563eb',
                  'CTP[c]': '#16a34a', 'UTP[c]': '#f59e0b'}
    has_ntp = False
    for ntp_name, color in ntp_colors.items():
        counts = _np_array([s.get('ntp_counts', {}).get(ntp_name, 0)
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
    return _fig_to_b64(fig)


def _plot_daughters_mass(daughters_meta, title=''):
    """Plot both daughters' mass fold change side by side (2 subplots)."""
    import matplotlib.pyplot as plt

    d1_snaps = daughters_meta.get('daughter_1_snapshots', [])
    d2_snaps = daughters_meta.get('daughter_2_snapshots', [])

    if not d1_snaps and not d2_snaps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No daughters data', ha='center', va='center')
        return _fig_to_b64(fig)

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
    return _fig_to_b64(fig)


def _np_array(data, **kwargs):
    """Thin wrapper so we don't import numpy at module level."""
    import numpy as np
    return np.array(data, **kwargs)


# ---------------------------------------------------------------------------
# WorkflowVisualization Step
# ---------------------------------------------------------------------------

class WorkflowVisualization(Visualization):
    """Render full cell lifecycle report from one trajectory.

    Accepts ``history`` (list of snapshot dicts from the single-cell sim)
    and ``metadata`` (dict of step-level results from the pipeline). Produces
    a multi-panel HTML with mass trajectories, growth metrics, chromosome
    state, ppGpp dynamics, division results, daughter simulations, and timing.

    This Step contains only rendering logic. All pipeline orchestration
    (parca → cache → composite → run → division → daughters) stays in
    reports/workflow_report.py.
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history":  "list[map[any]]",
            "metadata": "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        history = state.get("history") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v2ecoli workflow"
        body = self._render_body(history, meta, title)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, history: list[dict], meta: dict, title: str) -> str:
        """Build the full workflow report HTML body.

        ``history`` is a list of snapshot dicts (from the single-cell or
        combined trajectory).  ``meta`` is a dict with nested keys for each
        pipeline step — same shape as the ``step_results`` dict passed to
        ``generate_html_report`` in the legacy script.
        """
        # Lazy-import matplotlib (Task 4-5 pattern).
        import matplotlib
        if matplotlib.get_backend() != "Agg":
            matplotlib.use("Agg")

        # Unpack step-level metadata (same keys as legacy step_results)
        biocyc = meta.get('biocyc', {})
        raw = meta.get('raw_data', {})
        parca = meta.get('parca', {})
        parca_stats = parca.get('stats', {})
        model = meta.get('load_model', {})
        long = meta.get('single_cell', {})
        div = meta.get('division', {})
        daughters = meta.get('daughters', {})

        # For the common case where metadata is minimal / absent (unit test),
        # fall back to extracting basics from `history` directly.
        if not long and history:
            # Build a synthetic single_cell meta from the trajectory itself
            dry_masses = [s.get('dry_mass', 0) for s in history]
            long = {
                'duration': history[-1].get('time', 0) if history else 0,
                'wall_time': 0,
                'rate': 0,
                'bulk_changed': 0,
                'final_dry_mass': dry_masses[-1] if dry_masses else 0,
                'division_reached': False,
                'chromosome_snapshots': history,
            }
        if not parca:
            parca = {}
        if not model:
            model = {}
        if not div:
            div = {}

        # ----------------------------------------------------------------
        # Build plots from ``history`` (the single-cell trajectory snapshots)
        # ----------------------------------------------------------------
        plots: dict[str, str] = {}

        chrom_snaps = long.get('chromosome_snapshots', history)

        # Only generate chromosome/growth plots when snapshots have the
        # expected fields (n_chromosomes / fork_coords).  Simple mass-only
        # histories (e.g. unit-test input) fall through gracefully.
        has_chrom_fields = (chrom_snaps and
                            'n_chromosomes' in (chrom_snaps[0] if chrom_snaps else {}))

        if has_chrom_fields:
            dur = long.get('duration', 0)
            try:
                plots['chromosome_long'] = _plot_chromosome_state(
                    chrom_snaps, f'v2 Chromosome State (to t={dur:.0f}s)')
            except Exception:
                pass
            try:
                plots['growth_long'] = _plot_single_cell_growth(
                    chrom_snaps, f'Growth Metrics ({dur/60:.0f} min)')
            except Exception:
                pass
            if any(s.get('ppgpp_count', 0) > 0 for s in chrom_snaps):
                try:
                    plots['ppgpp_dynamics'] = _plot_ppgpp_dynamics(
                        chrom_snaps, f'ppGpp Dynamics ({dur/60:.0f} min)')
                except Exception:
                    pass
        elif chrom_snaps and len(chrom_snaps) >= 2:
            # Minimal trajectory (e.g. only time/mass/dry_mass columns).
            try:
                plots['mass_simple'] = _plot_mass(
                    chrom_snaps, title)
            except Exception:
                pass

        if not daughters.get('skipped') and (
                daughters.get('daughter_1_snapshots') or
                daughters.get('daughter_2_snapshots')):
            try:
                plots['daughters_mass'] = _plot_daughters_mass(
                    daughters, 'Daughters — Daughter Mass Fold Change')
            except Exception:
                pass

        # ----------------------------------------------------------------
        # HTML assembly (mirrors generate_html_report from the legacy script)
        # ----------------------------------------------------------------

        def _badge(m):
            ts = m.get('timestamp', '')
            return f'<span style="background:#dbeafe;color:#1e40af;padding:1px 6px;border-radius:3px;font-size:0.8em;font-weight:500;">cached {escape(ts)}</span>' if ts else ''

        long_dur = long.get('duration', 0)
        long_wall = long.get('wall_time', 0)
        long_rate = long.get('rate', 0)
        build_time = model.get('build_time', 0)
        d1_wall = daughters.get('daughter_1', {}).get('wall_time', 0)
        d2_wall = daughters.get('daughter_2', {}).get('wall_time', 0)
        daughters_wall = d1_wall + d2_wall
        daughters_dur = daughters.get('duration', 0)

        # Division unique molecule rows
        div_unique_rows = ''
        for name, info in div.get('unique_conservation', {}).items():
            ok = 'color:#16a34a' if info['conserved'] else 'color:#dc2626'
            div_unique_rows += (
                f'<tr><td>{escape(name)}</td>'
                f'<td>{info["mother"]}</td>'
                f'<td>{info["d1"]}</td>'
                f'<td>{info["d2"]}</td>'
                f'<td style="{ok}">{"Yes" if info["conserved"] else "No"}</td></tr>'
            )

        # parca provenance label
        src_map = {
            'vecoli_pickle': f'Loaded from vEcoli ParCa output (<code>{escape(str(parca.get("sim_data_path", "")))}</code>)',
            'workflow_pickle': f'Loaded from a prior workflow run (<code>{escape(str(parca.get("sim_data_path", "")))}</code>)',
            'cache': 'Loaded from existing cache',
            'parca_fixture': f'Hydrated from the shipped ParCa fixture in {parca.get("parca_time", 0):.1f}s',
            'parca_composite': f'Computed by v2ecoli.processes.parca composite in {parca.get("parca_time", 0):.1f}s',
        }
        parca_src_label = src_map.get(parca.get('simdata_source', ''),
                                      str(parca.get('simdata_source', 'unknown')))

        html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  max-width:1400px;margin:0 auto;padding:20px;background:#f8fafc;color:#1e293b;">

<style>
  .metrics {{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:10px 0;}}
  .metric {{background:white;border-radius:8px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,0.08);}}
  .metric .label {{font-size:0.7em;color:#64748b;text-transform:uppercase;letter-spacing:.05em;}}
  .metric .value {{font-size:1.3em;font-weight:700;margin-top:2px;}}
  .plot {{background:white;border-radius:8px;padding:12px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.08);text-align:center;}}
  .plot img {{max-width:100%;}}
  .section {{background:white;border-radius:8px;padding:15px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.08);}}
  h2 {{font-size:1.3em;margin:25px 0 10px;color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:6px;}}
  h3 {{font-size:1.05em;margin:15px 0 8px;color:#475569;}}
  details {{margin:5px 0;}} details>summary {{cursor:pointer;font-weight:600;color:#475569;padding:5px 0;}}
  table {{border-collapse:collapse;width:100%;font-size:0.82em;}}
  th,td {{border:1px solid #e2e8f0;padding:5px 8px;text-align:left;}}
  th {{background:#f1f5f9;font-weight:600;}}
  footer {{margin-top:30px;padding:15px 0;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.75em;text-align:center;}}
</style>

<h1 style="font-size:1.8em;margin:15px 0;color:#0f172a;">{escape(title)}</h1>
<p style="color:#64748b;font-size:0.9em;">{time.strftime('%Y-%m-%d %H:%M')} &middot;
Simulation pipeline &middot; process-bigraph <code>Composite.run()</code></p>

<div class="section">
  <p><strong>v2ecoli</strong> is a whole-cell <em>E. coli</em> model running natively on
  <a href="https://github.com/vivarium-collective/process-bigraph">process-bigraph</a>.
  This report covers the <strong>simulation</strong> phase: building the online
  model from a pre-fitted <code>sim_data</code>, running a single cell to
  division, and continuing into daughter simulations.</p>
</div>
"""

        if parca.get('simdata_source') or parca.get('cache_dir'):
            html += f"""
<div class="section" style="background:#eff6ff;border-left:4px solid #2563eb;padding:0.75em 1em;margin:10px 0;">
  <strong>sim_data source:</strong> {parca_src_label}
  &nbsp;&middot;&nbsp;
  Cache: <code>{escape(str(parca.get('cache_dir', '')))}</code>
  ({parca_stats.get('n_process_configs', '?')} process configs,
   {parca_stats.get('n_bulk_molecules', 0):,} bulk molecules)
</div>
"""

        html += f"""
<nav style="background:white;border-radius:8px;padding:12px 20px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,0.08);">
  <strong style="font-size:0.9em;color:#475569;">Simulation Steps</strong>
  <ol style="margin:6px 0 0 0;padding-left:20px;font-size:0.88em;columns:2;column-gap:30px;">
    <li><a href="#sec-model">Load Model</a></li>
    <li><a href="#sec-long">Single Cell Simulation ({long_dur/60:.0f} min)</a></li>
    <li><a href="#sec-division">Division</a></li>
    <li><a href="#sec-daughters">Daughter Simulations</a></li>
    <li><a href="#sec-timing">Timing Summary</a></li>
  </ol>
</nav>

<!-- ===== Load Model ===== -->
<h2 id="sec-model">1. Load Model {_badge(model)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">Build Time</div><div class="value" style="color:#2563eb;">{model.get('build_time', 0):.2f}s</div></div>
  <div class="metric"><div class="label">Steps</div><div class="value">{model.get('n_steps', 0)}</div></div>
  <div class="metric"><div class="label">Processes</div><div class="value">{model.get('n_processes', 0)}</div></div>
  <div class="metric"><div class="label">Bulk Molecules</div><div class="value">{model.get('n_bulk', 0):,}</div></div>
  <div class="metric"><div class="label">Unique Types</div><div class="value">{model.get('n_unique_types', 0)}</div></div>
  <div class="metric"><div class="label">Initial Dry Mass</div><div class="value">{model.get('initial_dry_mass', 0):.1f} fg</div></div>
</div>

<!-- ===== Single Cell Simulation ===== -->
<h2 id="sec-long">2. Single Cell Simulation ({long_dur/60:.0f} min) {_badge(long)}</h2>
<div class="metrics">
  <div class="metric"><div class="label">Sim Duration</div><div class="value">{long_dur:.0f}s</div></div>
  <div class="metric"><div class="label">Wall Time</div><div class="value" style="color:#2563eb;">{long_wall:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value" style="color:#16a34a;">{long_rate:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value" style="color:#7c3aed;">{long.get('bulk_changed', 0)}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{long.get('final_dry_mass', 0):.1f} fg</div></div>
  <div class="metric"><div class="label">Division</div><div class="value" style="color:{'#16a34a' if long.get('division_reached') else '#7c3aed'};">{'Reached' if long.get('division_reached') else 'Not reached'}</div></div>
</div>
"""

        if plots.get('chromosome_long'):
            html += f'<div class="plot"><img src="data:image/png;base64,{plots["chromosome_long"]}" alt="Chromosome State"></div>\n'
        if plots.get('growth_long'):
            html += f'<div class="plot"><img src="data:image/png;base64,{plots["growth_long"]}" alt="Growth Metrics"></div>\n'
        if plots.get('ppgpp_dynamics'):
            html += f'<div class="plot"><img src="data:image/png;base64,{plots["ppgpp_dynamics"]}" alt="ppGpp Dynamics"></div>\n'
        if plots.get('mass_simple'):
            html += f'<div class="plot"><img src="data:image/png;base64,{plots["mass_simple"]}" alt="Mass Trajectory"></div>\n'

        v2_wall = long.get('wall_time', 0)
        v2_rate = long_dur / v2_wall if v2_wall > 0 else 0
        v2_final_dry = long.get('final_dry_mass', 0)

        html += f"""
<h3>Single Cell Summary</h3>
<table style="margin:1em auto;border-collapse:collapse;font-size:0.9em;">
<tr style="background:#f3f4f6;"><th style="padding:6px 16px;text-align:left;">Metric</th>
    <th style="padding:6px 16px;">Value</th></tr>
<tr><td style="padding:4px 16px;">Sim duration</td>
    <td style="padding:4px 16px;text-align:center;">{long_dur:.0f}s</td></tr>
<tr><td style="padding:4px 16px;">Wall time</td>
    <td style="padding:4px 16px;text-align:center;">{v2_wall:.0f}s</td></tr>
<tr><td style="padding:4px 16px;">Speed (sim/wall)</td>
    <td style="padding:4px 16px;text-align:center;font-weight:bold;">{v2_rate:.1f}x</td></tr>
<tr><td style="padding:4px 16px;">Final dry mass</td>
    <td style="padding:4px 16px;text-align:center;">{v2_final_dry:.1f} fg</td></tr>
<tr><td style="padding:4px 16px;">Division reached</td>
    <td style="padding:4px 16px;text-align:center;">{'Yes' if long.get('division_reached') else 'No'}</td></tr>
<tr><td style="padding:4px 16px;">Snapshots</td>
    <td style="padding:4px 16px;text-align:center;">{len(long.get('chromosome_snapshots', []))}</td></tr>
</table>

<!-- ===== Division ===== -->
<h2 id="sec-division">3. Division {_badge(div)}</h2>

<div class="section">
  <h3>How Division Works</h3>
  <p>The Division step uses process-bigraph's native <code>_add</code>/<code>_remove</code> structural
  updates. When division is triggered (dry mass &ge; threshold with &ge; 2 chromosomes):</p>
  <ol style="margin:8px 0 8px 20px;font-size:0.9em;">
    <li><strong>State splitting</strong> &mdash; <code>divide_cell()</code> partitions the mother cell's state:
      <ul>
        <li>Bulk molecules: binomial distribution (p=0.5) on each molecule's count</li>
        <li>Chromosomes: alternating assignment (even&rarr;D1, odd&rarr;D2) with descendant domain tracking</li>
        <li>Chromosome-attached molecules: follow their domain</li>
        <li>RNAs: full transcripts binomial, partial transcripts follow RNAP domain</li>
        <li>Ribosomes: follow their mRNA</li>
      </ul>
    </li>
    <li><strong>Daughter cell construction</strong> &mdash; builds complete cell states with fresh
    process instances from the divided initial state + cached configs</li>
  </ol>
</div>
"""
        if div:
            prediv_descr = (
                f'pre-division state (t={int(div.get("prediv_time", 0))}s, '
                f'{div.get("n_chromosomes", 0)} chromosomes, '
                f'dry mass {round(div.get("dry_mass", 0))} fg)'
                if div.get('prediv_time', 0) > 0
                else 'initial state (t=0)'
            )
            html += f"""
<h3>Division Test Results</h3>
<div class="section">
  <p>Tests run on {prediv_descr}.</p>
</div>
<div class="metrics">
  <div class="metric"><div class="label">Bulk Conserved</div><div class="value" style="color:{'#16a34a' if div.get('bulk_conserved') else '#dc2626'};">{'Yes' if div.get('bulk_conserved') else 'No'}</div></div>
  <div class="metric"><div class="label">Mother Bulk</div><div class="value">{div.get('mother_bulk_count', 0):,}</div></div>
  <div class="metric"><div class="label">D1 Bulk</div><div class="value">{div.get('d1_bulk_count', 0):,}</div></div>
  <div class="metric"><div class="label">D2 Bulk</div><div class="value">{div.get('d2_bulk_count', 0):,}</div></div>
  <div class="metric"><div class="label">State Split</div><div class="value" style="color:#2563eb;">{div.get('split_time', 0)*1000:.0f} ms</div></div>
  <div class="metric"><div class="label">Daughter Build</div><div class="value" style="color:#2563eb;">{div.get('daughter_build_time', 0):.1f}s</div></div>
  <div class="metric"><div class="label">Daughter Viable</div><div class="value" style="color:{'#16a34a' if div.get('daughter_viable') else '#dc2626'};">{'Yes' if div.get('daughter_viable') else 'No'}</div></div>
</div>
"""
            if div_unique_rows:
                html += f"""
<details open>
<summary>Unique Molecule Conservation</summary>
<div class="section" style="overflow-x:auto;">
  <table>
    <thead><tr><th>Molecule</th><th>Mother (active)</th><th>Daughter 1</th><th>Daughter 2</th><th>Conserved</th></tr></thead>
    <tbody>{div_unique_rows}</tbody>
  </table>
</div>
</details>
"""

        # ===== Daughter Simulations =====
        html += f'\n<h2 id="sec-daughters">4. Daughter Simulations {_badge(daughters)}</h2>\n'

        if daughters.get('skipped'):
            html += f'<div class="section"><p>Skipped: {escape(str(daughters.get("reason", "unknown")))}</p></div>'
        elif daughters:
            d1 = daughters.get('daughter_1', {})
            d2 = daughters.get('daughter_2', {})
            html += f"""
<div class="section">
  <p>Two daughter cells from the pre-division state, each run for {daughters.get('duration', 0)}s
  (approximately half a generation).</p>
</div>
<h3>Daughter 1</h3>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value" style="color:#2563eb;">{d1.get('wall_time', 0):.0f}s</div></div>
  <div class="metric"><div class="label">Initial Dry Mass</div><div class="value">{d1.get('initial_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Final Dry Mass</div><div class="value">{d1.get('final_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Fold Change</div><div class="value" style="color:#16a34a;">{d1.get('fold_change', 0):.2f}x</div></div>
</div>
<h3>Daughter 2</h3>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value" style="color:#2563eb;">{d2.get('wall_time', 0):.0f}s</div></div>
  <div class="metric"><div class="label">Initial Dry Mass</div><div class="value">{d2.get('initial_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Final Dry Mass</div><div class="value">{d2.get('final_dry_mass', 0):.0f} fg</div></div>
  <div class="metric"><div class="label">Fold Change</div><div class="value" style="color:#16a34a;">{d2.get('fold_change', 0):.2f}x</div></div>
</div>
"""

        if plots.get('daughters_mass'):
            html += f'<div class="plot"><img src="data:image/png;base64,{plots["daughters_mass"]}" alt="Daughters Mass"></div>\n'

        # ===== Timing Summary =====
        html += f"""
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
</div>"""

        return html
