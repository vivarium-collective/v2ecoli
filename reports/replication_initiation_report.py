"""Replication-initiation workflow report.

Layout: a left side-menu navigates an overview, the nine integration phases
(Phase 0 — Region-tag DnaA boxes through Phase 8 — dnaA promoter
autoregulation), the molecular-reference panels sourced from
``docs/references/replication_initiation.md`` (and the underlying PDF),
the trajectory plots under the new ``replication_initiation`` architecture,
and the references list.

Per-phase status is **auto-detected from the codebase**: each phase
declares a check that inspects schema files, molecule_ids, process modules,
sim_data attributes, etc. As phases land, the sidebar pills flip from
``pending`` to ``in progress`` to ``done`` and the gap items in each
section get crossed off — no manual report-side bookkeeping.

Usage:
    python reports/replication_initiation_report.py
        # writes out/reports/replication_initiation_report.html
"""

from __future__ import annotations

import argparse
import ast
import base64
import functools
import html as html_lib
import io
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bump default font sizes — the per-phase before/after grid renders
# each plot inside a narrow column, so the matplotlib defaults (~10pt
# everywhere) end up unreadably small once the SVG / PNG is scaled to
# fit. These rcParams apply to every plot the report renders.
plt.rcParams.update({
    'font.size':       13,
    'axes.titlesize':  15,
    'axes.labelsize':  13,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0,
    'axes.titlepad':   10,
    'axes.labelpad':   6,
})
import numpy as np

from v2ecoli.data.replication_initiation import (
    CITATIONS,
    DARS1, DARS2, DATA, DNAA_BOX_CONSENSUS, DNAA_BOX_HIGHEST_AFFINITY,
    DNAA_NUCLEOTIDE_EQUILIBRIUM, DNAA_POOL_DRIVERS,
    DNAA_PROMOTER, ORIC, RIDA, SEQA,
    GENOME_LENGTH_BP, ORIC_ABS_CENTER_BP, TERC_ABS_CENTER_BP,
    REGION_BOUNDARIES_ABS,
    PER_REGION_PDF_COUNT, PER_REGION_STRICT_CONSENSUS_COUNT,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = 'out/cache'
WORKFLOW_DIR = 'out/workflow'
OUT_DIR = 'out/reports'
DEFAULT_DURATION = 3600.0  # 60 min — at least one full cell cycle
                            # so the mass-threshold pre_gate config has
                            # time to actually cross the critical mass
SNAPSHOT_INTERVAL = 50.0


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _img(b64, alt=''):
    if not b64:
        return ''
    return f'<img alt="{html_lib.escape(alt)}" src="data:image/png;base64,{b64}"/>'


def _read_file(rel_path):
    p = os.path.join(REPO_ROOT, rel_path)
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def _file_contains(rel_path, needle):
    text = _read_file(rel_path)
    return text is not None and needle in text


def _file_exists(rel_path):
    return os.path.isfile(os.path.join(REPO_ROOT, rel_path))


# ---------------------------------------------------------------------------
# Trajectory acquisition
# ---------------------------------------------------------------------------

def _load_cached_trajectory():
    path = os.path.join(WORKFLOW_DIR, 'single_cell.dill')
    if not os.path.exists(path):
        return None
    try:
        import dill
        with open(path, 'rb') as f:
            data = dill.load(f)
    except Exception as exc:
        print(f'  cached trajectory load failed: {exc}')
        return None
    snaps = data.get('snapshots') if isinstance(data, dict) else None
    return snaps or None


def _bool_entry_count(structured):
    if structured is None or not hasattr(structured, 'dtype'):
        return 0
    if '_entryState' not in structured.dtype.names:
        return 0
    return int(structured['_entryState'].view(np.bool_).sum())


def _extract_replication_signals(history):
    snaps = []
    for h in history:
        t = float(h.get('global_time', 0))
        if int(t) % int(SNAPSHOT_INTERVAL) != 0 and t > 1:
            continue
        mass = h.get('listeners', {}).get('mass', {}) if isinstance(h.get('listeners'), dict) else {}
        rep_listener = h.get('listeners', {}).get('replication_data', {}) if isinstance(h.get('listeners'), dict) else {}

        # The emitter history stores unique molecules under their schema
        # names; ``oriC`` is not one of them (replication_data emits the
        # count via ``number_of_oric`` — prefer that over a structured-
        # array fallback that would silently return zero).
        n_oric_listener = rep_listener.get('number_of_oric')
        if n_oric_listener is not None:
            n_oric = int(n_oric_listener)
        else:
            n_oric = _bool_entry_count(h.get('oriC'))
        n_chrom = _bool_entry_count(h.get('full_chromosome'))
        n_rep = _bool_entry_count(h.get('active_replisome'))

        rep = h.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active = rep[rep['_entryState'].view(np.bool_)]
            if len(active) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active['coordinates'].tolist()

        dnaA_total = 0
        dnaA_bound = 0
        boxes = h.get('DnaA_box')
        if boxes is not None and hasattr(boxes, 'dtype') and '_entryState' in boxes.dtype.names:
            entries = boxes[boxes['_entryState'].view(np.bool_)]
            dnaA_total = len(entries)
            if 'DnaA_bound' in boxes.dtype.names:
                dnaA_bound = int(entries['DnaA_bound'].sum())
        free_listener = rep_listener.get('free_DnaA_boxes')
        total_listener = rep_listener.get('total_DnaA_boxes')
        if total_listener is not None:
            dnaA_total = int(total_listener)
        if free_listener is not None and total_listener is not None:
            dnaA_bound = int(total_listener) - int(free_listener)

        rida_listener = h.get('listeners', {}).get('rida', {}) if isinstance(h.get('listeners'), dict) else {}
        dars_listener = h.get('listeners', {}).get('dars', {}) if isinstance(h.get('listeners'), dict) else {}
        ddah_listener = h.get('listeners', {}).get('ddah', {}) if isinstance(h.get('listeners'), dict) else {}
        binding_listener = h.get('listeners', {}).get('dnaA_binding', {}) if isinstance(h.get('listeners'), dict) else {}

        snaps.append({
            'time': t,
            'n_oriC': n_oric,
            'n_chromosomes': n_chrom,
            'n_replisomes': n_rep,
            'fork_coords': fork_coords,
            'dnaA_box_total': dnaA_total,
            'dnaA_box_bound': dnaA_bound,
            'dnaA_apo_count': rep_listener.get('dnaA_apo_count'),
            'dnaA_atp_count': rep_listener.get('dnaA_atp_count'),
            'dnaA_adp_count': rep_listener.get('dnaA_adp_count'),
            'rida_flux_atp_to_adp': rida_listener.get('flux_atp_to_adp'),
            'rida_active_replisomes': rida_listener.get('active_replisomes'),
            'dars_flux_adp_to_apo': dars_listener.get('flux_adp_to_apo'),
            'ddah_flux_atp_to_adp': ddah_listener.get('flux_atp_to_adp'),
            'binding_total_bound': binding_listener.get('total_bound'),
            'binding_total_active': binding_listener.get('total_active'),
            'binding_fraction_bound': binding_listener.get('fraction_bound'),
            'binding_bound_oric': binding_listener.get('bound_oric'),
            'binding_bound_dnaA_promoter': binding_listener.get('bound_dnaA_promoter'),
            'binding_bound_datA': binding_listener.get('bound_datA'),
            'binding_bound_DARS1': binding_listener.get('bound_DARS1'),
            'binding_bound_DARS2': binding_listener.get('bound_DARS2'),
            'binding_bound_other': binding_listener.get('bound_other'),
            'critical_mass_per_oriC': rep_listener.get('critical_mass_per_oriC'),
            'critical_initiation_mass': rep_listener.get('critical_initiation_mass'),
            'dry_mass': float(mass.get('dry_mass', 0)),
            'cell_mass': float(mass.get('cell_mass', 0)),
            'dna_mass': float(mass.get('dna_mass', 0)),
        })
    return snaps


def _run_sim(duration, make_composite_fn, label):
    if not os.path.isdir(CACHE_DIR):
        print(f'  cache dir {CACHE_DIR!r} not present — skipping {label} sim')
        return []
    print(f'  building {label} composite from {CACHE_DIR}')
    composite = make_composite_fn(cache_dir=CACHE_DIR, seed=0)
    cell = composite.state['agents']['0']
    em_edge = cell.get('emitter', {})
    emitter = em_edge.get('instance') if isinstance(em_edge, dict) else None
    t0 = time.time()
    elapsed = 0.0
    while elapsed < duration:
        chunk = min(SNAPSHOT_INTERVAL, duration - elapsed)
        try:
            composite.run(chunk)
        except Exception as exc:
            print(f'  {label} sim aborted at t={elapsed}s: '
                  f'{type(exc).__name__}: {exc}')
            break
        elapsed += chunk
    print(f'  ran {elapsed:.0f}s {label} sim in {time.time() - t0:.1f}s wall time')
    history = emitter.history if emitter is not None else []
    return _extract_replication_signals(history)


def _run_baseline_sim(duration):
    """Run the unmodified `baseline` architecture — no RIDA, no DARS,
    no equilibrium override."""
    from v2ecoli.composite import make_composite
    return _run_sim(duration, make_composite, label='baseline')


def _run_rida_only_sim(duration):
    """Phase 5's cumulative state: RIDA enabled, no DARS, no binding."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return _run_sim(
        duration,
        lambda **kw: make_replication_initiation_composite(
            enable_rida=True, enable_dars=False,
            enable_dnaA_box_binding=False, **kw),
        label='rida_only')


def _run_rida_dars_sim(duration):
    """Phase 7's cumulative state: RIDA + DARS, no binding yet."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return _run_sim(
        duration,
        lambda **kw: make_replication_initiation_composite(
            enable_rida=True, enable_dars=True,
            enable_dnaA_box_binding=False, **kw),
        label='rida_dars')


def _run_pre_gate_sim(duration):
    """Cumulative state right before Phase 3: RIDA + DARS + box
    binding, but the baseline mass-threshold ChromosomeReplication
    is still in place."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return _run_sim(
        duration,
        lambda **kw: make_replication_initiation_composite(
            enable_rida=True, enable_dars=True,
            enable_dnaA_box_binding=True,
            enable_dnaA_gated_initiation=False, **kw),
        label='pre_gate')


def _run_gated_no_seqA_sim(duration):
    """Cumulative state right before Phase 4: DnaA-gated initiation
    is in place but no SeqA sequestration. Lets the report compare
    the gate's behavior with and without the refractory window."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return _run_sim(
        duration,
        lambda **kw: make_replication_initiation_composite(
            enable_rida=True, enable_dars=True,
            enable_dnaA_box_binding=True,
            enable_dnaA_gated_initiation=True,
            enable_seqA_sequestration=False, **kw),
        label='gated_no_seqA')


def _run_pre_ddah_sim(duration):
    """Cumulative state right before Phase 6: everything except DDAH."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return _run_sim(
        duration,
        lambda **kw: make_replication_initiation_composite(
            enable_rida=True, enable_dars=True,
            enable_dnaA_box_binding=True,
            enable_dnaA_gated_initiation=True,
            enable_seqA_sequestration=True,
            enable_ddah=False, **kw),
        label='pre_ddah')


def _run_full_sim(duration):
    """Full replication_initiation architecture: RIDA + DARS + box
    binding + DnaA-gated initiation + SeqA sequestration + DDAH."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return _run_sim(
        duration, make_replication_initiation_composite, label='full')


# ---------------------------------------------------------------------------
# Trajectory disk cache: pickle snapshot lists keyed by config + duration so
# successive report regenerations can skip the sim runs.
# ---------------------------------------------------------------------------

def _trajectory_cache_path(config_label, duration):
    return os.path.join(
        OUT_DIR, f'_traj_{config_label}_d{int(duration)}.pkl')


def _load_or_run(config_label, duration, runner_fn, force=False):
    import pickle
    path = _trajectory_cache_path(config_label, duration)
    if not force and os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                snaps = pickle.load(f)
            print(f'  reused cached {config_label} trajectory '
                  f'({len(snaps)} snapshots) from {path}')
            return snaps
        except Exception as exc:
            print(f'  cache read failed for {path}: {exc} — re-running')
    snaps = runner_fn(duration)
    if snaps:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'wb') as f:
                pickle.dump(snaps, f)
        except Exception as exc:
            print(f'  cache write failed for {path}: {exc}')
    return snaps


def _run_replication_initiation_sim(duration):
    if not os.path.isdir(CACHE_DIR):
        print(f'  cache dir {CACHE_DIR!r} not present — skipping sim')
        return []
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    print(f'  building replication_initiation composite from {CACHE_DIR}')
    composite = make_replication_initiation_composite(cache_dir=CACHE_DIR, seed=0)
    cell = composite.state['agents']['0']
    em_edge = cell.get('emitter', {})
    emitter = em_edge.get('instance') if isinstance(em_edge, dict) else None
    t0 = time.time()
    elapsed = 0.0
    while elapsed < duration:
        chunk = min(SNAPSHOT_INTERVAL, duration - elapsed)
        try:
            composite.run(chunk)
        except Exception as exc:
            print(f'  sim aborted at t={elapsed}s: {type(exc).__name__}: {exc}')
            break
        elapsed += chunk
    print(f'  ran {elapsed:.0f}s sim in {time.time() - t0:.1f}s wall time')
    history = emitter.history if emitter is not None else []
    return _extract_replication_signals(history)


# ---------------------------------------------------------------------------
# Plot helpers (trajectory)
# ---------------------------------------------------------------------------

def plot_initiation_signals(snaps):
    if not snaps:
        return ''
    times = np.array([s['time'] / 60 for s in snaps])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Replication-initiation signals (current model)', fontsize=13)

    ax = axes[0, 0]
    ax.step(times, [s['n_oriC'] for s in snaps], where='post',
            color='#10b981', lw=1.8, label='oriC')
    ax.step(times, [s['n_chromosomes'] for s in snaps], where='post',
            color='#7c3aed', lw=1.4, ls='--', label='chromosomes')
    ax.set_ylabel('Count'); ax.set_xlabel('Time (min)')
    ax.set_title('oriC and chromosome counts')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[0, 1]
    ax.step(times, [s['n_replisomes'] for s in snaps], where='post',
            color='#f59e0b', lw=1.8)
    ax.set_ylabel('Replisome count'); ax.set_xlabel('Time (min)')
    ax.set_title('Active replisomes (forks / 2)')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[1, 0]
    bound = np.array([s['dnaA_box_bound'] for s in snaps])
    total = np.array([s['dnaA_box_total'] for s in snaps])
    free = total - bound
    ax.plot(times, total, color='#1e293b', lw=1.4, label='total DnaA boxes')
    ax.plot(times, bound, color='#dc2626', lw=1.4, label='bound')
    ax.plot(times, free, color='#2563eb', lw=1.4, label='free')
    ax.set_ylabel('Count'); ax.set_xlabel('Time (min)')
    ax.set_title('DnaA-box occupancy on the chromosome')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    cell_mass = np.array([s['cell_mass'] for s in snaps])
    dry_mass = np.array([s['dry_mass'] for s in snaps])
    n_oric = np.array([max(1, s['n_oriC']) for s in snaps])
    ax.plot(times, cell_mass / n_oric, color='#0891b2', lw=1.5,
            label='cell mass / oriC')
    ax.plot(times, dry_mass, color='#f59e0b', lw=1.0, alpha=0.6, ls='--',
            label='dry mass')
    ax.set_ylabel('Mass (fg)'); ax.set_xlabel('Time (min)')
    ax.set_title('Mass and mass-per-oriC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_fork_positions(snaps):
    if not snaps:
        return ''
    fig, ax = plt.subplots(figsize=(11, 4.0))
    fig.suptitle('Replication-fork positions over time', fontsize=12)
    has_data = False
    for s in snaps:
        for c in s.get('fork_coords') or []:
            has_data = True
            ax.scatter(s['time'] / 60, c, c='#f59e0b', s=12,
                       alpha=0.7, edgecolors='black', linewidths=0.3)
    if not has_data:
        ax.text(0.5, 0.5, 'No active replisomes during this sim window.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#475569')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fork coordinate (bp; oriC=0)')
    ax.axhline(0, color='#10b981', ls='--', lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Reusable circular-chromosome diagram (used by Phase 0, 3, 5, 7)
# ---------------------------------------------------------------------------

# Genome / origin / terminus constants for converting absolute bp to angle.
_GENOME_BP = GENOME_LENGTH_BP  # 4_641_652
_ORIC_ABS = ORIC_ABS_CENTER_BP
_TERC_ABS = TERC_ABS_CENTER_BP

# Color per regulatory region (matches Phase 2 plot palette).
_REGION_COLORS = {
    'oriC':           '#10b981',  # green
    'dnaA_promoter':  '#0891b2',  # cyan
    'datA':           '#7c3aed',  # purple
    'DARS1':          '#f59e0b',  # amber
    'DARS2':          '#dc2626',  # red
}


def _abs_to_angle(abs_bp: int) -> float:
    """Convert an absolute MG1655 coordinate to an angle on a circular
    chromosome where oriC sits at 12 o'clock (90 deg), terC at 6 o'clock
    (-90 deg / 270 deg)."""
    rel = (abs_bp - _ORIC_ABS) % _GENOME_BP
    # Half-genome to angle: rel=0 at oriC, rel=±half_genome at terC.
    angle_from_oric = 2 * np.pi * rel / _GENOME_BP
    return np.pi / 2 - angle_from_oric


def _rel_to_angle(rel_bp: int) -> float:
    """Convert a coordinate relative to oriC (DnaA-box and replisome
    coordinates are stored this way) to an angle on the circle."""
    abs_bp = (_ORIC_ABS + rel_bp) % _GENOME_BP
    return _abs_to_angle(abs_bp)


def _draw_chromosome_diagram(ax, snap, *,
                              show_regions: bool = True,
                              show_replisomes: bool = True,
                              R: float = 1.0,
                              cx: float = 0.0,
                              cy: float = 0.0):
    """Draw one chromosome circle with named regions, oriC, ter, and
    optionally active replisomes (forks) and a region legend."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + R * np.cos(theta), cy + R * np.sin(theta),
            color='#cbd5e1', lw=4, zorder=1)
    # oriC — green dot at 12 o'clock
    ax.plot(cx, cy + R, 'o', color='#10b981', ms=12, zorder=5,
            markeredgecolor='#065f46', markeredgewidth=1.0)
    # ter — red square at 6 o'clock
    ax.plot(cx, cy - R, 's', color='#ef4444', ms=10, zorder=5,
            markeredgecolor='#7f1d1d', markeredgewidth=1.0)

    if show_regions:
        # Draw arcs for each named region
        for region, (lo, hi) in REGION_BOUNDARIES_ABS.items():
            color = _REGION_COLORS.get(region, '#64748b')
            a_lo = _abs_to_angle(lo)
            a_hi = _abs_to_angle(hi)
            # Region arcs are tiny (~hundreds of bp on a ~4.6 Mb genome).
            # Render as a thick chord plus a labelled marker so they
            # show up at this scale.
            mid_angle = (a_lo + a_hi) / 2
            mx = cx + R * np.cos(mid_angle)
            my = cy + R * np.sin(mid_angle)
            ax.plot(mx, my, 'o', color=color, ms=11, zorder=4,
                    markeredgecolor='black', markeredgewidth=0.8)
            # Label outside the circle
            label_r = R * 1.18
            lx = cx + label_r * np.cos(mid_angle)
            ly = cy + label_r * np.sin(mid_angle)
            ha = 'left' if lx > cx + 0.05 else ('right' if lx < cx - 0.05 else 'center')
            va = 'bottom' if ly > cy else 'top'
            ax.text(lx, ly, region, ha=ha, va=va, fontsize=11,
                    color=color, fontweight='bold')

    if show_replisomes:
        fork_coords = snap.get('fork_coords') or []
        for c in fork_coords:
            ang = _rel_to_angle(int(c))
            fx = cx + (R + 0.07) * np.cos(ang)
            fy = cy + (R + 0.07) * np.sin(ang)
            ax.plot(fx, fy, '^', color='#f59e0b', ms=14, zorder=6,
                    markeredgecolor='black', markeredgewidth=0.7)


def _chromosome_timeline_plot(snaps, indices=None, title='',
                               annotate_events=True):
    """Draw a row of chromosome diagrams at selected snapshot indices.
    Bottom panel shows oriC + replisome counts over time so the user
    can read the timeline alongside the spatial state."""
    if not snaps:
        return _no_data_msg()
    if indices is None:
        n = len(snaps)
        if n >= 5:
            indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        elif n >= 3:
            indices = [0, n // 2, n - 1]
        else:
            indices = list(range(n))
    indices = sorted(set(indices))

    n_maps = len(indices)
    fig = plt.figure(figsize=(max(11, n_maps * 3.0), 7.5))
    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, n_maps, i + 1)
        snap = snaps[idx]
        _draw_chromosome_diagram(ax, snap)
        n_oric = snap.get('n_oriC') or 0
        n_rep = snap.get('n_replisomes') or 0
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(
            f't = {snap["time"] / 60:.1f} min\n'
            f'oriC={n_oric}  replisomes={n_rep}',
            fontsize=11)

    # Bottom panel: oriC + replisome counts over time, full width
    ax = fig.add_subplot(2, 1, 2)
    times = np.array([s['time'] / 60 for s in snaps])
    n_oric = np.array([s.get('n_oriC') or 0 for s in snaps])
    n_rep = np.array([s.get('n_replisomes') or 0 for s in snaps])
    ax.step(times, n_oric, where='post', color='#10b981', lw=2.4,
            label='oriC count')
    ax.step(times, n_rep, where='post', color='#f59e0b', lw=2.0,
            label='active replisomes')
    if annotate_events:
        for i in range(1, len(n_oric)):
            if n_oric[i] > n_oric[i - 1]:
                ax.axvline(times[i], color='#dc2626', ls='--',
                           lw=0.9, alpha=0.5)
    # Mark which snapshots are shown above
    for idx in indices:
        ax.axvline(times[idx], color='#94a3b8', ls=':', lw=1.0,
                   alpha=0.55, zorder=0)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Count')
    ax.set_title('oriC count + active replisomes over time '
                 '(grey dotted lines mark the snapshots above)')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'chromosome timeline')


def _chromosome_diagram_static():
    """Single labeled chromosome — used in Phase 0 to show region
    placement without any trajectory data. Doesn't need snaps."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    snap = {'fork_coords': [], 'n_oriC': 1, 'n_replisomes': 0}
    _draw_chromosome_diagram(ax, snap, show_replisomes=False)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(
        'Regulatory regions on the E. coli MG1655 chromosome\n'
        '(circle = full genome, ~4.64 Mb)',
        fontsize=12)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'chromosome region map')


# ---------------------------------------------------------------------------
# Phase status detection
# ---------------------------------------------------------------------------

def _check_phase0():
    """Region classifier — coordinate-based, no schema change."""
    text = _read_file('v2ecoli/data/replication_initiation/molecular_reference.py')
    has_classifier = bool(text and 'def region_for_coord' in text)
    has_boundaries = bool(text and 'REGION_BOUNDARIES' in text)
    if has_classifier and has_boundaries:
        return 'done', 'region_for_coord and REGION_BOUNDARIES in molecular_reference'
    if has_boundaries or has_classifier:
        return 'in_progress', (
            'partial: ' + ('boundaries' if has_boundaries else 'classifier')
            + ' present')
    if _file_exists('tests/test_dnaA_box_regions.py'):
        return 'in_progress', 'phase test file exists; classifier not yet added'
    return 'pending', 'no region_for_coord / REGION_BOUNDARIES in molecular_reference.py'


def _check_phase1():
    """Phase 1: expose DnaA pool counts AND drive them into the
    literature-observed range.

    The listener piece is straightforward (was: missing; now: present).
    But the *biology* — getting the DnaA-ATP fraction into the
    literature band — requires the missing kinetic drivers in Phases 5
    (RIDA) and 7 (DARS). Phase 1 stays ``in_progress`` until those
    drivers are wired and the observed ATP fraction sits in the band."""
    eq_path = 'v2ecoli/processes/parca/reconstruction/ecoli/flat/equilibrium_reactions.tsv'
    eq = _read_file(eq_path)
    has_atp_rxn = bool(eq and 'MONOMER0-160_RXN' in eq)
    has_adp_rxn = bool(eq and 'MONOMER0-4565_RXN' in eq)
    rd = _read_file('v2ecoli/steps/listeners/replication_data.py')
    has_listener = bool(rd and ('MONOMER0-4565' in rd or 'dnaA_adp_count' in rd))

    if not (has_atp_rxn and has_adp_rxn):
        return 'pending', 'DnaA-ATP/ADP equilibrium reactions not wired'
    if not has_listener:
        return 'pending', (
            'equilibrium reactions wired; listener does not yet emit '
            'DnaA-ATP / DnaA-ADP / apo-DnaA pool counts')

    # Listener present — but the pool dynamics are still off target
    # until RIDA (Phase 5) and DARS (Phase 7) close the gap.
    rida_status, _ = _check_phase5()
    dars_status, _ = _check_phase7()
    if rida_status == 'done' and dars_status == 'done':
        return 'done', (
            'listener emits all three pool counts; RIDA (Phase 5) and '
            'DARS (Phase 7) are wired, so the DnaA-ATP fraction can '
            'reach the literature band')
    return 'in_progress', (
        'listener wires both pool counts and surfaces the gap '
        '(observed DnaA-ATP fraction ~95%, vs literature band 30–70%); '
        'closing the gap requires Phase 5 (RIDA) and Phase 7 (DARS)')


def _check_phase2():
    """DnaA box binding process — Phase 2 listener that samples
    equilibrium occupancy per active box and emits per-region bound
    counts."""
    if _file_exists('v2ecoli/processes/dnaA_box_binding.py'):
        gen = _read_file('v2ecoli/generate_replication_initiation.py')
        if gen and '_splice_dnaA_box_binding' in gen:
            return 'done', (
                'DnaABoxBinding step wired into the architecture; '
                'per-region bound counts emitted in '
                'listeners.dnaA_binding')
        return 'in_progress', (
            'binding process file present but not spliced')
    return 'pending', 'no dnaA_box_binding process module'


def _check_phase3():
    """DnaA-gated chromosome-replication initiation."""
    if _file_exists('v2ecoli/processes/chromosome_replication_dnaA_gated.py'):
        gen = _read_file('v2ecoli/generate_replication_initiation.py')
        if gen and '_swap_in_dnaA_gated_chromosome_replication' in gen:
            return 'done', (
                'DnaAGatedChromosomeReplication subclass swapped in; '
                'initiation gates on DnaA-ATP-per-oriC instead of '
                'cell-mass-per-oriC')
        return 'in_progress', (
            'DnaA-gated subclass exists but the architecture does not '
            'swap it in')
    return 'pending', 'no DnaA-gated chromosome_replication subclass'


def _check_phase4():
    """SeqA sequestration — refractory window after each initiation
    event, modeling SeqA binding to hemimethylated GATC sites at the
    newly-replicated origin. Wired into the DnaA-gated subclass."""
    text = _read_file('v2ecoli/processes/chromosome_replication_dnaA_gated.py')
    gen = _read_file('v2ecoli/generate_replication_initiation.py')
    if (text and 'seqA_sequestration_window_s' in text
            and gen and 'enable_seqA_sequestration' in gen):
        return 'done', (
            'SeqA refractory window wired into the DnaA-gated '
            'chromosome-replication step (default 600s)')
    if text and 'seqA' in text.lower():
        return 'in_progress', 'SeqA fields present but flag not wired'
    return 'pending', 'no SeqA refractory window; no sequestration'


def _check_phase5():
    """RIDA — Hda + β-clamp + DnaA-ATP hydrolysis.

    Done = a dedicated RIDA Step process exists and the architecture
    deactivates the DnaA-ADP equilibrium reaction so RIDA's output is
    not instantly re-dissociated by mass-action."""
    if _file_exists('v2ecoli/processes/rida.py'):
        gen = _read_file('v2ecoli/generate_replication_initiation.py')
        if gen and 'MONOMER0-4565_RXN' in gen and '_splice_rida' in gen:
            return 'done', (
                'rida.RIDA process wired into the architecture; '
                'MONOMER0-4565_RXN equilibrium deactivated so RIDA flux '
                'accumulates DnaA-ADP')
        return 'in_progress', (
            'RIDA process present but DnaA-ADP equilibrium not deactivated; '
            'flux will be re-equilibrated away each tick')
    return 'pending', 'no rida process; DnaA-ADP cannot accumulate'


def _check_phase6():
    """DDAH — backup DnaA-ATP hydrolysis at the datA locus."""
    if _file_exists('v2ecoli/processes/ddah.py'):
        gen = _read_file('v2ecoli/generate_replication_initiation.py')
        if gen and '_splice_ddah' in gen:
            return 'done', (
                'DDAH process spliced into the architecture; constitutive '
                'first-order DnaA-ATP hydrolysis runs alongside RIDA')
        return 'in_progress', 'DDAH process file present but not spliced'
    return 'pending', 'no DDAH process'


def _check_phase7():
    """DARS1/2 reactivation — closes the DnaA cycle.

    Done = a dedicated DARS Step process is wired into the
    replication_initiation architecture, releasing ADP from DnaA-ADP
    so the still-active MONOMER0-160_RXN equilibrium can re-load it
    with cellular ATP."""
    if _file_exists('v2ecoli/processes/dars.py'):
        gen = _read_file('v2ecoli/generate_replication_initiation.py')
        if gen and '_splice_dars' in gen:
            return 'done', (
                'dars.DARS process wired into the architecture; '
                'closes the DnaA-ATP/ADP cycle with RIDA (Phase 5)')
        return 'in_progress', 'DARS process file present but not spliced'
    return 'pending', 'no dars process; cycle remains open-loop'


def _check_phase8():
    """dnaA promoter autoregulation."""
    text = _read_file('v2ecoli/processes/transcript_initiation.py')
    if text and ('dnaA_autoregulation' in text or 'dnaA_box_occupancy' in text):
        return 'done', 'dnaA-promoter autoregulation hook present in transcript_initiation'
    return 'pending', 'no DnaA-occupancy hook in transcript_initiation'


# ---------------------------------------------------------------------------
# Per-phase analysis renderers
# ---------------------------------------------------------------------------

def _placeholder(text):
    return (f'<div class="placeholder">{html_lib.escape(text)}</div>')


# ---------------------------------------------------------------------------
# Test-file introspection — pull docstrings from the AST so hover tooltips
# describe what each test asserts. For not-yet-existing test files we fall
# back to the planned description carried on the TestSpec.
# ---------------------------------------------------------------------------

def _extract_test_summary(rel_path):
    """Return (n_tests, [(name, summary), ...]) by parsing the test file's AST.

    ``summary`` is the first line of the docstring if present, else a short
    synthesized phrase from the function name. Returns (0, []) if the file
    is missing or fails to parse."""
    text = _read_file(rel_path)
    if text is None:
        return 0, []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0, []
    items = []

    def _docline(node):
        d = ast.get_docstring(node)
        if not d:
            return ''
        return d.strip().split('\n', 1)[0]

    def _humanize(name):
        return name.removeprefix('test_').replace('_', ' ')

    for top in tree.body:
        if isinstance(top, ast.ClassDef) and top.name.startswith('Test'):
            cls_doc = _docline(top)
            for sub in top.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and sub.name.startswith('test_'):
                    items.append((
                        f'{top.name}.{sub.name}',
                        _docline(sub) or cls_doc or _humanize(sub.name),
                    ))
        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and top.name.startswith('test_'):
            items.append((top.name, _docline(top) or _humanize(top.name)))
    return len(items), items


@dataclass
class TestSpec:
    path: str
    description: str  # planned summary; shown when the file does not exist


def _render_test_li(spec):
    """One <li> per test file with a hover tooltip showing per-test summaries."""
    n_tests, items = _extract_test_summary(spec.path)
    present = _file_exists(spec.path)
    mark = '✓' if present else '○'
    cls = 'present' if present else 'missing'

    if present:
        header = f'<strong>{n_tests} test{"" if n_tests == 1 else "s"}</strong>'
        rows = ''.join(
            f'<div class="t-row"><code>{html_lib.escape(name)}</code>'
            f' — {html_lib.escape(summary)}</div>'
            for name, summary in items
        ) or f'<div class="t-row note">{html_lib.escape(spec.description)}</div>'
        body = header + rows
    else:
        body = (f'<strong class="planned">Planned</strong>'
                f'<div class="t-row note">{html_lib.escape(spec.description)}</div>')

    return (f'<li class="test {cls}"><span class="mark">{mark}</span> '
            f'<code>{html_lib.escape(spec.path)}</code>'
            f'<span class="tooltip">{body}</span></li>')


# ---------------------------------------------------------------------------
# "Before / after" analysis blocks — generic renderer driven by per-phase data
# ---------------------------------------------------------------------------

def _expected_region_counts_table():
    rows = [
        ('oriC', len(ORIC.dnaA_boxes)),
        ('dnaA promoter', len(DNAA_PROMOTER.dnaA_boxes)),
        ('datA', DATA.n_dnaA_boxes),
        ('DARS1', len(DARS1.core_box_names) + len(DARS1.extra_box_names)),
        ('DARS2', len(DARS2.core_box_names) + len(DARS2.extra_box_names)),
    ]
    body = ''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td></tr>' for r in rows)
    return ('<table class="ref"><thead><tr><th>Region</th>'
            '<th>Expected DnaA-box count</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def _no_data_msg():
    return _placeholder('No simulation data — re-run with --duration N.')


# Per-phase "before" plots use only data already in the trajectory (so they
# render today even with the model in its current state). When the phase
# lands, the matching ``after_plot`` slot in the Phase dataclass swaps the
# placeholder for a real plot.

def _before_phase0(snaps):
    return ('<p class="note">Today the chromosome carries DnaA boxes from a '
            'single global motif set, with no per-region breakdown. Below is '
            'the expected per-region count when the classifier is applied to '
            'the existing init-state coordinates.</p>'
            + _expected_region_counts_table())


def _after_phase0(snaps):
    """Phase-0 result: per-region bar chart of strict-consensus search
    coverage vs the curated PDF counts."""
    regions = list(PER_REGION_PDF_COUNT.keys())
    pdf_counts = [PER_REGION_PDF_COUNT[r] for r in regions]
    strict_counts = [PER_REGION_STRICT_CONSENSUS_COUNT[r] for r in regions]
    pretty = {
        'oriC': 'oriC', 'dnaA_promoter': 'dnaA prom',
        'datA': 'datA', 'DARS1': 'DARS1', 'DARS2': 'DARS2',
    }
    labels = [pretty[r] for r in regions]

    fig, ax = plt.subplots(figsize=(11, 4.0))
    x = np.arange(len(regions))
    w = 0.38
    ax.bar(x - w / 2, pdf_counts, w, color='#1e3a8a',
           label='Curated PDF count (named boxes)')
    ax.bar(x + w / 2, strict_counts, w, color='#f59e0b',
           label='Bioinformatic strict-consensus (current model)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('DnaA-box count')
    ax.set_title('Phase 0 result: strict-consensus motif search vs PDF named boxes')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    # Annotate counts above each bar
    for i, (p, s) in enumerate(zip(pdf_counts, strict_counts)):
        ax.text(i - w / 2, p + 0.2, str(p), ha='center', fontsize=8)
        ax.text(i + w / 2, s + 0.2, str(s), ha='center', fontsize=8)
    fig.tight_layout()
    total_pdf = sum(pdf_counts)
    total_strict = sum(strict_counts)
    return (_img(fig_to_b64(fig), 'Phase 0 region counts') +
            f'<p class="note">Coverage: <strong>{total_strict}/{total_pdf}</strong> '
            f'named boxes are caught by the strict-consensus motif. '
            f'The remaining {total_pdf - total_strict} are non-consensus '
            f'low-affinity sites that Phase 2 must enrich into the box list.</p>')


def _dnaA_pool_traces(snaps):
    """Pull the DnaA apo / ATP / ADP pool counts from listener snapshots.

    Returns ``(times_min, apo, atp, adp)`` arrays, or ``None`` if the
    listener fields aren't present in the trajectory. Drops the very
    first snapshot — the equilibrium relaxes from apo=124, atp=0, adp=0
    in the first tick, and including that point compresses the
    post-relaxation dynamics into a tiny y-range."""
    if not snaps:
        return None
    if len(snaps) > 1:
        snaps = snaps[1:]
    times = [s.get('time', 0) / 60 for s in snaps]
    apo = [s.get('dnaA_apo_count') for s in snaps]
    atp = [s.get('dnaA_atp_count') for s in snaps]
    adp = [s.get('dnaA_adp_count') for s in snaps]
    if all(v is None for v in apo) and all(v is None for v in atp):
        return None
    return (np.array(times),
            np.array([0 if v is None else v for v in apo]),
            np.array([0 if v is None else v for v in atp]),
            np.array([0 if v is None else v for v in adp]))


def _drivers_table_html():
    """Audit table of every process that drives the DnaA pool, with a
    wired/missing flag per row."""
    def _row(d):
        cls = 'driver-on' if d.wired_in_v2ecoli else 'driver-off'
        mark = '✓' if d.wired_in_v2ecoli else '○'
        return (
            f'<tr class="{cls}">'
            f'<td>{mark}</td>'
            f'<td><strong>{html_lib.escape(d.process)}</strong></td>'
            f'<td><code>{html_lib.escape(d.direction)}</code></td>'
            f'<td>{html_lib.escape(d.represents)}</td>'
            f'</tr>'
        )
    rows = ''.join(_row(d) for d in DNAA_POOL_DRIVERS)
    return ('<table class="ref drivers"><thead>'
            '<tr><th></th><th>Process</th><th>Effect on pool</th>'
            '<th>Biological meaning</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _before_phase1(snaps):
    """Phase 1 Before = the *baseline* architecture's pool dynamics.
    No RIDA, no DARS, no equilibrium override. The DnaA-ATP fraction
    sits at the equilibrium ceiling (~95%), well above the literature
    30–70% band — this is the gap the new mechanisms close."""
    pulled = _dnaA_pool_traces(snaps)
    if pulled is None:
        return _placeholder(
            'No DnaA pool counts in trajectory — re-run with --duration N '
            'and without --no-baseline.')
    times, apo, atp, adp = pulled
    total = apo + atp + adp
    safe_total = np.where(total > 0, total, 1)
    atp_frac = atp / safe_total

    eq = DNAA_NUCLEOTIDE_EQUILIBRIUM
    band_min = eq.biological_atp_fraction_min.value
    band_max = eq.biological_atp_fraction_max.value

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.plot(times, apo, color='#0891b2', lw=1.4, label='apo-DnaA')
    ax.plot(times, atp, color='#16a34a', lw=1.8, label='DnaA-ATP')
    ax.plot(times, adp, color='#dc2626', lw=1.8, label='DnaA-ADP')
    ax.set_ylabel('Bulk count')
    ax.set_title('Observed DnaA pool counts')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.fill_between(
        [times.min(), times.max()], band_min, band_max,
        color='#bfdbfe', alpha=0.55,
        label=f'literature band ({band_min:.0%}–{band_max:.0%})')
    ax.plot(times, atp_frac, color='#16a34a', lw=2.0,
            label='observed DnaA-ATP fraction')
    ax.set_ylabel('DnaA-ATP / total')
    ax.set_xlabel('Time (min)')
    ax.set_ylim(0, 1.05)
    ax.set_title('Observed DnaA-ATP fraction vs literature band')
    ax.legend(fontsize=7, loc='center right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    final_atp_frac = float(atp_frac[-1]) if len(atp_frac) else 0.0
    if final_atp_frac > band_max:
        diagnosis = (
            f'<p class="note">Observed final DnaA-ATP fraction '
            f'<strong>{final_atp_frac:.0%}</strong> sits <em>above</em> the '
            f'literature band ({band_min:.0%}–{band_max:.0%}). The DnaA-ATP '
            f'equilibrium is dominating; <strong>RIDA (Phase 5)</strong> is '
            f'the missing kinetic force that hydrolyzes DnaA-ATP → DnaA-ADP.</p>'
        )
    elif final_atp_frac < band_min:
        diagnosis = (
            f'<p class="note">Observed final DnaA-ATP fraction '
            f'<strong>{final_atp_frac:.0%}</strong> sits <em>below</em> the '
            f'literature band ({band_min:.0%}–{band_max:.0%}). RIDA (Phase 5) '
            f'is depleting DnaA-ATP but the cycle is open-loop; '
            f'<strong>DARS (Phase 7)</strong> is the missing reactivation '
            f'that converts DnaA-ADP back into DnaA-ATP.</p>'
        )
    else:
        diagnosis = (
            f'<p class="note">Observed final DnaA-ATP fraction '
            f'<strong>{final_atp_frac:.0%}</strong> is inside the literature '
            f'band ({band_min:.0%}–{band_max:.0%}). The model has reached '
            f'biological steady-state for the DnaA nucleotide-state cycle.</p>'
        )
    return _img(fig_to_b64(fig), 'observed DnaA pool + ATP fraction') + diagnosis


def _after_phase1(snaps):
    """Phase 1 After = the *current* replication_initiation trajectory
    with RIDA + DARS wired. Same plot structure as Before so the user
    can compare side-by-side. The ATP fraction crosses into the band
    and stabilizes there."""
    pulled = _dnaA_pool_traces(snaps)
    if pulled is None:
        return _placeholder('No replication_initiation trajectory data.')
    times, apo, atp, adp = pulled
    total = apo + atp + adp
    safe = np.where(total > 0, total, 1)
    atp_frac = atp / safe

    eq = DNAA_NUCLEOTIDE_EQUILIBRIUM
    band_min = eq.biological_atp_fraction_min.value
    band_max = eq.biological_atp_fraction_max.value

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.plot(times, apo, color='#0891b2', lw=1.4, label='apo-DnaA')
    ax.plot(times, atp, color='#16a34a', lw=1.8, label='DnaA-ATP')
    ax.plot(times, adp, color='#dc2626', lw=1.8, label='DnaA-ADP')
    ax.set_ylabel('Bulk count')
    ax.set_title('replication_initiation: DnaA pool counts')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.fill_between(
        [times.min(), times.max()], band_min, band_max,
        color='#bfdbfe', alpha=0.55,
        label=f'literature band ({band_min:.0%}–{band_max:.0%})')
    ax.plot(times, atp_frac, color='#16a34a', lw=2.0,
            label='observed DnaA-ATP fraction')
    ax.set_ylabel('DnaA-ATP / total')
    ax.set_xlabel('Time (min)')
    ax.set_ylim(0, 1.05)
    ax.set_title('ATP fraction settles inside the literature band')
    ax.legend(fontsize=7, loc='center right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    final_atp_frac = float(atp_frac[-1]) if len(atp_frac) else 0.0
    if band_min <= final_atp_frac <= band_max:
        diagnosis = (
            f'<p class="note">Observed final DnaA-ATP fraction '
            f'<strong>{final_atp_frac:.0%}</strong> sits inside the '
            f'literature band ({band_min:.0%}–{band_max:.0%}). RIDA '
            f'(Phase 5) and DARS (Phase 7) hold the cycle in steady '
            f'state — the gap Phase 1 was diagnosing is closed.</p>'
        )
    elif final_atp_frac > band_max:
        diagnosis = (
            f'<p class="note">Observed final DnaA-ATP fraction '
            f'<strong>{final_atp_frac:.0%}</strong> sits above the '
            f'band; RIDA flux not strong enough to balance the '
            f'equilibrium. Tune k_rida / k_dars.</p>'
        )
    else:
        diagnosis = (
            f'<p class="note">Observed final DnaA-ATP fraction '
            f'<strong>{final_atp_frac:.0%}</strong> sits below the '
            f'band; DARS flux not strong enough to balance RIDA. '
            f'Tune k_rida / k_dars.</p>'
        )
    return _img(fig_to_b64(fig), 'replication_initiation DnaA dynamics') + diagnosis


def _phase1_provenance_html():
    eq = DNAA_NUCLEOTIDE_EQUILIBRIUM
    table = _provenance_table([
        ('biological steady-state min', eq.biological_atp_fraction_min),
        ('biological steady-state max', eq.biological_atp_fraction_max),
        ('pre-initiation peak', eq.peak_atp_fraction_pre_initiation),
        ('post-RIDA trough', eq.typical_atp_fraction_post_initiation),
        ('K_d(ATP)  [nM]', eq.kd_atp_nm),
        ('K_d(ADP)  [nM]', eq.kd_adp_nm),
        ('cellular ATP/ADP ratio', eq.typical_atp_adp_ratio),
    ])
    n_ext = sum(1 for cv in (
        eq.biological_atp_fraction_min, eq.biological_atp_fraction_max,
        eq.peak_atp_fraction_pre_initiation,
        eq.typical_atp_fraction_post_initiation,
        eq.kd_atp_nm, eq.kd_adp_nm, eq.typical_atp_adp_ratio,
    ) if not cv.in_curated_pdf)
    note = (
        f'<p class="note">{n_ext} of 7 reference values come from '
        'outside the curated PDF reference list. Review the flagged '
        'rows before relying on them as load-bearing parameters.</p>'
    )
    return note + table


def _phase1_extras(snaps, status):
    """Per-phase subsections rendered in the Phase 1 section."""
    return [
        ('Pool drivers — what affects DnaA-ATP / DnaA-ADP today',
         _drivers_table_html()),
        ('Reference-value provenance',
         _phase1_provenance_html()),
    ]


# ---------------------------------------------------------------------------
# Architecture-change table — used by Phase 5 (and future phases) to show
# the user exactly what each phase added, modified, or overrode.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchChange:
    """One architectural delta introduced by a phase.

    ``kind`` is one of:
      * ``new_process``     — a new Step instance is wired into the
                              cell_state and appended to flow_order.
      * ``override``        — an existing process / config is mutated
                              after instantiation.
      * ``data``            — a new constant, dataclass, or look-up
                              table is added.
      * ``listener``        — a listener field is added or modified.
    """

    kind: str
    summary: str
    file: str
    detail: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


_ARCH_KIND_LABELS = {
    'new_process': ('New process', '#16a34a'),
    'override':    ('Override',     '#f59e0b'),
    'data':        ('Data',         '#0891b2'),
    'listener':    ('Listener',     '#7c3aed'),
}


def _render_arch_changes_html(changes):
    """Render a styled table of architecture changes for one phase."""
    rows = []
    for c in changes:
        label, color = _ARCH_KIND_LABELS.get(
            c.kind, (c.kind, '#475569'))
        pill = (f'<span class="pill" style="background:{color};">'
                f'{html_lib.escape(label)}</span>')
        ports_html = ''
        if c.reads or c.writes:
            ports_html = (
                '<div class="ports">'
                + (f'<div><strong>reads:</strong> '
                   + ', '.join(f'<code>{html_lib.escape(r)}</code>' for r in c.reads)
                   + '</div>' if c.reads else '')
                + (f'<div><strong>writes:</strong> '
                   + ', '.join(f'<code>{html_lib.escape(w)}</code>' for w in c.writes)
                   + '</div>' if c.writes else '')
                + '</div>')
        rows.append(
            f'<tr>'
            f'<td>{pill}</td>'
            f'<td><strong>{html_lib.escape(c.summary)}</strong></td>'
            f'<td><code>{html_lib.escape(c.file)}</code></td>'
            f'<td>{html_lib.escape(c.detail)}{ports_html}</td>'
            f'</tr>'
        )
    return ('<table class="ref arch"><thead>'
            '<tr><th>Kind</th><th>Summary</th><th>File</th>'
            '<th>Detail / wiring</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ---------------------------------------------------------------------------
# Phase 5 architecture-change list + process-connections diagram
# ---------------------------------------------------------------------------

PHASE5_ARCH_CHANGES = (
    ArchChange(
        kind='new_process',
        summary='RIDA Step',
        file='v2ecoli/processes/rida.py',
        detail=('A dedicated EcoliStep that hydrolyzes DnaA-ATP → '
                'DnaA-ADP each tick. Rate = k · n_active_replisomes · '
                '[DnaA-ATP] · dt, drawn from a Poisson distribution.'),
        reads=(
            'bulk[MONOMER0-160[c]]',
            'bulk[MONOMER0-4565[c]]',
            'unique.active_replisome._entryState',
            'timestep',
        ),
        writes=(
            'bulk[MONOMER0-160[c]]',
            'bulk[MONOMER0-4565[c]]',
            'listeners.rida.flux_atp_to_adp',
            'listeners.rida.active_replisomes',
            'listeners.rida.rate_constant',
        ),
    ),
    ArchChange(
        kind='override',
        summary='Deactivate MONOMER0-4565_RXN equilibrium',
        file='v2ecoli/generate_replication_initiation.py',
        detail=('After the baseline document is built, the live '
                '`ecoli-equilibrium` step instance has its '
                '`stoichMatrix` column for `MONOMER0-4565_RXN` zeroed. '
                'The equilibrium SS solver still computes a flux for '
                'this reaction but the result is multiplied by an '
                'all-zero stoichiometry vector, so molecule counts '
                'are not coupled by mass-action.'),
        reads=('configs.ecoli-equilibrium.stoichMatrix',),
        writes=(
            'ecoli-equilibrium.instance.stoichMatrix',
            'ecoli-equilibrium.instance._deactivated_reactions',
        ),
    ),
    ArchChange(
        kind='listener',
        summary='New rida.* listener fields',
        file='v2ecoli/processes/rida.py',
        detail=('Each tick the RIDA Step emits its hydrolysis flux, '
                'the active replisome count it observed, and the '
                'configured rate constant.'),
        writes=(
            'listeners.rida.flux_atp_to_adp',
            'listeners.rida.active_replisomes',
            'listeners.rida.rate_constant',
        ),
    ),
    ArchChange(
        kind='data',
        summary='Bulk-ID constants for DnaA forms',
        file='v2ecoli/data/replication_initiation/molecular_reference.py',
        detail=('`DNAA_APO_BULK_ID = "PD03831[c]"`, '
                '`DNAA_ATP_BULK_ID = "MONOMER0-160[c]"`, '
                '`DNAA_ADP_BULK_ID = "MONOMER0-4565[c]"`. '
                'Imported by RIDA and the replication_data listener.'),
    ),
)


@functools.lru_cache(maxsize=1)
def _phase5_bigraph_svg():
    """Render the RIDA-relevant subset of the live bigraph using
    ``bigraph_viz.plot_bigraph``. Builds the architecture composite
    once, filters cell_state to RIDA + the two adjacent processes that
    touch the same DnaA pools (ecoli-equilibrium, ecoli-metabolism),
    plus the stores they all wire to."""
    if not os.path.isdir(CACHE_DIR):
        return _placeholder(
            f'cache dir {CACHE_DIR!r} not present; cannot render the '
            f'live bigraph. Rebuild the cache and re-run the report.')
    try:
        from bigraph_viz import plot_bigraph
    except ImportError:
        return _placeholder(
            'bigraph_viz not installed; install it to render the live '
            'process-connections diagram.')
    try:
        from v2ecoli.composite_replication_initiation import (
            make_replication_initiation_composite,
        )
        composite = make_replication_initiation_composite(
            cache_dir=CACHE_DIR, seed=0)
    except Exception as exc:
        return _placeholder(
            f'composite build failed: {type(exc).__name__}: {exc}')

    cell = composite.state['agents']['0']
    keep_processes = {'rida', 'dars', 'ecoli-equilibrium', 'ecoli-metabolism'}
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict) or '_type' not in edge:
            continue
        if name not in keep_processes:
            continue
        inputs = {p: w for p, w in edge.get('inputs', {}).items()
                  if not p.startswith('_layer') and not p.startswith('_flow')
                  and p != 'global_time' and p != 'timestep'}
        outputs = {p: w for p, w in edge.get('outputs', {}).items()
                   if not p.startswith('_layer') and not p.startswith('_flow')
                   and p != 'global_time' and p != 'timestep'}
        viz[name.replace('ecoli-', '')] = {
            '_type': edge['_type'],
            'inputs': inputs,
            'outputs': outputs,
        }
    # Empty stores so the bigraph has something to draw edges to.
    viz['bulk'] = {}
    viz['unique'] = {'active_replisome': {}}
    viz['listeners'] = {
        'rida': {}, 'equilibrium_listener': {}, 'fba_results': {}}

    viz_state = {'agents': {'0': viz}}

    out_dir = os.path.join(OUT_DIR, '_phase5_bigraph')
    os.makedirs(out_dir, exist_ok=True)
    # Match viva-munk's rendering recipe: PNG output, high DPI,
    # collapse_redundant_processes for a tighter graph, embed as <img>
    # with max-width CSS so the browser handles scaling cleanly.
    try:
        plot_bigraph(
            viz_state, remove_process_place_edges=True,
            rankdir='LR', dpi='140',
            collapse_redundant_processes=True,
            show_values=False,
            port_labels=True,
            out_dir=out_dir, filename='rida_subgraph',
            file_format='png')
    except Exception as exc:
        return _placeholder(
            f'plot_bigraph failed: {type(exc).__name__}: {exc}')
    png_path = os.path.join(out_dir, 'rida_subgraph.png')
    if not os.path.exists(png_path):
        return _placeholder('PNG output not produced by plot_bigraph')
    with open(png_path, 'rb') as f:
        png_b64 = base64.b64encode(f.read()).decode('utf-8')
    return (
        f'<div class="bigraph-img" id="phase5-bigraph">'
        f'<img alt="RIDA subgraph rendered via bigraph_viz" '
        f'src="data:image/png;base64,{png_b64}"/>'
        f'</div>'
        f'<p class="note" style="font-size:0.78em;">'
        f'Rendered by <code>bigraph_viz.plot_bigraph</code> from the '
        f'live <code>replication_initiation</code> composite. Three '
        f'Steps shown: the new <code>rida</code>, plus '
        f'<code>ecoli-equilibrium</code> and <code>ecoli-metabolism</code> '
        f'(which read/write the same DnaA bulk pools).</p>')


PHASE4_ARCH_CHANGES = (
    ArchChange(
        kind='override',
        summary='SeqA refractory window in DnaAGatedChromosomeReplication',
        file='v2ecoli/processes/chromosome_replication_dnaA_gated.py',
        detail=('The DnaA-gated subclass tracks the previous tick\'s '
                'oriC count; when it jumps, an initiation just fired '
                'and global_time is recorded. While '
                '(global_time - last_init) < seqA_sequestration_window_s '
                '(default 600s), `_compute_dnaA_gate` returns 0 — the '
                'gate is shut. Models SeqA binding to hemimethylated '
                'GATC sites at the newly-replicated origin.'),
        reads=(
            'states.oriCs._entryState',
            'states.global_time',
        ),
        writes=(
            'self._previous_n_oric',
            'self._last_initiation_time_s',
        ),
    ),
    ArchChange(
        kind='data',
        summary='SeqA already in bulk; no protein addition needed',
        file='v2ecoli/processes/parca/reconstruction/ecoli/flat/proteins.tsv',
        detail=('SeqA monomer (`EG12197-MONOMER`) is already expressed '
                'by the baseline transcription/translation pipeline at '
                '~1029 copies in the init state. Phase 4 wires its '
                '*activity* downstream of the gate — a future refinement '
                'would consume SeqA stoichiometrically (one bound '
                'multimer per sequestered origin) so SeqA scarcity '
                'could shorten the window.'),
    ),
    ArchChange(
        kind='override',
        summary='enable_seqA_sequestration flag (default True)',
        file='v2ecoli/generate_replication_initiation.py',
        detail=('build_replication_initiation_document gains '
                '`enable_seqA_sequestration` (default True). When True, '
                '`_swap_in_dnaA_gated_chromosome_replication` instantiates '
                'the subclass with `seqA_sequestration_window_s=600.0`. '
                'When False, window=0 — gate has no refractory.'),
    ),
)


def _phase4_extras(snaps, status):
    arch = _render_arch_changes_html(PHASE4_ARCH_CHANGES)
    note = (
        '<p class="note"><strong>Why the refractory window approach?</strong> '
        'Modeling hemimethylated-GATC tracking + SeqA-multimer binding '
        'on the unique-molecule store would conflict with '
        'chromosome_structure\'s in-tick add/delete (same set-update '
        'issue we hit in Phase 2). The refractory window captures the '
        'biology that matters most for cell-cycle timing — SeqA blocks '
        'rebinding for a fixed window after each initiation — without '
        'requiring a hemimethylation field on the oriC unique molecule. '
        'Future refinement: read the bulk SeqA count and decrement / '
        'reset the window if SeqA is depleted.</p>'
    )
    return [
        ('Architecture changes — what this phase adds, modifies, overrides',
         note + arch),
    ]


PHASE7_ARCH_CHANGES = (
    ArchChange(
        kind='new_process',
        summary='DARS Step',
        file='v2ecoli/processes/dars.py',
        detail=('First-order release of ADP from DnaA-ADP, regenerating '
                'apo-DnaA. Rate = k_dars · [DnaA-ADP] · dt, Poisson-drawn. '
                'The freed apo-DnaA is then reloaded with ATP by the still-'
                'active MONOMER0-160_RXN equilibrium, completing the cycle.'),
        reads=(
            'bulk[MONOMER0-4565[c]]',
            'bulk[PD03831[c]]',
            'timestep',
        ),
        writes=(
            'bulk[MONOMER0-4565[c]]',
            'bulk[PD03831[c]]',
            'listeners.dars.flux_adp_to_apo',
            'listeners.dars.rate_constant',
        ),
    ),
    ArchChange(
        kind='listener',
        summary='New dars.* listener fields',
        file='v2ecoli/processes/dars.py',
        detail='Per-tick DARS reactivation flux + configured rate constant.',
        writes=(
            'listeners.dars.flux_adp_to_apo',
            'listeners.dars.rate_constant',
        ),
    ),
)


def _phase7_extras(snaps, status):
    arch = _render_arch_changes_html(PHASE7_ARCH_CHANGES)
    note = (
        '<p class="note"><strong>How DARS closes the cycle.</strong> '
        'RIDA (Phase 5) drove DnaA-ATP → DnaA-ADP, but with no reverse '
        'path the ATP form depleted to zero. DARS adds the reverse path: '
        'DnaA-ADP → apo-DnaA → DnaA-ATP (the apo-to-ATP step is handled '
        'by the still-active <code>MONOMER0-160_RXN</code> equilibrium). '
        'At steady state the DARS flux balances the RIDA flux and the '
        'ATP fraction stabilizes inside the literature band. '
        'The live bigraph (which now includes <code>dars</code>) is '
        'rendered once in '
        '<a href="#phase5-bigraph">Phase 5 → Process connections</a>.</p>'
    )
    return [
        ('Architecture changes — what this phase adds, modifies, overrides',
         arch),
        ('How DARS closes the cycle', note),
    ]


def _phase5_extras(snaps, status):
    arch = _render_arch_changes_html(PHASE5_ARCH_CHANGES)
    bigraph = _phase5_bigraph_svg()
    note = (
        '<p class="note"><strong>Is RIDA a constraint on metabolism?</strong> '
        'No. The FBA reaction <code>RXN0-7444</code> exists in '
        '<code>flat/metabolic_reactions.tsv</code> and is registered with '
        'metabolism (stoichiometry, catalyst, base reaction) but it carries '
        'no kinetic constraint and no biomass demand, so FBA pushes zero '
        'flux through it. The dedicated <code>RIDA</code> Step runs in '
        'parallel to metabolism: both Steps read and write the shared '
        '<code>bulk</code> store directly, but only RIDA produces the '
        'observable DnaA-ATP → DnaA-ADP flux today. The bigraph below is '
        'the live wiring extracted from the composite.</p>'
    )
    chromosome = _chromosome_timeline_plot(
        snaps,
        title='Replisome activity across the trajectory '
              '(amber triangles = active forks)')
    chromosome_note = (
        '<p class="note">RIDA\'s rate is gated on the active-replisome '
        'count. The disc at each snapshot shows how many forks are '
        'currently engaged with the chromosome — that\'s the substrate '
        'pool RIDA reads each tick.</p>'
    )
    return [
        ('Architecture changes — what this phase adds, modifies, overrides',
         arch),
        ('Process connections (live bigraph subset)',
         note + bigraph),
        ('Chromosome dynamics — replisomes drive the RIDA flux',
         chromosome + chromosome_note),
    ]


def _provenance_table(cited_pairs):
    """Render a small audit table for a list of (label, CitedValue)."""
    rows = []
    for label, cv in cited_pairs:
        cit = CITATIONS.get(cv.citation_key, {})
        full_cite = cit.get('cite', cv.citation_key)
        flag = ('<span class="cite-pdf">in curated PDF</span>'
                if cv.in_curated_pdf
                else '<span class="cite-ext">outside curated PDF</span>')
        note_html = (f'<div class="cite-note">{html_lib.escape(cv.note)}</div>'
                     if cv.note else '')
        rows.append(
            f'<tr>'
            f'<td>{html_lib.escape(label)}</td>'
            f'<td><strong>{cv.value:g}</strong></td>'
            f'<td>{flag}</td>'
            f'<td><span class="cite-text">{html_lib.escape(full_cite)}</span>'
            f'{note_html}</td>'
            f'</tr>'
        )
    return ('<table class="ref provenance"><thead>'
            '<tr><th>Quantity</th><th>Value</th><th>Source</th>'
            '<th>Citation + caveat</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _before_phase2(snaps):
    """Phase 2 'Before' = state at the cumulative point just before
    Phase 2 (rida_dars). The replication_data listener still reports
    bound=0 because no process has ever sampled / written DnaA_bound."""
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    bound = np.array([s['dnaA_box_bound'] for s in snaps])
    total = np.array([s['dnaA_box_total'] for s in snaps])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(times, total, color='#1e293b', lw=1.4, label='total active boxes')
    ax.plot(times, bound, color='#dc2626', lw=1.4,
            label='bound (replication_data listener — always 0)')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('DnaA boxes')
    ax.set_title('Pre-Phase-2: DnaA_bound flat at zero')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'DnaA-box occupancy (pre-Phase-2)')


def _after_phase2(snaps):
    """Phase 2 'After' = the binding listener's per-region occupancy
    over the trajectory."""
    if not snaps:
        return _no_data_msg()
    # Drop t=0 — the binding step hasn't fired yet, so its listener
    # fields are at their `_default` of 0, which makes a meaningless
    # spike at the left of the plot.
    if len(snaps) > 1:
        snaps = snaps[1:]
    times = np.array([s['time'] / 60 for s in snaps])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [2, 3]})

    ax = axes[0]
    total_bound = np.array(
        [s.get('binding_total_bound') or 0 for s in snaps])
    total_active = np.array(
        [s.get('binding_total_active') or 1 for s in snaps])
    fraction = total_bound / np.where(total_active > 0, total_active, 1)
    ax.plot(times, fraction, color='#16a34a', lw=2.0,
            label='global fraction bound')
    ax.set_ylabel('Bound / active'); ax.set_ylim(0, 1.05)
    ax.set_title('Per-tick equilibrium occupancy across all DnaA boxes')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    region_specs = [
        ('binding_bound_oric',          'oriC',          '#10b981'),
        ('binding_bound_dnaA_promoter', 'dnaA promoter', '#0891b2'),
        ('binding_bound_datA',          'datA',          '#7c3aed'),
        ('binding_bound_DARS1',         'DARS1',         '#f59e0b'),
        ('binding_bound_DARS2',         'DARS2',         '#dc2626'),
        ('binding_bound_other',         'other (low-aff)', '#94a3b8'),
    ]
    for field, label, color in region_specs:
        vals = np.array([s.get(field) or 0 for s in snaps])
        ax.plot(times, vals, color=color, lw=1.5, label=label)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Bound count')
    ax.set_title('Per-region bound count (high-affinity regions saturated)')
    ax.legend(fontsize=7, loc='center right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    note = (
        '<p class="note">High-affinity regions (oriC / dnaA promoter / '
        'DARS1 / DARS2 — Kd ~ 1 nM) saturate at 100% occupancy. '
        '"Other" boxes (the bulk of the strict-consensus motif hits '
        'across the genome, falling back to a low-affinity Kd ~ 100 nM '
        'rule) sit at ~50–70% occupancy depending on the cellular '
        '[DnaA-ATP].</p>'
    )
    return _img(fig_to_b64(fig), 'per-region bound counts') + note


PHASE2_ARCH_CHANGES = (
    ArchChange(
        kind='new_process',
        summary='DnaABoxBinding Step',
        file='v2ecoli/processes/dnaA_box_binding.py',
        detail=('Listener-only process: per active DnaA box, samples '
                'bound/unbound from the equilibrium occupancy '
                'p_bound = [DnaA] / (Kd + [DnaA]). Per-region Kd and '
                'nucleotide preference come from the curated reference. '
                'Does *not* write back to DnaA_bound on the unique '
                'store (the set update mode conflicts with '
                'chromosome_structure\'s add/delete on the same tick); '
                'Phase 3 reads the listener counts directly.'),
        reads=(
            'bulk[MONOMER0-160[c]]',
            'bulk[MONOMER0-4565[c]]',
            'unique.DnaA_box.coordinates',
            'unique.DnaA_box._entryState',
        ),
        writes=(
            'listeners.dnaA_binding.total_bound',
            'listeners.dnaA_binding.total_active',
            'listeners.dnaA_binding.fraction_bound',
            'listeners.dnaA_binding.bound_<region>',
        ),
    ),
    ArchChange(
        kind='data',
        summary='REGION_BINDING_RULES + DEFAULT_REGION_BINDING_RULE',
        file='v2ecoli/data/replication_initiation/molecular_reference.py',
        detail=('Per-region (affinity_class, binds_atp, binds_adp) tuples '
                'driving Phase 2\'s occupancy formula. All named regions '
                'currently use the high-affinity class (Kd ~1 nM); '
                'the default rule for unnamed regions is low-affinity '
                'ATP-preferential (Kd ~100 nM).'),
    ),
    ArchChange(
        kind='listener',
        summary='New listeners.dnaA_binding.* fields',
        file='v2ecoli/processes/dnaA_box_binding.py',
        detail=('total_bound / total_active / fraction_bound plus '
                'bound_oric / bound_dnaA_promoter / bound_datA / '
                'bound_DARS1 / bound_DARS2 / bound_other.'),
        writes=(
            'listeners.dnaA_binding.total_bound',
            'listeners.dnaA_binding.bound_oric',
            'listeners.dnaA_binding.bound_dnaA_promoter',
            'listeners.dnaA_binding.bound_datA',
            'listeners.dnaA_binding.bound_DARS1',
            'listeners.dnaA_binding.bound_DARS2',
            'listeners.dnaA_binding.bound_other',
        ),
    ),
)


def _phase2_extras(snaps, status):
    arch = _render_arch_changes_html(PHASE2_ARCH_CHANGES)
    note = (
        '<p class="note"><strong>Why a listener-only process?</strong> '
        'The unique-array <code>set</code> update mode requires the new '
        'value array to match the active-box count exactly. '
        '<code>chromosome_structure</code> adds and deletes DnaA_box '
        'entries during fork passage in the same tick that binding '
        'samples occupancy, so the apply-time count differs from the '
        'sample-time count and numpy raises a size mismatch. The '
        'listener pattern dodges that conflict — Phase 3 (initiation '
        'gating) and the report can both read the listener directly.</p>'
    )
    return [
        ('Architecture changes — what this phase adds, modifies, overrides',
         note + arch),
    ]


def _phase3_initiation_panel(snaps, title):
    """Show actual initiation events and the gate signal that drives
    them. Used by both Phase 3 Before (pre_gate / mass threshold) and
    After (full / DnaA-ATP gate)."""
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    n_oric = np.array([s.get('n_oriC') or 0 for s in snaps])
    n_rep = np.array([s.get('n_replisomes') or 0 for s in snaps])
    gate = np.array([s.get('critical_mass_per_oriC') or 0.0 for s in snaps])

    init_events = [
        times[i] for i in range(1, len(n_oric))
        if n_oric[i] > n_oric[i - 1]
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.step(times, n_oric, where='post', color='#10b981', lw=2.0,
            label='oriC count')
    ax.step(times, n_rep, where='post', color='#f59e0b', lw=1.6,
            label='active replisomes')
    for i, t in enumerate(init_events):
        ax.axvline(t, color='#dc2626', ls='--', lw=0.9, alpha=0.6,
                   label='initiation event' if i == 0 else None)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='center left')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[1]
    ax.plot(times, gate, color='#7c3aed', lw=1.8,
            label='gate ratio (critical_mass_per_oriC listener)')
    ax.axhline(1.0, color='#dc2626', ls='--', lw=1.0, alpha=0.7,
               label='trigger threshold')
    for t in init_events:
        ax.axvline(t, color='#dc2626', ls='--', lw=0.9, alpha=0.4)
    ax.set_ylabel('Gate ratio')
    ax.set_xlabel('Time (min)')
    ax.set_title('Initiation gate signal — fires at ratio ≥ 1.0')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    note = (
        f'<p class="note">Observed <strong>{len(init_events)}</strong> '
        f'initiation event(s) over {times[-1]:.0f} min of sim. Each red '
        f'dashed line marks one (oriC count steps up). The gate ratio in '
        f'the lower panel crosses 1.0 at each event and drops back below '
        f'afterward — the per-oriC denominator provides the self-limiting '
        f'feedback that both gates share.</p>'
    )
    return _img(fig_to_b64(fig), 'initiation events + gate signal') + note


def _before_phase3(snaps):
    """Phase 3 Before = mass-threshold gate (pre_gate config)."""
    return _phase3_initiation_panel(
        snaps, title='Mass-threshold gate (pre-Phase-3)')


def _after_phase3(snaps):
    """Phase 3 After = DnaA-ATP-per-oriC gate (full config)."""
    return _phase3_initiation_panel(
        snaps, title='DnaA-ATP-per-oriC gate (Phase 3 active)')


def _phase4_gate_panel(snaps, title, sequestration_window_min=10.0,
                        show_window=False):
    """oriC count + gate ratio for Phase 4. Optionally shades the
    SeqA sequestration window after each initiation event."""
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    n_oric = np.array([s.get('n_oriC') or 0 for s in snaps])
    gate = np.array([s.get('critical_mass_per_oriC') or 0.0 for s in snaps])

    init_events = [
        times[i] for i in range(1, len(n_oric))
        if n_oric[i] > n_oric[i - 1]
    ]
    # Treat t=0 as an initiation event for gated configs (the gate
    # fires in the first tick — see Phase 3 panel).
    if len(times) > 0 and not init_events and n_oric[0] >= 4:
        init_events = [times[0]]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.step(times, n_oric, where='post', color='#10b981', lw=2.2,
            label='oriC count')
    for i, t in enumerate(init_events):
        ax.axvline(t, color='#dc2626', ls='--', lw=1.0, alpha=0.6,
                   label='initiation event' if i == 0 else None)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[1]
    if show_window:
        for i, t in enumerate(init_events):
            ax.axvspan(
                t, t + sequestration_window_min,
                color='#fde68a', alpha=0.45,
                label=(f'SeqA window (~{sequestration_window_min:.0f} min)'
                       if i == 0 else None))
    ax.plot(times, gate, color='#7c3aed', lw=2.2,
            label='gate ratio')
    ax.axhline(1.0, color='#dc2626', ls='--', lw=1.0, alpha=0.7,
               label='trigger threshold')
    for t in init_events:
        ax.axvline(t, color='#dc2626', ls='--', lw=1.0, alpha=0.4)
    ax.set_ylabel('Gate ratio')
    ax.set_xlabel('Time (min)')
    ax.set_title('DnaA-ATP-per-oriC gate signal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if show_window:
        note = (
            f'<p class="note">After the initiation event, the gate is '
            f'forced to 0 for the SeqA sequestration window '
            f'(<strong>~{sequestration_window_min:.0f} min</strong>, '
            f'shaded amber). Once the window expires, the gate resumes '
            f'reading the actual DnaA-ATP-per-oriC ratio — and '
            f'because RIDA+DARS dynamics keep the per-oriC value '
            f'below 1.0, no immediate re-initiation fires.</p>'
        )
    else:
        note = (
            '<p class="note">Without SeqA, the gate signal is just the '
            'DnaA-ATP-per-oriC ratio at every tick. Reinitiation timing '
            'is set entirely by the DnaA pool dynamics — there is no '
            'refractory window after a firing event.</p>'
        )
    return _img(fig_to_b64(fig), 'Phase 4 gate panel') + note


def _before_phase4(snaps):
    """Phase 4 'Before' = the DnaA gate without SeqA sequestration.
    The gate ratio is the raw DnaA-ATP-per-oriC at every tick."""
    return _phase4_gate_panel(
        snaps,
        title='Without SeqA: gate ratio reads DnaA-ATP-per-oriC always',
        show_window=False)


def _after_phase4(snaps):
    """Phase 4 'After' = SeqA sequestration window forces the gate
    to 0 for ~10 min after each initiation event."""
    return _phase4_gate_panel(
        snaps,
        title='With SeqA: refractory window after each initiation',
        show_window=True)


def _before_phase5(snaps):
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    n_rep = np.array([s['n_replisomes'] for s in snaps])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.step(times, n_rep, where='post', color='#f59e0b', lw=1.6)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Active replisomes')
    ax.set_title('Current: replisome activity is not coupled to DnaA-ATP hydrolysis')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'replisome activity (current)')


def _after_phase5(snaps):
    """Phase 5 'After' = the RIDA-driven dynamics: DnaA-ATP fraction
    leaving the equilibrium ceiling, twin-axis with RIDA flux."""
    pulled = _dnaA_pool_traces(snaps)
    if pulled is None:
        return _placeholder('No DnaA pool counts in trajectory.')
    times, apo, atp, adp = pulled
    total = apo + atp + adp
    safe = np.where(total > 0, total, 1)
    atp_frac = atp / safe

    # _dnaA_pool_traces drops the first snapshot; align other arrays.
    aligned = snaps[1:] if len(snaps) > 1 else snaps
    flux = np.array([s.get('rida_flux_atp_to_adp') or 0 for s in aligned])
    n_rep = np.array([s.get('n_replisomes') or 0 for s in aligned])

    eq = DNAA_NUCLEOTIDE_EQUILIBRIUM
    band_min = eq.biological_atp_fraction_min.value
    band_max = eq.biological_atp_fraction_max.value

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.fill_between(
        [times.min(), times.max()], band_min, band_max,
        color='#bfdbfe', alpha=0.5,
        label=f'literature band ({band_min:.0%}–{band_max:.0%})')
    ax.plot(times, atp_frac, color='#16a34a', lw=2.0,
            label='observed DnaA-ATP fraction')
    ax.set_ylabel('DnaA-ATP / total DnaA')
    ax.set_ylim(0, 1.05)
    ax.set_title('After Phase 5: DnaA-ATP fraction crosses into the literature band')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(times, flux, width=0.5, color='#dc2626', alpha=0.7,
           label='RIDA flux (DnaA-ATP → DnaA-ADP / step)')
    ax2.step(times, n_rep, where='post', color='#f59e0b', lw=1.4,
             label='active replisomes')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('RIDA flux (molecules / step)')
    ax2.set_ylabel('Active replisomes')
    ax.set_title('RIDA flux scales with active replisome count')
    ax.legend(fontsize=8, loc='upper left')
    ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    final_frac = float(atp_frac[-1]) if len(atp_frac) else 0.0
    note = (
        f'<p class="note">Without DARS (Phase 7) the cycle is open-loop: '
        f'RIDA monotonically depletes DnaA-ATP, ending the run at '
        f'<strong>{final_frac:.0%}</strong>. The dip into the literature '
        f'band is real but transient; Phase 7 closes the cycle by '
        f'regenerating DnaA-ATP from DnaA-ADP via DARS1/2.</p>'
    )
    return _img(fig_to_b64(fig), 'RIDA-driven DnaA-ATP dynamics') + note


def _phase6_panel(snaps, title, show_ddah=True):
    """Phase 6 view: DnaA-ATP fraction trace + RIDA / DDAH fluxes.
    With DDAH, the ATP fraction sits a bit lower than without it
    (an additional drain on top of RIDA)."""
    pulled = _dnaA_pool_traces(snaps)
    if pulled is None:
        return _placeholder('No DnaA pool data in trajectory.')
    times, apo, atp, adp = pulled
    total = apo + atp + adp
    safe = np.where(total > 0, total, 1)
    atp_frac = atp / safe

    aligned = snaps[1:] if len(snaps) > 1 else snaps
    rida_flux = np.array([s.get('rida_flux_atp_to_adp') or 0 for s in aligned])
    ddah_flux = np.array([s.get('ddah_flux_atp_to_adp') or 0 for s in aligned])

    eq = DNAA_NUCLEOTIDE_EQUILIBRIUM
    band_min = eq.biological_atp_fraction_min.value
    band_max = eq.biological_atp_fraction_max.value

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.fill_between(
        [times.min(), times.max()], band_min, band_max,
        color='#bfdbfe', alpha=0.5,
        label=f'literature band ({band_min:.0%}–{band_max:.0%})')
    ax.plot(times, atp_frac, color='#16a34a', lw=2.2,
            label='DnaA-ATP fraction')
    ax.set_ylabel('DnaA-ATP / total')
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    width = 0.4
    ax.bar(times - width / 2, rida_flux, width=width,
           color='#dc2626', alpha=0.75,
           label='RIDA flux')
    if show_ddah:
        ax.bar(times + width / 2, ddah_flux, width=width,
               color='#7c3aed', alpha=0.75,
               label='DDAH flux')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Hydrolysis flux (molecules / step)')
    ax.set_title('DnaA-ATP hydrolysis flux contributors')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if show_ddah:
        cumulative_ddah = int(ddah_flux.sum())
        cumulative_rida = int(rida_flux.sum())
        note = (
            f'<p class="note">Over this trajectory, RIDA contributed '
            f'<strong>{cumulative_rida}</strong> hydrolysis events vs '
            f'DDAH\'s <strong>{cumulative_ddah}</strong>. RIDA dominates '
            f'while replisomes are active; DDAH provides a steady '
            f'background drain that runs even between replication '
            f'rounds.</p>'
        )
    else:
        note = (
            '<p class="note">Without DDAH, RIDA is the only DnaA-ATP '
            'hydrolyzer. The ATP fraction sits closer to the upper '
            'edge of the literature band because nothing pulls it '
            'down between replication pulses.</p>'
        )
    return _img(fig_to_b64(fig), 'Phase 6 panel') + note


def _before_phase6(snaps):
    """Phase 6 'Before' = pre_ddah: gate + SeqA, no DDAH."""
    return _phase6_panel(
        snaps,
        title='Without DDAH: RIDA alone hydrolyzes DnaA-ATP',
        show_ddah=False)


def _after_phase6(snaps):
    """Phase 6 'After' = full: gate + SeqA + DDAH."""
    return _phase6_panel(
        snaps,
        title='With DDAH: a constant background drain alongside RIDA',
        show_ddah=True)


PHASE6_ARCH_CHANGES = (
    ArchChange(
        kind='new_process',
        summary='DDAH Step',
        file='v2ecoli/processes/ddah.py',
        detail=('A constitutive first-order DnaA-ATP -> DnaA-ADP '
                'hydrolyzer modeling the catalytic effect of the '
                'datA-IHF complex. Rate = k_ddah * dnaA_atp_count, '
                'Poisson-drawn each tick. Default rate is ~10x slower '
                'than RIDA per replisome so DDAH stays a backup.'),
        reads=(
            'bulk[MONOMER0-160[c]]',
            'bulk[MONOMER0-4565[c]]',
            'timestep',
        ),
        writes=(
            'bulk[MONOMER0-160[c]]',
            'bulk[MONOMER0-4565[c]]',
            'listeners.ddah.flux_atp_to_adp',
            'listeners.ddah.rate_constant',
        ),
    ),
    ArchChange(
        kind='listener',
        summary='New listeners.ddah.* fields',
        file='v2ecoli/processes/ddah.py',
        detail='Per-tick DDAH hydrolysis flux + configured rate constant.',
        writes=(
            'listeners.ddah.flux_atp_to_adp',
            'listeners.ddah.rate_constant',
        ),
    ),
    ArchChange(
        kind='override',
        summary='enable_ddah flag (default True)',
        file='v2ecoli/generate_replication_initiation.py',
        detail=('build_replication_initiation_document gains '
                '`enable_ddah` (default True). When True, '
                '`_splice_ddah` adds the step to flow_order alongside '
                'RIDA / DARS / box binding.'),
    ),
)


def _phase6_extras(snaps, status):
    arch = _render_arch_changes_html(PHASE6_ARCH_CHANGES)
    note = (
        '<p class="note"><strong>What\'s deferred for the first cut:</strong> '
        'DDAH currently fires constitutively — it does not gate on '
        'IHF binding at datA, and the bioinformatic strict-consensus '
        'search finds 0 boxes in the datA window so per-box occupancy '
        'is not used. Adding datA region coordinates to '
        '<code>motif_coordinates</code> and gating on the IHF '
        'heterodimer count are the natural follow-ups.</p>'
    )
    return [
        ('Architecture changes — what this phase adds, modifies, overrides',
         note + arch),
    ]


def _before_phase7(snaps):
    """Phase 7 'Before' = the open-loop RIDA-only state, where
    DnaA-ATP monotonically depletes because nothing regenerates it
    from DnaA-ADP. Pulls from the post-Phase-5 trajectory if
    available — the trajectory now includes both rida and dars flux,
    so the 'before' plot is illustrative rather than measured."""
    pulled = _dnaA_pool_traces(snaps)
    if pulled is None:
        return _placeholder('No trajectory data — re-run with --duration N.')
    times, apo, atp, adp = pulled
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(times, atp, color='#16a34a', lw=1.6, label='DnaA-ATP')
    ax.plot(times, adp, color='#dc2626', lw=1.6, label='DnaA-ADP')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Bulk count')
    ax.set_title('Without DARS: DnaA-ATP would deplete monotonically')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    note = (
        '<p class="note">Phase 5 alone (RIDA without DARS) drains '
        'DnaA-ATP toward zero over the cell cycle because there is no '
        'reverse path. The "After" panel shows what DARS does — flux '
        'from DnaA-ADP back into apo-DnaA, then ATP-loading via the '
        'still-active equilibrium reaction.</p>'
    )
    return _img(fig_to_b64(fig), 'pre-DARS dynamics') + note


def _after_phase7(snaps):
    """Phase 7 'After' = the RIDA + DARS steady state. DnaA-ATP
    fraction lives inside the literature band; both flux traces
    settle into a balanced cycle."""
    pulled = _dnaA_pool_traces(snaps)
    if pulled is None:
        return _placeholder('No trajectory data.')
    times, apo, atp, adp = pulled
    total = apo + atp + adp
    safe = np.where(total > 0, total, 1)
    atp_frac = atp / safe
    aligned = snaps[1:] if len(snaps) > 1 else snaps
    rida_flux = np.array([s.get('rida_flux_atp_to_adp') or 0 for s in aligned])
    dars_flux = np.array([s.get('dars_flux_adp_to_apo') or 0 for s in aligned])

    eq = DNAA_NUCLEOTIDE_EQUILIBRIUM
    band_min = eq.biological_atp_fraction_min.value
    band_max = eq.biological_atp_fraction_max.value

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    ax.fill_between(
        [times.min(), times.max()], band_min, band_max,
        color='#bfdbfe', alpha=0.5,
        label=f'literature band ({band_min:.0%}–{band_max:.0%})')
    ax.plot(times, atp_frac, color='#16a34a', lw=2.0,
            label='observed DnaA-ATP fraction')
    ax.set_ylabel('DnaA-ATP / total DnaA')
    ax.set_ylim(0, 1.05)
    ax.set_title('After Phase 7: DnaA-ATP fraction stable inside the band')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.bar(times - 0.2, rida_flux, width=0.4, color='#dc2626',
           alpha=0.7, label='RIDA flux (ATP → ADP)')
    ax.bar(times + 0.2, dars_flux, width=0.4, color='#1d4ed8',
           alpha=0.7, label='DARS flux (ADP → apo)')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Flux (molecules / step)')
    ax.set_title('Balanced cycle: RIDA out ≈ DARS in')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    final_frac = float(atp_frac[-1]) if len(atp_frac) else 0.0
    note = (
        f'<p class="note">Cycle closed. Final DnaA-ATP fraction '
        f'<strong>{final_frac:.0%}</strong>. The two fluxes balance at '
        f'steady state, holding the ATP-fraction inside the band — '
        f'the biology that Phase 1 was diagnosing as missing.</p>'
    )
    return _img(fig_to_b64(fig), 'RIDA + DARS steady-state cycle') + note


def _before_phase8(snaps):
    return _placeholder(
        'dnaA transcription rate is constant (no occupancy feedback). After '
        'Phase 8, this panel plots dnaA mRNA against DnaA-ATP/ADP occupancy '
        'at p1/p2 — the closed feedback loop.')


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    number: int
    title: str
    goal: str
    status_check: Callable
    tests: tuple[TestSpec, ...]
    gap_items: tuple[str, ...]
    viz_plan: str
    before_plot: Callable
    after_description: str
    after_plot: Optional[Callable] = None  # filled in once phase lands
    extra_sections: Optional[Callable] = None
    """Optional callable returning ``[(heading, html), ...]`` — extra
    subsections rendered between the gap list and the before/after
    grid. Used by Phase 1 + 5 + 7 for driver tables, provenance audits,
    and bigraph subsets."""

    # Trajectory configurations to plug into before / after panels.
    # Three configurations are run during a report regeneration:
    #   'baseline'  — unmodified baseline architecture
    #   'rida_only' — baseline + RIDA, no DARS (the cumulative state
    #                  immediately after Phase 5 ships, before Phase 7)
    #   'full'      — current replication_initiation (RIDA + DARS)
    # Phases that don't compare cumulative slices default to 'full' for
    # both panels — the before_plot / after_plot callable still controls
    # what gets rendered.
    before_config: str = 'full'
    after_config: str = 'full'

    @property
    def slug(self) -> str:
        return f'phase-{self.number}'


PHASES: list[Phase] = [
    Phase(
        number=0,
        title='Region classifier — partition the 307 consensus boxes',
        goal=('The bioinformatic strict-consensus search at '
              '`flat/sequence_motifs.tsv` finds ~307 distinct DnaA boxes '
              '(TTWTNCACA variants) across the chromosome. Phase 0 adds '
              '`REGION_BOUNDARIES_ABS` and `region_for_coord(bp)` to '
              'partition them into oriC / dnaA_promoter / datA / DARS1 / '
              'DARS2 / `other`. No schema change — region is derived from '
              'each box\'s existing `coordinates` field.'),
        status_check=_check_phase0,
        tests=(
            TestSpec(
                'tests/test_replication_initiation_reference.py',
                'Locks in the curated PDF facts (counts, sequences, affinity '
                'classes, motif compatibility) as data assertions.'),
            TestSpec(
                'tests/test_dnaA_box_regions.py',
                'Apply region_for_coord over the actual init-state DnaA-box '
                'coordinates from sim_data and assert the count per region '
                'matches the curated reference (oriC=11, dnaA promoter=7, '
                'datA=4, DARS1=3, DARS2=5).'),
        ),
        gap_items=(
            'Of ~307 strict-consensus boxes total, only ~8 fall in the '
            'five named regulatory regions; the remaining ~99% are in '
            '`other` (genomic background). These background boxes are '
            'where most DnaA titration happens.',
            'Per-box affinity differentiation (R1/R2/R4 vs the named '
            'low-affinity oriC sites) is not yet supported — the '
            'strict-consensus search picks out high-affinity boxes only. '
            'Enriching `motif_coordinates` with named non-consensus boxes '
            'is a follow-up.',
        ),
        viz_plan=(
            'Bar chart of DnaA-box counts per region, with the curated '
            'reference counts overlaid as horizontal markers. Highlights '
            'any gap between what the motif search returns and what the '
            'literature says should be there.'),
        before_plot=_before_phase0,
        after_description=(
            'After Phase 0: live bar chart of DnaA-box counts per region '
            'computed from sim_data.process.replication.motif_coordinates '
            'via region_for_coord — the same lookup the binding process '
            'will use.'),
        after_plot=_after_phase0,
        extra_sections=lambda snaps, status: [(
            'Where the named regions sit on the chromosome',
            (_chromosome_diagram_static() +
             '<p class="note">A schematic view of the E. coli chromosome '
             '(~4.64 Mb circle) with oriC at 12 o\'clock, ter at 6, and '
             'each named region marked at its absolute bp position. The '
             'arc between any two regions is to scale, so the sparsity '
             'of named regions vs. the genomic background (~99% of the '
             '307 strict-consensus boxes) is visible at a glance.</p>'),
        )],
    ),
    Phase(
        number=1,
        title='Expose the DnaA nucleotide-state pools',
        goal=('Wire a listener that emits apo / DnaA-ATP / DnaA-ADP pool '
              'counts every step. Pool dynamics were already running '
              'underneath (the equilibrium reactions for both '
              'nucleotide-bound forms have been firing in the cache '
              'since the start) but were invisible from the trajectory; '
              'Phase 1 surfaces them so the rest of the work can be '
              'measured against the literature steady-state band.'),
        status_check=_check_phase1,
        tests=(
            TestSpec(
                'tests/test_dnaA_nucleotide_pool.py',
                'After a short sim, apo-DnaA drains into DnaA-ATP via '
                '`MONOMER0-160_RXN`. DnaA-ADP stays near zero in the '
                'absence of RIDA flux. The listener exposes all three '
                'pool counts in `listeners.replication_data` and the '
                'values match direct bulk reads. Total DnaA pool is '
                'conserved modulo translation / degradation.'),
        ),
        gap_items=(
            'Phase 2 (DnaA-box binding) is wired as a listener-only '
            'process today — bound boxes are counted but not yet '
            'subtracted from the free DnaA pool. The observed pool '
            'counts represent total DnaA, not free DnaA.',
        ),
        viz_plan=(
            'Before: observed DnaA pool counts and ATP-fraction trace, '
            'with the literature target band shaded. After: a schematic '
            'of the cell-cycle ATP-fraction pattern (trough → rise → '
            'peak) that Phase 5 + Phase 7 will produce in real data.'),
        before_plot=_before_phase1,
        after_description=(
            'Once the missing kinetic drivers (Phase 5 RIDA, Phase 7 '
            'DARS) land, this panel becomes a real measured trajectory '
            'with the cell-cycle ATP-fraction pattern instead of a '
            'schematic.'),
        after_plot=_after_phase1,
        extra_sections=_phase1_extras,
        before_config='baseline',
        after_config='full',
    ),
    Phase(
        number=2,
        title='DnaA-box binding + chromosomal titration',
        goal=('Add a DnaABoxBinding step that samples per-box equilibrium '
              'occupancy each tick from the DnaA-ATP/ADP bulk pools and '
              'per-region affinity classes. Emits per-region bound counts '
              '(including the ~99 background boxes in `other`, which '
              'sequester most of the DnaA pool — the chromosomal '
              'titration buffer described in the curated reference). '
              'Currently a listener: bound counts are reported but not '
              'yet subtracted from the free pool.'),
        status_check=_check_phase2,
        tests=(
            TestSpec(
                'tests/test_dnaA_binding.py',
                'Step is in cell_state. listener.dnaA_binding emits '
                'per-region counts. Total bound > 0. High-affinity '
                'regions (oriC, dnaA promoter, DARS) saturate; low-affinity '
                "'other' boxes are partially bound (Kd ~100 nM)."),
        ),
        gap_items=(
            '<strong>Titration is not yet wired into the dynamics.</strong> '
            'The listener counts ~250+ bound boxes at steady state but '
            'does not decrement the free DnaA-ATP / DnaA-ADP pool. '
            'Phases 3, 5, 7 see the un-buffered total pool, which '
            'inflates the available DnaA-ATP for initiation by ~5–10×. '
            'The follow-up is to move bound counts into the bulk delta '
            'each tick.',
            'No write-back to the `DnaA_bound` field on the unique-'
            'molecule store — the set-update / add-delete conflict with '
            'chromosome_structure is left as a follow-up.',
            'All named regions currently use the high-affinity rule. '
            'Per-box affinity differentiation (R1/R2/R4 vs named low-'
            'affinity oriC sites) and context-dependent binding (variants '
            'near high-affinity clusters acting as high-affinity sites '
            'via cooperativity) are follow-ups.',
        ),
        viz_plan='',
        before_plot=_before_phase2,
        after_description=(
            'After Phase 2: per-region bound-count traces from the '
            'binding listener.'),
        after_plot=_after_phase2,
        extra_sections=_phase2_extras,
        before_config='rida_dars',
        after_config='pre_gate',
    ),
    Phase(
        number=3,
        title='Replace mass-threshold initiation gate',
        goal=('Substitute `DnaAGatedChromosomeReplication` for the baseline '
              '`ChromosomeReplication`. The new gate fires when '
              '`DnaA-ATP / n_oriC >= threshold`, matching the structure '
              'of the mass-per-oriC gate (per-oriC division gives the '
              'same self-limiting feedback) but routed through the '
              'DnaA-ATP pool that Phases 5 + 7 set up.'),
        status_check=_check_phase3,
        tests=(
            TestSpec(
                'tests/test_initiation_dnaA_gate.py',
                'DnaAGatedChromosomeReplication is in cell_state at the '
                'baseline step name. Initiation fires at least once over '
                '30 min of sim. The gate self-limits: oriC count does '
                "not run away. enable_dnaA_gated_initiation=False falls "
                'back to baseline.'),
        ),
        gap_items=(
            'The threshold (60 DnaA-ATP / oriC) is calibrated against the '
            'un-buffered DnaA pool. Once Phase 2 wires titration so the '
            'gate sees the *free* pool only, this number will need to '
            'come down by an order of magnitude.',
            'A more refined gate would read the binding listener directly '
            'for the R1/R2/R4 high-affinity occupancy + the low-affinity '
            'DnaA-ATP filament configuration the curated reference '
            'describes, instead of the bulk-count proxy.',
        ),
        viz_plan='',
        before_plot=_before_phase3,
        after_description=(
            'After Phase 3: oriC count over time under the DnaA-gated '
            'initiation; the listener `critical_mass_per_oriC` field now '
            'reads as DnaA-ATP-per-oriC / threshold.'),
        after_plot=_after_phase3,
        extra_sections=lambda snaps, status: [(
            'Chromosome dynamics — initiation events in space',
            _chromosome_timeline_plot(
                snaps,
                title='Chromosome state across the trajectory '
                      '(post-Phase-3 DnaA gate)') +
            '<p class="note">Each disc shows the chromosome at one '
            'snapshot. Green dot = oriC, red square = ter, amber '
            'triangles outside the circle = active replisomes (forks). '
            'The named regulatory regions are marked around the rim. '
            'The bottom strip plots oriC + replisome counts vs time '
            'so you can see when initiations fired (red dashed lines '
            'mark step-ups in oriC count).</p>',
        )],
        before_config='pre_gate',
        after_config='gated_no_seqA',
    ),
    Phase(
        number=4,
        title='SeqA sequestration window after initiation',
        goal=('SeqA the protein (`EG12197-MONOMER`) is already expressed '
              '(~1029 copies in the init state); Phase 4 wires its '
              '*activity*. The DnaA-gated step records the time of each '
              'initiation event and forces its gate ratio to 0 for the '
              'next ~10 minutes — modeling SeqA binding to '
              'hemimethylated GATC sites at the newly-replicated origin '
              'and blocking DnaA rebinding during the window.'),
        status_check=_check_phase4,
        tests=(
            TestSpec(
                'tests/test_seqA_sequestration.py',
                'SeqA monomer is already in bulk (~1029 copies). The '
                'DnaA-gated step has seqA_sequestration_window_s = 600 '
                'with the default flag. After t=0 initiation, the gate '
                'ratio is 0 throughout the 600s window. At t=900s '
                '(window expired) the gate resumes reading the actual '
                'DnaA-ATP-per-oriC ratio. enable_seqA_sequestration='
                'False falls back to the unblocked gate.'),
        ),
        gap_items=(
            'No hemimethylated-GATC tracking on the oriC unique '
            'molecule — the refractory window is a coarser proxy for '
            'the SeqA-binding biology. Adding a hemimethylation field '
            'would conflict with chromosome_structure\'s in-tick '
            'add/delete (same set-update issue as Phase 2).',
            'No stoichiometric coupling to bulk SeqA. The window is a '
            'fixed-duration cap; depleted SeqA does not shorten it. '
            'A future refinement would scale the window by the bulk '
            'SeqA count.',
        ),
        viz_plan='',
        before_plot=_before_phase4,
        after_description=(
            'After Phase 4: gate ratio held at 0 for ~10 min after the '
            'initiation event (shaded amber); resumes normal DnaA-ATP-'
            'per-oriC reading once the window closes.'),
        after_plot=_after_phase4,
        extra_sections=_phase4_extras,
        before_config='gated_no_seqA',
        after_config='pre_ddah',
    ),
    Phase(
        number=5,
        title='RIDA — DnaA-ATP hydrolysis at the replisome',
        goal=('Add a dedicated `RIDA` Step that hydrolyzes DnaA-ATP to '
              'DnaA-ADP at a rate ∝ active replisome count, and '
              'deactivate the DnaA-ADP equilibrium reaction so RIDA\'s '
              'output accumulates instead of being instantly re-dissociated. '
              'Hda + Hda-β-clamp complex (CPLX0-10342) and the FBA-level '
              'reaction (RXN0-7444) were already in the data; Phase 5 wires '
              'the kinetic process and the equilibrium override.'),
        status_check=_check_phase5,
        tests=(
            TestSpec(
                'tests/test_rida.py',
                'RIDA Step is in the cell_state. The MONOMER0-4565_RXN '
                'equilibrium is deactivated. After 60s of sim, the '
                'DnaA-ATP fraction drops into the literature 30–70% band. '
                'DnaA-ADP rises monotonically while replisomes are active '
                '(open-loop until DARS in Phase 7).'),
        ),
        gap_items=(
            'Without DARS (Phase 7), the cycle is open-loop: DnaA-ATP '
            'monotonically depletes during a sim. The dip into the '
            'literature band is transient.',
            'The rate constant `rate_per_replisome_per_s = 0.005` is a '
            'first-pass tuning parameter; should be re-verified against '
            'measured DnaA-ATP cell-cycle dynamics.',
        ),
        viz_plan=(
            'Two panels. Top: observed DnaA-ATP fraction with literature '
            'band. Bottom: RIDA flux per timestep on left axis, active '
            'replisome count on right. The flux scales with replisomes; '
            'the fraction crosses into the band within ~60 s.'),
        before_plot=_before_phase5,
        after_description=(
            'After Phase 5: live two-panel plot showing DnaA-ATP fraction '
            'crossing into the literature band, with RIDA flux scaling on '
            'replisome count.'),
        after_plot=_after_phase5,
        extra_sections=_phase5_extras,
        before_config='baseline',
        after_config='rida_only',
    ),
    Phase(
        number=6,
        title='DDAH — backup DnaA-ATP hydrolysis at datA',
        goal=('Add a DDAH Step that hydrolyzes DnaA-ATP at a small '
              'constitutive rate, modeling the catalytic effect of the '
              'datA-IHF complex described in the curated reference. '
              'Pairs with Phase 5 (RIDA): RIDA dominates while replisomes '
              'are active, DDAH provides a steady background drain that '
              'runs even between replication rounds.'),
        status_check=_check_phase6,
        tests=(
            TestSpec(
                'tests/test_ddah.py',
                'DDAH step in cell_state. listener.ddah emits a non-zero '
                'rate constant. Cumulative flux > 0 over a 20-min window. '
                'enable_ddah=False removes the step.'),
        ),
        gap_items=(
            'datA region coordinates not yet in `motif_coordinates` — the '
            'strict-consensus motif search finds 0 boxes in the datA '
            'window. Per-box occupancy is therefore not used; DDAH fires '
            'at a constant rate independent of datA box count.',
            'IHF heterodimer count is not yet read by DDAH. The biology '
            'gates DDAH on IHF binding at the datA IBS; we approximate '
            'with a constant rate instead.',
            'Rate constant (`rate_per_s = 0.0005`) is a first-pass tune. '
            'With it, the ATP fraction can dip slightly below the lower '
            'edge of the literature band — would need recalibration once '
            'the IHF gating is wired.',
        ),
        viz_plan='',
        before_plot=_before_phase6,
        after_description=(
            'After Phase 6: ATP-fraction trace + side-by-side RIDA / DDAH '
            'flux bars showing both hydrolyzers contributing.'),
        after_plot=_after_phase6,
        extra_sections=_phase6_extras,
        before_config='pre_ddah',
        after_config='full',
    ),
    Phase(
        number=7,
        title='DARS reactivation — closes the DnaA cycle',
        goal=('Add a dedicated `DARS` Step that releases ADP from DnaA-ADP '
              'at a first-order rate, regenerating apo-DnaA. Combined with '
              'the still-active `MONOMER0-160_RXN` equilibrium that '
              're-loads apo-DnaA with cellular ATP, this closes the loop '
              'opened in Phase 5. Steady-state DnaA-ATP fraction lands '
              'inside the literature 30–70% band. Per-locus DARS1 vs DARS2 '
              'and IHF/Fis modulation are deferred follow-ups.'),
        status_check=_check_phase7,
        tests=(
            TestSpec(
                'tests/test_dars.py',
                'DARS Step is in cell_state. After 5 min of sim, the '
                'DnaA-ATP fraction sits inside the literature band and '
                'does not drop sharply from its t=60s value. DnaA-ADP no '
                'longer monotonically grows — the cycle has closed. The '
                'dars listener emits a non-zero rate constant.'),
        ),
        gap_items=(
            'DARS1 and DARS2 are not yet differentiated — a single rate '
            'constant covers both loci.',
            'IHF and Fis modulation of DARS2 (cell-cycle gating) is not '
            'yet wired; the rate is constant in time.',
            'DARS region coordinates are not loaded into '
            'motif_coordinates as their own motif type — the existing '
            'EcoCyc dna_sites entries are recognized only via the '
            'region classifier from Phase 0.',
        ),
        viz_plan=(
            'Two panels. Top: DnaA-ATP fraction stable inside the band. '
            'Bottom: side-by-side bars for the RIDA flux (out) and DARS '
            'flux (in), showing the balanced cycle.'),
        before_plot=_before_phase7,
        after_description=(
            'After Phase 7: balanced flux cycle and stable in-band ATP '
            'fraction. Per-locus DARS1 / DARS2 split + IHF/Fis modulation '
            'come later.'),
        after_plot=_after_phase7,
        extra_sections=_phase7_extras,
        before_config='rida_only',
        after_config='rida_dars',
    ),
    Phase(
        number=8,
        title='dnaA promoter autoregulation',
        goal=('transcript_initiation reads DnaA-ATP/ADP occupancy at the p1/p2 '
              'boxes and modulates dnaA transcription rate.'),
        status_check=_check_phase8,
        tests=(
            TestSpec(
                'tests/test_dnaA_autoregulation.py',
                'dnaA mRNA level decreases when DnaA-ATP occupancy at p1/p2 '
                'boxes is elevated and recovers when the DnaA pool drops; '
                'p2/p1 promoter activity ratio ≈ 3 at low DnaA occupancy.'),
        ),
        gap_items=(
            'dnaA gene is transcribed via generic `transcript_initiation`; '
            'no DnaA-occupancy feedback on its own promoter.',
        ),
        viz_plan=(
            'Twin-axis plot: dnaA mRNA level (left axis) and DnaA-ATP+ADP '
            'occupancy at p1/p2 boxes (right axis). The two should be '
            'anti-correlated, closing the autoregulatory loop.'),
        before_plot=_before_phase8,
        after_description=(
            'After Phase 8: anti-correlated dnaA mRNA and p1/p2 occupancy '
            'traces; p2/p1 ratio panel.'),
    ),
]


# ---------------------------------------------------------------------------
# Narrative: phases grouped into four Parts
#
# Phase numbers track implementation order (and so commits / tests).
# The Parts impose a *narrative* order on top: a reader following the
# story from the top of the report goes through Foundations → the
# nucleotide-state cycle → cytoplasm-to-chromosome → regulatory
# feedbacks. Within a Part the phases are re-ordered to read in
# dependency order, not by number.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Part:
    number: int
    title: str
    intro: str
    phase_numbers: tuple[int, ...]

    @property
    def slug(self) -> str:
        return f'part-{self.number}'


PARTS: tuple[Part, ...] = (
    Part(
        number=1,
        title='Part I — Foundations: codify what\'s already there',
        intro=(
            'Before adding any new biology, codify the curated reference '
            'and surface the relevant pools. The bioinformatic motif '
            'search at <code>flat/sequence_motifs.tsv</code> already '
            'finds ~307 strict-consensus DnaA boxes (TTWTNCACA) across '
            'the chromosome — far more than the 30 named in the curated '
            'PDF. Phase 0 partitions these into the named regulatory '
            'regions plus an "other" bucket for the genomic background; '
            'Phase 1 wires a listener so the apo / DnaA-ATP / DnaA-ADP '
            'pool counts are visible in the trajectory.'
        ),
        phase_numbers=(0, 1),
    ),
    Part(
        number=2,
        title='Part II — Nucleotide-state cycle: drive DnaA-ATP / DnaA-ADP dynamics',
        intro=(
            'The DnaA nucleotide state is the central control variable. '
            'RIDA pulls DnaA-ATP into DnaA-ADP during ongoing replication '
            '(coupled to the DNA-loaded β-clamp + Hda complex); DARS '
            'pulls DnaA-ADP back into the apo form, which is then re-'
            'loaded with cellular ATP via the existing equilibrium '
            'reaction. Together they form a closed cycle whose steady-'
            'state ATP-fraction lands in the literature-observed '
            '30–70% band.'
        ),
        phase_numbers=(5, 7),
    ),
    Part(
        number=3,
        title='Part III — Cytoplasm to chromosome: bind, titrate, gate',
        intro=(
            'DnaA acts on initiation by binding to chromosomal sites at '
            'oriC. Phase 2 wires an equilibrium-occupancy binding model '
            'over the ~307 consensus boxes; the genomic-background '
            'boxes ("other") sequester the bulk of the DnaA pool '
            '(<strong>titration</strong>). Phase 3 replaces the baseline '
            'mass-threshold initiation gate with a DnaA-ATP-per-oriC '
            'gate — same per-oriC self-limiting feedback as the mass '
            'gate, but now driven by the DnaA biology that Part II '
            'sets up. Note: the binding listener counts bound boxes '
            'today but does not yet decrement the free DnaA pool — '
            'the titration force is visible only in the listener, not '
            'in downstream dynamics. Adding that decrement is a '
            'follow-up.'
        ),
        phase_numbers=(2, 3),
    ),
    Part(
        number=4,
        title='Part IV — Regulatory feedbacks: refine the timing',
        intro=(
            'Three regulatory layers refine the cell-cycle timing: '
            'SeqA sequesters newly-replicated origins for ~10 min '
            '(blocking premature reinitiation), DDAH provides backup '
            'DnaA-ATP hydrolysis at the datA locus (with IHF as the '
            'gating cofactor), and the dnaA gene autoregulates its '
            'own expression via DnaA binding at the boxes between p1 '
            'and p2. None of these are wired yet.'
        ),
        phase_numbers=(4, 6, 8),
    ),
)


def _phase_by_number(n: int) -> Phase:
    for p in PHASES:
        if p.number == n:
            return p
    raise KeyError(n)


# ---------------------------------------------------------------------------
# HTML rendering — sidebar + main content
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    'done':        ('#16a34a', 'done'),
    'in_progress': ('#f59e0b', 'in progress'),
    'pending':     ('#94a3b8', 'pending'),
}


def _status_pill(status):
    color, label = _STATUS_COLORS.get(status, ('#475569', status))
    return (f'<span class="pill" style="background:{color};">'
            f'{html_lib.escape(label)}</span>')


def _render_oric_table():
    rows = []
    for b in ORIC.dnaA_boxes:
        seq = b.sequence or '—'
        rows.append(
            f'<tr><td>{html_lib.escape(b.name)}</td>'
            f'<td>{html_lib.escape(b.affinity_class)}</td>'
            f'<td><code>{html_lib.escape(seq)}</code></td>'
            f'<td>{html_lib.escape(b.nucleotide_preference)}</td></tr>'
        )
    return ('<table class="ref"><thead><tr><th>Box</th><th>Class</th>'
            '<th>Sequence</th><th>Nucleotide preference</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def _render_promoter_table():
    rows = []
    for b in DNAA_PROMOTER.dnaA_boxes:
        seq = b.sequence or '—'
        rows.append(
            f'<tr><td>{html_lib.escape(b.name)}</td>'
            f'<td>{html_lib.escape(b.affinity_class)}</td>'
            f'<td><code>{html_lib.escape(seq)}</code></td>'
            f'<td>{html_lib.escape(b.nucleotide_preference)}</td></tr>'
        )
    return ('<table class="ref"><thead><tr><th>Box</th><th>Class</th>'
            '<th>Sequence</th><th>Nucleotide preference</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def _render_dars_table():
    rows = []
    for d in (DARS1, DARS2):
        rows.append(
            f'<tr><td>{html_lib.escape(d.name)}</td>'
            f'<td>{d.length_bp}</td>'
            f'<td>{", ".join(d.core_box_names)}</td>'
            f'<td>{", ".join(d.extra_box_names) or "—"}</td>'
            f'<td>{d.n_ihf_sites}</td>'
            f'<td>{d.n_fis_sites}</td>'
            f'<td>{"yes" if d.is_dominant_in_vivo else "no"}</td></tr>'
        )
    return ('<table class="ref"><thead><tr><th>Locus</th><th>bp</th>'
            '<th>Core boxes</th><th>Extra boxes</th>'
            '<th>IHF sites</th><th>Fis sites</th>'
            '<th>Dominant</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


_CONFIG_LABELS = {
    'baseline':       'baseline (no extras; mass-threshold initiation gate)',
    'rida_only':      '+ RIDA (no DARS, no binding; mass gate)',
    'rida_dars':      '+ RIDA + DARS (no binding; mass gate)',
    'pre_gate':       '+ RIDA + DARS + binding (mass gate still active)',
    'gated_no_seqA':  'DnaA-gated initiation (no SeqA sequestration)',
    'pre_ddah':       'gate + SeqA, no DDAH backup hydrolysis',
    'full':           'replication_initiation (gate + SeqA + DDAH)',
}


def _render_phase_section(phase: Phase, statuses, trajectories):
    status, evidence = statuses[phase.number]
    trajectories = trajectories or {}
    test_rows = ''.join(_render_test_li(t) for t in phase.tests)
    gap_rows = ''.join(
        f'<li>{html_lib.escape(g)}</li>' for g in phase.gap_items
    )

    full_snaps = trajectories.get('full', []) or []
    before_snaps = trajectories.get(phase.before_config) or full_snaps
    after_snaps = trajectories.get(phase.after_config) or full_snaps

    before_block = phase.before_plot(before_snaps)
    if phase.after_plot is not None:
        after_block = phase.after_plot(after_snaps)
    else:
        after_block = _placeholder(phase.after_description)

    extras_html = ''
    if phase.extra_sections is not None:
        # Extras get the 'full' (current) trajectory — they're meant to
        # describe the architecture as it stands now.
        for heading, body in phase.extra_sections(full_snaps, status):
            extras_html += f'<h3>{html_lib.escape(heading)}</h3>{body}'

    if phase.before_config == phase.after_config:
        before_label = 'Before — current model behavior'
        after_label = (
            f'After — Phase {phase.number} lands'
            if phase.after_plot is None
            else f'After — Phase {phase.number} (current state)')
    else:
        before_desc = _CONFIG_LABELS.get(phase.before_config, phase.before_config)
        after_desc = _CONFIG_LABELS.get(phase.after_config, phase.after_config)
        before_label = f'Before — {before_desc}'
        after_label = f'After — {after_desc}'

    return f"""
<section id="{phase.slug}">
  <div class="phase-header">
    <h2>Phase {phase.number} — {html_lib.escape(phase.title)}</h2>
    {_status_pill(status)}
  </div>
  <p class="goal">{html_lib.escape(phase.goal)}</p>
  <p class="evidence">Status check: <em>{html_lib.escape(evidence)}</em></p>
  <h3>Tests <span class="hint">(hover for descriptions)</span></h3>
  <ul class="tests">{test_rows}</ul>
  <h3>Mechanisms still missing</h3>
  <ul class="gaps">{gap_rows}</ul>
  {extras_html}
  <div class="before-after">
    <div class="ba-col">
      <h4>{html_lib.escape(before_label)}</h4>
      {before_block}
    </div>
    <div class="ba-col">
      <h4>{html_lib.escape(after_label)}</h4>
      {after_block}
    </div>
  </div>
</section>
"""


def _render_overview_table(statuses):
    """Overview grouped by Part. Phases listed in narrative order
    (the order they appear in PARTS), not by phase number."""
    out = ['<table class="overview"><thead>'
           '<tr><th>#</th><th>Phase</th><th>Status</th>'
           '<th>What it does</th></tr></thead><tbody>']
    for part in PARTS:
        out.append(
            f'<tr class="part-row"><td colspan="4">'
            f'<a href="#{part.slug}"><strong>{html_lib.escape(part.title)}</strong></a>'
            f'</td></tr>')
        for n in part.phase_numbers:
            phase = _phase_by_number(n)
            status, _ = statuses[phase.number]
            out.append(
                f'<tr><td>P{phase.number}</td>'
                f'<td><a href="#{phase.slug}">{html_lib.escape(phase.title)}</a></td>'
                f'<td>{_status_pill(status)}</td>'
                f'<td>{html_lib.escape(phase.goal)}</td></tr>'
            )
    out.append('</tbody></table>')
    return ''.join(out)


def _render_sidebar(statuses):
    out = ['<aside class="sidebar">'
           '<h3>Navigation</h3>'
           '<a href="#overview" class="nav-link">Overview</a>']
    for part in PARTS:
        out.append(f'<h3>{html_lib.escape(part.title)}</h3>')
        out.append('<div class="phase-list">')
        for n in part.phase_numbers:
            phase = _phase_by_number(n)
            status, _ = statuses[phase.number]
            color = _STATUS_COLORS[status][0]
            out.append(
                f'<a href="#{phase.slug}" class="phase-link">'
                f'<span class="dot" style="background:{color};"></span>'
                f'<span class="num">P{phase.number}</span> '
                f'<span class="ttl">{html_lib.escape(phase.title)}</span></a>')
        out.append('</div>')
    out.append(
        '<h3>Reference</h3>'
        '<a href="#ref-oriC" class="nav-link">oriC</a>'
        '<a href="#ref-promoter" class="nav-link">dnaA promoter</a>'
        '<a href="#ref-datA" class="nav-link">datA</a>'
        '<a href="#ref-dars" class="nav-link">DARS1 / DARS2</a>'
        '<a href="#ref-seqA-rida" class="nav-link">SeqA / RIDA</a>'
        '<a href="#ref-motifs" class="nav-link">Consensus motifs</a>'
        '<h3>Data</h3>'
        '<a href="#trajectory" class="nav-link">Trajectory plots</a>'
        '<a href="#references" class="nav-link">References</a>'
        '</aside>')
    return ''.join(out)


def _render_part_intro(part: Part) -> str:
    return (
        f'<section id="{part.slug}" class="part-intro">'
        f'<h2 class="part-title">{html_lib.escape(part.title)}</h2>'
        f'<p class="part-intro-text">{part.intro}</p>'
        f'</section>'
    )


def _ref_path(out_path, target):
    """Compute a relative path from the report's output dir to a file
    under ``docs/references/``. Works whether the report lives at
    ``out/reports/...`` (local) or at ``docs/...`` (published)."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    target_abs = os.path.join(REPO_ROOT, 'docs', 'references', target)
    return os.path.relpath(target_abs, out_dir)


def render_html(trajectories, sim_meta, out_path):
    """trajectories is a dict mapping config name -> list of snapshots.
    Expected keys: 'baseline', 'rida_only', 'full'. Missing keys fall
    back to 'full' so phases can still render their plots."""
    statuses = {p.number: p.status_check() for p in PHASES}
    n_done = sum(1 for s, _ in statuses.values() if s == 'done')
    n_in_progress = sum(1 for s, _ in statuses.values() if s == 'in_progress')
    n_total = len(PHASES)

    full_snaps = trajectories.get('full', []) or []

    pdf_link = _ref_path(out_path, 'replication_initiation_molecular_info.pdf')
    md_link = _ref_path(out_path, 'replication_initiation.md')

    try:
        from v2ecoli.generate_replication_initiation import ARCHITECTURE_NAME
    except Exception:
        ARCHITECTURE_NAME = '(import failed)'

    sidebar_html = _render_sidebar(statuses)
    overview_table = _render_overview_table(statuses)
    # Render Parts in narrative order: Part intro, then each phase in
    # the Part's declared order. Phase numbers in PHASES no longer
    # control display order — PARTS does.
    rendered_phase_numbers: set[int] = set()
    sections = []
    for part in PARTS:
        sections.append(_render_part_intro(part))
        for n in part.phase_numbers:
            phase = _phase_by_number(n)
            sections.append(_render_phase_section(phase, statuses, trajectories))
            rendered_phase_numbers.add(n)
    # Safety net: any Phase not assigned to a Part still gets rendered
    # at the bottom so nothing silently disappears if PARTS gets out
    # of sync with PHASES.
    for phase in PHASES:
        if phase.number not in rendered_phase_numbers:
            sections.append(_render_phase_section(phase, statuses, trajectories))
    phase_sections = '\n'.join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli — replication initiation report</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 0; background: #f8fafc; color: #0f172a; line-height: 1.5; }}
.layout {{ display: flex; min-height: 100vh; }}
.sidebar {{ width: 240px; background: #0f172a; color: #e2e8f0;
            padding: 20px 16px; position: sticky; top: 0;
            height: 100vh; overflow-y: auto; flex-shrink: 0; }}
.sidebar h3 {{ font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.04em;
               color: #94a3b8; margin: 18px 0 6px; }}
.sidebar a {{ display: block; color: #cbd5e1; text-decoration: none;
              font-size: 0.88em; padding: 4px 6px; border-radius: 4px; }}
.sidebar a:hover {{ background: #1e293b; color: #fff; }}
.phase-link {{ display: flex !important; align-items: baseline; gap: 6px; }}
.phase-link .dot {{ display: inline-block; width: 8px; height: 8px;
                    border-radius: 50%; flex-shrink: 0; }}
.phase-link .num {{ font-family: monospace; font-size: 0.78em; color: #64748b;
                    flex-shrink: 0; }}
.phase-link .ttl {{ flex: 1; }}
.main {{ flex: 1; padding: 28px 36px; max-width: 1200px; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 8px;
      margin-top: 0; }}
h2 {{ color: #1e3a8a; margin-top: 0; font-size: 1.2em; }}
h3 {{ color: #1e293b; font-size: 0.98em; margin-top: 1.2em;
      margin-bottom: 0.4em; }}
.banner {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
           padding: 10px 16px; font-size: 0.9em; }}
.kv {{ font-family: monospace; font-size: 0.9em; }}
section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px;
           padding: 18px 22px; margin: 16px 0; }}
.phase-header {{ display: flex; align-items: center; gap: 12px; }}
.phase-header h2 {{ margin: 0; }}
.pill {{ display: inline-block; color: white; border-radius: 12px;
         padding: 1px 10px; font-size: 0.78em; font-weight: 600; }}
.goal {{ color: #334155; font-size: 0.92em; margin-top: 8px; }}
.evidence {{ color: #475569; font-size: 0.85em; }}
.tests {{ list-style: none; padding-left: 0; margin: 4px 0 8px; }}
.tests li.test {{ font-family: monospace; font-size: 0.85em; padding: 2px 0;
                   position: relative; cursor: help; }}
.tests .mark {{ display: inline-block; width: 16px;
                color: #16a34a; font-weight: 700; }}
.tests li.missing .mark {{ color: #94a3b8; }}
.tests li.missing code {{ color: #64748b; }}
.tests .tooltip {{ display: none; position: absolute;
                    left: 24px; top: 100%; z-index: 20;
                    background: #0f172a; color: #e2e8f0;
                    border-radius: 6px; padding: 10px 14px;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 0.82em; line-height: 1.5;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
                    width: 520px; max-width: 80vw; }}
.tests li.test:hover .tooltip {{ display: block; }}
.tests .tooltip strong {{ color: #fff; display: inline-block;
                           margin-bottom: 4px; }}
.tests .tooltip strong.planned {{ color: #fbbf24; }}
.tests .tooltip .t-row {{ margin: 3px 0; }}
.tests .tooltip .t-row code {{ background: #1e293b; color: #93c5fd;
                                padding: 1px 5px; border-radius: 3px; }}
.tests .tooltip .t-row.note {{ color: #cbd5e1; font-style: italic; }}
.hint {{ font-size: 0.75em; font-weight: normal; color: #94a3b8;
         font-family: -apple-system, sans-serif; }}
.gaps {{ margin: 4px 0 8px; padding-left: 22px; font-size: 0.9em;
         color: #334155; }}
.gaps li {{ margin-bottom: 3px; }}
.viz-plan {{ font-size: 0.92em; color: #1e293b;
             background: #fef3c7; border-left: 3px solid #f59e0b;
             padding: 8px 14px; margin: 4px 0 10px;
             border-radius: 0 4px 4px 0; }}
table.drivers tr.driver-on td:first-child {{ color: #16a34a;
                                               font-weight: 700; }}
table.drivers tr.driver-off td:first-child {{ color: #94a3b8;
                                                font-weight: 700; }}
table.drivers tr.driver-off {{ background: #f8fafc; }}
table.drivers tr.driver-off td {{ color: #475569; }}
table.provenance .cite-text {{ font-size: 0.82em; color: #334155; }}
table.provenance .cite-note {{ font-size: 0.78em; color: #64748b;
                                font-style: italic; margin-top: 3px; }}
.cite-pdf {{ display: inline-block; background: #dcfce7;
             color: #166534; border-radius: 10px; padding: 1px 8px;
             font-size: 0.76em; font-weight: 600; }}
.cite-ext {{ display: inline-block; background: #fee2e2;
             color: #991b1b; border-radius: 10px; padding: 1px 8px;
             font-size: 0.76em; font-weight: 600; }}
table.arch td:first-child {{ white-space: nowrap; width: 90px; }}
table.arch .ports {{ font-size: 0.82em; color: #475569;
                      margin-top: 6px;
                      border-left: 2px solid #cbd5e1;
                      padding-left: 8px; }}
table.arch .ports code {{ background: #f1f5f9; }}
.bigraph-img {{ background: white; border: 1px solid #e2e8f0;
                border-radius: 6px; padding: 12px; margin: 6px 0;
                text-align: center; overflow-x: auto; }}
.bigraph-img img {{ max-width: 95%; height: auto;
                     display: block; margin: 0 auto; }}
.before-after {{ display: flex; flex-direction: column;
                  gap: 18px; margin-top: 10px; }}
.ba-col {{ background: #f8fafc; border: 1px solid #e2e8f0;
            border-radius: 6px; padding: 16px; }}
.ba-col:nth-child(1) {{ border-left: 4px solid #94a3b8; }}
.ba-col:nth-child(2) {{ border-left: 4px solid #16a34a; }}
.ba-col h4 {{ margin: 0 0 12px; font-size: 1.0em;
              color: #1e293b; text-transform: uppercase;
              letter-spacing: 0.05em;
              border-bottom: 1px solid #e2e8f0;
              padding-bottom: 6px; }}
.ba-col img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0;
                border-radius: 4px; background: white; display: block;
                margin: 0 auto; }}
.placeholder {{ background: #f1f5f9; border: 1px dashed #cbd5e1;
                border-radius: 6px; padding: 14px 18px; color: #475569;
                font-size: 0.95em; font-style: italic; }}
table {{ border-collapse: collapse; width: 100%;
         font-size: 0.88em; margin: 6px 0 12px; }}
table th, table td {{ border: 1px solid #cbd5e1; padding: 5px 9px;
                       text-align: left; vertical-align: top; }}
table thead {{ background: #e2e8f0; }}
table.overview td:first-child {{ width: 28px; text-align: center;
                                  font-family: monospace; }}
table.overview tr.part-row td {{ background: #1e3a8a; color: #fff;
                                  border: 1px solid #1e3a8a;
                                  padding: 6px 10px; }}
table.overview tr.part-row td a {{ color: #fff; text-decoration: none; }}
table.overview tr.part-row td a:hover {{ text-decoration: underline; }}
section.part-intro {{ background: #eff6ff; border: 1px solid #bfdbfe;
                       border-left: 4px solid #1e3a8a; }}
section.part-intro h2.part-title {{ color: #1e3a8a; margin-top: 0; }}
section.part-intro .part-intro-text {{ color: #1e293b; font-size: 0.95em;
                                        line-height: 1.55; margin-bottom: 0; }}
.sidebar h3 {{ font-size: 0.74em; }}
table code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
.note {{ font-size: 0.85em; color: #475569; }}
.refs {{ font-size: 0.85em; color: #334155; }}
.refs li {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="layout">
{sidebar_html}
<main class="main">

<h1>Replication-initiation report</h1>
<div class="banner">
<strong>Architecture:</strong> <span class="kv">{html_lib.escape(ARCHITECTURE_NAME)}</span>
&nbsp;|&nbsp; <strong>Snapshots:</strong> {len(full_snaps)}
&nbsp;|&nbsp; <strong>Phase progress:</strong>
{n_done}/{n_total} done, {n_in_progress} in progress
&nbsp;|&nbsp; <strong>Source:</strong>
<a href="{html_lib.escape(pdf_link)}">PDF</a>
&middot; <a href="{html_lib.escape(md_link)}">Markdown summary</a>
</div>

<section id="overview">
<h2>Overview</h2>
<p class="note">
The work is grouped into four <strong>Parts</strong>, each answering one
question: codify what's already there → drive the DnaA-ATP/ADP cycle →
bind to the chromosome and gate initiation → add regulatory feedbacks.
Phase numbers track implementation order; the table below renders them
in <em>narrative</em> order under each Part. Status pills are
auto-detected from the codebase (schema fields, molecule_ids entries,
process modules, sim_data attributes), so the indicators cannot drift
from reality.
</p>
{overview_table}
</section>

{phase_sections}

<section id="ref-oriC">
<h2>oriC</h2>
<p class="note">
{ORIC.length_bp} bp, {len(ORIC.dnaA_boxes)} DnaA boxes, {len(ORIC.ihf_sites)} IHF sites.
High-affinity boxes bind both DnaA-ATP and DnaA-ADP (Kd ~1 nM). Low-affinity
boxes prefer DnaA-ATP and bind cooperatively (Kd &gt; 100 nM). Ordered DnaA-ATP
loading on the right arm: {' &rarr; '.join(ORIC.ordered_oligomerization_right_arm)}.
</p>
{_render_oric_table()}
</section>

<section id="ref-promoter">
<h2>dnaA promoter</h2>
<p class="note">
{DNAA_PROMOTER.length_bp} bp; p2 ≈ {DNAA_PROMOTER.p2_to_p1_strength_ratio:g}× p1;
~{DNAA_PROMOTER.promoter_separation_bp} bp separation.
</p>
{_render_promoter_table()}
</section>

<section id="ref-datA">
<h2>datA</h2>
<p class="kv">
{DATA.length_bp} bp at {DATA.chromosomal_position_min} min &nbsp;|&nbsp;
DnaA boxes: {DATA.n_dnaA_boxes} &nbsp;|&nbsp;
essential: {', '.join(DATA.essential_dnaA_box_names)} &nbsp;|&nbsp;
stimulatory: {', '.join(DATA.stimulatory_dnaA_box_names)} &nbsp;|&nbsp;
IHF sites: {DATA.n_ihf_sites}
</p>
</section>

<section id="ref-dars">
<h2>DARS1 / DARS2</h2>
{_render_dars_table()}
</section>

<section id="ref-seqA-rida">
<h2>SeqA / RIDA</h2>
<p class="kv">
SeqA sequestration window: {SEQA.sequestration_window_minutes:g} min
(~{SEQA.fraction_of_doubling_time_at_rapid_growth:.2f} of doubling time at rapid growth).
SeqA binds {SEQA.binds_state} GATC sites (&gt;{SEQA.n_gatc_sites_oriC_lower_bound} per origin).
</p>
<p class="kv">
RIDA: {RIDA.clamp_protein} (β-clamp) + {RIDA.hda_nucleotide_state}-{RIDA.catalytic_partner}
(N-terminal clamp-binding motif) catalyzes <code>{html_lib.escape(RIDA.reaction)}</code>.
</p>
</section>

<section id="ref-motifs">
<h2>Consensus motifs</h2>
<p class="kv">
Consensus: <code>{DNAA_BOX_CONSENSUS}</code> (W = A|T, N = any).
Highest-affinity 9-mer: <code>{DNAA_BOX_HIGHEST_AFFINITY}</code> (Kd ~1 nM).
</p>
</section>

<section id="trajectory">
<h2>Trajectory under the new architecture</h2>
{('<p class="note">' + html_lib.escape(sim_meta) + '</p>') if sim_meta else ''}
{(_img(plot_initiation_signals(full_snaps), 'replication-initiation signals')
   if full_snaps else _placeholder('No simulation data — re-run with the ParCa cache present.'))}
{_img(plot_fork_positions(full_snaps), 'fork positions over time') if full_snaps else ''}
</section>

<section id="references">
<h2>Key references</h2>
<ul class="refs">
<li>Katayama, Kasho, Kawakami. <em>Frontiers in Microbiology</em> 8 (2017): 2496.</li>
<li>Kasho, Ozaki, Katayama. <em>Int. J. Mol. Sci.</em> 24 (2023): 11572.</li>
<li>Speck, Weigel, Messer. <em>EMBO J.</em> 18 (1999): 6169–6176.</li>
<li>Schaper &amp; Messer. <em>J. Biol. Chem.</em> 270 (1995): 17622–17626.</li>
<li>Hansen et al. <em>J. Mol. Biol.</em> 355 (2006): 85–95.</li>
<li>Olivi et al. <em>Nat. Commun.</em> 16 (2025): 7813.</li>
<li>Riber et al. <em>Front. Mol. Biosci.</em> 3 (2016): 29.</li>
<li>Hansen &amp; Atlung. <em>Front. Microbiol.</em> 9 (2018): 319.</li>
<li>Fujimitsu, Senriuchi, Katayama. <em>Genes &amp; Dev.</em> 23 (2009): 1221–1233.</li>
<li>Kasho et al. <em>Nucleic Acids Res.</em> 42 (2014): 13134–13149.</li>
</ul>
</section>

</main>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION)
    parser.add_argument('--no-sim', action='store_true')
    parser.add_argument('--force-resim', action='store_true',
                        help='ignore disk-cached trajectories and re-run sims')
    parser.add_argument('--out',
                        default=os.path.join(OUT_DIR, 'replication_initiation_report.html'))
    args = parser.parse_args()

    trajectories: dict[str, list] = {}
    sim_meta = ''
    if args.no_sim:
        sim_meta = 'Reference-only mode (--no-sim).'
    else:
        sim_meta = (f'Ran (or reused cached) single-cell sims for three '
                    f'cumulative configurations: baseline → +RIDA → +DARS '
                    f'(full). Each up to {args.duration:.0f}s.')
        configs = [
            ('baseline', _run_baseline_sim),
            ('rida_only', _run_rida_only_sim),
            ('rida_dars', _run_rida_dars_sim),
            ('pre_gate', _run_pre_gate_sim),
            ('gated_no_seqA', _run_gated_no_seqA_sim),
            ('pre_ddah', _run_pre_ddah_sim),
            ('full', _run_full_sim),
        ]
        for label, runner in configs:
            trajectories[label] = _load_or_run(
                label, args.duration, runner, force=args.force_resim)

    # We keep t=0 in the trajectory so the per-phase plots can show
    # the very first initiation event (which often fires within the
    # first equilibrium tick). The few plots that suffer from the
    # initial relaxation discontinuity drop it locally instead.

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(render_html(trajectories, sim_meta, args.out))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
