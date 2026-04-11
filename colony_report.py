"""
Colony Report: 1 whole-cell E. coli + many adder-based cells

Runs a mixed colony simulation with:
- 1 whole-cell E. coli model (v2ecoli, 55 biological steps)
- N adder-based growth/division cells (simple Monod kinetics)
- 2D pymunk physics (collisions, spatial arrangement)

Generates:
- GIF animation of the colony
- HTML report with cell counts, mass, timing

Usage:
    python colony_report.py                # default: 1 wc-ecoli + 9 adder cells, 60 min
    python colony_report.py --n-adder 20   # more adder cells
    python colony_report.py --duration 30  # shorter sim (minutes)
"""

import os
import sys
import time
import json
import argparse
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires

from multi_cell import core_import
from multi_cell.processes.multibody import (
    make_initial_state, make_rng, build_microbe)
from multi_cell.processes.grow_divide import (
    add_adder_grow_divide_to_agents, make_adder_grow_divide_process,
    make_grow_divide_process)
from multi_cell.plots.multibody_plots import simulation_to_gif

from v2ecoli.bridge import EcoliWCM
from v2ecoli.types import ECOLI_TYPES


REPORT_DIR = 'out/colony'


def make_colony_document(
    n_adder=9,
    env_size=40,
    physics_interval=10.0,
    ecoli_interval=60.0,  # WCM runs every 60s (heavy — don't run too often)
    cache_dir='out/cache',
    seed=0,
):
    """Build a colony with 1 wc-ecoli + n adder cells."""
    rng = make_rng(seed)
    cx, cy = env_size / 2, env_size / 2  # center

    # Build all cells clustered at center
    cells = {}
    total = n_adder + 1  # adder + 1 ecoli

    # Place cells in a tight grid near center
    cols = int(np.ceil(np.sqrt(total)))
    spacing = 3.0  # µm between cell centers

    positions = []
    for i in range(total):
        row, col = divmod(i, cols)
        x = cx + (col - cols / 2) * spacing + rng.uniform(-0.3, 0.3)
        y = cy + (row - cols / 2) * spacing + rng.uniform(-0.3, 0.3)
        positions.append((x, y))

    # Randomly assign one position to ecoli
    ecoli_idx = rng.randint(0, total)

    for i in range(total):
        x, y = positions[i]
        angle = rng.uniform(0, 2 * np.pi)
        length = 1.5 + rng.uniform(0, 1.0)
        radius = 0.4 + rng.uniform(0, 0.2)

        if i == ecoli_idx:
            # This is the whole-cell ecoli
            ecoli_id, ecoli_body = build_microbe(
                rng, env_size=env_size,
                x=x, y=y, angle=angle,
                length=2.0, radius=0.5, density=0.02,
            )
            # Set initial physical mass to match WCM dry mass
            ecoli_body['mass'] = 380.0  # fg, matches v2ecoli initial dry mass
        else:
            # Adder surrogate cell
            aid, body = build_microbe(
                rng, env_size=env_size,
                x=x, y=y, angle=angle,
                length=length, radius=radius, density=0.02,
            )
            body['grow_divide'] = make_adder_grow_divide_process(
                config={'agents_key': 'cells'},
                agents_key='cells',
                interval=physics_interval,
            )
            cells[aid] = body

    # Embed EcoliWCM process inside the ecoli cell
    ecoli_body['ecoli'] = {
        '_type': 'process',
        'address': 'local:EcoliWCM',
        'config': {
            'cache_dir': cache_dir,
            'seed': seed,
        },
        'interval': ecoli_interval,
        'inputs': {
            'local': ['local'],
            'agent_id': ['id'],
            'location': ['location'],
            'angle': ['angle'],
        },
        'outputs': {
            'mass': ['mass'],
            'length': ['length'],
            'volume': ['volume'],
            'exchange': ['exchange'],
            'agents': ['..', '..', 'cells'],
        },
    }
    ecoli_body.setdefault('local', {})
    ecoli_body.setdefault('volume', 0.0)
    ecoli_body.setdefault('exchange', {})

    cells[ecoli_id] = ecoli_body

    document = {
        'cells': cells,
        'particles': {},
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'elasticity': 0.1,
            },
            'interval': physics_interval,
            'inputs': {
                'segment_cells': ['cells'],
                'circle_particles': ['particles'],
            },
            'outputs': {
                'segment_cells': ['cells'],
                'circle_particles': ['particles'],
            },
        },

        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }

    return document, ecoli_id


# ---------------------------------------------------------------------------
# Chromosome state visualization
# ---------------------------------------------------------------------------

MAX_COORD = 2_320_826  # Half-genome in bp


def _coord_to_angle(coord):
    frac = coord / MAX_COORD
    return np.pi / 2 - frac * np.pi


def _draw_chromosome(ax, cx, cy, R, rnap_coords, fork_coords):
    """Draw one circular chromosome."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + R * np.cos(theta), cy + R * np.sin(theta),
            color='#cbd5e1', lw=3, zorder=1)
    # OriC (top) and Ter (bottom)
    ax.plot(cx, cy + R, 'o', color='#10b981', ms=7, zorder=5)
    ax.plot(cx, cy - R, 's', color='#ef4444', ms=5, zorder=5)
    # RNAPs
    if rnap_coords:
        angles = [_coord_to_angle(c) for c in rnap_coords]
        rx = [cx + R * np.cos(a) for a in angles]
        ry = [cy + R * np.sin(a) for a in angles]
        ax.scatter(rx, ry, c='#3b82f6', s=3, alpha=0.3, zorder=3)
    # Forks
    for coord in fork_coords:
        angle = _coord_to_angle(coord)
        fx = cx + (R + 0.08) * np.cos(angle)
        fy = cy + (R + 0.08) * np.sin(angle)
        ax.plot(fx, fy, 'v', color='#f59e0b', ms=8, zorder=4)


def _generate_chromosome_gif_from_history(history, out_path, frame_duration_ms=100):
    """Generate chromosome GIF from synchronized history.

    Shows all ecoli cells' chromosomes in a single figure. Before division,
    one panel shows the mother's chromosomes (1→2 during replication).
    After division, two panels show each daughter's chromosomes.

    Args:
        history: list of (time, {cell_id: chrom_state_dict})
        out_path: output GIF path
        frame_duration_ms: milliseconds per frame (matches colony GIF)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    if not history:
        return

    # Determine max panels needed across all frames
    max_panels = max((len(cells) for _, cells in history if cells), default=1)
    max_panels = max(1, max_panels)

    # Fixed figure size for consistency across frames
    fig_w = 4.5 * max_panels
    fig_h = 4.5

    images = []
    for t, ecoli_cells in history:
        fig, axes = plt.subplots(1, max_panels, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(f't = {t:.0f}s  ({t/60:.1f} min)', fontsize=13, y=0.98,
                     fontweight='bold')

        # Sort cells: mother first, then daughters by name
        sorted_cells = sorted(ecoli_cells.items(), key=lambda x: x[0])

        for i in range(max_panels):
            ax = axes[0][i]
            if i < len(sorted_cells):
                aid, chrom = sorted_cells[i]
                n_chrom = chrom.get('n_chromosomes', 1)
                forks = chrom.get('fork_coords', [])
                rnaps = chrom.get('rnap_coords', [])
                dm = chrom.get('dry_mass', 0)
                rna_mass = chrom.get('rna_mass', 0)
                protein_mass = chrom.get('protein_mass', 0)

                # Draw chromosomes
                R = 0.8
                spacing = 2.2
                total_w = (n_chrom - 1) * spacing
                for c in range(n_chrom):
                    cx = -total_w / 2 + c * spacing
                    _draw_chromosome(ax, cx, 0, R, rnaps, forks)

                # Label: determine if mother or daughter
                if '_' in aid:
                    parts = aid.split('_')
                    daughter_num = parts[-1]
                    label = f'Daughter {daughter_num}'
                else:
                    label = 'Mother'

                ax.set_title(
                    f'{label}\n'
                    f'{dm:.0f} fg dry mass · {n_chrom} chromosome{"s" if n_chrom > 1 else ""}\n'
                    f'{len(forks)} forks · {len(rnaps)} RNAPs',
                    fontsize=9, pad=8)

                xlim = max(2.0, 1.5 + (n_chrom - 1) * spacing)
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-1.5, 1.5)
            else:
                # Empty panel placeholder
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        transform=ax.transAxes, fontsize=20, color='#ddd')
                ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 1.5)

            ax.set_aspect('equal')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())

    if images:
        # Ensure all frames are the same size (pad to largest)
        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)
        uniform = []
        for img in images:
            if img.width != max_w or img.height != max_h:
                padded = Image.new('RGB', (max_w, max_h), (255, 255, 255))
                padded.paste(img, (0, 0))
                uniform.append(padded)
            else:
                uniform.append(img)

        uniform[0].save(out_path, save_all=True, append_images=uniform[1:],
                        duration=frame_duration_ms, loop=0)


def _generate_chromosome_gif(results, ecoli_id, out_path, skip=1):
    """Generate a GIF showing chromosome state over time for ecoli cells."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    frames_data = []
    for entry in results[::skip]:
        if isinstance(entry, tuple):
            t, data = entry
        else:
            t = entry.get('time', 0)
            data = entry

        # Read from top-level chromosome_states store
        chrom_states = data.get('chromosome_states', {})
        ecoli_cells = {}
        for aid, chrom in chrom_states.items():
            if isinstance(chrom, dict) and chrom.get('n_chromosomes', 0) > 0:
                ecoli_cells[aid] = chrom

        frames_data.append((float(t), ecoli_cells))

    if not frames_data:
        return

    # Render frames
    images = []
    for t, ecoli_cells in frames_data:
        n_panels = max(1, len(ecoli_cells))
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
        fig.suptitle(f't = {t:.0f}s ({t/60:.1f} min)', fontsize=12, y=0.98)

        if not ecoli_cells:
            ax = axes[0][0]
            ax.text(0.5, 0.5, 'No chromosome data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#888')
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal'); ax.axis('off')
        else:
            for i, (aid, chrom) in enumerate(ecoli_cells.items()):
                ax = axes[0][i]
                n_chrom = chrom.get('n_chromosomes', 1)
                forks = chrom.get('fork_coords', [])
                rnaps = chrom.get('rnap_coords', [])
                dm = chrom.get('dry_mass', 0)

                # Draw chromosomes side by side
                R = 0.8
                spacing = 2.2
                total_w = (n_chrom - 1) * spacing
                for c in range(n_chrom):
                    cx = -total_w / 2 + c * spacing
                    _draw_chromosome(ax, cx, 0, R, rnaps, forks)

                # Label
                label = aid.replace(ecoli_id, 'ecoli')
                ax.set_title(f'{label}\n{dm:.0f} fg, {n_chrom} chr, '
                             f'{len(forks)} forks, {len(rnaps)} RNAPs',
                             fontsize=8)
                ax.set_xlim(-2.5, 2.5 + (n_chrom - 1) * spacing)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal'); ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())

    if images:
        images[0].save(out_path, save_all=True, append_images=images[1:],
                       duration=200, loop=0)


def run_colony(duration_min=60, n_adder=9, env_size=40, seed=0):
    """Run the colony simulation and generate report."""
    duration = duration_min * 60  # convert to seconds

    print(f"Building colony: 1 wc-ecoli + {n_adder} adder cells...")
    t0 = time.time()

    core = core_import()
    core.register_types(ECOLI_TYPES)
    core.register_link('EcoliWCM', EcoliWCM)

    doc, ecoli_id = make_colony_document(
        n_adder=n_adder,
        env_size=env_size,
        seed=seed,
    )

    sim = Composite({'state': doc}, core=core)
    build_time = time.time() - t0
    print(f"Built in {build_time:.1f}s")

    n_initial = len(sim.state['cells'])
    print(f"Initial cells: {n_initial} (1 wc-ecoli '{ecoli_id}' + {n_adder} adder)")

    # Run in chunks, collecting chromosome snapshots directly
    chunk = 120  # 2 min chunks
    total = 0
    t0 = time.time()
    chromosome_history = []  # list of (time, {cell_id: chrom_state})

    while total < duration:
        step = min(chunk, duration - total)
        try:
            sim.run(step)
        except Exception as e:
            print(f"  Warning at t={total+step}s: {type(e).__name__}")
        total += step

        # Collect chromosome state directly from EcoliWCM instances
        colony_cells = sim.state.get('cells', {})
        chrom_snap = {}
        for aid, cell in colony_cells.items():
            if aid == ecoli_id or aid.startswith(ecoli_id + '_'):
                ecoli_proc = cell.get('ecoli', {})
                inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
                if inst and hasattr(inst, '_composite') and inst._composite is not None:
                    chrom_snap[aid] = inst._read_chromosome_state()
        if chrom_snap:
            chromosome_history.append((total, chrom_snap))

        n_cells = len(colony_cells)
        ecoli_alive = ecoli_id in colony_cells
        # Check for daughter cells (ecoli_id_0, ecoli_id_1)
        ecoli_daughters = [k for k in colony_cells if k.startswith(ecoli_id + '_')]
        if ecoli_alive:
            status = 'alive'
        elif ecoli_daughters:
            status = f'divided → {len(ecoli_daughters)} daughters'
        else:
            status = 'gone'
        print(f"  t={total}s ({total/60:.0f}min): {n_cells} cells, ecoli={status}")

    wall_time = time.time() - t0
    n_final = len(sim.state.get('cells', {}))
    print(f"\nDone: {total}s sim in {wall_time:.0f}s wall ({total/wall_time:.1f}x realtime)")
    print(f"Final cells: {n_final}")

    # Extract emitter data
    print("Extracting emitter data...")
    try:
        results = gather_emitter_results(sim)[('emitter',)]
    except Exception:
        results = []

    # Generate GIFs — use same skip factor for synchronization
    os.makedirs(REPORT_DIR, exist_ok=True)
    gif_skip = max(1, len(results) // 100)  # shared skip factor
    gif_duration_ms = 100  # ms per frame, shared

    gif_path = os.path.join(REPORT_DIR, 'colony.gif')
    print(f"Generating colony GIF ({len(results)} frames, skip={gif_skip})...")
    try:
        # Hybrid coloring: fixed green base for ecoli, grey for surrogates
        # Daughters get hue-shifted variants of the ecoli color
        from colorsys import hsv_to_rgb, rgb_to_hsv
        import random as _random
        ecoli_base_rgb = (0.2, 0.75, 0.3)  # green
        ecoli_base_hsv = rgb_to_hsv(*ecoli_base_rgb)
        _color_rng = _random.Random(42)

        # Build daughter color map with mutations from the ecoli base
        _ecoli_colors = {ecoli_id: ecoli_base_rgb}

        def _mutate(hsv):
            h, s, v = hsv
            h = (h + _color_rng.gauss(0, 0.08)) % 1.0
            s = max(0.3, min(1.0, s + _color_rng.gauss(0, 0.05)))
            v = max(0.4, min(1.0, v + _color_rng.gauss(0, 0.05)))
            return (h, s, v)

        def _get_ecoli_color(aid):
            if aid in _ecoli_colors:
                return _ecoli_colors[aid]
            # Find parent
            parent = '_'.join(aid.rsplit('_', 1)[:-1])
            parent_rgb = _get_ecoli_color(parent) if parent in _ecoli_colors else ecoli_base_rgb
            parent_hsv = rgb_to_hsv(*parent_rgb)
            child_hsv = _mutate(parent_hsv)
            child_rgb = hsv_to_rgb(*child_hsv)
            _ecoli_colors[aid] = child_rgb
            return child_rgb

        def _hybrid_color(aid, ent=None):
            if aid == ecoli_id or aid.startswith(ecoli_id + '_'):
                return _get_ecoli_color(aid)
            return (0.7, 0.7, 0.7)

        simulation_to_gif(
            results,
            config={'env_size': env_size},
            agents_key='agents',
            filename='colony.gif',
            out_dir=REPORT_DIR,
            skip_frames=gif_skip,
            show_time_title=True,
            color_fn=_hybrid_color,
            frame_duration_ms=gif_duration_ms,
        )
        print(f"GIF saved: {gif_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"GIF generation failed: {e}")
        gif_path = None

    # Build synchronized chromosome timeline from EcoliWCM instances
    # Collect all chromosome histories keyed by cell ID
    all_chrom_histories = {}  # {cell_id: [(time, chrom_dict), ...]}
    colony_cells = sim.state.get('cells', {})
    for aid, cell in colony_cells.items():
        if aid == ecoli_id or aid.startswith(ecoli_id + '_'):
            ecoli_proc = cell.get('ecoli', {})
            inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
            if inst and hasattr(inst, 'chromosome_history'):
                all_chrom_histories[aid] = list(inst.chromosome_history)

    # Also check chromosome_history collected during the run
    # (for the mother cell that may have been removed at division)
    if chromosome_history:
        for t, chrom_snap in chromosome_history:
            for aid, chrom in chrom_snap.items():
                if aid not in all_chrom_histories:
                    all_chrom_histories[aid] = []
                all_chrom_histories[aid].append((t, chrom))

    # Build frame-synchronized chromosome data using emitter timestamps
    frame_times = []
    for entry in results:
        if isinstance(entry, tuple):
            t, _ = entry
        else:
            t = entry.get('time', 0)
        frame_times.append(float(t))

    # For each emitter frame, find the closest chromosome snapshot per cell
    synced_chrom = []  # list of (time, {cell_id: chrom_dict})
    for ft in frame_times:
        frame_chrom = {}
        for aid, hist in all_chrom_histories.items():
            if not hist:
                continue
            # Find closest snapshot to this frame time
            best = min(hist, key=lambda h: abs(h[0] - ft))
            if abs(best[0] - ft) < 120:  # within 2 min
                frame_chrom[aid] = best[1]
        synced_chrom.append((ft, frame_chrom))

    # Generate chromosome GIF synchronized with colony GIF (same skip)
    chrom_gif_path = os.path.join(REPORT_DIR, 'chromosome.gif')
    print(f"Generating chromosome GIF ({len(synced_chrom)} frames, skip={gif_skip})...")
    try:
        _generate_chromosome_gif_from_history(
            synced_chrom[::gif_skip], chrom_gif_path,
            frame_duration_ms=gif_duration_ms)
        print(f"Chromosome GIF saved: {chrom_gif_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Chromosome GIF failed: {e}")
        chrom_gif_path = None

    # Generate HTML report
    report_path = os.path.join(REPORT_DIR, 'colony_report.html')

    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>E. coli Colony Simulation</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #16a34a; padding-bottom: 8px; }}
h2 {{ color: #166534; margin-top: 2em; }}
h3 {{ color: #334155; }}
p {{ line-height: 1.6; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ padding: 6px 16px; border: 1px solid #e2e8f0; }}
th {{ background: #f1f5f9; }}
.media {{ text-align: center; margin: 1.5em 0; }}
.media img {{ max-width: 100%; border: 2px solid #e2e8f0; border-radius: 8px; }}
.media-label {{ font-size: 0.85em; color: #64748b; margin-top: 0.5em; }}
.legend {{ display: flex; gap: 2em; justify-content: center; margin: 1em 0; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 0.5em; }}
.legend-swatch {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid #ccc; }}
.section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5em; margin: 1em 0; }}
</style>
</head><body>

<h1>E. coli Colony Simulation</h1>

<div class="section">
<p>This simulation places a <strong>whole-cell <em>E. coli</em> model</strong> — with 55 biological
processes including metabolism, transcription, translation, DNA replication, and chromosome
segregation — inside a <strong>2D colony</strong> alongside simpler surrogate cells.</p>

<p>Each whole-cell <em>E. coli</em> is implemented as an <code>EcoliWCM</code> process — a
process-bigraph <code>Process</code> that holds an internal <code>Composite</code> connected
via a bridge. The bridge maps external colony ports (mass, length, volume) to internal
whole-cell stores. The whole-cell model (v2ecoli) runs the full mechanistic simulation of
intracellular biology, while the colony framework (pymunk-process) handles spatial physics:
cell body collisions, growth-driven elongation, and division mechanics.</p>

<p>The <strong style="color:rgb(51,191,77);">green cell</strong> is the whole-cell
<em>E. coli</em> — its length and mass are driven by the internal biological simulation. The
<span style="color:#999;">grey cells</span> are surrogate cells using a simple adder growth model.
When the whole-cell model reaches its division threshold (~702 fg dry mass, ~42 min), the bridge
removes the mother cell and adds two daughter cells, each with a fresh copy of the whole-cell
model. Daughters are shown with color-shifted variants of the mother&rsquo;s green.</p>
</div>

<h2>Colony Dynamics</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:rgb(51,191,77)"></div> Whole-cell <em>E. coli</em> (v2ecoli, 55 processes)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#b0b0b0"></div> Surrogate cell (adder growth/division)</div>
</div>
""")

        if gif_path and os.path.exists(gif_path):
            f.write(f"""
<div class="media">
  <img src="colony.gif" alt="Colony simulation — spatial view">
  <div class="media-label">Colony spatial view: {n_initial} initial cells → {n_final} final cells over {duration_min} min.
  Colored cell = whole-cell <em>E. coli</em>; grey = surrogates.</div>
</div>
""")

        if chrom_gif_path and os.path.exists(chrom_gif_path):
            f.write(f"""
<h2>Chromosome State</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#10b981; border-radius:50%"></div> OriC (origin of replication)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ef4444"></div> Ter (terminus)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#3b82f6; border-radius:50%"></div> RNA polymerase</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f59e0b"></div> Replication fork</div>
</div>
<p>The circular chromosome of each whole-cell <em>E. coli</em>, synchronized with the colony
animation above. Chromosome replication initiates around ~23 min, producing 2 chromosomes
visible as separate circles. Each frame shows the current number of chromosomes, replication
forks, and active RNA polymerases.</p>
<div class="media">
  <img src="chromosome.gif" alt="Chromosome state over time">
  <div class="media-label">Chromosome state: replication forks traverse the circular genome,
  RNA polymerases transcribe genes along the chromosome.</div>
</div>
""")

        f.write(f"""
<h2>Simulation Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Duration</td><td>{duration_min} min ({duration}s)</td></tr>
<tr><td>Whole-cell <em>E. coli</em></td><td>1 cell (EcoliWCM bridge, 55 steps)</td></tr>
<tr><td>Surrogate cells</td><td>{n_adder} (AdderGrowDivide)</td></tr>
<tr><td>Environment</td><td>{env_size} × {env_size} µm</td></tr>
<tr><td>Physics interval</td><td>{10}s</td></tr>
<tr><td>WCM update interval</td><td>{60}s</td></tr>
</table>

<h2>Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Build time</td><td>{build_time:.1f}s</td></tr>
<tr><td>Wall time</td><td>{wall_time:.0f}s ({wall_time/60:.1f} min)</td></tr>
<tr><td>Speed</td><td>{duration/wall_time:.1f}× realtime</td></tr>
<tr><td>Initial cells</td><td>{n_initial}</td></tr>
<tr><td>Final cells</td><td>{n_final}</td></tr>
<tr><td>Emitter frames</td><td>{len(results)}</td></tr>
</table>


<footer style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; color:#94a3b8; font-size:0.85em;">
v2ecoli colony · pure process-bigraph · <a href="https://github.com/vivarium-collective/v2ecoli">github</a>
</footer>
</body></html>""")

    print(f"Report: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='E. coli colony report')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in minutes (default: 60)')
    parser.add_argument('--n-adder', type=int, default=9,
                        help='Number of adder cells (default: 9)')
    parser.add_argument('--env-size', type=int, default=40,
                        help='Environment size in µm (default: 40)')
    args = parser.parse_args()

    report = run_colony(
        duration_min=args.duration,
        n_adder=args.n_adder,
        env_size=args.env_size,
    )

    import subprocess
    subprocess.run(['open', report], capture_output=True)


if __name__ == '__main__':
    main()
