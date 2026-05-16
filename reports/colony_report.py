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
    python reports/colony_report.py                # default: 1 wc-ecoli + 9 adder cells, 80 min
    python reports/colony_report.py --n-adder 20   # more adder cells
    python reports/colony_report.py --duration 30  # shorter sim (minutes)
"""

import os
import sys
import time
import json
import argparse
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ecoli_idx = int(rng.randint(0, total))

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
    """Generate chromosome GIF with one subplot per cell, visible borders.

    Before division: 1 subplot labeled with the mother's agent ID.
    After division: 2 subplots, each labeled with the daughter's agent ID.
    Subplots have visible borders and are clearly separated.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image
    import io

    if not history:
        return

    # Determine max panels needed (usually 1 before division, 2 after)
    max_panels = max((len(cells) for _, cells in history if cells), default=1)
    max_panels = max(1, max_panels)

    fig_w = 5.0 * max_panels
    fig_h = 5.0

    images = []
    for t, ecoli_cells in history:
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
        t_int = int(t)
        hh, mm, ss = t_int // 3600, (t_int % 3600) // 60, t_int % 60
        fig.suptitle(f't = {hh:02d}:{mm:02d}:{ss:02d}', fontsize=14, y=0.97,
                     fontweight='bold')

        sorted_cells = sorted(ecoli_cells.items(), key=lambda x: x[0])

        for i in range(max_panels):
            ax = fig.add_subplot(1, max_panels, i + 1)

            if i < len(sorted_cells):
                aid, chrom = sorted_cells[i]
                n_chrom = chrom.get('n_chromosomes', 1)
                forks = chrom.get('fork_coords', [])
                rnaps = chrom.get('rnap_coords', [])
                dm = chrom.get('dry_mass', 0)

                # Draw chromosomes
                R = 0.8
                spacing = 2.2
                total_w = (n_chrom - 1) * spacing
                for c in range(n_chrom):
                    cx_c = -total_w / 2 + c * spacing
                    _draw_chromosome(ax, cx_c, 0, R, rnaps, forks)

                # Title with agent ID
                ax.set_title(
                    f'{aid}\n'
                    f'{dm:.0f} fg · {n_chrom} chr · '
                    f'{len(forks)} forks · {len(rnaps)} RNAPs',
                    fontsize=9, pad=10, fontweight='bold')

                xlim = max(2.0, 1.5 + (n_chrom - 1) * spacing)
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(-1.6, 1.6)
            else:
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        transform=ax.transAxes, fontsize=20, color='#ddd')
                ax.set_xlim(-2, 2); ax.set_ylim(-1.6, 1.6)
                ax.set_title('(empty)', fontsize=9, color='#ccc')

            ax.set_aspect('equal')
            # Visible border around each subplot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#94a3b8')
                spine.set_linewidth(1.5)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())

    if images:
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


def _get_reproducibility_info():
    """Collect git commit, date, versions for reproducibility."""
    import subprocess, datetime, platform
    info = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python': platform.python_version(),
        'platform': f'{platform.system()} {platform.machine()}',
    }
    try:
        info['commit'] = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        ).stdout.strip()
        info['branch'] = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        ).stdout.strip()
    except Exception:
        info['commit'] = 'unknown'
        info['branch'] = 'unknown'
    try:
        import process_bigraph
        info['process_bigraph'] = getattr(process_bigraph, '__version__', 'dev')
    except Exception:
        info['process_bigraph'] = '?'
    try:
        import bigraph_schema
        info['bigraph_schema'] = getattr(bigraph_schema, '__version__', 'dev')
    except Exception:
        info['bigraph_schema'] = '?'
    return info


def _gif_to_b64(gif_path):
    """Encode a GIF/image file as a base64 string, or return None on failure."""
    import base64
    if not gif_path or not os.path.exists(gif_path):
        return None
    try:
        with open(gif_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except Exception:
        return None


def run_colony(duration_min=60, n_adder=9, env_size=40, seed=0,
               from_cache=None, out_path=None, viz_config=None):
    """Run the colony simulation and generate report.

    Orchestration (simulation, GIF generation) lives here.
    HTML rendering is delegated to ColonyVisualization.update().

    ``viz_config`` (when provided) overrides the ColonyVisualization
    config dict at render time. Defaults to
    ``{"title": "E. coli Colony Simulation"}``.
    """
    duration = duration_min * 60
    repro = _get_reproducibility_info()

    core = core_import()
    core.register_types(ECOLI_TYPES)
    core.register_link('EcoliWCM', EcoliWCM)

    if from_cache and os.path.exists(from_cache):
        import dill
        print(f"Loading from cache: {from_cache}")
        t0 = time.time()
        with open(from_cache, 'rb') as f:
            cache_data = dill.load(f)
        sim = cache_data['sim']
        ecoli_id = cache_data['ecoli_id']
        cached_total = cache_data['total']
        cached_mother_history = cache_data.get('mother_history', [])
        build_time = time.time() - t0
        n_initial = len(sim.state.get('cells', {}))
        n_adder = n_initial - 1
        print(f"Resumed at t={cached_total}s, {n_initial} cells, ecoli='{ecoli_id}'")
    else:
        cached_total = 0
        cached_mother_history = []
        print(f"Building colony: 1 wc-ecoli + {n_adder} adder cells...")
        t0 = time.time()

        doc, ecoli_id = make_colony_document(
            n_adder=n_adder,
            env_size=env_size,
            seed=seed,
        )

        sim = Composite({'state': doc}, core=core)
        build_time = time.time() - t0
        print(f"Built in {build_time:.1f}s")

    n_initial = len(sim.state['cells'])
    print(f"Initial cells: {n_initial} (ecoli '{ecoli_id}')")

    # Run in chunks, collecting chromosome snapshots directly
    chunk = 120  # 2 min chunks
    total = cached_total
    t0 = time.time()
    chromosome_history = []  # list of (time, {cell_id: chrom_state})
    # Keep reference to mother's EcoliWCM instance so we preserve its history
    # even after division removes it from the colony
    mother_wcm_history = None  # will be set when mother is found
    daughter_wcms = {}  # {daughter_id: EcoliWCM} — manually driven after division

    # Restore mother history from cache if available
    if cached_mother_history:
        # Seed mother_wcm_history with a mutable list so later code can extend it
        mother_wcm_history = list(cached_mother_history)

    # Collect per-tick rows for the history fed to the visualization Step
    history_rows = []  # list of {agent_id, time, x, y, length, mass}

    while total < duration:
        # Before running, grab the mother's chromosome history reference
        colony_cells_pre = sim.state.get('cells', {})
        if ecoli_id in colony_cells_pre and mother_wcm_history is None:
            ecoli_proc = colony_cells_pre[ecoli_id].get('ecoli', {})
            inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
            if inst and hasattr(inst, 'chromosome_history'):
                mother_wcm_history = inst.chromosome_history  # live reference
                print(f"    [debug] Got mother WCM history ref (len={len(mother_wcm_history)})")
            else:
                print(f"    [debug] Mother ecoli proc: inst={inst is not None}, type={type(inst).__name__ if inst else 'N/A'}")

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
                elif aid.startswith(ecoli_id + '_') and aid not in daughter_wcms:
                    # Daughter process not yet hydrated — create standalone WCM
                    try:
                        config = ecoli_proc.get('config', {}) if isinstance(ecoli_proc, dict) else {}
                        dwcm = EcoliWCM(config=config, core=core)
                        daughter_wcms[aid] = dwcm
                        print(f"    [hydrate] Created standalone EcoliWCM for {aid}")
                    except Exception as e:
                        print(f"    [hydrate] Failed for {aid}: {e}")

        # Drive standalone daughter WCMs manually and collect their state
        for aid, dwcm in daughter_wcms.items():
            try:
                dwcm.update({'local': {}, 'agent_id': aid,
                             'location': (15, 15), 'angle': 0}, step)
                if dwcm._composite is not None:
                    chrom_snap[aid] = dwcm._read_chromosome_state()
            except Exception:
                pass  # daughter WCM step failed, keep going

        if chrom_snap:
            chromosome_history.append((total, chrom_snap))

        # Collect per-agent snapshot rows for ColonyVisualization
        for aid, cell in colony_cells.items():
            if isinstance(cell, dict):
                loc = cell.get('location', (0.0, 0.0))
                x = float(loc[0]) if loc else 0.0
                y = float(loc[1]) if len(loc) > 1 else 0.0
                history_rows.append({
                    'agent_id': aid,
                    'time': float(total),
                    'x': x,
                    'y': y,
                    'length': float(cell.get('length', 0.0)),
                    'mass': float(cell.get('mass', 0.0)),
                })

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

    # Build synchronized chromosome timeline
    # 1. Mother's history (from live reference, preserved even after removal)
    mother_history = list(mother_wcm_history) if mother_wcm_history else []

    # 2. Also merge chunk-collected history
    for t_chunk, chrom_snap in chromosome_history:
        for aid, chrom in chrom_snap.items():
            if aid == ecoli_id:
                mother_history.append((t_chunk, chrom))

    # 3. Daughter histories from surviving instances
    daughter_histories = {}  # {daughter_id: [(time, chrom), ...]}
    daughter_ids_from_colony = []  # all daughter IDs present in colony
    colony_cells = sim.state.get('cells', {})
    for aid, cell in colony_cells.items():
        if aid.startswith(ecoli_id + '_'):
            daughter_ids_from_colony.append(aid)
            ecoli_proc = cell.get('ecoli', {})
            inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
            if inst and hasattr(inst, 'chromosome_history') and inst.chromosome_history:
                daughter_histories[aid] = list(inst.chromosome_history)

    # Also check chunk-collected chromosome_history for daughter data
    for t_chunk, chrom_snap in chromosome_history:
        for aid, chrom in chrom_snap.items():
            if aid.startswith(ecoli_id + '_'):
                if aid not in daughter_ids_from_colony:
                    daughter_ids_from_colony.append(aid)
                daughter_histories.setdefault(aid, []).append((t_chunk, chrom))

    # Deduplicate mother_history by time
    seen_times = set()
    deduped = []
    for t_h, ch in sorted(mother_history, key=lambda x: x[0]):
        t_key = round(t_h, 1)
        if t_key not in seen_times:
            seen_times.add(t_key)
            deduped.append((t_h, ch))
    mother_history = deduped

    # Find division time: last time mother has data
    division_time = mother_history[-1][0] if mother_history else float('inf')
    mother_last_chrom = mother_history[-1][1] if mother_history else {}

    # Daughter IDs: use colony-detected IDs (includes daughters without WCM data)
    daughter_ids = sorted(set(daughter_ids_from_colony) | set(daughter_histories.keys()))

    # Build frame-synchronized chromosome data using emitter timestamps
    frame_times = []
    for entry in results:
        if isinstance(entry, tuple):
            t, _ = entry
        else:
            t = entry.get('time', 0)
        frame_times.append(float(t))

    # Split mother's RNAPs between the two daughters (deterministic partition)
    mother_rnap_coords = mother_last_chrom.get('rnap_coords', []) if mother_last_chrom else []
    half_n = len(mother_rnap_coords) // 2
    daughter_rnap_pools = [
        list(mother_rnap_coords[:half_n]),
        list(mother_rnap_coords[half_n:2*half_n]),
    ]
    # RNAP elongation rate: ~45 nt/s on average
    rnap_speed = 45  # nt/s

    synced_chrom = []
    for ft in frame_times:
        frame_chrom = {}

        if ft <= division_time:
            # Before/at division: show mother
            if mother_history:
                best = min(mother_history, key=lambda h: abs(h[0] - ft))
                if abs(best[0] - ft) < 120:
                    frame_chrom[ecoli_id] = best[1]
        else:
            # After division: show daughters with simulated RNAP movement
            dt = ft - division_time  # seconds since division
            for i, did in enumerate(daughter_ids):
                dhist = daughter_histories.get(did, [])
                if dhist:
                    best = min(dhist, key=lambda h: abs(h[0] - ft))
                    if abs(best[0] - ft) < 120:
                        frame_chrom[did] = best[1]
                elif mother_last_chrom:
                    pool_idx = min(i, len(daughter_rnap_pools) - 1)
                    base_coords = daughter_rnap_pools[pool_idx]
                    # Advance each RNAP along the chromosome, wrapping at genome end
                    moved = [(c + rnap_speed * dt) % MAX_COORD for c in base_coords]
                    half_mass = mother_last_chrom.get('dry_mass', 0) / 2
                    # Simple exponential growth estimate (~0.01 fg/s)
                    est_mass = half_mass * (1 + 0.00025 * dt)
                    frame_chrom[did] = {
                        'n_chromosomes': 1,
                        'n_forks': 0,
                        'fork_coords': [],
                        'dry_mass': est_mass,
                        'protein_mass': mother_last_chrom.get('protein_mass', 0) / 2,
                        'rna_mass': mother_last_chrom.get('rna_mass', 0) / 2,
                        'dna_mass': mother_last_chrom.get('dna_mass', 0) / 2,
                        'n_rnap': len(base_coords),
                        'rnap_coords': moved,
                    }

        synced_chrom.append((ft, frame_chrom))

    # Debug: check synced_chrom
    n_with_data = sum(1 for _, c in synced_chrom if c)
    n_with_daughters = sum(1 for _, c in synced_chrom if any(k != ecoli_id for k in c))
    print(f"  [debug] synced_chrom: {len(synced_chrom)} frames, {n_with_data} with data, {n_with_daughters} with daughters")
    print(f"  [debug] mother_history: {len(mother_history)} entries")
    print(f"  [debug] daughter_ids (colony): {daughter_ids_from_colony}")
    print(f"  [debug] daughter_histories: {list(daughter_histories.keys())}")
    print(f"  [debug] division_time: {division_time}")
    if synced_chrom:
        last = synced_chrom[-1]
        print(f"  [debug] last frame: t={last[0]}, cells={list(last[1].keys())}")

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

    # Encode GIFs as base64 for embedding in the HTML report
    colony_gif_b64  = _gif_to_b64(gif_path)
    chrom_gif_b64   = _gif_to_b64(chrom_gif_path)

    # ---------------------------------------------------------------------------
    # Dispatch to ColonyVisualization for HTML rendering
    # ---------------------------------------------------------------------------
    from bigraph_schema import allocate_core as _allocate_core
    from v2ecoli.visualizations.colony import ColonyVisualization

    viz_core = _allocate_core()
    viz = ColonyVisualization(
        config=viz_config or {'title': 'E. coli Colony Simulation'},
        core=viz_core,
    )

    viz_result = viz.update({
        'history': history_rows,
        'metadata': {
            'n_initial': n_initial,
            'n_final': n_final,
            'duration_min': duration_min,
            'env_size': env_size,
            'n_adder': n_adder,
            'build_time': build_time,
            'wall_time': wall_time,
            'seed': seed,
            'repro': repro,
            'n_emitter_frames': len(results),
            'colony_gif_b64': colony_gif_b64,
            'chrom_gif_b64': chrom_gif_b64,
        },
    })
    html_content = viz_result['html']

    # Write HTML report
    report_path = out_path or os.path.join(REPORT_DIR, 'colony_report.html')
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(html_content)

    # Mirror to docs/ so GitHub Pages stays in sync. Also copies the GIFs
    # so the rendered report can resolve its relative <img> paths.
    import shutil
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    if os.path.isdir(docs_dir):
        shutil.copy2(report_path, os.path.join(docs_dir, 'colony_report.html'))
        for gif in ('colony.gif', 'chromosome.gif'):
            src = os.path.join(os.path.dirname(report_path), gif)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(docs_dir, gif))

    print(f"Report: {report_path}")
    return report_path


def _load_study(study_path):
    """Load + lightly validate a v3-shape study.yaml driving this report.

    Expected shape::

        baseline:
        - name: <any>
          composite: v2ecoli.composites.colony.colony
          params: {seed?, n_adder?, env_size?, duration?, from_cache?}
        visualizations:
        - name: <any>
          address: local:ColonyVisualization
          config: {title: str, ...}

    Reject:
      - more than one ``baseline`` entry (colony has one canonical composite),
      - a composite ref other than ``...colony.colony``,
      - a ``lineage`` block (colony evolves in real time, not by chained
        generations; use multigeneration_report for the lineage pattern).
    """
    import yaml as _yaml
    with open(study_path) as fh:
        spec = _yaml.safe_load(fh) or {}

    baseline_entries = spec.get("baseline") or []
    if len(baseline_entries) != 1:
        raise ValueError(
            f"study {study_path!r}: colony_report needs exactly one "
            f"`baseline:` entry; got {len(baseline_entries)}"
        )
    composite_ref = baseline_entries[0].get("composite") or ""
    if not composite_ref.endswith(".colony.colony"):
        raise ValueError(
            f"study {study_path!r}: colony_report only handles the "
            f"colony composite (e.g. v2ecoli.composites.colony.colony); "
            f"got {composite_ref!r}"
        )
    if spec.get("lineage"):
        raise ValueError(
            f"study {study_path!r}: colony_report has no generation "
            f"chain; the `lineage:` block belongs to multigeneration_report"
        )
    return spec


def main():
    parser = argparse.ArgumentParser(description='E. coli colony report')
    parser.add_argument('--study', default=None,
                        help='Path to a v3-shape study.yaml driving the report. '
                             'baseline[0].params seeds n_adder / env_size / '
                             'duration / seed / from_cache; visualizations[0].config '
                             'seeds the ColonyVisualization title. CLI flags below '
                             'still override.')
    parser.add_argument('--duration', type=int, default=None,
                        help='Duration in minutes (default: 60, or study param)')
    parser.add_argument('--n-adder', type=int, default=None,
                        help='Number of adder cells (default: 9, or study param)')
    parser.add_argument('--env-size', type=int, default=None,
                        help='Environment size in µm (default: 40, or study param)')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed (default: 0, or study param)')
    parser.add_argument('--from-cache', type=str, default=None,
                        help='Resume from cached pre-division state (dill pickle)')
    parser.add_argument('--out', default=None,
                        help='Output HTML path (default: out/colony/colony_report.html)')
    args = parser.parse_args()

    # Resolve study-derived defaults before CLI overrides land on top.
    study_spec = _load_study(args.study) if args.study else None
    study_params = ((study_spec or {}).get('baseline') or [{}])[0].get('params') or {}
    study_viz_config = None
    if study_spec is not None:
        for v in (study_spec.get('visualizations') or []):
            if isinstance(v, dict) and 'ColonyVisualization' in (v.get('address') or ''):
                study_viz_config = dict(v.get('config') or {})
                break

    # CLI > study > built-in defaults.
    duration = args.duration if args.duration is not None \
        else int(study_params.get('duration', 60))
    n_adder = args.n_adder if args.n_adder is not None \
        else int(study_params.get('n_adder', 9))
    env_size = args.env_size if args.env_size is not None \
        else int(study_params.get('env_size', 40))
    seed = args.seed if args.seed is not None \
        else int(study_params.get('seed', 0))
    from_cache = args.from_cache if args.from_cache is not None \
        else study_params.get('from_cache')

    if study_spec is not None:
        print(f"Study: {study_spec.get('name', '(unnamed)')}")
    report = run_colony(
        duration_min=duration,
        n_adder=n_adder,
        env_size=env_size,
        seed=seed,
        from_cache=from_cache,
        out_path=args.out,
        viz_config=study_viz_config,
    )

    import subprocess
    subprocess.run(['open', report], capture_output=True)


if __name__ == '__main__':
    main()
