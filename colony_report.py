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
    python colony_report.py                # default: 1 wc-ecoli + 9 adder cells, 80 min
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
from multi_cell.processes.pressure import make_pressure_process
from multi_cell.plots.multibody_plots import simulation_to_gif

from v2ecoli.bridge import EcoliWCM
from v2ecoli.types import ECOLI_TYPES
from v2ecoli.viz.network import render_html as _viz_render_html


REPORT_DIR = 'out/colony'


# ---------------------------------------------------------------------------
# Colony network graph (Cytoscape, same viewer as other network reports)
# ---------------------------------------------------------------------------

_COLONY_PROCESS_COLORS = {
    'PymunkProcess': '#2563eb',   # blue — physics
    'Pressure':      '#be185d',   # magenta — mechanics
    'GrowDivide':    '#15803d',   # green — cell-cycle
    'AdderGrowDivide': '#15803d',
    'EcoliWCM':      '#ea580c',   # orange — whole-cell biology
    'emitter':       '#64748b',   # slate
}


def _colony_proc_color(addr_tail):
    return _COLONY_PROCESS_COLORS.get(addr_tail, '#9333ea')


def _wire_to_path(wire):
    """Normalize a wire (list of segments, possibly with '..') into a
    canonical store path string. Returns None if the wire is unusable."""
    if not isinstance(wire, (list, tuple)) or not wire:
        return None
    segs = [s for s in wire if s != '..' and isinstance(s, str) and s and not s.startswith('_')]
    return '.'.join(segs) if segs else None


def build_colony_graph(document):
    """Build a Cytoscape-compatible graph dict from the colony document.

    Walks the nested dict; every sub-dict with ``_type`` in {process, step}
    becomes a process node and its ``inputs``/``outputs`` wires become edges
    to store nodes. Stores referenced via ``['..', '..', 'cells']`` resolve
    to the top-level ``cells`` store so per-cell processes and the multibody
    process share the same anchor.
    """
    nodes = {}
    edges = {}

    def add_store(path):
        if not path:
            return
        if path not in nodes:
            nodes[path] = {'data': {
                'id': path, 'label': path.split('.')[-1],
                'kind': 'store', 'subsystem': 'store', 'color': '#FFFFFF',
            }}

    def add_edge(src, dst, direction, port):
        key = (src, dst, direction)
        if key not in edges:
            edges[key] = {'data': {
                'id': f'{src}__{dst}__{direction}',
                'source': src, 'target': dst,
                'direction': direction, 'ports': [],
            }}
        edges[key]['data']['ports'].append(port)

    def walk(path_segs, node):
        if not isinstance(node, dict):
            return
        t = node.get('_type')
        if t in ('process', 'step'):
            addr = node.get('address', '') or ''
            addr_tail = addr.rsplit(':', 1)[-1] or addr
            proc_id = '.'.join(path_segs) or addr_tail
            label = addr_tail if len(path_segs) <= 1 else f'{path_segs[-1]} ({addr_tail})'
            nodes[proc_id] = {'data': {
                'id': proc_id, 'label': label,
                'kind': 'process', 'subsystem': addr_tail,
                'color': _colony_proc_color(addr_tail),
                'klass': t, 'interval': node.get('interval'),
            }}
            for direction, ports in (
                    ('input', node.get('inputs') or {}),
                    ('output', node.get('outputs') or {})):
                for port, wire in ports.items():
                    store = _wire_to_path(wire)
                    if not store:
                        continue
                    add_store(store)
                    if direction == 'input':
                        add_edge(store, proc_id, 'input', port)
                    else:
                        add_edge(proc_id, store, 'output', port)
            return  # do not recurse into processes' own config
        for k, v in node.items():
            if k.startswith('_') or k in ('inputs', 'outputs', 'config'):
                continue
            walk(path_segs + [k], v)

    walk([], document)

    return {
        'nodes': list(nodes.values()),
        'edges': list(edges.values()),
        'layers': [],
        'legend': {p: c for p, c in _COLONY_PROCESS_COLORS.items()},
    }


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

    # Random placement around center (like the original rigid-cell version).
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

        # Density=1.0 so per-sub-body mass is high enough for stable bending
        # physics. viva-munk's bending example stays stable because it runs a
        # single cell; with 10+ bending cells that divide, low-mass sub-bodies
        # accumulate numerical error at every pivot rebuild and fly apart.
        if i == ecoli_idx:
            # This is the whole-cell ecoli
            ecoli_id, ecoli_body = build_microbe(
                rng, env_size=env_size,
                x=x, y=y, angle=angle,
                length=2.0, radius=0.5, density=50.0,
            )
            # Set initial physical mass to match WCM dry mass
            ecoli_body['mass'] = 380.0  # fg, matches v2ecoli initial dry mass
        else:
            # Adder surrogate cell — match ecoli's density/mass regime
            aid, body = build_microbe(
                rng, env_size=env_size,
                x=x, y=y, angle=angle,
                length=length, radius=radius, density=1.0,
            )
            body['mass'] = 380.0  # fg, match v2ecoli initial dry mass
            body['grow_divide'] = make_adder_grow_divide_process(
                config={'agents_key': 'cells', 'pressure_k': 2.5},
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
                'n_bending_segments': 4,
                'bending_stiffness': 120.0,
                'bending_damping': 20.0,
                'jitter_per_second': 0.0,
            },
            'interval': physics_interval,
            'inputs': {
                'bending_cells': ['cells'],
                'circle_particles': ['particles'],
            },
            'outputs': {
                'bending_cells': ['cells'],
                'circle_particles': ['particles'],
            },
        },

        'pressure': make_pressure_process(
            agents_key='cells',
            contact_slack=0.2,
            pressure_scale=1.0,
            env_size=env_size,
            wall_weight=2.0,
        ),

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

    # Determine max panels needed (usually 1 before division, 2+ after).
    # Lay out 2+ panels into a 2-row grid so wide rows don't squish subplots.
    max_panels = max((len(cells) for _, cells in history if cells), default=1)
    max_panels = max(1, max_panels)
    if max_panels <= 2:
        nrows, ncols = 1, max_panels
    else:
        nrows = 2
        ncols = (max_panels + 1) // 2  # ceil

    fig_w = 5.0 * ncols
    fig_h = 5.0 * nrows

    images = []
    for t, ecoli_cells in history:
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
        t_int = int(t)
        hh, mm, ss = t_int // 3600, (t_int % 3600) // 60, t_int % 60
        fig.suptitle(f't = {hh:02d}:{mm:02d}:{ss:02d}', fontsize=14, y=0.97,
                     fontweight='bold')

        sorted_cells = sorted(ecoli_cells.items(), key=lambda x: x[0])

        for i in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1)

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


def run_colony(duration_min=60, n_adder=9, env_size=40, seed=0,
               from_cache=None):
    """Run the colony simulation and generate report."""
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
    # Ecoli-lineage ids: any cell that ever had an 'ecoli' process attached,
    # including granddaughters born from internal WCM division (whose ids do
    # not prefix-match the original ecoli_id).
    ecoli_lineage = {ecoli_id}
    # Keep reference to mother's EcoliWCM instance so we preserve its history
    # even after division removes it from the colony
    mother_wcm_history = None  # will be set when mother is found

    # Restore mother history from cache if available
    if cached_mother_history:
        # Seed mother_wcm_history with a mutable list so later code can extend it
        mother_wcm_history = list(cached_mother_history)

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

        # Hydrate daughter process specs that were added via _add during
        # division. process-bigraph does not auto-instantiate nested
        # process specs that appear mid-simulation, so we walk the cells
        # dict and instantiate any grow_divide process whose `instance` is
        # missing by looking up its class in core.link_registry.
        _hydrated_any = False
        # First update lineage based on pre-hydration cells so the gate below
        # knows which cells are ecoli descendants. (The lineage-refresh pass
        # further down runs after this block.)
        for _aid in list(sim.state.get('cells', {}).keys()):
            for root in list(ecoli_lineage):
                if _aid == root or _aid.startswith(root + '_'):
                    ecoli_lineage.add(_aid)
                    break
        for aid, cell in sim.state.get('cells', {}).items():
            for proc_key in ('grow_divide', 'ecoli'):
                # Both 'ecoli' and 'grow_divide' keys are broadcast to every
                # cell by bigraph-schema's map-schema fill. Gate each
                # hydration to the cell's real lineage: ecoli ancestors get
                # WCM, everything else gets grow_divide.
                if proc_key == 'ecoli' and aid not in ecoli_lineage:
                    continue
                if proc_key == 'grow_divide' and aid in ecoli_lineage:
                    continue
                spec = cell.get(proc_key)
                if not (isinstance(spec, dict)
                        and spec.get('_type') == 'process'
                        and not spec.get('instance')):
                    continue
                addr = spec.get('address', '')
                if isinstance(addr, dict):
                    cls_name = addr.get('data', '')
                else:
                    cls_name = addr.rsplit(':', 1)[-1] if addr else ''
                cls = core.link_registry.get(cls_name)
                if cls is None:
                    print(f"    [hydrate] no class for {aid}.{proc_key} (addr={addr})")
                    continue
                try:
                    inst = cls(config=spec.get('config') or {}, core=core)
                    spec['instance'] = inst
                    _hydrated_any = True
                    print(f"    [hydrate] {aid}.{proc_key} -> {cls_name}")
                except Exception as ex:
                    print(f"    [hydrate] {aid}.{proc_key} failed: {ex}")
        if _hydrated_any:
            sim.find_instance_paths(sim.state)
            sim.edge_paths = {**sim.process_paths, **sim.step_paths}
            sim.expire_process_paths([('cells',)])

        # Collect chromosome state directly from EcoliWCM instances
        colony_cells = sim.state.get('cells', {})
        # Refresh ecoli lineage by colony-path descent from a known ecoli
        # ancestor. We CANNOT detect lineage by "has an 'ecoli' key" because
        # bigraph-schema's fill() broadcasts the cells map's value-schema
        # default (which contains the full EcoliWCM spec once any daughter
        # carries it) to every cell in the store. The bridge names daughters
        # {parent}_0 / {parent}_1, so lineage is identifiable by prefix.
        for _aid in list(colony_cells.keys()):
            for root in list(ecoli_lineage):
                if _aid == root or _aid.startswith(root + '_'):
                    ecoli_lineage.add(_aid)
                    break
        chrom_snap = {}
        for aid, cell in colony_cells.items():
            if aid not in ecoli_lineage:
                continue
            ecoli_proc = cell.get('ecoli', {})
            inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
            if inst and hasattr(inst, '_composite') and inst._composite is not None:
                chrom_snap[aid] = inst._read_chromosome_state()
            # No standalone-WCM fallback: the in-composite hydrate pass above
            # instantiates the daughter's ecoli process inside `sim`, so it
            # runs on the next tick with state routed back into the cell
            # store. A parallel standalone WCM would fire its own independent
            # division and spawn phantom granddaughters.

        if chrom_snap:
            chromosome_history.append((total, chrom_snap))

        n_cells = len(colony_cells)
        ecoli_alive = ecoli_id in colony_cells
        # Check for daughter cells (ecoli_id_0, ecoli_id_1)
        ecoli_daughters = [k for k in colony_cells
                           if k != ecoli_id and k in ecoli_lineage]
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
            if aid in ecoli_lineage:
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
        if aid != ecoli_id and aid in ecoli_lineage:
            daughter_ids_from_colony.append(aid)
            ecoli_proc = cell.get('ecoli', {})
            inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
            if inst and hasattr(inst, 'chromosome_history') and inst.chromosome_history:
                daughter_histories[aid] = list(inst.chromosome_history)

    # Also check chunk-collected chromosome_history for daughter data
    for t_chunk, chrom_snap in chromosome_history:
        for aid, chrom in chrom_snap.items():
            if aid != ecoli_id:
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

    # Build frame-synced chromosome data by matching each emitter frame to the
    # nearest chromosome_history snapshot (which captures all live ecoli
    # lineage cells, including 2nd-gen daughters). Falls back to the mother
    # history / daughter_histories when the per-chunk snapshot is missing.
    synced_chrom = []
    history_times = [t for t, _ in chromosome_history]
    for ft in frame_times:
        frame_chrom = {}

        if history_times:
            # nearest-in-time chunk snapshot (covers all live ecoli-lineage cells)
            best_idx = min(range(len(history_times)),
                           key=lambda i: abs(history_times[i] - ft))
            if abs(history_times[best_idx] - ft) < 120:
                frame_chrom.update(chromosome_history[best_idx][1])

        # Fill from mother_history if the chunk snapshot missed the mother
        if ecoli_id not in frame_chrom and mother_history:
            best = min(mother_history, key=lambda h: abs(h[0] - ft))
            if abs(best[0] - ft) < 120:
                frame_chrom[ecoli_id] = best[1]

        # NOTE: per-daughter WCM chromosome_history cannot be used here
        # because each daughter's internal composite has its own clock that
        # starts at 0, so its h[0] timestamps are not comparable to the
        # outer simulation time. The chunk-captured chromosome_history
        # (global time) is authoritative for daughters.

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

    # Colony network viz (Cytoscape, same viewer as other network reports).
    network_html = os.path.join(REPORT_DIR, 'colony_network.html')
    try:
        graph = build_colony_graph(doc)
        n_proc = sum(1 for n in graph['nodes'] if n['data']['kind'] == 'process')
        n_store = sum(1 for n in graph['nodes'] if n['data']['kind'] == 'store')
        with open(network_html, 'w') as nf:
            nf.write(_viz_render_html(
                graph,
                title='Colony composition',
                subtitle=f'{n_proc} processes · {n_store} stores · {len(graph["edges"])} edges',
            ))
        print(f"Network viz saved: {network_html}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Network viz failed: {e}")
        network_html = None

    # Mass timeseries GIF: one line per cell, colored the same as the colony
    # GIF. When a cell divides its line stops and its daughters start ~half
    # its mass with lineage-matched colors.
    mass_gif_path = os.path.join(REPORT_DIR, 'mass_timeseries.gif')
    print(f"Generating mass timeseries GIF ({len(results)} frames, skip={gif_skip})...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        import io as _io

        # Keep raw frame_ts for the animation cursor, but deduplicate each
        # trajectory to only (time, mass) points where mass actually changed.
        # Without this, slow processes (ecoli WCM at 60s) produce 6 identical
        # 10s-spaced points and render as flat-then-jump; fast processes
        # (adder at 10s) render smoother, so the visual cadences mismatch.
        raw = {}
        frame_ts = []
        for entry in results:
            t_f = float(entry.get('time', 0) or 0)
            frame_ts.append(t_f)
            for aid, a in (entry.get('agents') or {}).items():
                raw.setdefault(aid, []).append((t_f, float(a.get('mass', 0) or 0)))
        trajectories = {}
        for aid, pts in raw.items():
            keep = []
            for t, m in pts:
                if not keep or abs(m - keep[-1][1]) > 1e-9:
                    keep.append((t, m))
            # Ensure a terminal point at the final observed time so the
            # line extends fully even if mass was constant at the end.
            if keep and pts and keep[-1][0] != pts[-1][0]:
                keep.append((pts[-1][0], keep[-1][1]))
            trajectories[aid] = keep

        t_max_s = max(frame_ts) if frame_ts else 1.0
        m_max = max((m for traj in trajectories.values() for _, m in traj), default=1.0)
        m_max = max(m_max * 1.15, 1.0)

        frames_idx = list(range(0, len(results), max(1, gif_skip)))
        images = []
        for fi in frames_idx:
            t_now = frame_ts[fi]
            fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white', dpi=90)
            for aid, traj in trajectories.items():
                pts = [(t, m) for t, m in traj if t <= t_now]
                if not pts:
                    continue
                xs = [p[0] / 60.0 for p in pts]
                ys = [p[1] for p in pts]
                rgb = _hybrid_color(aid)
                is_ecoli = aid in ecoli_lineage
                lw = 2.2 if is_ecoli else 1.1
                alpha = 1.0 if is_ecoli else 0.75
                ax.plot(xs, ys, color=rgb, lw=lw, alpha=alpha)
                # mark current mass with a dot at the leading edge
                ax.plot(xs[-1], ys[-1], 'o', color=rgb, ms=4,
                        alpha=alpha, zorder=5)
            ax.set_xlim(0, t_max_s / 60.0)
            ax.set_ylim(0, m_max)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Dry mass (fg)')
            ax.grid(alpha=0.25)
            t_int = int(t_now)
            ax.set_title(
                f't = {t_int // 3600:02d}:{(t_int % 3600) // 60:02d}:{t_int % 60:02d}',
                fontsize=11, fontweight='bold')
            buf = _io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            images.append(Image.open(buf).copy())

        if images:
            images[0].save(mass_gif_path, save_all=True,
                           append_images=images[1:],
                           duration=gif_duration_ms, loop=0)
            print(f"Mass GIF saved: {mass_gif_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Mass GIF failed: {e}")
        mass_gif_path = None

    # Generate HTML report
    report_path = os.path.join(REPORT_DIR, 'colony_report.html')

    from v2ecoli.library.repro_banner import banner_html
    repro_banner = banner_html()

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

{repro_banner}

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

        if mass_gif_path and os.path.exists(mass_gif_path):
            f.write("""
<h2>Cell Mass Over Time</h2>
<p>Each line traces one cell&rsquo;s dry mass. The thick coloured line is the
whole-cell <em>E. coli</em> lineage; thin grey lines are surrogates. When a
cell divides its line ends and two daughter lines start at roughly half the
mother&rsquo;s mass, each inheriting a hue-shifted variant of the lineage
colour (green family for <em>E. coli</em>, grey for surrogates).</p>
<div class="media">
  <img src="mass_timeseries.gif" alt="Cell mass over time">
  <div class="media-label">Per-cell dry mass trajectories, synchronized with the colony animation.</div>
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
""")

        if network_html and os.path.exists(network_html):
            f.write("""
<h2>Composition Network</h2>
<p>Interactive Cytoscape graph of the full colony composite — the multibody
physics process, the pressure step, each cell&rsquo;s internal processes
(grow/divide for surrogates, EcoliWCM for the whole-cell <em>E. coli</em>),
and the shared stores that wire them together.</p>
<div class="media"><p><a href="colony_network.html" target="_blank">Open interactive colony network &rarr;</a></p></div>
""")

        f.write("""

<footer style="margin-top:2em; padding-top:1em; border-top:1px solid #e2e8f0; color:#94a3b8; font-size:0.85em;">
v2ecoli colony · pure process-bigraph · <a href="https://github.com/vivarium-collective/v2ecoli">github</a>
</footer>
</body></html>""")

    # Mirror to docs/ so GitHub Pages stays in sync. Also copies the GIFs
    # so the rendered report can resolve its relative <img> paths.
    import shutil
    docs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'docs')
    if os.path.isdir(docs_dir):
        shutil.copy2(report_path, os.path.join(docs_dir, 'colony_report.html'))
        for gif in ('colony.gif', 'chromosome.gif', 'mass_timeseries.gif',
                    'colony_network.html'):
            src = os.path.join(os.path.dirname(report_path), gif)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(docs_dir, gif))

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
    parser.add_argument('--from-cache', type=str, default=None,
                        help='Resume from cached pre-division state (dill pickle)')
    args = parser.parse_args()

    report = run_colony(
        duration_min=args.duration,
        n_adder=args.n_adder,
        env_size=args.env_size,
        from_cache=args.from_cache,
    )

    import subprocess
    subprocess.run(['open', report], capture_output=True)


if __name__ == '__main__':
    main()
