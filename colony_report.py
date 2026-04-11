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
    add_adder_grow_divide_to_agents, make_adder_grow_divide_process)
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

    # Create adder cells
    initial = make_initial_state(
        n_microbes=n_adder,
        n_particles=0,
        env_size=env_size,
        microbe_length_range=(1.5, 2.5),
        microbe_radius_range=(0.4, 0.6),
        microbe_mass_density=0.02,
    )

    # Add mass-based growth/division
    add_adder_grow_divide_to_agents(
        initial,
        agents_key='cells',
        config={'agents_key': 'cells'},
    )

    cells = initial['cells']

    # Add 1 whole-cell E. coli
    ecoli_id, ecoli_body = build_microbe(
        rng, env_size=env_size,
        x=env_size / 2, y=env_size / 2, angle=0,
        length=2.0, radius=0.5, density=0.02,
    )

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
        },
        'outputs': {
            'mass': ['mass'],
            'volume': ['volume'],
            'exchange': ['exchange'],
        },
    }
    ecoli_body.setdefault('local', {})
    ecoli_body.setdefault('volume', 0.0)
    ecoli_body.setdefault('exchange', {})

    cells[ecoli_id] = ecoli_body

    document = {
        'cells': cells,
        'particles': initial.get('particles', {}),

        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'jitter_per_second': 0.5,
                'damping_per_second': 0.3,
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

    # Run in chunks, reporting progress
    chunk = 120  # 2 min chunks
    total = 0
    t0 = time.time()

    while total < duration:
        step = min(chunk, duration - total)
        try:
            sim.run(step)
        except Exception as e:
            print(f"  Warning at t={total+step}s: {type(e).__name__}")
        total += step

        n_cells = len(sim.state.get('cells', {}))
        ecoli_alive = ecoli_id in sim.state.get('cells', {})
        print(f"  t={total}s ({total/60:.0f}min): {n_cells} cells, "
              f"ecoli={'alive' if ecoli_alive else 'gone'}")

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

    # Generate GIF
    os.makedirs(REPORT_DIR, exist_ok=True)
    gif_path = os.path.join(REPORT_DIR, 'colony.gif')
    print(f"Generating GIF ({len(results)} frames)...")
    try:
        simulation_to_gif(
            results,
            config={'env_size': env_size},
            agents_key='agents',
            filename='colony.gif',
            out_dir=REPORT_DIR,
            skip_frames=max(1, len(results) // 100),
            color_by_phylogeny=True,
            show_time_title=True,
        )
        print(f"GIF saved: {gif_path}")
    except Exception as e:
        print(f"GIF generation failed: {e}")
        gif_path = None

    # Generate HTML report
    report_path = os.path.join(REPORT_DIR, 'colony_report.html')

    # Collect cell count history
    cell_counts = []
    for entry in results:
        if isinstance(entry, tuple) and len(entry) == 2:
            t_val, data = entry
            agents = data.get('agents', {})
        elif isinstance(entry, dict):
            t_val = entry.get('time', 0)
            agents = entry.get('agents', {})
        else:
            continue
        cell_counts.append((float(t_val), len(agents)))

    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>E. coli Colony Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f8fafc; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #16a34a; padding-bottom: 8px; }}
h2 {{ color: #166534; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ padding: 6px 16px; border: 1px solid #e2e8f0; }}
th {{ background: #f1f5f9; }}
.gif {{ text-align: center; margin: 1em 0; }}
.gif img {{ max-width: 100%; border: 2px solid #e2e8f0; border-radius: 8px; }}
</style>
</head><body>
<h1>E. coli Colony Simulation</h1>
<p>Mixed colony: 1 whole-cell E. coli (v2ecoli, 55 biological steps) +
{n_adder} adder-based growth/division cells. 2D pymunk physics.</p>

<h2>Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Duration</td><td>{duration_min} min ({duration}s)</td></tr>
<tr><td>WC-Ecoli cells</td><td>1 (EcoliWCM bridge)</td></tr>
<tr><td>Adder cells</td><td>{n_adder}</td></tr>
<tr><td>Environment</td><td>{env_size} x {env_size} µm</td></tr>
<tr><td>Physics interval</td><td>30s</td></tr>
<tr><td>WC-Ecoli interval</td><td>30s</td></tr>
</table>

<h2>Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Build time</td><td>{build_time:.1f}s</td></tr>
<tr><td>Wall time</td><td>{wall_time:.0f}s</td></tr>
<tr><td>Speed</td><td>{duration/wall_time:.1f}x realtime</td></tr>
<tr><td>Initial cells</td><td>{n_initial}</td></tr>
<tr><td>Final cells</td><td>{n_final}</td></tr>
<tr><td>Emitter frames</td><td>{len(results)}</td></tr>
</table>
""")

        if gif_path and os.path.exists(gif_path):
            f.write(f"""
<h2>Colony Animation</h2>
<div class="gif"><img src="colony.gif" alt="Colony simulation"></div>
""")

        f.write("""
<footer>v2ecoli colony · pure process-bigraph</footer>
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
