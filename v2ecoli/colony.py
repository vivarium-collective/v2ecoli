"""
E. coli Colony Simulation

Places whole-cell E. coli models (v2ecoli) inside the multi-cell
pymunk physics framework. Each cell in the colony has:
- Full v2ecoli whole-cell model (55 biological steps) via EcoliWCM bridge
- Physical body in 2D pymunk space (capsule shape)
- Whole-cell dry mass drives the physical body's mass
- Spatial interactions (collisions, jitter) via pymunk 2D physics

Usage:
    from v2ecoli.colony import make_colony
    colony = make_colony(n_cells=2, env_size=30)
    colony.run(100.0)
"""

import os
import numpy as np
from process_bigraph import Composite
from process_bigraph.emitter import emitter_from_wires

from multi_cell import core_import
from multi_cell.processes.multibody import (
    PymunkProcess, build_microbe, make_rng)

from v2ecoli.bridge import EcoliWCM, ecoli_document
from v2ecoli.types import ECOLI_TYPES


def make_colony_document(
    n_cells=1,
    env_size=30,
    physics_interval=1.0,
    ecoli_interval=1.0,
    cache_dir='out/cache',
    seed=0,
):
    """Build a colony document with n whole-cell E. coli agents.

    Each cell has:
    - Physical body (pymunk_agent capsule)
    - Embedded EcoliWCM process (whole-cell model via bridge)
    - Mass output from EcoliWCM drives physical body mass

    Args:
        n_cells: Number of initial cells.
        env_size: Size of the 2D environment (micrometers).
        physics_interval: Seconds between physics updates.
        ecoli_interval: Seconds between whole-cell model updates.
        cache_dir: Path to v2ecoli sim_data cache.
        seed: Random seed.

    Returns:
        Document dict for Composite().
    """
    rng = make_rng(seed)

    cells = {}
    for i in range(n_cells):
        x = env_size / 2 + rng.uniform(-5, 5)
        y = env_size / 2 + rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)

        agent_id, cell_body = build_microbe(
            rng, env_size=env_size,
            x=x, y=y, angle=angle,
            length=2.0, radius=0.5, density=0.02,
        )

        # Embed EcoliWCM process inside each cell
        cell_body['ecoli'] = {
            '_type': 'process',
            'address': 'local:EcoliWCM',
            'config': {
                'cache_dir': cache_dir,
                'seed': seed + i,
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

        # Initialize stores that EcoliWCM writes to
        cell_body.setdefault('local', {})
        cell_body.setdefault('volume', 0.0)
        cell_body.setdefault('exchange', {})

        cells[agent_id] = cell_body

    document = {
        'cells': cells,

        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'jitter_per_second': 0.5,
                'damping_per_second': 0.5,
            },
            'interval': physics_interval,
            'inputs': {
                'segment_cells': ['cells'],
            },
            'outputs': {
                'segment_cells': ['cells'],
            },
        },

        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'time': ['global_time'],
        }),
    }

    return document


def make_colony(n_cells=1, env_size=30, cache_dir='out/cache', seed=0):
    """Create a colony Composite ready to run.

    Returns:
        process_bigraph.Composite instance.
    """
    core = core_import()
    core.register_types(ECOLI_TYPES)
    # Register EcoliWCM so Composite can resolve 'local:EcoliWCM'
    core.register_link('EcoliWCM', EcoliWCM)

    doc = make_colony_document(
        n_cells=n_cells,
        env_size=env_size,
        cache_dir=cache_dir,
        seed=seed,
    )

    return Composite({'state': doc}, core=core)


if __name__ == '__main__':
    import time
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    print(f"Building colony with {n} E. coli cell(s)...")
    t0 = time.time()
    colony = make_colony(n_cells=n, env_size=30)
    print(f"Built in {time.time()-t0:.1f}s")

    print(f"Cells: {list(colony.state['cells'].keys())}")

    print(f"Running {dur}s...")
    t0 = time.time()
    colony.run(dur)
    wall = time.time() - t0
    print(f"Done in {wall:.1f}s ({dur/wall:.1f}x realtime)")

    cells = colony.state['cells']
    for cid, cell in cells.items():
        loc = cell.get('location', (0, 0))
        mass = cell.get('mass', 0)
        vol = cell.get('volume', 0)
        print(f"  {cid}: loc=({loc[0]:.1f},{loc[1]:.1f}), mass={mass:.1f}fg, vol={vol:.4f}fL")
