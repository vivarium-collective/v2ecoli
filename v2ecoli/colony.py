"""
E. coli Colony Simulation

Places whole-cell E. coli models (v2ecoli) inside the multi-cell
pymunk physics framework. Each cell in the colony is a full
whole-cell model with partitioned processes, coupled to 2D physics
for spatial arrangement, growth-driven size changes, and division.

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
    PymunkProcess, build_microbe, make_initial_state, make_rng)

from v2ecoli.composite import _build_core, make_composite
from v2ecoli.types import ECOLI_TYPES


def make_colony_document(
    n_cells=1,
    env_size=30,
    physics_interval=1.0,
    seed=0,
):
    """Build a colony document with n whole-cell E. coli agents.

    Each cell has:
    - Full v2ecoli whole-cell model (partitioned processes)
    - Physical body in 2D pymunk space (capsule shape)
    - Mass drives physical size (growth → expansion)

    Args:
        n_cells: Number of initial cells.
        env_size: Size of the 2D environment (micrometers).
        physics_interval: Seconds between physics updates.
        seed: Random seed.

    Returns:
        Document dict for Composite().
    """
    rng = make_rng(seed)

    # Build initial physical state for cells
    cells = {}
    for i in range(n_cells):
        # Spread cells in the environment
        x = env_size / 2 + rng.uniform(-5, 5)
        y = env_size / 2 + rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)

        agent_id, cell_body = build_microbe(
            rng, env_size=env_size,
            x=x, y=y, angle=angle,
            length=2.0, radius=0.5, density=0.02,
        )
        cells[agent_id] = cell_body

    # Build the document
    document = {
        # Cell physical states (pymunk agents)
        'cells': cells,

        # 2D physics engine
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

        # Emitter for visualization
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'time': ['global_time'],
        }),
    }

    return document


def make_colony(n_cells=1, env_size=30, seed=0):
    """Create a colony Composite ready to run.

    Returns:
        process_bigraph.Composite instance.
    """
    core = core_import()
    # Register ecoli types alongside multi_cell types
    core.register_types(ECOLI_TYPES)

    doc = make_colony_document(
        n_cells=n_cells,
        env_size=env_size,
        seed=seed,
    )

    return Composite({'state': doc}, core=core)


if __name__ == '__main__':
    import time

    print("Building colony with 3 cells...")
    colony = make_colony(n_cells=3, env_size=30)

    print(f"Cells: {list(colony.state['cells'].keys())}")

    print("Running 10s...")
    t0 = time.time()
    colony.run(10.0)
    print(f"Done in {time.time()-t0:.2f}s")

    cells = colony.state['cells']
    for cid, cell in cells.items():
        loc = cell.get('location', (0, 0))
        mass = cell.get('mass', 0)
        print(f"  {cid}: location=({loc[0]:.1f}, {loc[1]:.1f}), mass={mass:.3f}")
