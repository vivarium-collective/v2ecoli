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

from viva_munk import core_import
from viva_munk.processes.multibody import (
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
    jitter_per_second=1e-4,
    damping_per_second=0.5,
    init_mass=None,
    transport='local',
    emit_cells=True,
    phenotype_store=None,
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
        jitter_per_second: Brownian impulse std applied by pymunk each substep.
            Defaults to viva-munk's 1e-4. (Was hard-coded to 0.5 — ~5000x that —
            which, with the tiny density-seeded body mass, flung cells around in
            the colony.gif. See ``init_mass``.)
        damping_per_second: pymunk velocity damping factor per second.
        init_mass: If set, seed each body's mass to this value (fg) instead of
            build_microbe's density(0.02)-derived mass (~0.04 pymunk units). The
            EcoliWCM ``mass`` output is a fg DELTA that accumulates onto the body
            mass; seeding a realistic dry mass (~200 fg) keeps units coherent
            rather than summing pymunk-density units with femtograms.
        transport: Per-cell EcoliWCM transport — ``'local'`` (single-threaded,
            GIL-bound) or ``'ray'`` (one OS process per cell via the
            process-bigraph Ray protocol). Daughters inherit this through the
            cell config so dynamically-added cells use the same transport.

    Returns:
        Document dict for Composite().
    """
    rng = make_rng(seed)
    address = f'{transport}:EcoliWCM'

    cells = {}
    for i in range(n_cells):
        x = env_size / 2 + rng.uniform(-5, 5)
        y = env_size / 2 + rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)

        # Deterministic initial-cell id (a_0, a_1, …). build_microbe's default id
        # is randomly generated (untied to the seed), so it changes on every
        # build — which breaks resolving a cell by id across two builds (the loom
        # Explorer re-builds the colony to instantiate + drill into a cell).
        # Seeding a stable id keeps the composite reproducible; daughters spawned
        # at division still get fresh unique ids.
        agent_id, cell_body = build_microbe(
            rng, agent_id=f'a_{i}', env_size=env_size,
            x=x, y=y, angle=angle,
            length=2.0, radius=0.5, density=0.02,
        )
        if init_mass is not None:
            cell_body['mass'] = float(init_mass)

        # Embed EcoliWCM process inside each cell. Wiring matches what
        # _handle_division produces for daughters (v2ecoli/bridge.py) so
        # initial cells can themselves divide cleanly: agent_id/location/
        # angle drive daughter placement, and `agents` is the wire the
        # division update writes `{_remove, _add}` to. ``transport`` is
        # threaded into config so daughters inherit local-vs-ray.
        cell_body['ecoli'] = {
            '_type': 'process',
            'address': address,
            'config': {
                'cache_dir': cache_dir,
                'seed': seed + i,
                'transport': transport,
                'init_mass': init_mass,
                'env_size': env_size,
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
                'jitter_per_second': jitter_per_second,
                'damping_per_second': damping_per_second,
            },
            'interval': physics_interval,
            'inputs': {
                'segment_cells': ['cells'],
            },
            'outputs': {
                'segment_cells': ['cells'],
            },
        },

        # Outer colony emitter.
        #
        # The legacy full cells-map capture (emit_cells=True) wires the whole
        # `cells` map into a RAMEmitter, which appends every cell's
        # numpy-array-heavy state to an in-RAM history EVERY tick — measured at
        # ~1.5 MB/tick vs ~0.09 MB/tick with emit_cells=False, i.e. ~94% of the
        # "colony RAM leak" the investigation chased as a native/C leak (it only
        # looked native because numpy buffers are invisible to tracemalloc).
        #
        # Preferred path: pass `phenotype_store` (a runs.<id>.zarr path). The
        # colony then streams a bounded per-cell PHENOTYPE PANEL (mass, length,
        # volume, x, y, angle) to zarr via ColonyPhenotypeEmitter — O(1) RAM AND
        # the queryable per-cell timeseries the phenotype studies need. The
        # legacy emit_cells flag is kept only for back-compat callers.
        'emitter': (
            {
                '_type': 'step',
                'address': 'local:ColonyPhenotypeEmitter',
                'config': {
                    # TYPED shallow schema: the engine gathers only these scalar
                    # per-cell fields, NOT each cell's heavy `ecoli` sub-state.
                    # (Wiring the whole cells map deep-copied ~1.6 MB/tick.)
                    'emit': {
                        'cells': {
                            '_type': 'map',
                            '_value': {
                                'mass': 'float',
                                'length': 'float',
                                'volume': 'float',
                                'location': 'list',
                                'angle': 'float',
                            },
                        },
                        'global_time': 'float',
                    },
                    'out_uri': phenotype_store,
                },
                'inputs': {'cells': ['cells'], 'global_time': ['global_time']},
            }
            if phenotype_store else
            emitter_from_wires(
                {'agents': ['cells'], 'time': ['global_time']}
                if emit_cells else
                {'time': ['global_time']}
            )
        ),
    }

    return document


def make_colony(
    n_cells=1,
    env_size=30,
    cache_dir='out/cache',
    seed=0,
    jitter_per_second=1e-4,
    damping_per_second=0.5,
    init_mass=None,
    transport='local',
    parallel_processes=None,
    emit_cells=True,
    phenotype_store=None,
):
    """Create a colony Composite ready to run.

    Args:
        transport: ``'local'`` (default, single-threaded) or ``'ray'`` (one OS
            process per cell via the process-bigraph Ray protocol).
        parallel_processes: Whether the engine dispatches per-tick process
            updates concurrently. Defaults to True when ``transport='ray'``,
            False otherwise.
        jitter_per_second / damping_per_second / init_mass: physics-fidelity
            knobs — see make_colony_document.

    Returns:
        process_bigraph.Composite instance.
    """
    core = core_import()
    core.register_types(ECOLI_TYPES)
    # Register EcoliWCM so Composite can resolve 'local:EcoliWCM' AND so the
    # Ray protocol's _resolve_target can find it in core.link_registry.
    core.register_link('EcoliWCM', EcoliWCM)
    # Bounded-RAM per-cell phenotype sink (streams to zarr instead of
    # accumulating the cells map in RAM). Registered on the colony's own core
    # so 'local:ColonyPhenotypeEmitter' resolves even outside build_core().
    from v2ecoli.colony_emitter import ColonyPhenotypeEmitter
    core.register_link('ColonyPhenotypeEmitter', ColonyPhenotypeEmitter)

    if transport == 'ray':
        from process_bigraph.protocols import ray as ray_protocol
        ray_protocol.register_types(core)          # register the 'ray:' address
        ray_protocol.register_process_class('EcoliWCM', EcoliWCM)
        if parallel_processes is None:
            parallel_processes = True
    elif parallel_processes is None:
        parallel_processes = False

    doc = make_colony_document(
        n_cells=n_cells,
        env_size=env_size,
        cache_dir=cache_dir,
        seed=seed,
        jitter_per_second=jitter_per_second,
        damping_per_second=damping_per_second,
        init_mass=init_mass,
        transport=transport,
        emit_cells=emit_cells,
        phenotype_store=phenotype_store,
    )

    return Composite(
        {'state': doc, 'parallel_processes': bool(parallel_processes)},
        core=core,
    )


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
