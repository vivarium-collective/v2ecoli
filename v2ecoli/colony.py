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

        # Colony emitter: global_time only whenever we're capturing per-cell
        # data the leak-free way (or when emit_cells is off).
        #
        # The heavy per-cell capture does NOT go through an emitter. Emitters
        # deep-copy the wired `cells` map every tick — each cell's embedded
        # 55-process WCM state — costing ~1.6 MB/tick of RSS that the OS never
        # reclaims (the "colony RAM leak", ~94% of it, invisible to tracemalloc
        # because it's numpy buffers). A *process* gets a cheap typed VIEW of the
        # cells instead (that's why PymunkProcess, also wired to ['cells'], is
        # leak-free). So per-cell phenotypes are recorded by a process
        # (ColonyPhenotypeRecorder, added below when `phenotype_store` is set).
        # The legacy emit_cells=True full-map RAMEmitter is kept only for
        # back-compat callers that read the emitted trajectory (e.g. the GIF).
        'emitter': emitter_from_wires(
            {'time': ['global_time']}
            if (phenotype_store or not emit_cells) else
            {'agents': ['cells'], 'time': ['global_time']}
        ),
    }

    # Leak-free per-cell phenotype capture: a process typed map[pymunk_agent]
    # sees a shallow view of the cells (never the deep `ecoli` state), so it
    # streams mass/length/volume/x/y/angle to zarr at O(1) RAM.
    if phenotype_store:
        document['phenotype_recorder'] = {
            '_type': 'process',
            'address': 'local:ColonyPhenotypeRecorder',
            'config': {'out_uri': phenotype_store},
            'interval': ecoli_interval,
            'inputs': {
                'cells': ['cells'],
                'global_time': ['global_time'],
            },
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
    # so 'local:ColonyPhenotypeRecorder' resolves even outside build_core().
    from v2ecoli.colony_emitter import ColonyPhenotypeRecorder
    core.register_link('ColonyPhenotypeRecorder', ColonyPhenotypeRecorder)

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
