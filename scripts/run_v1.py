"""Run vEcoli and collect snapshots. Called as subprocess by reports/v1_v2_report.py."""
import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings('ignore')

# Args: duration, interval, result_path
duration = int(sys.argv[1])
interval = int(sys.argv[2])
result_path = sys.argv[3]

vecoli_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vEcoli')
os.chdir(vecoli_dir)
sys.path.insert(0, vecoli_dir)

# Suppress C-level warnings
fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(fd, 2)

from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.composites.ecoli_composite import build_composite_native
from ecoli.library.bigraph_types import ECOLI_TYPES
from process_bigraph import Composite
from bigraph_schema import allocate_core

sim = EcoliSim.from_cli()
sim.processes = sim._retrieve_processes(sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes)
sim.topology = sim._retrieve_topology(sim.topology, sim.processes, sim.swap_processes, sim.log_updates)
sim.process_configs = sim._retrieve_process_configs(sim.process_configs, sim.processes)

core = allocate_core()
core.register_types(ECOLI_TYPES)

t0 = time.time()
state = build_composite_native(core, sim.config)
ecoli = Composite(dict(schema=dict(), state=state), core=core)
ecoli.to_run = []
load_time = time.time() - t0


def snap(t, cell):
    mass = cell.get('listeners', {}).get('mass', {})
    unique = cell.get('unique', {})
    fc = unique.get('full_chromosome')
    n_chrom = 0
    if fc is not None and hasattr(fc, 'dtype') and '_entryState' in fc.dtype.names:
        n_chrom = int(fc['_entryState'].sum())
    rep = unique.get('active_replisome')
    n_forks = 0
    if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
        n_forks = int(rep['_entryState'].sum())
    return {
        'time': t,
        'dry_mass': float(mass.get('dry_mass', 0)),
        'cell_mass': float(mass.get('cell_mass', 0)),
        'protein_mass': float(mass.get('protein_mass', 0)),
        'rna_mass': float(mass.get('rRna_mass', 0)) + float(mass.get('tRna_mass', 0)) + float(mass.get('mRna_mass', 0)),
        'rRna_mass': float(mass.get('rRna_mass', 0)),
        'tRna_mass': float(mass.get('tRna_mass', 0)),
        'mRna_mass': float(mass.get('mRna_mass', 0)),
        'dna_mass': float(mass.get('dna_mass', 0)),
        'smallMolecule_mass': float(mass.get('smallMolecule_mass', 0)),
        'water_mass': float(mass.get('water_mass', 0)),
        'volume': float(mass.get('volume', 0)),
        'instantaneous_growth_rate': float(mass.get('instantaneous_growth_rate', 0)),
        'n_chromosomes': n_chrom,
        'n_forks': n_forks,
    }


snapshots = []
cell = ecoli.state['agents']['0']
snapshots.append(snap(0, cell))

t0 = time.time()
total = 0
while total < duration:
    chunk = min(interval, duration - total)
    ecoli.run(chunk)
    total += chunk
    cell = ecoli.state['agents']['0']
    snapshots.append(snap(total, cell))

wall_time = time.time() - t0

result = {
    'engine': 'vEcoli (vivarium/composite)',
    'load_time': load_time,
    'wall_time': wall_time,
    'sim_time': total,
    'speed': total / wall_time if wall_time > 0 else 0,
    'snapshots': snapshots,
}

with open(result_path, 'w') as f:
    json.dump(result, f)
