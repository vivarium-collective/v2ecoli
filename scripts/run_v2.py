"""Run v2ecoli and collect snapshots. Called as subprocess by compare_v1_v2.py."""
import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings('ignore')

duration = int(sys.argv[1])
interval = int(sys.argv[2])
result_path = sys.argv[3]

v2ecoli_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(v2ecoli_dir)
sys.path.insert(0, v2ecoli_dir)

# Suppress C-level warnings
fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(fd, 2)

from v2ecoli.composite import make_composite


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


t0 = time.time()
composite = make_composite(cache_dir='out/cache', seed=0)
load_time = time.time() - t0

cell = composite.state['agents']['0']
snapshots = [snap(0, cell)]

t0 = time.time()
total = 0
while total < duration:
    chunk = min(interval, duration - total)
    try:
        composite.run(chunk)
    except Exception:
        total += chunk
        # Non-fatal error — collect snapshot and continue
        cell = composite.state.get('agents', {}).get('0')
        if cell is not None:
            snapshots.append(snap(total, cell))
        continue
    total += chunk

    cell = composite.state.get('agents', {}).get('0')
    if cell is None:
        break
    snapshots.append(snap(total, cell))

wall_time = time.time() - t0

result = {
    'engine': 'v2ecoli (process-bigraph)',
    'load_time': load_time,
    'wall_time': wall_time,
    'sim_time': total,
    'speed': total / wall_time if wall_time > 0 else 0,
    'snapshots': snapshots,
}

with open(result_path, 'w') as f:
    json.dump(result, f)
