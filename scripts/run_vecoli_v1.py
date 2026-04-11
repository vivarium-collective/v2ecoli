"""Run vEcoli 1.0 (vivarium engine, master branch) and collect snapshots."""
import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings('ignore')

duration = int(sys.argv[1])
interval = int(sys.argv[2])
result_path = sys.argv[3]

vecoli_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vEcoli')
os.chdir(vecoli_dir)
sys.path.insert(0, vecoli_dir)

# Suppress C-level warnings
fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(fd, 2)

# Must be on master branch for v1 engine
import subprocess
current_branch = subprocess.run(['git', 'branch', '--show-current'],
    capture_output=True, text=True).stdout.strip()

if current_branch != 'master':
    subprocess.run(['git', 'stash'], capture_output=True)
    subprocess.run(['git', 'checkout', 'master'], capture_output=True)

try:
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    t0 = time.time()
    sim = EcoliSim.from_cli()
    sim.build_ecoli()
    load_time = time.time() - t0

    def snap(t):
        """Extract snapshot from the vivarium engine state."""
        state = sim.query()
        agents = state.get('agents', {})
        cell = next(iter(agents.values()), {}) if agents else {}
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

    # Build and run via vivarium Engine
    from vivarium.core.engine import Engine

    experiment = Engine(
        processes=sim.ecoli.processes,
        steps=sim.ecoli.steps,
        flow=sim.ecoli.flow,
        topology=sim.ecoli.topology,
        initial_state=sim.generated_initial_state,
        progress_bar=False,
        emitter='timeseries',
    )

    snapshots = []
    t0 = time.time()
    total = 0
    while total < duration:
        chunk = min(interval, duration - total)
        experiment.update(chunk)
        total += chunk
        snapshots.append(snap(total))

    wall_time = time.time() - t0

    result = {
        'engine': 'vEcoli 1.0 (vivarium)',
        'load_time': load_time,
        'wall_time': wall_time,
        'sim_time': total,
        'speed': total / wall_time if wall_time > 0 else 0,
        'snapshots': snapshots,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f)

finally:
    # Restore branch
    if current_branch != 'master':
        subprocess.run(['git', 'checkout', current_branch], capture_output=True)
        subprocess.run(['git', 'stash', 'pop'], capture_output=True)
