"""Run vEcoli 1.0 (vivarium engine, master branch) and collect snapshots."""
import os, sys, json, time, warnings
import numpy as np

# np.in1d was removed in NumPy 2.x; shim for vEcoli master compatibility
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

warnings.filterwarnings('ignore')

duration = int(sys.argv[1])
interval = int(sys.argv[2])
result_path = sys.argv[3]

# Strip our extra args so EcoliSim.from_cli() doesn't choke on them
sys.argv = sys.argv[:1]

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
        from ecoli.library.schema import not_a_process
        state = experiment.state.get_value(condition=not_a_process)
        agents = state.get('agents', {})
        cell = next(iter(agents.values()), {}) if agents else {}
        mass = cell.get('listeners', {}).get('mass', {})
        unique = cell.get('unique', {})
        # Per-type active unique-molecule counts (rich species detail).
        uc = {}
        for nm, arr in (unique or {}).items():
            if hasattr(arr, 'dtype') and getattr(arr.dtype, 'names', None) \
                    and '_entryState' in arr.dtype.names:
                uc[nm] = int(arr['_entryState'].sum())
        n_chrom = uc.get('full_chromosome', 0)
        n_forks = uc.get('active_replisome', 0)
        # Bulk molecular-species summary.
        bulk = cell.get('bulk')
        if bulk is not None and hasattr(bulk, 'dtype'):
            bc = bulk['count'] if (bulk.dtype.names and 'count' in bulk.dtype.names) else bulk
            bc = np.asarray(bc)
            bulk_total = int(bc.sum())
            bulk_nonzero = int((bc > 0).sum())
            bulk_n = int(bc.size)
        else:
            bulk_total = bulk_nonzero = bulk_n = 0
        return {
            'unique_counts': uc,
            'bulk_total': bulk_total,
            'bulk_species_nonzero': bulk_nonzero,
            'bulk_n_species': bulk_n,
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
        try:
            experiment.update(chunk)
        except Exception:
            total += chunk
            break
        total += chunk
        try:
            snapshots.append(snap(total))
        except Exception:
            pass  # snapshot failed but keep simulating

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
