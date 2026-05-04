"""Run v2ecoli with sigma factor competition + sustained ppGpp stress.

Simulates stationary phase: ppGpp clamped at 250,000 molecules (~360 µM).
"""
import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings('ignore')

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
interval = int(sys.argv[2]) if len(sys.argv) > 2 else 1
result_path = sys.argv[3] if len(sys.argv) > 3 else 'out/sigma_stress_test.json'

v2ecoli_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(v2ecoli_dir)
sys.path.insert(0, v2ecoli_dir)

fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(fd, 2)

from v2ecoli.composite import make_composite


def snap(t, cell):
    mass = cell.get('listeners', {}).get('mass', {})
    sigma = cell.get('listeners', {}).get('sigma_factors', {})
    ox = cell.get('listeners', {}).get('oxidative_stress', {})
    return {
        'time': t,
        'dry_mass': float(mass.get('dry_mass', 0)),
        'cell_mass': float(mass.get('cell_mass', 0)),
        'growth_rate': float(mass.get('instantaneous_growth_rate', 0)),
        'f_sigma70': float(sigma.get('f_sigma70', 0)),
        'f_sigma38': float(sigma.get('f_sigma38', 0)),
        'f_sigma32': float(sigma.get('f_sigma32', 0)),
        'f_sigma24': float(sigma.get('f_sigma24', 0)),
        'f_sigma54': float(sigma.get('f_sigma54', 0)),
        'ppgpp_uM': float(sigma.get('ppgpp_uM', 0)),
        'phase': float(sigma.get('phase', 0)),
        'h2o2_uM': float(ox.get('h2o2_uM', 0)),
        'oxyr_fold_change': float(ox.get('oxyr_fold_change', 1)),
        'soxrs_fold_change': float(ox.get('soxrs_fold_change', 1)),
    }


print(f"Running STRESS simulation: {duration}s, {interval}s intervals")
print("Features: ppgpp_regulation + sigma_factor_competition + sustained_stress")

t0 = time.time()
composite = make_composite(
    cache_dir='out/cache',
    seed=0,
    features=['ppgpp_regulation', 'sigma_factor_competition', 'sustained_stress'],
)
load_time = time.time() - t0
print(f"Composite loaded in {load_time:.1f}s")

cell = composite.state['agents']['0']
snapshots = [snap(0, cell)]

t0 = time.time()
total = 0
while total < duration:
    chunk = min(interval, duration - total)
    try:
        composite.run(chunk)
    except Exception as e:
        print(f"  Division/error at t={total}: {e}")
        break
    total += chunk
    cell = composite.state.get('agents', {}).get('0')
    if cell is None:
        print(f"  Cell divided at t={total}")
        break
    s = snap(total, cell)
    snapshots.append(s)

wall_time = time.time() - t0

result = {
    'engine': 'v2ecoli sigma+stress',
    'features': ['ppgpp_regulation', 'sigma_factor_competition', 'sustained_stress'],
    'load_time': load_time,
    'wall_time': wall_time,
    'sim_time': total,
    'snapshots': snapshots,
}

os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nDone: {total}s in {wall_time:.1f}s")
print(f"Results: {result_path}")
