"""
Benchmark: v2ecoli (new, partitioned) vs vEcoli composite branch.

Runs each engine in a separate subprocess to avoid type conflicts.
"""

import subprocess
import sys
import os
import json

DURATION = 60.0

V2ECOLI_SCRIPT = f"""
import time, json, os
os.chdir('/Users/eranagmon/code/v2ecoli')
from v2ecoli.composite import make_composite

t0 = time.time()
composite = make_composite(cache_dir='out/cache', seed=0)
load_time = time.time() - t0

t0 = time.time()
composite.run({DURATION})
run_time = time.time() - t0

cell = composite.state['agents']['0']
dm = float(cell.get('listeners', {{}}).get('mass', {{}}).get('dry_mass', 0))
cm = float(cell.get('listeners', {{}}).get('mass', {{}}).get('cell_mass', 0))

print(json.dumps({{'load': load_time, 'run': run_time, 'dry_mass': dm, 'cell_mass': cm}}))
"""

VECOLI_SCRIPT = f"""
import time, json, os
os.chdir('/Users/eranagmon/code/vEcoli')
from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.composites.ecoli_composite import build_composite_native
from ecoli.library.bigraph_types import ECOLI_TYPES
from process_bigraph import Composite
from bigraph_schema import allocate_core

sim = EcoliSim.from_cli()
sim.processes = sim._retrieve_processes(
    sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes)
sim.topology = sim._retrieve_topology(
    sim.topology, sim.processes, sim.swap_processes, sim.log_updates)
sim.process_configs = sim._retrieve_process_configs(
    sim.process_configs, sim.processes)

core = allocate_core()
core.register_types(ECOLI_TYPES)

t0 = time.time()
state = build_composite_native(core, sim.config)
ecoli = Composite({{'schema': {{}}, 'state': state}}, core=core)
ecoli.to_run = []
load_time = time.time() - t0

t0 = time.time()
ecoli.run({DURATION})
run_time = time.time() - t0

cell = ecoli.state['agents']['0']
dm = float(cell.get('listeners', {{}}).get('mass', {{}}).get('dry_mass', 0))
cm = float(cell.get('listeners', {{}}).get('mass', {{}}).get('cell_mass', 0))

print(json.dumps({{'load': load_time, 'run': run_time, 'dry_mass': dm, 'cell_mass': cm}}))
"""


def run_benchmark(name, script):
    print(f"\n--- {name} ---")
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        stderr = result.stderr.strip().split('\n')
        for line in stderr[-5:]:
            print(f"  {line}")
        return None

    for line in result.stdout.strip().split('\n'):
        try:
            data = json.loads(line)
            print(f"  Load: {data['load']:.2f}s, Run: {data['run']:.2f}s")
            print(f"  dry_mass={data['dry_mass']:.1f}fg, cell_mass={data['cell_mass']:.1f}fg")
            print(f"  Speed: {DURATION/data['run']:.1f}x faster than real-time")
            return data
        except (json.JSONDecodeError, KeyError):
            continue
    print("  FAILED: no valid output")
    return None


if __name__ == '__main__':
    print("=" * 60)
    print(f"BENCHMARK: {DURATION}s simulation")
    print("=" * 60)

    v1 = run_benchmark("vEcoli (composite branch)", VECOLI_SCRIPT)
    v2 = run_benchmark("v2ecoli (new, partitioned)", V2ECOLI_SCRIPT)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if v1 and v2:
        print(f"  {'':20s} {'vEcoli':>12s} {'v2ecoli':>12s}")
        print(f"  {'Load time':20s} {v1['load']:>10.2f}s {v2['load']:>10.2f}s")
        print(f"  {'Run time':20s} {v1['run']:>10.2f}s {v2['run']:>10.2f}s")
        print(f"  {'dry_mass':20s} {v1['dry_mass']:>10.1f}fg {v2['dry_mass']:>10.1f}fg")
        print(f"  {'Speed':20s} {DURATION/v1['run']:>9.1f}x {DURATION/v2['run']:>9.1f}x")
        ratio = v2['run'] / v1['run']
        print(f"\n  v2ecoli / vEcoli = {ratio:.2f}x")
        mass_diff = abs(v2['dry_mass'] - v1['dry_mass']) / v1['dry_mass'] * 100
        print(f"  Mass difference: {mass_diff:.1f}%")
        if ratio <= 1.2:
            print("  EXCELLENT: v2ecoli matches vEcoli performance")
        elif ratio <= 1.5:
            print("  GOOD: v2ecoli within 1.5x of vEcoli")
        elif ratio <= 2.0:
            print("  OK: v2ecoli within 2x of vEcoli")
        else:
            print(f"  SLOW: v2ecoli is {ratio:.1f}x slower")
    else:
        for name, data in [("vEcoli", v1), ("v2ecoli", v2)]:
            if data:
                print(f"  {name}: run={data['run']:.2f}s, dry_mass={data['dry_mass']:.1f}fg")
            else:
                print(f"  {name}: FAILED")
