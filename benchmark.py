"""
Benchmark: v2ecoli (new, partitioned) vs vEcoli composite branch.

Runs both engines for a short simulation and compares wall-clock time.
Requires:
- v2ecoli cache at out/cache/
- vEcoli on the composite branch with simData available
"""

import time
import sys
import os

DURATION = 10.0  # seconds of simulation time


def benchmark_v2ecoli():
    """Run the new v2ecoli partitioned composite."""
    print("=" * 60)
    print(f"v2ecoli (new, partitioned) — {DURATION}s simulation")
    print("=" * 60)

    from v2ecoli.composite import make_composite

    print("Loading composite from cache...")
    t0 = time.time()
    composite = make_composite(cache_dir='out/cache', seed=0)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")

    print(f"Running {DURATION}s...")
    t0 = time.time()
    composite.run(DURATION)
    run_time = time.time() - t0

    cell = composite.state['agents']['0']
    gt = cell.get('global_time', 0.0)
    mass = cell.get('listeners', {}).get('mass', {})
    dm = mass.get('dry_mass', 0)

    print(f"  Completed: t={gt}s, dry_mass={float(dm):.1f}fg")
    print(f"  Wall time: {run_time:.2f}s")
    print(f"  Ratio: {run_time/DURATION:.2f}x real-time")
    print()
    return run_time, float(dm)


def benchmark_vecoli():
    """Run vEcoli composite branch engine."""
    print("=" * 60)
    print(f"vEcoli (composite branch) — {DURATION}s simulation")
    print("=" * 60)

    # Add vEcoli to path
    vecoli_path = os.path.join(os.path.dirname(__file__), '..', 'vEcoli')
    vecoli_path = os.path.abspath(vecoli_path)
    if vecoli_path not in sys.path:
        sys.path.insert(0, vecoli_path)

    from ecoli.composites.ecoli_composite import build_composite_native
    from ecoli.library.bigraph_types import ECOLI_TYPES
    from process_bigraph import Composite
    from bigraph_schema import allocate_core

    # We need to build the sim config from scratch — use vEcoli's default
    # config loading. This requires the full vEcoli machinery.
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    print("Building EcoliSim config...")
    t0 = time.time()
    sim = EcoliSim.from_cli([
        '--total_time', str(int(DURATION)),
        '--engine', 'composite',
    ])
    config_time = time.time() - t0
    print(f"  Config built in {config_time:.2f}s")

    core = allocate_core()
    core.register_types(ECOLI_TYPES)

    print("Building composite document...")
    t0 = time.time()
    state = build_composite_native(core, sim.config)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    print("Creating Composite...")
    t0 = time.time()
    ecoli = Composite({'schema': {}, 'state': state}, core=core)
    ecoli.to_run = []
    create_time = time.time() - t0
    print(f"  Created in {create_time:.2f}s")

    print(f"Running {DURATION}s...")
    t0 = time.time()
    ecoli.run(float(DURATION))
    run_time = time.time() - t0

    cell = ecoli.state['agents']['0']
    gt = cell.get('global_time', 0.0)
    mass = cell.get('listeners', {}).get('mass', {})
    dm = mass.get('dry_mass', 0)

    print(f"  Completed: t={gt}s, dry_mass={float(dm):.1f}fg")
    print(f"  Wall time: {run_time:.2f}s")
    print(f"  Ratio: {run_time/DURATION:.2f}x real-time")
    print()
    return run_time, float(dm)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = {}

    # Run v2ecoli first (lighter imports)
    try:
        v2_time, v2_mass = benchmark_v2ecoli()
        results['v2ecoli'] = (v2_time, v2_mass)
    except Exception as e:
        print(f"v2ecoli FAILED: {e}")
        import traceback; traceback.print_exc()
        results['v2ecoli'] = None

    # Run vEcoli composite
    try:
        v1_time, v1_mass = benchmark_vecoli()
        results['vEcoli'] = (v1_time, v1_mass)
    except Exception as e:
        print(f"vEcoli FAILED: {e}")
        import traceback; traceback.print_exc()
        results['vEcoli'] = None

    # Compare
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for name, res in results.items():
        if res:
            t, m = res
            print(f"  {name:20s}: {t:.2f}s wall, dry_mass={m:.1f}fg")
        else:
            print(f"  {name:20s}: FAILED")

    if results.get('v2ecoli') and results.get('vEcoli'):
        v2t, _ = results['v2ecoli']
        v1t, _ = results['vEcoli']
        ratio = v2t / v1t
        print(f"\n  v2ecoli / vEcoli = {ratio:.2f}x")
        if ratio < 1.5:
            print("  -> GOOD: v2ecoli is within 1.5x of vEcoli")
        elif ratio < 2.0:
            print("  -> OK: v2ecoli is within 2x of vEcoli")
        else:
            print(f"  -> SLOW: v2ecoli is {ratio:.1f}x slower than vEcoli")
