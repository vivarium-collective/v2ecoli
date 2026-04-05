"""
Comprehensive v1/v2 comparison to diagnose mass growth discrepancy.

Runs both v1 and v2 for a short time, comparing:
1. Per-timestep bulk molecule counts
2. Per-timestep mass from listeners
3. Per-process update deltas (what each process adds/removes)
4. Metabolism inputs/outputs specifically
"""

import os
import sys
import time
import json
import copy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def run_v2_traced(n_steps=10, cache_dir='out/cache'):
    """Run v2 simulation, collecting per-step bulk deltas and mass."""
    from v2ecoli.composite import make_composite

    composite = make_composite(cache_dir=cache_dir)
    cell = composite.state['agents']['0']

    # Collect per-timestep data
    records = []
    for step_i in range(n_steps):
        # Snapshot before this timestep
        bulk_before = cell['bulk']['count'].copy()
        mass_before = copy.deepcopy(cell['listeners'].get('mass', {}))

        composite.run(1.0)

        bulk_after = cell['bulk']['count'].copy()
        mass_after = copy.deepcopy(cell['listeners'].get('mass', {}))

        delta = bulk_after - bulk_before
        records.append({
            't': step_i + 1,
            'bulk_sum_before': int(bulk_before.sum()),
            'bulk_sum_after': int(bulk_after.sum()),
            'bulk_delta_sum': int(delta.sum()),
            'bulk_delta_nonzero': int(np.count_nonzero(delta)),
            'bulk_delta_positive': int((delta > 0).sum()),
            'bulk_delta_negative': int((delta < 0).sum()),
            'dry_mass_before': float(mass_before.get('dry_mass', 0)),
            'dry_mass_after': float(mass_after.get('dry_mass', 0)),
            'dry_mass_delta': float(mass_after.get('dry_mass', 0) - mass_before.get('dry_mass', 0)),
            'protein_mass': float(mass_after.get('protein_mass', 0)),
            'rna_mass': float(mass_after.get('rna_mass', 0)),
            'dna_mass': float(mass_after.get('dna_mass', 0)),
            'smallMolecule_mass': float(mass_after.get('smallMolecule_mass', 0)),
            'bulk_counts': bulk_after.copy(),
        })

    return records, composite


def run_v1_traced(n_steps=10):
    """Run v1 simulation, collecting per-step bulk deltas and mass."""
    try:
        from ecoli.experiments.ecoli_master_sim import EcoliSim
    except ImportError:
        print('V1 not available')
        return None

    # Temporarily clear sys.argv to prevent EcoliSim's argparse from
    # consuming our arguments
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    sim = EcoliSim.from_cli()
    sys.argv = saved_argv

    # Point to simData
    sim_data_path = os.environ.get(
        'SIM_DATA_PATH',
        os.path.expanduser('~/code/vEcoli/out/kb/simData.cPickle'))
    if not os.path.exists(sim_data_path):
        print(f'V1 simData not found at {sim_data_path}')
        return None
    sim.sim_data_path = sim_data_path
    sim.total_time = n_steps
    sim.log_to_disk_every = 1
    sim.build_ecoli()
    sim.run()

    ts = sim.query()
    agent = ts.get('agents', {}).get('0', {})
    listeners = agent.get('listeners', {})
    mass_ts = listeners.get('mass', {})
    bulk_ts = agent.get('bulk', [])

    records = []
    for i in range(min(n_steps, len(bulk_ts))):
        bulk = bulk_ts[i]
        if isinstance(bulk, (list, np.ndarray)):
            bulk = np.array(bulk)
        elif hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
            bulk = bulk['count']

        record = {'t': i + 1}
        record['bulk_sum'] = int(np.sum(bulk)) if hasattr(bulk, 'sum') else 0

        for key in ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']:
            vals = mass_ts.get(key, [])
            if isinstance(vals, list) and i < len(vals):
                record[key] = float(vals[i])
            else:
                record[key] = 0.0

        record['bulk_counts'] = bulk
        records.append(record)

    return records


def compare_and_report(v2_records, v1_records, outdir='out/comparison'):
    """Generate comparison report."""
    os.makedirs(outdir, exist_ok=True)

    n = min(len(v2_records), len(v1_records) if v1_records else 0)
    if n == 0 and not v1_records:
        # V2-only report
        print('\n=== V2-Only Mass Report ===')
        for r in v2_records:
            print(f"  t={r['t']}: dry_mass={r['dry_mass_after']:.2f} "
                  f"(delta={r['dry_mass_delta']:.4f}), "
                  f"bulk_delta_sum={r['bulk_delta_sum']}, "
                  f"nonzero={r['bulk_delta_nonzero']}")
        return

    print(f'\n=== V1/V2 Comparison ({n} timesteps) ===')

    # Compare bulk counts per timestep
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Bulk sum over time
    v2_sums = [r['bulk_sum_after'] for r in v2_records[:n]]
    v1_sums = [r['bulk_sum'] for r in v1_records[:n]]
    times = list(range(1, n + 1))

    axes[0, 0].plot(times, v2_sums, 'b-o', label='v2', markersize=3)
    axes[0, 0].plot(times, v1_sums, 'r--x', label='v1', markersize=3)
    axes[0, 0].set_title('Bulk Count Sum')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time (s)')

    # 2. Dry mass over time
    v2_dm = [r['dry_mass_after'] for r in v2_records[:n]]
    v1_dm = [r['dry_mass'] for r in v1_records[:n]]
    axes[0, 1].plot(times, v2_dm, 'b-o', label='v2', markersize=3)
    axes[0, 1].plot(times, v1_dm, 'r--x', label='v1', markersize=3)
    axes[0, 1].set_title('Dry Mass (fg)')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Time (s)')

    # 3. Mass delta per timestep
    v2_deltas = [r['dry_mass_delta'] for r in v2_records[:n]]
    axes[0, 2].plot(times, v2_deltas, 'b-o', label='v2 mass delta', markersize=3)
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_title('V2 Dry Mass Delta per Step')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('fg')

    # 4. Per-timestep correlation of bulk counts
    correlations = []
    for i in range(n):
        v2_bulk = v2_records[i]['bulk_counts']
        v1_bulk = v1_records[i]['bulk_counts']
        if hasattr(v2_bulk, 'dtype') and 'count' in v2_bulk.dtype.names:
            v2_bulk = v2_bulk['count']
        v1_bulk = np.array(v1_bulk).flatten()
        v2_bulk = np.array(v2_bulk).flatten()
        if len(v1_bulk) == len(v2_bulk) and len(v1_bulk) > 0:
            corr, _ = pearsonr(v1_bulk.astype(float), v2_bulk.astype(float))
            correlations.append(corr)
        else:
            correlations.append(0)
    axes[1, 0].plot(times[:len(correlations)], correlations, 'g-o', markersize=3)
    axes[1, 0].set_ylim(0.999, 1.001)
    axes[1, 0].set_title('Per-Timestep Bulk Correlation')
    axes[1, 0].set_xlabel('Time (s)')

    # 5. Protein mass comparison
    v2_prot = [r['protein_mass'] for r in v2_records[:n]]
    v1_prot = [r.get('protein_mass', 0) for r in v1_records[:n]]
    axes[1, 1].plot(times, v2_prot, 'b-o', label='v2', markersize=3)
    axes[1, 1].plot(times, v1_prot, 'r--x', label='v1', markersize=3)
    axes[1, 1].set_title('Protein Mass (fg)')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Time (s)')

    # 6. RNA mass comparison
    v2_rna = [r['rna_mass'] for r in v2_records[:n]]
    v1_rna = [r.get('rna_mass', 0) for r in v1_records[:n]]
    axes[1, 2].plot(times, v2_rna, 'b-o', label='v2', markersize=3)
    axes[1, 2].plot(times, v1_rna, 'r--x', label='v1', markersize=3)
    axes[1, 2].set_title('RNA Mass (fg)')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'v1_v2_comparison.png'), dpi=150)
    plt.close()

    # Print summary
    for i in range(min(5, n)):
        v2 = v2_records[i]
        v1 = v1_records[i]
        print(f"  t={i+1}: v2_dm={v2['dry_mass_after']:.2f}, v1_dm={v1['dry_mass']:.2f}, "
              f"diff={v2['dry_mass_after']-v1['dry_mass']:.4f}, "
              f"corr={correlations[i]:.6f}")

    # Find biggest bulk count differences at last timestep
    v2_final = v2_records[n-1]['bulk_counts']
    v1_final = v1_records[n-1]['bulk_counts']
    if hasattr(v2_final, 'dtype') and 'count' in v2_final.dtype.names:
        v2_counts = v2_final['count']
    else:
        v2_counts = np.array(v2_final).flatten()
    v1_counts = np.array(v1_final).flatten()

    if len(v1_counts) == len(v2_counts):
        diffs = v2_counts.astype(float) - v1_counts.astype(float)
        top_idx = np.argsort(np.abs(diffs))[-20:][::-1]
        print(f'\n  Top 20 bulk count differences at t={n}:')
        for idx in top_idx:
            if diffs[idx] != 0:
                # Try to get molecule name
                name = ''
                if hasattr(v2_final, 'dtype') and 'id' in v2_final.dtype.names:
                    name = v2_final['id'][idx].decode() if isinstance(v2_final['id'][idx], bytes) else str(v2_final['id'][idx])
                print(f'    [{idx}] {name}: v2={v2_counts[idx]}, v1={v1_counts[idx]}, diff={diffs[idx]:.0f}')

    print(f'\nPlot saved to {outdir}/v1_v2_comparison.png')


if __name__ == '__main__':
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    print(f'Running v2 for {n_steps} steps...')
    t0 = time.time()
    v2_records, _ = run_v2_traced(n_steps)
    print(f'V2 done in {time.time()-t0:.1f}s')

    print(f'Running v1 for {n_steps} steps...')
    t0 = time.time()
    v1_records = run_v1_traced(n_steps)
    if v1_records:
        print(f'V1 done in {time.time()-t0:.1f}s')
    else:
        print('V1 not available, running v2-only report')

    compare_and_report(v2_records, v1_records)
