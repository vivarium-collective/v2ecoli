"""Compare a simulation run against the saved baseline."""
import json
import numpy as np
from v2ecoli.composite import make_composite


def run_and_compare(label='test', duration=60):
    """Run simulation and compare against baseline."""
    baseline_mass = json.load(open('out/baseline_mass.json'))
    baseline_bulk = np.load('out/baseline_bulk.npy')

    composite = make_composite(cache_dir='out/cache')

    snapshots = []
    for t in range(duration):
        composite.run(1.0)
        cell = composite.state['agents']['0']
        mass = cell.get('listeners', {}).get('mass', {})
        snapshots.append({
            'time': t + 1,
            'cell_mass': float(mass.get('cell_mass', 0)),
            'dry_mass': float(mass.get('dry_mass', 0)),
            'protein_mass': float(mass.get('protein_mass', 0)),
            'rna_mass': float(mass.get('rna_mass', 0)),
            'dna_mass': float(mass.get('dna_mass', 0)),
            'smallmolecule_mass': float(mass.get('smallmolecule_mass', 0)),
            'water_mass': float(mass.get('water_mass', 0)),
            'growth': float(mass.get('growth', 0)),
        })

    bulk_final = composite.state['agents']['0']['bulk']['count'].copy()

    # --- Compare ---
    print(f'\n{"="*60}')
    print(f'  Comparison: {label} vs baseline ({duration}s)')
    print(f'{"="*60}')

    # Mass trajectories
    metrics = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallmolecule_mass']
    for m in metrics:
        base_vals = np.array([s[m] for s in baseline_mass[:duration]])
        test_vals = np.array([s[m] for s in snapshots])
        if base_vals.sum() == 0:
            continue
        pct_err = np.abs(test_vals - base_vals) / np.maximum(np.abs(base_vals), 1e-10) * 100
        max_err = pct_err.max()
        mean_err = pct_err.mean()
        final_diff = test_vals[-1] - base_vals[-1]
        match = 'MATCH' if max_err < 0.01 else ('OK' if max_err < 1.0 else 'DIFF')
        print(f'  {m:25s}: mean_err={mean_err:.4f}%, max_err={max_err:.4f}%, '
              f'final_diff={final_diff:+.4f}fg  [{match}]')

    # Bulk correlation
    base_changed = baseline_bulk != baseline_bulk  # will recompute
    both_nonzero = (baseline_bulk != 0) | (bulk_final != 0)
    if both_nonzero.sum() > 0:
        corr = np.corrcoef(baseline_bulk.astype(float), bulk_final.astype(float))[0, 1]
        exact_match = (baseline_bulk == bulk_final).sum()
        total = len(baseline_bulk)
        diff_count = total - exact_match
        print(f'  {"bulk_correlation":25s}: {corr:.6f}')
        print(f'  {"bulk_exact_match":25s}: {exact_match}/{total} ({diff_count} differ)')

    # Growth comparison
    base_growth = sum(s['growth'] for s in baseline_mass[:duration])
    test_growth = sum(s['growth'] for s in snapshots)
    print(f'  {"total_growth":25s}: baseline={base_growth:.4f}, test={test_growth:.4f}, '
          f'diff={test_growth-base_growth:+.4f}fg')

    print(f'{"="*60}\n')
    return snapshots, bulk_final


if __name__ == '__main__':
    run_and_compare('current')
