"""
Per-step, per-timestep v1/v2 diagnostic comparison.

Traces the exact bulk delta from each process at each timestep
to find where v1 and v2 diverge.
"""

import numpy as np
import sys
import copy
import types

# Numpy compat
if not hasattr(np, 'in1d'):
    np.in1d = np.isin


def run_v2_traced(n_timesteps=3, cache_dir='out/cache'):
    """Run v2, capture per-step bulk deltas at each timestep."""
    from v2ecoli.composite import make_composite
    from process_bigraph.composite import get_path

    c = make_composite(cache_dir=cache_dir)
    cell = c.state['agents']['0']
    bulk_ids = cell['bulk']['id']

    all_timesteps = []

    for ts in range(n_timesteps):
        step_deltas = {}
        orig = c.run_steps.__func__

        def make_traced(step_deltas_ref, cell_ref):
            def traced(self, step_paths):
                if step_paths:
                    bulk_before_batch = cell_ref['bulk']['count'].copy()
                    updates = []
                    for step_path in step_paths:
                        step = get_path(self.state, step_path)
                        state = self.core.view(self.schema, self.state, step_path, 'inputs')
                        step_update = self.process_update(step_path, step, state, -1.0, 'outputs')
                        updates.append(step_update)
                    update_paths = self.apply_updates(updates)

                    # Single batch delta (all steps applied atomically)
                    bulk_after = cell_ref['bulk']['count'].copy()
                    delta = bulk_after - bulk_before_batch
                    if np.any(delta != 0):
                        batch_name = '+'.join(p[-1] for p in step_paths)
                        step_deltas_ref[batch_name] = {
                            'delta': delta.copy(),
                            'changed': int((delta != 0).sum()),
                            'net': int(delta.sum()),
                            'n_steps': len(step_paths),
                        }

                    self.expire_process_paths(update_paths)
                    to_run = self.cycle_step_state()
                    if to_run:
                        self.run_steps(to_run)
                    else:
                        self.steps_run = set()
                else:
                    self.steps_run = set()
            return traced

        c.run_steps = types.MethodType(make_traced(step_deltas, cell), c)
        c.run(1.0)

        all_timesteps.append({
            'time': ts + 1,
            'step_deltas': step_deltas,
            'bulk': cell['bulk']['count'].copy(),
            'mass': copy.deepcopy(cell['listeners'].get('mass', {})),
        })

    return all_timesteps, bulk_ids


def run_v1_traced(n_timesteps=3):
    """Run v1, capture per-timestep bulk and mass."""
    from contextlib import chdir
    from wholecell.utils.filepath import ROOT_PATH
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.library.schema import not_a_process

    saved = sys.argv
    sys.argv = [sys.argv[0]]
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file()
        sim.max_duration = n_timesteps
        sim.emitter = 'timeseries'
        sim.divide = False
        sim.build_ecoli()
        v1_initial = sim.generated_initial_state['bulk']['count'].copy()
        sim.run()
    sys.argv = saved

    ts = sim.query()

    all_timesteps = []
    prev_bulk = v1_initial
    for t in range(1, n_timesteps + 1):
        snap = ts.get(float(t), {})
        bulk_struct = snap.get('bulk')
        if bulk_struct is not None:
            if hasattr(bulk_struct, 'dtype') and 'count' in (bulk_struct.dtype.names or []):
                bulk = bulk_struct['count'].copy()
            else:
                bulk = np.array(bulk_struct, dtype=np.int64)
        else:
            bulk = prev_bulk.copy()

        mass = snap.get('listeners', {}).get('mass', {})
        mass_clean = {}
        for k, v in mass.items():
            try:
                mass_clean[k] = float(v)
            except (TypeError, ValueError):
                pass

        all_timesteps.append({
            'time': t,
            'bulk': bulk,
            'delta': bulk - prev_bulk,
            'mass': mass_clean,
        })
        prev_bulk = bulk

    return all_timesteps


def compare(v2_data, v1_data, bulk_ids):
    """Compare v2 and v1 step-by-step."""
    n = min(len(v2_data), len(v1_data))

    for t in range(n):
        v2 = v2_data[t]
        v1 = v1_data[t]

        v2_bulk = v2['bulk']
        v1_bulk = v1['bulk']
        total_diff = v2_bulk.astype(np.int64) - v1_bulk.astype(np.int64)
        diffs = (total_diff != 0).sum()

        v2_dm = v2['mass'].get('dry_mass', 0)
        v1_dm = v1['mass'].get('dry_mass', 0)

        print(f"\n{'='*80}")
        print(f"TIMESTEP {t+1}")
        print(f"{'='*80}")
        print(f"  Bulk diffs: {diffs}")
        print(f"  V2 dry_mass: {float(v2_dm):.4f}")
        print(f"  V1 dry_mass: {float(v1_dm):.4f}")
        print(f"  Mass diff: {float(v2_dm) - float(v1_dm):+.4f} fg")

        # Mass components
        print(f"\n  Mass components:")
        for k in ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']:
            v2v = float(v2['mass'].get(k, 0))
            v1v = float(v1['mass'].get(k, 0))
            pct = abs(v2v - v1v) / max(abs(v1v), 1e-10) * 100
            marker = ' ***' if pct > 1 else ''
            label = k.replace('_mass', '').replace('_', ' ').title()
            print(f"    {label:20s}  v2={v2v:10.2f}  v1={v1v:10.2f}  diff={v2v-v1v:+8.2f}  ({pct:.2f}%){marker}")

        # V2 per-batch deltas
        print(f"\n  V2 per-batch bulk deltas:")
        for batch_name, info in v2['step_deltas'].items():
            if info['changed'] > 0:
                label = batch_name if len(batch_name) < 60 else batch_name[:57] + '...'
                steps = f" [{info['n_steps']} steps]" if info['n_steps'] > 1 else ""
                print(f"    {label:<60s} changed={info['changed']:>5d}  net={info['net']:>12d}{steps}")

        # Top molecule differences
        if diffs > 0:
            print(f"\n  Top 10 molecule differences (v2 - v1):")
            top = np.argsort(np.abs(total_diff))[-10:][::-1]
            for idx in top:
                if total_diff[idx] != 0:
                    name = str(bulk_ids[idx])
                    print(f"    {name:40s}  diff={total_diff[idx]:>+10d}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    import warnings
    warnings.filterwarnings('ignore')

    print(f"Running v2 for {n} timesteps...")
    v2_data, bulk_ids = run_v2_traced(n)
    print(f"Running v1 for {n} timesteps...")
    v1_data = run_v1_traced(n)

    compare(v2_data, v1_data, bulk_ids)
