"""Integration tests for v2ecoli.

Generates a document, runs the v2ecoli simulation, compares with v1 vEcoli,
and produces a comparison report.
"""

import time
import numpy as np
from contextlib import chdir

from wholecell.utils.filepath import ROOT_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.library.schema import not_a_process

from v2ecoli import generate_document, load_simulation


DOCUMENT_PATH = 'out/ecoli.pickle'
DURATION = 10.0


def test_generate():
    """Generate an E. coli document from EcoliSim."""
    generate_document(DOCUMENT_PATH)
    print('test_generate PASSED')


def test_run():
    """Load document and run simulation for 10 seconds."""
    ecoli = load_simulation(DOCUMENT_PATH)
    bulk_before = ecoli.state['agents']['0']['bulk']['count'].copy()

    ecoli.run(DURATION)

    bulk_after = ecoli.state['agents']['0']['bulk']['count']
    changed = (bulk_before != bulk_after).sum()

    print(f"  global_time: {ecoli.state['global_time']}")
    print(f"  bulk molecules changed: {changed} / {len(bulk_before)}")

    assert ecoli.state['global_time'] == DURATION
    assert changed > 0, "No bulk molecules changed"
    print('test_run PASSED')


def test_compare_v1():
    """Compare v2ecoli results against original vEcoli v1 simulation."""

    # --- Run v1 ---
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file()
        sim.max_duration = int(DURATION)
        sim.emitter = 'timeseries'
        sim.divide = False
        sim.build_ecoli()
        v1_initial = sim.generated_initial_state['bulk']['count'].copy()
        t0 = time.time()
        sim.run()
        v1_runtime = time.time() - t0

    v1_state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
    v1_bulk = v1_state['bulk']['count'].copy()

    # --- Run v2 ---
    ecoli = load_simulation(DOCUMENT_PATH)
    v2_initial = ecoli.state['agents']['0']['bulk']['count'].copy()
    t0 = time.time()
    ecoli.run(DURATION)
    v2_runtime = time.time() - t0
    v2_bulk = ecoli.state['agents']['0']['bulk']['count'].copy()

    # --- Compare bulk counts ---
    assert np.array_equal(v1_initial, v2_initial), "Initial states differ"

    both = (v1_initial != v1_bulk) & (v2_initial != v2_bulk)
    if both.sum() > 0:
        d1 = v1_bulk[both] - v1_initial[both]
        d2 = v2_bulk[both] - v2_initial[both]
        bulk_corr = np.corrcoef(d1.astype(float), d2.astype(float))[0, 1]
    else:
        bulk_corr = 0.0

    v1_changed = (v1_initial != v1_bulk).sum()
    v2_changed = (v2_initial != v2_bulk).sum()

    # --- Print report ---
    print()
    print("=" * 60)
    print("  v2ecoli vs vEcoli Comparison Report")
    print("=" * 60)
    print(f"  Duration:      {DURATION}s simulated")
    print(f"  v1 runtime:    {v1_runtime:.2f}s")
    print(f"  v2 runtime:    {v2_runtime:.2f}s ({v2_runtime/v1_runtime:.1f}x)")
    print(f"  v1 changed:    {v1_changed} / {len(v1_initial)} bulk molecules")
    print(f"  v2 changed:    {v2_changed} / {len(v2_initial)} bulk molecules")
    print(f"  Both changed:  {both.sum()}")
    print(f"  Correlation:   {bulk_corr:.4f}")
    print("=" * 60)
    print()

    assert bulk_corr > 0.90, f"Bulk correlation too low: {bulk_corr:.4f}"
    print('test_compare_v1 PASSED')


if __name__ == '__main__':
    test_generate()
    test_run()
    test_compare_v1()
