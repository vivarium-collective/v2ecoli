"""Cache colony state just before E. coli division (~40 min).

Saves the Composite + metadata so reports/colony_report.py can resume
from this checkpoint instead of re-running 40 min of simulation.

Usage:
    python scripts/cache_predivision.py
    python reports/colony_report.py --from-cache out/colony/predivision_cache.pkl
"""
import os, sys, time, warnings, dill

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_bigraph import Composite
from process_bigraph.emitter import emitter_from_wires
from multi_cell import core_import
from v2ecoli.bridge import EcoliWCM
from v2ecoli.types import ECOLI_TYPES
from reports.colony_report import make_colony_document

CACHE_PATH = 'out/colony/predivision_cache.pkl'

def main():
    core = core_import()
    core.register_types(ECOLI_TYPES)
    core.register_link('EcoliWCM', EcoliWCM)

    doc, ecoli_id = make_colony_document(n_adder=5, env_size=40, seed=0)
    sim = Composite({'state': doc}, core=core)
    print(f"Built colony, ecoli_id={ecoli_id}")

    # Run to 2400s (40 min) — just before division at ~42-44 min
    chunk = 120
    total = 0
    target = 2400

    # Grab mother WCM reference
    mother_wcm_history = None

    while total < target:
        cells_pre = sim.state.get('cells', {})
        if ecoli_id in cells_pre and mother_wcm_history is None:
            ecoli_proc = cells_pre[ecoli_id].get('ecoli', {})
            inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
            if inst and hasattr(inst, 'chromosome_history'):
                mother_wcm_history = inst.chromosome_history

        step = min(chunk, target - total)
        sim.run(step)
        total += step

        n_cells = len(sim.state.get('cells', {}))
        ecoli_alive = ecoli_id in sim.state.get('cells', {})
        print(f"  t={total}s ({total/60:.0f}min): {n_cells} cells, ecoli={'alive' if ecoli_alive else 'GONE'}")

    # Collect mother's chromosome history
    mother_history = list(mother_wcm_history) if mother_wcm_history else []

    # Get mother's EcoliWCM instance for state extraction
    mother_inst = None
    cells = sim.state.get('cells', {})
    if ecoli_id in cells:
        ecoli_proc = cells[ecoli_id].get('ecoli', {})
        inst = ecoli_proc.get('instance') if isinstance(ecoli_proc, dict) else None
        if inst:
            mother_inst = inst

    # Save cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache = {
        'sim': sim,
        'ecoli_id': ecoli_id,
        'total': total,
        'mother_history': mother_history,
        'mother_inst': mother_inst,
    }
    with open(CACHE_PATH, 'wb') as f:
        dill.dump(cache, f)

    print(f"\nCached at t={total}s → {CACHE_PATH}")
    print(f"Mother history: {len(mother_history)} entries")
    print(f"Mother inst: {mother_inst is not None}")


if __name__ == '__main__':
    main()
