"""
Composite loading for v2ecoli (reconciled architecture).

Mirrors composite.py but uses the reconciled document generator.
Each allocator layer is a single ReconciledStep that collects
requests, reconciles them proportionally against available supply,
then runs evolve_state with fair allocations.
"""

import os

import dill
from bigraph_schema import allocate_core
from process_bigraph import Composite

from v2ecoli.types import ECOLI_TYPES
from v2ecoli.cache import load_initial_state


def _build_core():
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    return core


def make_reconciled_composite(cache_dir=None, seed=0, core=None):
    """Create a reconciled Composite from cache."""
    if core is None:
        core = _build_core()

    from v2ecoli.generate_reconciled import build_reconciled_document

    initial_state = load_initial_state(
        os.path.join(cache_dir, 'initial_state.json'))

    with open(os.path.join(cache_dir, 'sim_data_cache.dill'), 'rb') as f:
        cache = dill.load(f)

    document = build_reconciled_document(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
    )

    return Composite(document, core=core)
