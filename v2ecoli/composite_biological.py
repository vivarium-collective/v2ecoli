"""
Composite loading for v2ecoli (biological architecture — pilot).

Uses the biological document generator: one merged ReconciledStep for
all three bulk-pool-competing processes, biologically-ordered layers,
fewer flushes.
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


def make_biological_composite(cache_dir=None, seed=0, core=None):
    """Create a biological-architecture Composite from cache."""
    if core is None:
        core = _build_core()

    from v2ecoli.generate_biological import build_biological_document

    initial_state = load_initial_state(
        os.path.join(cache_dir, 'initial_state.json'))

    with open(os.path.join(cache_dir, 'sim_data_cache.dill'), 'rb') as f:
        cache = dill.load(f)

    document = build_biological_document(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
    )

    return Composite(document, core=core)
