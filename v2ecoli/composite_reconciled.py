"""
Composite loading for v2ecoli (reconciled architecture).

Mirrors composite.py but uses the reconciled document generator.
Each allocator layer is a single ReconciledStep that collects
requests, reconciles them proportionally against available supply,
then runs evolve_state with fair allocations.
"""

from process_bigraph import Composite

from v2ecoli.composite import _build_core, _load_cache_bundle


def make_reconciled_composite(cache_dir=None, seed=0, core=None):
    """Create a reconciled Composite from cache."""
    if core is None:
        core = _build_core()

    from v2ecoli.generate_reconciled import build_reconciled_document

    initial_state, cache = _load_cache_bundle(cache_dir)
    document = build_reconciled_document(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
    )

    return Composite(document, core=core)
