"""
Composite loading for v2ecoli (departitioned architecture).

Mirrors composite.py but uses the departitioned document generator.
"""

from process_bigraph import Composite

from v2ecoli.composite import _build_core, _load_cache_bundle


def make_departitioned_composite(cache_dir=None, seed=0, core=None):
    """Create a departitioned Composite from cache."""
    if core is None:
        core = _build_core()

    from v2ecoli.generate_departitioned import build_departitioned_document

    initial_state, cache = _load_cache_bundle(cache_dir)
    document = build_departitioned_document(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
    )

    return Composite(document, core=core)
