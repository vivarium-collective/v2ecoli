"""
Composite loading for v2ecoli (replication-initiation architecture).

A v2ecoli-only architecture (no upstream vEcoli counterpart) that diverges
from ``baseline`` to incorporate a more biologically detailed model of
chromosome-replication initiation: explicit DnaA-ATP / DnaA-ADP species,
oriC binding-state-driven initiation, RIDA, DDAH, DARS1/2, and SeqA
sequestration.

Curated reference for the new biology lives in
``docs/references/replication_initiation.md`` (and the underlying PDF),
codified for assertions in
``v2ecoli.data.replication_initiation.molecular_reference``.

This composite is **functionally identical to ``baseline``** at the time of
this scaffolding PR — divergent processes will land in follow-up PRs as
laid out in the draft-PR plan.
"""

from process_bigraph import Composite

from v2ecoli.composite import _build_core, _load_cache_bundle


def make_replication_initiation_composite(
    cache_dir=None, seed=0, core=None, features=None,
):
    """Create a replication-initiation Composite from cache.

    Mirrors ``make_composite`` but routes through
    ``v2ecoli.generate_replication_initiation.build_replication_initiation_document``
    so future divergent wiring can land without touching the baseline path.
    """
    if core is None:
        core = _build_core()

    from v2ecoli.generate_replication_initiation import (
        build_replication_initiation_document,
    )

    initial_state, cache = _load_cache_bundle(cache_dir)
    document = build_replication_initiation_document(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
        features=features,
    )

    return Composite(document, core=core)
