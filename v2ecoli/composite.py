"""
Composite loading for v2ecoli.

Uses process-bigraph's Composite directly — no custom simulation engine.

Supports three loading modes:
1. From cache directory: make_composite(cache_dir='out/cache')
2. From pre-built document: make_composite(document={...})
3. From pre-loaded state/configs: make_composite(initial_state=..., configs=...)
"""

import os

import dill
from bigraph_schema import allocate_core
from process_bigraph import Composite

from v2ecoli.types import ECOLI_TYPES
from v2ecoli.cache import load_initial_state, save_initial_state, save_json


def _build_core():
    """Create and configure a bigraph-schema core with ecoli types."""
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    return core


def make_composite(document=None, cache_dir=None,
                   initial_state=None, configs=None, unique_names=None,
                   dry_mass_inc_dict=None, seed=0, core=None,
                   features=None):
    """Create a Composite from a document, cache, or configs.

    Loading modes (in priority order):
    1. document: Pre-built document dict
    2. cache_dir: Directory with initial_state.json + sim_data_cache.dill
    3. initial_state + configs: Direct state and config dicts

    Args:
        features: List of feature module names to enable (e.g. ['supercoiling']).
            Defaults to generate.DEFAULT_FEATURES if None.

    Returns:
        A Composite ready for .run(interval).
    """
    if core is None:
        core = _build_core()

    if document is None:
        if cache_dir and os.path.isdir(cache_dir):
            document = _build_from_cache(cache_dir, core, seed,
                                         features=features)
        elif initial_state is not None and configs is not None:
            from v2ecoli.generate import build_document
            document = build_document(
                initial_state=initial_state,
                configs=configs,
                unique_names=unique_names or [],
                dry_mass_inc_dict=dry_mass_inc_dict,
                core=core,
                seed=seed,
                features=features,
            )
        else:
            raise ValueError(
                "Provide one of: document, cache_dir, or "
                "(initial_state + configs)")

    composite = Composite(document, core=core)
    return composite


def _build_from_cache(cache_dir, core, seed=0, features=None):
    """Build a document from cached initial state and process configs."""
    from v2ecoli.generate import build_document

    initial_state = load_initial_state(
        os.path.join(cache_dir, 'initial_state.json'))

    cache_path = os.path.join(cache_dir, 'sim_data_cache.dill')
    with open(cache_path, 'rb') as f:
        cache = dill.load(f)
    # Pint Quantities round-tripped through dill can land on a stale
    # UnitRegistry if a side-effectful import (e.g. vEcoli bigraph_types)
    # has replaced pint.application_registry. Rebind every Quantity in
    # the loaded cache to the shared v2ecoli ureg so downstream
    # arithmetic stays on a single registry.
    from v2ecoli.library.unit_bridge import rebind_cache_quantities
    rebind_cache_quantities(cache)

    return build_document(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
        features=features,
    )


def save_cache(sim_data_path, cache_dir='out/cache', seed=0):
    """Generate and save cache files from simData (vEcoli ParCa output).

    Creates:
    - cache_dir/initial_state.json
    - cache_dir/sim_data_cache.dill
    - cache_dir/metadata.json
    """
    from v2ecoli.library.sim_data import LoadSimData

    os.makedirs(cache_dir, exist_ok=True)

    loader = LoadSimData(sim_data_path=sim_data_path, seed=seed)

    state = loader.generate_initial_state()
    save_initial_state(state, os.path.join(cache_dir, 'initial_state.json'))

    configs = {}
    for name in [
        'post-division-mass-listener', 'ecoli-mass-listener', 'media_update',
        'exchange_data', 'ecoli-tf-unbinding', 'ecoli-tf-binding',
        'ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation',
        'ecoli-transcript-initiation', 'ecoli-polypeptide-initiation',
        'ecoli-chromosome-replication', 'ecoli-protein-degradation',
        'ecoli-rna-degradation', 'ecoli-complexation',
        'ecoli-transcript-elongation', 'ecoli-polypeptide-elongation',
        'ecoli-chromosome-structure', 'ecoli-metabolism',
        'RNA_counts_listener', 'rna_synth_prob_listener',
        'monomer_counts_listener', 'dna_supercoiling_listener',
        'ribosome_data_listener', 'rnap_data_listener',
        'unique_molecule_counts', 'allocator',
    ]:
        try:
            configs[name] = loader.get_config_by_name(name)
        except Exception:
            pass

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule
        .unique_molecule_definitions.keys())

    dry_mass_inc = getattr(loader.sim_data, 'expectedDryMassIncreaseDict', {})

    cache = {
        'configs': configs,
        'unique_names': unique_names,
        'dry_mass_inc_dict': dry_mass_inc,
    }
    cache_path = os.path.join(cache_dir, 'sim_data_cache.dill')
    with open(cache_path, 'wb') as f:
        dill.dump(cache, f)

    save_json({'unique_names': unique_names}, os.path.join(cache_dir, 'metadata.json'))
    print(f"Cache saved to {cache_dir}")
