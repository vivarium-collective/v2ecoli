"""Shared infrastructure for v2ecoli composites.

Provides the bigraph-schema ``core`` (with ECOLI_TYPES registered) and the
cache-loading/saving helpers used by every architecture's
``@composite_generator`` function in ``v2ecoli/composites/``.
"""

from __future__ import annotations

import copy
import functools
import os
from typing import Any

import dill
from bigraph_schema import allocate_core

from v2ecoli.cache import load_initial_state, save_initial_state, save_json
from v2ecoli.library.cache_version import (
    StaleCacheError,
    verify_cache_version,
    write_cache_version,
)
# Import at module load so the shared pint UnitRegistry has
# nucleotide/amino_acid/count defined before any dill.load hydrates
# a Quantity whose unit string references those names.
from v2ecoli.library.unit_bridge import rebind_cache_quantities  # noqa: F401
from v2ecoli.types import ECOLI_TYPES


__all__ = [
    "build_core",
    "load_cache_bundle",
    "save_cache",
    "save_sim_input",
    "StaleCacheError",
]


def build_core():
    """Create and configure a bigraph-schema core with ecoli types."""
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    return core


@functools.lru_cache(maxsize=4)
def _load_cache_bundle_cached(cache_dir):
    """Raw loader — memoized by cache_dir."""
    initial_state = load_initial_state(
        os.path.join(cache_dir, 'initial_state.json'))
    cache_path = os.path.join(cache_dir, 'sim_data_cache.dill')
    # Side-effectful imports (upstream vEcoli's bigraph_types) can
    # replace pint.application_registry after unit_bridge has
    # registered nucleotide/amino_acid/count on our ureg. Reassert
    # the app registry so dill.load's Quantity unpickle resolves
    # those custom units against the registry that has them.
    import pint
    from v2ecoli.types.quantity import ureg
    pint.set_application_registry(ureg)
    with open(cache_path, 'rb') as f:
        cache = dill.load(f)
    rebind_cache_quantities(cache)
    return initial_state, cache


def load_cache_bundle(cache_dir: str) -> dict[str, Any]:
    """Load the ParCa cache bundle from ``cache_dir``.

    Returns a flat dict containing ``initial_state`` plus whatever keys the
    underlying dill cache provides (typically ``configs``, ``unique_names``,
    ``dry_mass_inc_dict``).

    The heavy work (reading the dill, rebinding pint Quantities onto the
    shared UnitRegistry) is memoized per ``cache_dir``; ``initial_state`` is
    deep-copied because ``build_document`` mutates it, while the cache dict
    is returned by reference (read-only).

    Fails fast with ``StaleCacheError`` if ``cache_version.json`` doesn't
    match the current sim_data / unit-bridge / composite-wiring inputs —
    without this, a stale cache surfaces as obscure AttributeErrors deep
    inside mass_listener / equilibrium.
    """
    verify_cache_version(cache_dir)
    initial_state, cache = _load_cache_bundle_cached(cache_dir)
    return {"initial_state": copy.deepcopy(initial_state), **cache}


_CACHE_CONFIG_NAMES = [
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
]

def _write_sim_input_bundle(loader, bundle_dir):
    """Write the simulation-input bundle from an instantiated LoadSimData.

    Shared body of ``save_cache`` (path-based) and ``save_sim_input``
    (live-object). Emits ``initial_state.json``, ``sim_data_cache.dill``,
    ``metadata.json``, and the cache-version marker into ``bundle_dir``.
    """
    os.makedirs(bundle_dir, exist_ok=True)

    state = loader.generate_initial_state()
    save_initial_state(state, os.path.join(bundle_dir, 'initial_state.json'))

    config_names = list(_CACHE_CONFIG_NAMES)
    if getattr(loader, 'has_plasmid', False):
        config_names.append('ecoli-plasmid-replication')

    configs = {}
    failed: list[tuple[str, Exception]] = []
    for name in config_names:
        try:
            configs[name] = loader.get_config_by_name(name)
        except Exception as e:  # noqa: BLE001 — surface the name+error, don't lose it
            # Don't crash the whole cache build on one bad config: the
            # legacy vEcoli sim_data doesn't always expose every
            # config-getter's inputs (e.g. ``ecoli-metabolism-redux``
            # needs redux-specific attrs).  But DO print each failure
            # loudly — silently dropping ``ecoli-mass-listener`` /
            # ``ecoli-metabolism`` produces a cache that crashes the
            # online sim's Equilibrium step with a divide-by-zero on
            # ``listeners.mass.cell_mass``, which is obscure to debug.
            failed.append((name, e))
    if failed:
        print(f"  sim-input bundle: {len(failed)} config(s) failed to build "
              f"and were omitted:")
        for name, exc in failed:
            print(f"    - {name}: {type(exc).__name__}: {exc}")

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule
        .unique_molecule_definitions.keys())

    dry_mass_inc = getattr(loader.sim_data, 'expectedDryMassIncreaseDict', {})

    cache = {
        'configs': configs,
        'unique_names': unique_names,
        'dry_mass_inc_dict': dry_mass_inc,
    }
    cache_path = os.path.join(bundle_dir, 'sim_data_cache.dill')
    with open(cache_path, 'wb') as f:
        dill.dump(cache, f)

    save_json({'unique_names': unique_names},
              os.path.join(bundle_dir, 'metadata.json'))
    write_cache_version(bundle_dir)
    print(f"Sim-input bundle saved to {bundle_dir}")


def save_cache(sim_data_path, cache_dir='out/cache', seed=0,
               has_plasmid=False, mechanistic_replisome=False,
               condition=None, critical_mass_scale=1.0,
               c_period_minutes=None, d_period_minutes=None,
               dnaa_txn_scale=1.0, dnaa_constitutive=False,
               dnaa_stable=False, dnaa_translation_efficiency=None):
    """Generate the simulation-input bundle from a dilled SimulationDataEcoli.

    Prefer ``save_sim_input(sim_data, ...)`` when the SimulationDataEcoli is
    already in memory — this entry point exists for callers that only have a
    pickle path (legacy vEcoli ``simData.cPickle``).

    Set ``has_plasmid=True`` to bake the ecoli-plasmid-replication config
    (used by scripts/build_plasmid_cache.py). ``mechanistic_replisome=True``
    requires the full replisome subunit complement before chromosome /
    plasmid replication will initiate (matches LoadSimData's stricter
    initiation gate). Pass ``condition`` to bake a non-basal growth
    condition (e.g. ``"acetate"``) into the cache; if omitted, the bundle
    inherits whatever ``sim_data.condition`` is already set to.
    """
    from v2ecoli.library.sim_data import LoadSimData
    loader = LoadSimData(sim_data_path=sim_data_path, seed=seed,
                         condition=condition,
                         has_plasmid=has_plasmid,
                         mechanistic_replisome=mechanistic_replisome,
                         critical_mass_scale=critical_mass_scale,
                         c_period_minutes=c_period_minutes,
                         d_period_minutes=d_period_minutes,
                         dnaa_txn_scale=dnaa_txn_scale,
                         dnaa_constitutive=dnaa_constitutive,
                         dnaa_stable=dnaa_stable,
                         dnaa_translation_efficiency=dnaa_translation_efficiency)
    _write_sim_input_bundle(loader, cache_dir)


def save_sim_input(sim_data, bundle_dir='out/cache', seed=0,
                   has_plasmid=False, mechanistic_replisome=False,
                   condition=None, critical_mass_scale=1.0,
                   c_period_minutes=None, d_period_minutes=None,
                   dnaa_txn_scale=1.0, dnaa_constitutive=False,
                   dnaa_stable=False, dnaa_translation_efficiency=None):
    """Generate the simulation-input bundle from a live ``SimulationDataEcoli``.

    Skips the ~300 MB dill round-trip that ``save_cache`` performs to load
    sim_data from disk. Use this when you already have a hydrated
    ``SimulationDataEcoli`` in hand (e.g. straight off the parca composite or
    its fixture) — the resulting bundle is byte-for-byte equivalent to what
    ``save_cache`` would produce from the same sim_data dilled to a file.

    See ``save_cache`` for ``has_plasmid`` / ``mechanistic_replisome`` /
    ``condition``.
    """
    from v2ecoli.library.sim_data import LoadSimData
    loader = LoadSimData(sim_data=sim_data, seed=seed,
                         condition=condition,
                         has_plasmid=has_plasmid,
                         mechanistic_replisome=mechanistic_replisome,
                         critical_mass_scale=critical_mass_scale,
                         c_period_minutes=c_period_minutes,
                         d_period_minutes=d_period_minutes,
                         dnaa_txn_scale=dnaa_txn_scale,
                         dnaa_constitutive=dnaa_constitutive,
                         dnaa_stable=dnaa_stable,
                         dnaa_translation_efficiency=dnaa_translation_efficiency)
    _write_sim_input_bundle(loader, bundle_dir)
