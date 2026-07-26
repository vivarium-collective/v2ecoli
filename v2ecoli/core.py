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
    # Register emitters as links so they're discoverable (dashboard scans
    # core.link_registry for Emitter subclasses). Parquet stays the default
    # (workspace.yaml runtime.default_emitter = parquet). Be defensive: these
    # carry heavy optional deps — register what imports successfully and never
    # let an emitter import break build_core.
    try:
        from process_bigraph.emitter import SQLiteEmitter, RAMEmitter
        core.register_link("SQLiteEmitter", SQLiteEmitter)
        core.register_link("RAMEmitter", RAMEmitter)
    except Exception:
        pass
    try:
        from pbg_emitters import ParquetEmitter
        core.register_link("ParquetEmitter", ParquetEmitter)
    except Exception:
        pass
    try:
        from pbg_emitters import XArrayEmitter
        core.register_link("XArrayEmitter", XArrayEmitter)
    except Exception:
        pass
    try:
        # Bounded-RAM per-cell phenotype sink for the colony composite (streams
        # a scalar panel to zarr instead of accumulating the cells map in RAM).
        from v2ecoli.colony_emitter import ColonyPhenotypeEmitter
        core.register_link("ColonyPhenotypeEmitter", ColonyPhenotypeEmitter)
    except Exception:
        pass
    # Placeholder labeled-vector types so a SERIALIZED composite (the dashboard
    # composite-runner builds documents via build_generator, which doesn't run
    # the derivers' initialize() side-effects) can resolve ``overwrite[<vec>]``
    # port schemas at realize time. CountsDeriver / RnapData re-register these
    # with their real shape + element labels in initialize(); the labels are
    # what the output_metadata walker reads. Resolution only needs the type to
    # exist, so a bare array placeholder is enough.
    for _vec in ('monomer_counts_vec', 'rna_init_event_per_cistron_vec'):
        try:
            core.register_type(_vec, {'_inherit': 'array', '_data': 'int64'})
        except Exception:
            pass
    # Pulled-in external composites (pbg-ketchup): register its Process classes
    # so local:KetchupEstimator resolves in dashboard runs. Guarded — a missing
    # pbg_ketchup must never break build_core for the rest of v2ecoli.
    try:
        from pbg_ketchup import KetchupEstimator, KetchupDynamicEstimator
        core.register_link("KetchupEstimator", KetchupEstimator)
        core.register_link("KetchupDynamicEstimator", KetchupDynamicEstimator)
    except Exception:
        pass
    # Pulled-in external reactor physics (pbg-bioreactordesign): register
    # BiRDTransportProcess so local:BiRDTransportProcess resolves in the mbp-03
    # coupled-reactor composite. Guarded — a missing pbg_bioreactordesign must
    # never break build_core for the rest of v2ecoli.
    try:
        from pbg_bioreactordesign import BiRDTransportProcess
        core.register_link("BiRDTransportProcess", BiRDTransportProcess)
        # BiRDTransportHours: the seconds->hours time-base adapter the coupled
        # composite actually wires (v2ecoli steps in seconds; the transport math
        # is in hours). See v2ecoli/steps/bird_transport_hours.py.
        from v2ecoli.steps.bird_transport_hours import BiRDTransportHours
        core.register_link("BiRDTransportHours", BiRDTransportHours)
    except Exception:
        pass
    # Report-card Steps (resolved by name in the comparison harness).
    try:
        from scripts._compare.report_cards import REPORT_CARD_STEPS
        core.register_links(REPORT_CARD_STEPS)
    except Exception:  # noqa: BLE001 — never let card registration break build_core
        pass
    return core


# Importing v2ecoli.core also registers the pulled-in pbg-ketchup composite
# *generators* (the @composite_generator decorators fire on import), so the
# dashboard's run subprocess — which does `from v2ecoli.core import build_core`
# then looks up the generator in the registry — can resolve ketchup_baseline /
# ketchup_dynamic. Guarded so it's a no-op when pbg-ketchup isn't installed.
try:
    import pbg_ketchup.composites  # noqa: F401
except Exception:
    pass


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
    """
    initial_state, cache = _load_cache_bundle_cached(cache_dir)
    # The translation_supply / trna_charging selector flags were removed from
    # the elongation process config_schema (model choice is now a wiring
    # choice). Strip them from pre-existing caches so old caches still build.
    # Idempotent; safe to mutate the memoized cache once.
    pe_cfg = cache.get("configs", {}).get("ecoli-polypeptide-elongation")
    if isinstance(pe_cfg, dict):
        pe_cfg.pop("translation_supply", None)
        pe_cfg.pop("trna_charging", None)
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

    configs = {}
    failed: list[tuple[str, Exception]] = []
    for name in _CACHE_CONFIG_NAMES:
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

    # Also dump the RAW SimulationData as vEcoli's classic simData.cPickle so the
    # vEcoli composite (ecoli.library.sim_data.LoadSimData, via sim_data_path)
    # can consume the SAME ParCa output as the v2ecoli composite — identical
    # sim_data isolates the MODEL difference in the v2ecoli↔vEcoli comparison.
    # v2ecoli's own composite uses the `configs` dict above and ignores this file.
    import pickle as _pickle
    simdata_path = os.path.join(bundle_dir, 'simData.cPickle')
    with open(simdata_path, 'wb') as f:
        _pickle.dump(loader.sim_data, f, protocol=_pickle.HIGHEST_PROTOCOL)

    # Record the media this bundle was BUILT for. A per-condition bundle bakes
    # media_id into its configs at generation; recording it here lets a later
    # run cheaply detect a stale/wrong-media bundle (e.g. a with_aa bundle that
    # baked 'minimal' before the condition->media fix) without loading the
    # 164MB dill. Pull it from the same configs the composite actually runs on.
    _bundle_media_id = None
    try:
        _bundle_media_id = configs.get('media_update', {}).get('media_id')
    except AttributeError:
        pass
    save_json({'unique_names': unique_names, 'media_id': _bundle_media_id},
              os.path.join(bundle_dir, 'metadata.json'))
    write_cache_version(bundle_dir)
    print(f"Sim-input bundle saved to {bundle_dir}")


def save_cache(sim_data_path, cache_dir='out/cache', seed=0):
    """Generate the simulation-input bundle from a dilled SimulationDataEcoli.

    Prefer ``save_sim_input(sim_data, ...)`` when the SimulationDataEcoli is
    already in memory — this entry point exists for callers that only have a
    pickle path (legacy vEcoli ``simData.cPickle``).
    """
    from v2ecoli.library.sim_data import LoadSimData
    loader = LoadSimData(sim_data_path=sim_data_path, seed=seed)
    _write_sim_input_bundle(loader, cache_dir)


def save_sim_input(sim_data, bundle_dir='out/cache', seed=0,
                   condition=None, fixed_media=None):
    """Generate the simulation-input bundle from a live ``SimulationDataEcoli``.

    Skips the ~300 MB dill round-trip that ``save_cache`` performs to load
    sim_data from disk. Use this when you already have a hydrated
    ``SimulationDataEcoli`` in hand (e.g. straight off the parca composite or
    its fixture) — the resulting bundle is byte-for-byte equivalent to what
    ``save_cache`` would produce from the same sim_data dilled to a file.

    ``condition`` (e.g. "acetate") and ``fixed_media`` (e.g. "minimal_acetate")
    select a non-default nutrient condition so the generated initial state
    reflects that condition's growth rate / doubling time — both already live in
    the ParCa state's ``condition_to_doubling_time`` / saved media, so no refit
    is needed. Omitted → default basal / minimal (glucose).
    """
    from v2ecoli.library.sim_data import LoadSimData
    kwargs = {"sim_data": sim_data, "seed": seed}
    if condition is not None:
        kwargs["condition"] = condition
    if fixed_media is not None:
        kwargs["fixed_media"] = fixed_media
    loader = LoadSimData(**kwargs)
    _write_sim_input_bundle(loader, bundle_dir)
