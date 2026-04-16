"""Convenience loader for the pre-computed ParCa state shipped in
``models/parca/parca_state.pkl.gz``.

Lets downstream callers skip the ~70-minute ParCa run.  The file ships
as a gzipped pickle; the bigraph-schema ``.pbg`` refactor that will
replace it is tracked in ``models/parca/README.md``.
"""

from __future__ import annotations

import gzip
import importlib
import pickle
import sys
from pathlib import Path
from typing import Any


# ``v2ecoli/processes/parca/data_loader.py`` → repo root is four parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _REPO_ROOT / 'models' / 'parca' / 'parca_state.pkl.gz'

# Module-name remapping applied during unpickling.  Pickles created by
# v2parca or vEcoli embed module paths like ``v2parca.reconstruction.ecoli…``
# or bare ``reconstruction.ecoli…``.  This table rewrites them to the
# canonical ``v2ecoli.processes.parca.*`` path **without** touching
# ``sys.modules`` globally — which is critical because v2ecoli's own
# code depends on the pypi ``wholecell.*`` / ``reconstruction.*`` from
# vEcoli[dev], and global aliasing would clobber those.
_MODULE_REMAP_PREFIXES = [
    ('v2parca.',          'v2ecoli.processes.parca.'),
    ('vparca.',           'v2ecoli.processes.parca.'),
    ('reconstruction.',   'v2ecoli.processes.parca.reconstruction.'),
    ('wholecell.',        'v2ecoli.processes.parca.wholecell.'),
    ('ecoli.',            'v2ecoli.processes.parca.ecoli.'),
]


def _ensure_parca_modules_loaded() -> None:
    """Pre-import the v2ecoli.processes.parca submodules that the pickle's
    class references resolve to, so ``find_class`` finds them in
    ``sys.modules``."""
    for modpath in (
        'v2ecoli.processes.parca',
        'v2ecoli.processes.parca.reconstruction.ecoli.simulation_data',
        'v2ecoli.processes.parca.reconstruction.ecoli.dataclasses',
        'v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.process',
        'v2ecoli.processes.parca.wholecell.utils.units',
        'v2ecoli.processes.parca.ecoli.library.schema',
    ):
        try:
            importlib.import_module(modpath)
        except Exception:
            pass


class _RemappingUnpickler(pickle.Unpickler):
    """Unpickler that rewrites legacy module names to the merged
    ``v2ecoli.processes.parca.*`` namespace on the fly.

    This avoids global ``sys.modules`` aliasing, so the pypi
    ``wholecell.*`` / ``reconstruction.*`` from vEcoli[dev] remain
    intact for v2ecoli's own non-parca code."""

    def find_class(self, module: str, name: str):
        for old_prefix, new_prefix in _MODULE_REMAP_PREFIXES:
            if module == old_prefix.rstrip('.') or module.startswith(old_prefix):
                module = new_prefix + module[len(old_prefix):]
                break
        return super().find_class(module, name)


def _install_legacy_pickle_aliases() -> None:
    """Install ``v2parca.*`` / ``vparca.*`` aliases in ``sys.modules``
    so that code outside the loader (e.g. dill.load, compare scripts)
    can also unpickle legacy artifacts.

    Only installs v2parca/vparca aliases — does NOT touch the bare
    ``wholecell.*`` / ``reconstruction.*`` names to avoid clobbering
    the pypi vEcoli versions that v2ecoli's own processes depend on."""
    _ensure_parca_modules_loaded()
    base = 'v2ecoli.processes.parca'
    for name, mod in list(sys.modules.items()):
        if name == base or name.startswith(base + '.'):
            tail = name[len(base):]
            for legacy_root in ('v2parca', 'vparca'):
                sys.modules.setdefault(legacy_root + tail, mod)


def load_parca_state(path: str | Path | None = None) -> dict[str, Any]:
    """Return the gzipped, pre-computed v2ecoli-ParCa composite state.

    Uses a custom ``Unpickler`` that remaps legacy module names
    (``v2parca.*``, ``vparca.*``, bare ``reconstruction.*`` /
    ``wholecell.*``) to the canonical ``v2ecoli.processes.parca.*``
    path — without modifying ``sys.modules`` globally.

    Args:
        path: optional path to a ``parca_state.pkl.gz`` file.  Defaults
            to ``<repo>/models/parca/parca_state.pkl.gz``.

    Returns:
        Dict of top-level stores (``process``, ``cell_specs``, ``mass``,
        ``constants``, …).  See ``models/parca/README.md`` for layout.
    """
    p = Path(path) if path is not None else _DEFAULT_PATH
    _ensure_parca_modules_loaded()
    with gzip.open(p, 'rb') as f:
        return _RemappingUnpickler(f).load()


# Sibling composite stores that are produced by Steps 1–9 but never
# installed onto ``sim_data_root`` during the run (the ParCa composite
# keeps pure-data leaves at sibling store paths; see
# ``v2ecoli/processes/parca/composite.py`` STORE_PATH).  Downstream
# online-sim code reaches these via ``sim_data.<attr>``, so the
# extraction step must copy them over.  Each tuple is
# ``(store_key, sim_data_attr, install_if_present_and_empty_only)``.
_SIM_DATA_SIBLING_STORES: tuple[tuple[str, str, bool], ...] = (
    # Step 8 emits this; never lands on sim_data_root.
    ('expected_dry_mass_increase_dict', 'expectedDryMassIncreaseDict', False),
    # Step 1 initializes an empty dict on sim_data_root; steps populate
    # the sibling store.  Overwrite only if the root's copy is empty.
    ('translation_supply_rate', 'translation_supply_rate', True),
)


def hydrate_sim_data_from_state(state: dict[str, Any]) -> Any:
    """Extract ``sim_data_root`` and install sibling composite stores.

    The ParCa composite stores per-nutrient / per-condition data leaves
    (``expected_dry_mass_increase_dict``, ``translation_supply_rate``,
    …) as *siblings* of ``sim_data_root`` in the store tree, not as
    attributes of the sim_data object itself.  Downstream online-sim
    code — e.g. ``LoadSimData.get_mass_listener_config`` and
    ``LoadSimData.get_metabolism_config`` — reaches these via
    ``sim_data.<attr>``, so extraction without hydration leaves them
    missing or empty and produces a cache that crashes the online
    simulation's ``Equilibrium`` / ``PolypeptideElongation`` steps.

    Args:
        state: the dict returned by ``load_parca_state`` or by
            ``composite.state`` after ``build_parca_composite`` runs.

    Returns:
        The ``sim_data_root`` object, with the sibling stores installed
        as attributes in-place so the returned object is fully
        consumable by the online pipeline.
    """
    sim_data = state['sim_data_root']
    for store_key, sd_attr, install_only_if_empty in _SIM_DATA_SIBLING_STORES:
        if store_key not in state:
            continue
        value = state[store_key]
        if install_only_if_empty:
            existing = getattr(sim_data, sd_attr, None)
            # Only overwrite when the root's copy is empty/missing.
            if existing and existing != value:
                continue
        setattr(sim_data, sd_attr, value)

    # Defensive cleanup: an older ``condition_defs.tsv`` shipped with a
    # trailing ``""`` row that parsed into ``condition_to_doubling_time[""] = ""``.
    # That empty-string doubling time crashes
    # ``LoadSimData._precompute_biomass_concentrations`` with
    # ``AttributeError: 'str' object has no attribute 'to'`` because
    # ``unum_to_pint('')`` blows up.  The TSV is fixed going forward,
    # but existing pickles still carry the ghost row — strip it here.
    ctdt = getattr(sim_data, 'condition_to_doubling_time', None)
    if isinstance(ctdt, dict) and '' in ctdt and not ctdt['']:
        del ctdt['']

    return sim_data
