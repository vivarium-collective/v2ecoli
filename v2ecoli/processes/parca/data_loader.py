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
