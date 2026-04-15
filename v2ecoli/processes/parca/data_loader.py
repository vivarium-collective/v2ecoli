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


def _install_legacy_pickle_aliases() -> None:
    """Register ``v2parca.*`` and ``vparca.*`` as aliases of the current
    ``v2ecoli.processes.parca.*`` modules so pickles made under the old
    package names unpickle without errors.  Also aliases the bare
    ``reconstruction.*`` / ``wholecell.*`` / ``ecoli.*`` names used by
    vEcoli-originated pickles."""
    # Import the v2ecoli.processes.parca subpackages so they're in
    # sys.modules before we install aliases that point at them.
    for modpath in (
        'v2ecoli.processes.parca',
        'v2ecoli.processes.parca.reconstruction.ecoli.simulation_data',
        'v2ecoli.processes.parca.reconstruction.ecoli.dataclasses',
        'v2ecoli.processes.parca.wholecell.utils.units',
        'v2ecoli.processes.parca.ecoli.library.schema',
    ):
        try:
            importlib.import_module(modpath)
        except Exception:
            pass
    # Alias v2ecoli.processes.parca.X → v2parca.X, vparca.X, and the bare
    # top-level name (X) when X is reconstruction/wholecell/ecoli.
    base = 'v2ecoli.processes.parca'
    for name, mod in list(sys.modules.items()):
        if name == base or name.startswith(base + '.'):
            tail = name[len(base):]  # '' or '.reconstruction....'
            for legacy_root in ('v2parca', 'vparca'):
                sys.modules.setdefault(legacy_root + tail, mod)
            # bare top-level (reconstruction/wholecell/ecoli)
            if tail.startswith('.'):
                bare = tail.lstrip('.')
                first = bare.split('.', 1)[0]
                if first in ('reconstruction', 'wholecell', 'ecoli'):
                    sys.modules.setdefault(bare, mod)


def load_parca_state(path: str | Path | None = None) -> dict[str, Any]:
    """Return the gzipped, pre-computed v2ecoli-ParCa composite state.

    Args:
        path: optional path to a ``parca_state.pkl.gz`` file.  Defaults
            to ``<repo>/models/parca/parca_state.pkl.gz``.

    Returns:
        Dict of top-level stores (``process``, ``cell_specs``, ``mass``,
        ``constants``, …).  See ``models/parca/README.md`` for layout.
    """
    p = Path(path) if path is not None else _DEFAULT_PATH
    _install_legacy_pickle_aliases()
    with gzip.open(p, 'rb') as f:
        return pickle.load(f)
