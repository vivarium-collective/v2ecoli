"""Shared cell-division helpers for the comparison multigen runners.

Used by BOTH the genuine-vEcoli (vivarium-process) runner
(:mod:`v2ecoli.library.vivarium_ecoli_engine`) and the v2ecoli runner:
daughter-id phylogeny + dry-mass / chromosome-count / mass-increment
readers off a cell's state. The old upstream COLONY-wrapper
(``UpstreamDivision`` step + ``build_upstream_agents_composite``) was
removed when the comparison consolidated on the single supported vEcoli
loader (vivarium-process); only these stateless helpers remain.
"""
from __future__ import annotations

import numpy as np


def daughter_phylogeny_id(mother_id):
    return [str(mother_id) + "0", str(mother_id) + "1"]


def _dry_mass(states) -> float:
    listeners = states.get("listeners") or {}
    mass = listeners.get("mass") or {} if isinstance(listeners, dict) else {}
    dm = mass.get("dry_mass", 0.0) if isinstance(mass, dict) else 0.0
    try:
        return float(dm)
    except (TypeError, ValueError):
        return 0.0


def _n_chromosomes(unique) -> int:
    if not isinstance(unique, dict):
        return 0
    fc = unique.get("full_chromosome")
    if fc is None or not hasattr(fc, "dtype") or fc.dtype.names is None:
        return 0
    if "_entryState" in fc.dtype.names:
        return int(fc["_entryState"].view(np.bool_).sum())
    return len(fc)


def _inc_to_fg(inc) -> float:
    """Coerce an ``expectedDryMassIncreaseDict`` value to a plain fg float.

    Upstream stores these as Unum quantities (``333.8 [fg]``); convert via
    ``asNumber(units.fg)`` when available, else assume already-fg float.
    """
    if hasattr(inc, "asNumber"):
        try:
            from wholecell.utils import units
            return float(inc.asNumber(units.fg))
        except Exception:
            try:
                return float(inc.magnitude)  # pint fallback
            except Exception:
                return 0.0
    try:
        return float(inc)
    except (TypeError, ValueError):
        return 0.0
