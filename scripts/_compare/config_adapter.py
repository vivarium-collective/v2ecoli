"""Resolve a vEcoli config and translate/diff it against v2ecoli's schema."""
from __future__ import annotations

from typing import Any


def schema_diff(vecoli: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Partition keys: only-in-vEcoli, only-in-v2, shared-but-different.

    ``different`` maps each differing shared key to a (vecoli_value,
    v2_value) tuple. Only top-level keys are compared.
    """
    vkeys, v2keys = set(vecoli), set(v2)
    different = {
        k: (vecoli[k], v2[k])
        for k in (vkeys & v2keys)
        if vecoli[k] != v2[k]
    }
    return {
        "only_in_vecoli": sorted(vkeys - v2keys),
        "only_in_v2": sorted(v2keys - vkeys),
        "different": different,
    }
