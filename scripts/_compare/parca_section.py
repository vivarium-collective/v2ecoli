"""ParCa / sim_data comparison rows for the harness report.

Per-step diffing is delegated to the existing scripts/parca_compare.py;
this module adds a final-sim_data field-by-field diff using the same attr
paths that comparator already curates.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts._compare.stats import compare_series

# Attr paths mirror scripts/parca_compare.py SCALARS + DISTRIBUTIONS.
_SCALARS = [
    ("mass.avg_cell_dry_mass_init", ("mass", "avg_cell_dry_mass_init")),
    ("mass.avg_cell_dry_mass", ("mass", "avg_cell_dry_mass")),
    ("mass.avg_cell_water_mass_init", ("mass", "avg_cell_water_mass_init")),
    ("mass.fitAvgSolubleTargetMolMass", ("mass", "fitAvgSolubleTargetMolMass")),
    ("constants.darkATP", ("constants", "darkATP")),
]
_DISTRIBUTIONS = [
    ("RNA expression — basal",
     ("process", "transcription", "rna_expression", "basal")),
]


def _reach(obj: Any, path: tuple[str, ...]):
    """Follow an attr/key path; return None if any hop is missing."""
    cur = obj
    for p in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return cur


def _row(label: str, left, right, rel_tol: float) -> dict[str, Any]:
    if left is None or right is None:
        return {"label": label, "left": "n/a", "right": "n/a",
                "verdict": "not_compared",
                "reason": "attribute missing on one side"}
    r = compare_series(np.atleast_1d(left), np.atleast_1d(right),
                       rel_tol=rel_tol)
    return {"label": label,
            "left": np.array2string(np.atleast_1d(left), threshold=4),
            "right": np.array2string(np.atleast_1d(right), threshold=4),
            **r}


def final_sim_data_diff(left, right, *, rel_tol: float) -> list[dict[str, Any]]:
    """Diff curated scalar + distribution fields of two sim_data objects."""
    rows = []
    for label, path in _SCALARS + _DISTRIBUTIONS:
        rows.append(_row(label, _reach(left, path), _reach(right, path),
                         rel_tol))
    return rows
