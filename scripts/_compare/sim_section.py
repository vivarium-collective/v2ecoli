"""Read a sim's emitted parquet history and compare observables.

Both engines emit the same ``<out>/<experiment_id>/history/**/*.pq`` layout
and both expose ``read_stacked_columns``; the same reader works for either
side — only the importing module differs.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scripts._compare.stats import compare_series

# Observables grouped into the four families from the design. ``column`` is
# the emitter field path (dotted) read via read_stacked_columns.
OBSERVABLES: list[dict[str, str]] = [
    {"family": "mass_growth", "key": "dry_mass",
     "column": "listeners__mass__dry_mass"},
    {"family": "mass_growth", "key": "cell_mass",
     "column": "listeners__mass__cell_mass"},
    {"family": "mass_growth", "key": "growth_rate",
     "column": "listeners__mass__instantaneous_growth_rate"},
    {"family": "molecule_counts", "key": "bulk_counts",
     "column": "bulk"},
    {"family": "listeners", "key": "active_ribosomes",
     "column": "listeners__ribosome_data__effective_elongation_rate"},
    {"family": "listeners", "key": "active_rnap",
     "column": "listeners__rnap_data__active_rnap_coordinates"},
    {"family": "division_lineage", "key": "division_time",
     "column": "listeners__mass__cell_mass"},  # divisions inferred from drops
]


def read_observables(
    out_dir: str,
    experiment_id: str,
    reader: Callable[..., Any],
    keys: list[str],
) -> dict[str, np.ndarray]:
    """Read named observable series from a parquet history dir.

    ``reader`` is the engine's ``read_stacked_columns`` (vEcoli or v2ecoli).
    A column that is absent/unreadable is simply omitted from the result so
    the comparison reports it as ``not_compared``.
    """
    import glob
    import os

    history_glob = os.path.join(out_dir, experiment_id, "history", "**", "*.pq")
    files = glob.glob(history_glob, recursive=True)
    by_key = {o["key"]: o for o in OBSERVABLES}
    out: dict[str, np.ndarray] = {}
    for key in keys:
        col = by_key[key]["column"]
        try:
            arr = reader(files, [col])
            out[key] = np.asarray(arr).ravel()
        except Exception:
            continue
    return out


def compare_observables(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    keys: list[str],
    rel_tol: float,
) -> list[dict[str, Any]]:
    """Build report rows comparing each requested observable key."""
    rows = []
    for key in keys:
        l, r = left.get(key), right.get(key)
        if l is None or r is None:
            rows.append({"label": key, "left": "n/a", "right": "n/a",
                         "verdict": "not_compared",
                         "reason": "observable missing on one side"})
            continue
        res = compare_series(l, r, rel_tol=rel_tol)
        rows.append({
            "label": key,
            "left": np.array2string(l, threshold=4),
            "right": np.array2string(r, threshold=4),
            **res,
        })
    return rows
