"""Read a sim's emitted parquet history and compare observables.

Both engines emit the same ``<out>/<experiment_id>/history/**/*.pq`` layout.
``read_stacked_columns`` from v2ecoli.library.parquet_emitter reads either
engine's parquet; it expects a DuckDB SQL subquery string + connection, not
a glob list.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts._compare.stats import compare_series

# Tracks columns that could not be read, for visibility in logs.
_SKIPPED: list[str] = []

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


def read_observables(out_dir: str, experiment_id: str,
                     keys: list[str]) -> dict[str, np.ndarray]:
    """Read named observable series from a sim's emitted parquet history.

    Builds the DuckDB history-SQL with ``dataset_sql`` and reads each column
    via ``read_stacked_columns``. A column that is absent/unreadable is
    omitted (and its name collected in the module-level ``_SKIPPED`` list for
    visibility) so the comparison reports it as ``not_compared`` rather than
    crashing.
    """
    from v2ecoli.library.parquet_emitter import (
        dataset_sql, create_duckdb_conn, read_stacked_columns)
    import numpy as np

    by_key = {o["key"]: o for o in OBSERVABLES}
    out: dict[str, np.ndarray] = {}
    try:
        history_sql, _c, _s = dataset_sql(out_dir, [experiment_id])
        conn = create_duckdb_conn()
    except Exception as e:
        _SKIPPED.append(f"{out_dir}: dataset open failed: {e}")
        return out
    for key in keys:
        col = by_key[key]["column"]
        try:
            df = read_stacked_columns(history_sql, [col], conn=conn)
            out[key] = np.asarray(df[col].to_list(), dtype=float).ravel()
        except Exception as e:
            _SKIPPED.append(f"{out_dir}:{col}: {type(e).__name__}: {e}")
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
    fam = {o["key"]: o["family"] for o in OBSERVABLES}
    rows = []
    for key in keys:
        l, r = left.get(key), right.get(key)
        if l is None or r is None:
            rows.append({"label": key, "left": "n/a", "right": "n/a",
                         "verdict": "not_compared",
                         "reason": "observable missing on one side",
                         "group": fam.get(key, "")})
            continue
        res = compare_series(l, r, rel_tol=rel_tol)
        rows.append({
            "label": key,
            "left": np.array2string(l, threshold=4),
            "right": np.array2string(r, threshold=4),
            "group": fam.get(key, ""),
            **res,
        })
    return rows
