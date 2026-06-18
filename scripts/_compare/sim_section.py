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

# Observables grouped into the four families from the design. ``exprs`` is a
# list of candidate DuckDB SQL expressions tried in order (first that binds
# wins) — this lets one observable span engine-specific schemas and derived
# quantities:
#   * bulk: vEcoli emits one `bulk` BIGINT[]; v2ecoli emits `bulk__count`.
#     `list_sum(...)` gives the total bulk molecule count per timestep on each.
#   * active RNAP: both emit `active_rnap_unique_indexes` as an array; the
#     active-RNAP COUNT is its length.
OBSERVABLES: list[dict] = [
    {"family": "mass_growth", "key": "dry_mass",
     "exprs": ["listeners__mass__dry_mass"]},
    {"family": "mass_growth", "key": "cell_mass",
     "exprs": ["listeners__mass__cell_mass"]},
    {"family": "mass_growth", "key": "growth_rate",
     "exprs": ["listeners__mass__instantaneous_growth_rate"]},
    {"family": "molecule_counts", "key": "bulk_total_count",
     "exprs": ["list_sum(bulk)", "list_sum(bulk__count)"]},
    {"family": "listeners", "key": "ribosome_elong_rate",
     "exprs": ["listeners__ribosome_data__effective_elongation_rate"]},
    {"family": "listeners", "key": "active_rnap_count",
     "exprs": ["len(listeners__rnap_data__active_rnap_unique_indexes)"]},
    {"family": "division_lineage", "key": "cell_mass_for_division",
     "exprs": ["listeners__mass__cell_mass"]},  # divisions inferred from drops
]


def read_observables(out_dir: str, experiment_id: str,
                     keys: list[str]) -> dict[str, np.ndarray]:
    """Read named observable columns from a sim's emitted parquet history.

    Reads each value column DIRECTLY with DuckDB over the
    ``<out_dir>/<experiment_id>/history/**/*.pq`` glob, rather than via
    ``read_stacked_columns`` — the latter hardcodes a ``time`` id-column that
    vEcoli emits but v2ecoli does not (it emits ``global_time``). Since the
    comparison uses order/length-independent summary stats, we only need the
    raw value column. A column that is absent/non-numeric (e.g. the
    list-valued ``active_rnap_coordinates``) is omitted and recorded in the
    module-level ``_SKIPPED`` list so it surfaces as ``not_compared``.
    """
    import glob
    import os

    import duckdb
    import numpy as np

    by_key = {o["key"]: o for o in OBSERVABLES}
    out: dict[str, np.ndarray] = {}
    files = glob.glob(
        os.path.join(out_dir, experiment_id, "history", "**", "*.pq"),
        recursive=True)
    if not files:
        _SKIPPED.append(f"{out_dir}/{experiment_id}: no history parquet found")
        return out
    con = duckdb.connect()
    for key in keys:
        last_err = None
        for expr in by_key[key]["exprs"]:
            try:
                res = con.execute(
                    f"SELECT {expr} AS v "
                    "FROM read_parquet(?, union_by_name=true)", [files]
                ).fetchnumpy()
                out[key] = np.asarray(res["v"], dtype=float).ravel()
                break
            except Exception as e:  # try the next candidate expression
                last_err = e
        else:
            _SKIPPED.append(
                f"{out_dir}:{key}: {type(last_err).__name__}: {last_err}")
    return out


# Time columns to try, in order: vEcoli emits ``time``, v2ecoli ``global_time``.
_TIME_EXPRS = ("time", "global_time")


def read_observable_xy(out_dir: str, experiment_id: str, key: str,
                       max_points: int = 600) -> list[tuple[float, float]]:
    """Return time-ordered ``[(t, v), ...]`` for one observable, for plotting.

    Unlike :func:`read_observables` (which returns an unordered flat array for
    summary stats), this orders rows by the simulation time column so the
    trajectory plots against a real time axis instead of an arbitrary row index.
    Tries each time column in ``_TIME_EXPRS`` and each candidate value
    expression; downsamples to ~``max_points`` points for a compact SVG.
    """
    import glob
    import os

    import duckdb
    import numpy as np

    by_key = {o["key"]: o for o in OBSERVABLES}
    if key not in by_key:
        return []
    files = glob.glob(
        os.path.join(out_dir, experiment_id, "history", "**", "*.pq"),
        recursive=True)
    if not files:
        return []
    con = duckdb.connect()
    for texpr in _TIME_EXPRS:
        for vexpr in by_key[key]["exprs"]:
            try:
                res = con.execute(
                    f"SELECT {texpr} AS t, {vexpr} AS v "
                    "FROM read_parquet(?, union_by_name=true) ORDER BY t",
                    [files]).fetchnumpy()
            except Exception:  # this (time, value) pair did not bind; try next
                continue
            t = np.asarray(res["t"], dtype=float).ravel()
            v = np.asarray(res["v"], dtype=float).ravel()
            pts = [(float(a), float(b)) for a, b in zip(t, v) if b == b]
            if len(pts) > max_points:
                step = max(1, len(pts) // max_points)
                pts = pts[::step]
            return pts
    return []


def _summary(a: "np.ndarray") -> "np.ndarray":
    """Order/length-independent fingerprint of a trajectory: mean, min, max.

    The two engines emit different time sampling and trajectory lengths, so a
    raw element-wise comparison is ill-posed. Summary statistics give a
    well-defined cross-engine agreement check.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.array([float(a.mean()), float(a.min()), float(a.max())])


def compare_observables(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    keys: list[str],
    rel_tol: float,
) -> list[dict[str, Any]]:
    """Build report rows comparing each requested observable key."""
    fam = {o["key"]: o["family"] for o in OBSERVABLES}

    def _fmt(a):
        s = _summary(a)
        return f"mean={s[0]:.4g}  min={s[1]:.4g}  max={s[2]:.4g}  (n={a.size})"

    rows = []
    for key in keys:
        l, r = left.get(key), right.get(key)
        if l is None or r is None:
            rows.append({"label": key, "left": "n/a", "right": "n/a",
                         "verdict": "not_compared",
                         "reason": "observable missing on one side",
                         "group": fam.get(key, "")})
            continue
        # Compare order/length-independent summary stats (mean, min, max).
        res = compare_series(_summary(l), _summary(r), rel_tol=rel_tol)
        rows.append({
            "label": key,
            "left": _fmt(l),
            "right": _fmt(r),
            "group": fam.get(key, ""),
            **res,
        })
    return rows
