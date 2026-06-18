"""Build a grouped catalog of every unit-bearing readout in the composite.

Reuses ``units_resolver.build_units_index`` for the path->unit map, groups by
physical dimension, and (optionally) samples a run's parquet for example
magnitude + min/max. Descriptive only — no acceptance gates.
"""
from __future__ import annotations

from typing import Any, Optional

from v2ecoli.library.units_resolver import build_units_index

# Unit string -> physical dimension. Extend as new units appear.
_DIMENSION_BY_UNIT = {
    "fg": "mass", "g": "mass", "pg": "mass",
    "s": "time", "min": "time", "h": "time",
    "mM": "concentration", "mmol/L": "concentration", "M": "concentration",
    "uM": "concentration", "nM": "concentration", "M/L": "concentration",
    "1/s": "rate", "1/h": "rate", "1/min": "rate",
    "nt": "count", "aa": "count", "count": "count",
    "L": "volume", "fL": "volume",
    "m": "length", "nm": "length", "um": "length",
}


def dimension_of(unit: str) -> str:
    """Map a unit string to a coarse physical dimension; unknown -> 'other'."""
    return _DIMENSION_BY_UNIT.get(unit, "other")


# One-line description of what each physical dimension captures in the model.
_DIMENSION_DESC = {
    "mass": "Cell and component masses — the whole-cell mass budget the growth "
            "and division logic balances (cell, dry, protein, RNA mass).",
    "concentration": "Molecular concentrations — metabolite pools, FBA target "
                     "and updated concentrations, and growth-limiting species "
                     "the metabolism solver reads and writes.",
    "rate": "Per-time rates — reaction and process velocities (e.g. equilibrium "
            "reaction rates) expressed per second.",
    "time": "Durations and timesteps — the simulation clock and per-process "
            "step sizes that pace the cell cycle.",
    "count": "Molecule counts — discrete copy numbers (nucleotides, amino "
             "acids, bulk species) the stochastic processes act on.",
    "volume": "Cell volume — the compartment size that links counts to "
              "concentrations.",
    "length": "Lengths — spatial extents (e.g. cell length).",
    "other": "Composite or flux-like units that don't fall into a single base "
             "dimension (e.g. g·s/L, mmol/g/h).",
}


def dimension_description(dim: str) -> str:
    """Human-readable one-liner for a physical dimension; '' if unknown."""
    return _DIMENSION_DESC.get(dim, "")


def format_magnitude(x: Optional[float]) -> str:
    """Format a sampled magnitude compactly: ~4 sig figs, scientific for
    very small/large values, an en-dash for missing values."""
    if x is None:
        return "—"
    try:
        ax = abs(float(x))
    except (TypeError, ValueError):
        return "—"
    if ax == 0:
        return "0"
    if ax < 1e-3 or ax >= 1e6:
        return f"{x:.3e}"
    return f"{x:.4g}"


def build_atlas(run_dir: Optional[Any] = None) -> dict:
    """Return ``{dimension: [row, ...], '_flags': [...]}``.

    Each row: ``{'path', 'unit', 'example', 'min', 'max'}``. When ``run_dir`` is
    None, magnitude fields are ``None``. ``_flags`` lists readouts the scan
    could not assign a unit (best-effort; empty here since the index only holds
    unit-bearing leaves).
    """
    index = build_units_index()
    atlas: dict = {}
    samples = _sample_magnitudes(run_dir, list(index)) if run_dir else {}
    for path, unit in sorted(index.items()):
        dim = dimension_of(unit)
        s = samples.get(path, {})
        atlas.setdefault(dim, []).append({
            "path": path,
            "unit": unit,
            "example": s.get("example"),
            "min": s.get("min"),
            "max": s.get("max"),
        })
    atlas["_flags"] = []
    return atlas


def _sample_magnitudes(run_dir: Any, paths: list[str]) -> dict:
    """Best-effort: read example/min/max per path from a run's parquet history.

    Uses the existing parquet loader; any failure yields an empty sample for
    that path (magnitudes stay None). Column name is the dotted path with '.'
    replaced by '__' (parquet convention).
    """
    out: dict = {}
    try:
        from pathlib import Path
        from v2ecoli.library.parquet_viz import load_run_history
        df = load_run_history(Path(run_dir)) if run_dir else None
    except Exception:
        return out
    if df is None:
        return out
    for path in paths:
        col = path.replace(".", "__")
        if col not in df.columns:
            continue
        try:
            series = df[col].drop_nulls()
            if series.len() == 0:
                continue
            out[path] = {
                "example": float(series[-1]) if series.dtype.is_numeric() else None,
                "min": float(series.min()) if series.dtype.is_numeric() else None,
                "max": float(series.max()) if series.dtype.is_numeric() else None,
            }
        except Exception:
            continue
    return out
