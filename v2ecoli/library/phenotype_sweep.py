"""Generic variant-sweep phenotype extractor.

Engine-agnostic: consumes already-loaded per-run observable series (a run =
one variant/sweep index) and reshapes them for sweep-axis comparison. Knows
nothing about what any observable means — paths are pure inputs.
"""
from __future__ import annotations


def collect_sweep(runs: list, observable_paths: list) -> dict:
    """Reshape ``[{label, series:{path:[...]}}]`` into ``{path: {label: [...]}}``.

    A path absent from every run is skipped (not an error). A path present in
    some runs only appears for the runs that have it.
    """
    out: dict = {}
    for path in observable_paths:
        col = {}
        for run in runs:
            series = (run.get("series") or {})
            if path in series:
                col[run["label"]] = series[path]
        if col:
            out[path] = col
    return out


def sweep_endpoints(sweep: dict) -> dict:
    """Last value of each series — the dose-response point per (path, label)."""
    return {
        path: {label: (vals[-1] if vals else float("nan"))
               for label, vals in cols.items()}
        for path, cols in sweep.items()
    }
