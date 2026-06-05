"""Cell-level extraction of vector observables for the report card.

Transcriptome / proteome / exchange-flux axes are *vectors* (one value per
gene / protein / exchange reaction). Following the cell-level discipline used
everywhere in the card: time-average each vector **within** a cell (one vector
per cell), then take the ensemble mean across the N cells that pass burn-in.
The ensemble-mean vector is what the card grades (R^2 vs the pinned reference)
and plots (scatter).

This reads the sweep parquet directly (the array columns aren't carried in the
scalar per-cell records). It is heavier than the scalar analysis (~minute over
a 4x8 ensemble), so it runs at report-render time, not in the workflow step.
"""
from __future__ import annotations

# observable column -> card path it populates
_VECTOR_COLS = {
    "listeners__rna_counts__mRNA_cistron_counts": ("omics", "transcriptome"),
    "listeners__monomer_counts": ("omics", "proteome"),
    "listeners__fba_results__external_exchange_fluxes": ("fluxes", "exchange"),
}


def extract_vectors(sweep_dir: str, generation_lower_bound: int = 0) -> dict:
    """Return ``{group: {name: {"vector": [...], "n_cells": int}}}`` of
    ensemble-mean vectors, aggregated cell-first (time-mean within cell, then
    mean across cells)."""
    import glob
    import os
    from collections import defaultdict

    import duckdb
    import numpy as np

    files = glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        return {}
    con = duckdb.connect()
    rel = "read_parquet(" + repr(files) + ", hive_partitioning=true)"
    cols = ", ".join(_VECTOR_COLS)
    rows = con.sql(
        f"SELECT lineage_seed, generation, agent_id, {cols} FROM {rel} "
        f"WHERE generation >= {int(generation_lower_bound)}"
    ).fetchall()

    # group timesteps by cell, keep a running list of each vector column
    per_cell: dict[tuple, list] = defaultdict(lambda: [[] for _ in _VECTOR_COLS])
    for r in rows:
        cell = (r[0], r[1], r[2])
        for i, val in enumerate(r[3:]):
            per_cell[cell][i].append(val)

    out: dict[str, dict] = {}
    for i, (col, (group, name)) in enumerate(_VECTOR_COLS.items()):
        # per-cell time-mean vector -> matrix [n_cells x n_features]
        per_cell_means = np.array(
            [np.mean(np.asarray(per_cell[c][i], float), axis=0) for c in per_cell])
        ensemble_mean = per_cell_means.mean(axis=0)
        out.setdefault(group, {})[name] = {
            "vector": [float(x) for x in ensemble_mean],
            "n_cells": len(per_cell),
        }
    return out
