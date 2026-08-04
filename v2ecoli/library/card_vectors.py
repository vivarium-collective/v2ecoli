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

# Bump whenever the aggregation semantics change (the column set, the ragged-row
# rule, the cell-first order). It is part of the sim_vector_cache key, so a bump
# invalidates every cached vector rather than silently serving one built by
# older code — the vector is a function of this code as much as of the run.
EXTRACTOR_VERSION = 1

# observable column -> card path it populates
_VECTOR_COLS = {
    "listeners__rna_counts__mRNA_cistron_counts": ("omics", "transcriptome"),
    "listeners__monomer_counts": ("omics", "proteome"),
    "listeners__fba_results__external_exchange_fluxes": ("fluxes", "exchange"),
}


def extract_vectors(sweep_dir: str, generation_lower_bound: int = 0) -> dict:
    """Return ``{group: {name: {...}}}`` of cell-first aggregated vectors
    (time-mean within cell, then mean across cells).

    Each node carries the ensemble-mean ``vector`` and ``n_cells``. For the
    ``fluxes`` group it additionally carries ``per_cell`` (the n_cells x
    n_features matrix of per-cell time-mean vectors) so named flux KPIs can be
    sliced out and graded with the same ttest/violin path as the scalar axes.

    Ragged/empty array rows are dropped per column: ``external_exchange_fluxes``
    emits a ``[]`` default on some timesteps, so only rows whose array matches
    the column's modal length are averaged."""
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
    col_len = [0] * len(_VECTOR_COLS)  # modal (max) feature length per column
    for r in rows:
        cell = (r[0], r[1], r[2])
        for i, val in enumerate(r[3:]):
            per_cell[cell][i].append(val)
            if val is not None:
                col_len[i] = max(col_len[i], len(val))

    out: dict[str, dict] = {}
    for i, (col, (group, name)) in enumerate(_VECTOR_COLS.items()):
        n = col_len[i]
        # per-cell time-mean vector over rows whose array is full-length (drops
        # the [] empties); skip cells with no full-length rows.
        cell_means = []
        for c in per_cell:
            full = [v for v in per_cell[c][i] if v is not None and len(v) == n]
            if full:
                cell_means.append(np.asarray(full, float).mean(axis=0))
        per_cell_means = np.array(cell_means)
        ensemble_mean = per_cell_means.mean(axis=0)
        node = {
            "vector": [float(x) for x in ensemble_mean],
            "n_cells": len(cell_means),
        }
        if group == "fluxes":
            node["per_cell"] = [[float(x) for x in row] for row in per_cell_means]
        out.setdefault(group, {})[name] = node
    return out
