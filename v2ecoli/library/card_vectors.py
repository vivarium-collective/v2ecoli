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

``sweep_dir`` may be a local path or an ``s3://`` URI (see
:mod:`v2ecoli.library.sweep_io`). The S3 form lets an equivalence reference be
pinned — and a measured card rendered — against a sweep that never lands on the
local disk, which is what makes a multi-condition card affordable: the parquet
stays in object storage and DuckDB's column projection reads only the three
array columns below.
"""
from __future__ import annotations

# Bump whenever a node's CONTENT changes — the aggregation semantics (the column
# set, the ragged-row rule, the cell-first order) *or* the set of keys a node
# carries. It is part of the sim_vector_cache key, so a bump invalidates every
# cached vector rather than silently serving one built by older code — the vector
# is a function of this code as much as of the run.
#
# v1 -> v2: every group now carries ``per_cell`` (was: fluxes only) and every node
# declares its ``units``. Neither changes a single number, which is exactly why
# the bump is needed rather than optional: a v1 envelope and a v2 envelope are
# numerically identical, so nothing else would tell a consumer asking for
# ``per_cell`` that this cache simply predates it. Without the bump, "the run
# recorded no per-cell samples" and "this file was written by older code" are
# indistinguishable — and the first is a fact about the run, the second a fact
# about the tooling.
EXTRACTOR_VERSION = 2

#: Rows pulled from DuckDB per batch. Bounds peak memory during extraction: the
#: rows themselves are released after each batch, so what persists is the
#: per-cell running sums, not the run. Large enough that per-batch overhead is
#: irrelevant, small enough that a batch of wide omics rows is megabytes.
_FETCH_BATCH_ROWS = 2000

#: observable column -> ``(group, name, units)``.
#:
#: **Units are declared HERE, beside the parquet column they describe, and
#: nowhere else.** They are a property of what the listener emits — monomer
#: counts are copies/cell whatever anybody does with them downstream — so a
#: consumer that names its own units is stating a belief about this column
#: rather than reading a fact from it. ``scripts/bake_model_omics.py`` and the
#: committed fixtures already declare exactly these strings; this is the
#: declaration they should eventually read rather than a fourth copy.
_VECTOR_COLS = {
    "listeners__rna_counts__mRNA_cistron_counts": ("omics", "transcriptome", "counts/cell"),
    "listeners__monomer_counts": ("omics", "proteome", "copies/cell"),
    "listeners__fba_results__external_exchange_fluxes": ("fluxes", "exchange", "mmol/gDCW/h"),
}

#: ``(group, name) -> units``. The lookup a resolver uses when it holds a card
#: path rather than a parquet column — see ``operands.run_operand``, which stamps
#: units from HERE rather than from the cached node, so that a hand-built or
#: pre-v2 envelope cannot produce an operand that silently declares nothing.
VECTOR_UNITS = {(group, name): units for group, name, units in _VECTOR_COLS.values()}


def extract_vectors(sweep_dir: str, generation_lower_bound: int = 0) -> dict:
    """Return ``{group: {name: {...}}}`` of cell-first aggregated vectors
    (time-mean within cell, then mean across cells).

    Each node carries the ensemble-mean ``vector``, ``n_cells``, its ``units``,
    and ``per_cell`` — the n_cells x n_features matrix of per-cell time-mean
    vectors whose column mean IS ``vector``.

    ``per_cell`` used to be emitted for the ``fluxes`` group alone, so that named
    flux KPIs could be sliced out and graded with the same ttest/violin path as
    the scalar axes. It is now emitted for every group, because the matrix was
    always computed for every group and a comparison that grades DISTRIBUTIONS
    (rather than two centres) cannot be written without it: a mean vector is not
    a distribution, so any distributional statistic over omics had nothing to
    consume.

    ⚠ **The cost is real and was measured before this was made unconditional.**
    At the ensemble sizes in use the omics matrices dominate the artifact: 104
    cells x (4345 + 4309) features is ~900k floats, taking a cached envelope from
    ~279 KB to ~17 MB. That is 0.03% of the sweep it is written beside and it
    never enters git (the cache is gitignored by contract, see
    :mod:`v2ecoli.library.sim_vector_cache`), which is what makes unconditional
    affordable.

    ★ **It is unconditional rather than opt-in, and that is a deliberate refusal
    of the cheaper design.** An ``include_per_cell`` flag would sit OUTSIDE the
    cache key — and the key is the whole integrity story here. A caller passing
    the flag would get a cache HIT on an envelope written without it and see no
    per-cell data at all, with nothing distinguishing "this run has none" from
    "the file was written by a caller that didn't ask". Saving ~16 MB is not
    worth buying that class of silence. Callers who do not want the matrix in
    memory opt out at the point of USE (``operands.run_operand``'s
    ``with_per_cell``), where the choice is a pure function of the arguments and
    no cache can serve a stale answer.

    Ragged/empty array rows are dropped per column: ``external_exchange_fluxes``
    emits a ``[]`` default on some timesteps, so only rows whose array matches
    the column's modal length are averaged."""
    import numpy as np

    from v2ecoli.library.sweep_io import connect_for, history_files

    files = history_files(sweep_dir)
    if not files:
        return {}
    con = connect_for(sweep_dir)
    rel = "read_parquet(" + repr(files) + ", hive_partitioning=true)"
    cols = ", ".join(_VECTOR_COLS)
    result = con.sql(
        f"SELECT lineage_seed, generation, agent_id, {cols} FROM {rel} "
        f"WHERE generation >= {int(generation_lower_bound)}"
    )

    # ⛔ STREAMED IN BATCHES, AND THIS IS NOT A MICRO-OPTIMISATION.
    #
    # This read used to be a single ``.fetchall()``, which materialises every
    # timestep x every cell x every vector column into Python objects at once,
    # and was then grouped into a SECOND full copy keyed by cell. Peak footprint
    # was therefore ~2x the sweep's vector columns, in Python objects.
    #
    # ⚠ Measured 2026-08-21 on an 8x16 basal sweep (50 GB of history parquet):
    # 52 GB resident, 63 GB peak, 42.5 GB of 44 GB swap consumed, the process
    # pinned in uninterruptible I/O wait. It could not have completed, and it
    # took the machine down with it rather than failing on its own.
    #
    # The fix is to never hold the rows. We accumulate a RUNNING SUM per
    # (cell, column, array-length) and divide at the end, so peak memory is a
    # function of the ENSEMBLE (cells x features), not of the RUN (timesteps x
    # cells x features). For that sweep that is ~900k floats instead of billions.
    #
    # Bucketing by array length rather than resolving the modal length first is
    # what keeps this a SINGLE pass: the modal length is not known until every
    # row has been seen, and a second pass over 50 GB to learn it would trade
    # the memory problem for an I/O one. Distinct lengths are few (a column is
    # ragged only where a listener emits its ``[]`` default), so the buckets
    # cost nothing.
    per_cell_sum: dict[tuple, "np.ndarray"] = {}
    per_cell_n: dict[tuple, int] = {}
    cell_order: list[tuple] = []          # first-appearance order, which IS the
    seen_cells: set[tuple] = set()        # row order of the per_cell matrix
    col_len = [0] * len(_VECTOR_COLS)     # modal (max) feature length per column

    while True:
        batch = result.fetchmany(_FETCH_BATCH_ROWS)
        if not batch:
            break
        for r in batch:
            cell = (r[0], r[1], r[2])
            if cell not in seen_cells:
                seen_cells.add(cell)
                cell_order.append(cell)
            for i, val in enumerate(r[3:]):
                if val is None:
                    continue
                length = len(val)
                if length > col_len[i]:
                    col_len[i] = length
                key = (cell, i, length)
                acc = per_cell_sum.get(key)
                if acc is None:
                    per_cell_sum[key] = np.array(val, dtype=float)  # copy: acc is mutated in place
                    per_cell_n[key] = 1
                else:
                    acc += val
                    per_cell_n[key] += 1

    out: dict[str, dict] = {}
    for i, (col, (group, name, units)) in enumerate(_VECTOR_COLS.items()):
        n = col_len[i]
        # per-cell time-mean vector over rows whose array is full-length (drops
        # the [] empties); skip cells with no full-length rows.
        cell_means = []
        for c in cell_order:
            key = (c, i, n)
            count = per_cell_n.get(key)
            if count:
                cell_means.append(per_cell_sum[key] / count)
        per_cell_means = np.array(cell_means)
        ensemble_mean = per_cell_means.mean(axis=0)
        node = {
            "vector": [float(x) for x in ensemble_mean],
            "n_cells": len(cell_means),
            "units": units,
            "per_cell": [[float(x) for x in row] for row in per_cell_means],
        }
        out.setdefault(group, {})[name] = node
    return out
