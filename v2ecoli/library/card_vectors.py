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
#
# v2 -> v3: the extraction no longer assumes things about the run that PRODUCED
# the sweep. Two changes, and BOTH independently require the bump:
#   * a node's CONTENT can change — cells whose row count is far below the
#     sweep's longest are excluded from the ensemble, so ``vector``, ``n_cells``
#     and ``per_cell`` all move for any sweep containing birth stubs;
#   * a node carries a new KEY, ``n_cells_excluded_partial``.
# ⛔ Without this bump the fix would appear to do NOTHING wherever it matters
# most: an envelope extracted by v2 against a stub-contaminated sweep is exactly
# what a v2-keyed lookup would keep serving, so the corrected code would never
# run for the runs that need it. The stale envelope carries no marker — its
# numbers are plausible, which is the whole problem.
EXTRACTOR_VERSION = 3

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

#: A cell is admitted to the ensemble only if it contributed at least this
#: fraction of the sweep's LONGEST cell, measured in rows.
#:
#: ⛔ **This exists because a sweep can contain rows for a cell that never lived
#: a cell cycle**, and averaging them beside real cells is silent, not loud.
#: A runner that hand-drives its generations emits both daughters at each
#: division and then follows only one; the abandoned daughter leaves a short
#: birth stub in its own ``agent_id`` partition. Grouping by
#: ``(lineage_seed, generation, agent_id)`` — which is the correct grouping —
#: therefore yields cells that are a few dozen ticks of newborn state alongside
#: cells that ran a full cycle, and the ensemble mean weights them equally.
#:
#: ⚠ **The previous behaviour was not a bug in any one runner.** Sweeps produced
#: with the workflow's ``single_daughters`` setting contain no stubs at all, so
#: the "one cell per (seed, generation)" assumption held for every sweep this
#: function had been used on. It is an assumption about the PRODUCER that was
#: satisfied by configuration and asserted nowhere — which is why it went
#: unnoticed rather than being caught by a test.
#:
#: **A fraction rather than an absolute row count, because the quantity must be
#: scale-free**: it cannot change meaning with the timestep, the generation
#: length, or the units of either. Measured on a real sweep the two populations
#: separate by roughly two orders of magnitude (tens of rows against thousands),
#: so nothing depends on where in that gap the cut sits.
#:
#: ⛔⛔ **Scaled off the MAXIMUM, and NOT off the median — the median version was
#: written first, tested against a real sweep, and FAILED SILENTLY.** When stubs
#: outnumber full cells the median is itself a stub: on a 4-generation sweep the
#: per-cell row counts were ``25, 25, 29, 31, 45, 3217, 3333, 3511``, whose
#: median is 38, giving a floor of 3.8 — and admitting all eight. **A robust
#: statistic is the wrong tool here precisely because the contaminant can be the
#: majority.** The longest cell in a sweep is the only scale that a stub
#: population cannot drag down.
#:
#: ⚠ **The cost of using a maximum, stated:** one anomalously long cell raises
#: the floor for everyone. At this fraction that needs better than a 10x spread
#: in cell-cycle length before a genuine cell is at risk — and a sweep with a
#: 10x spread in cycle length has a finding in it that matters more than this
#: rule does.
#:
#: ⚠ **Limitation, stated rather than hidden: this is a RELATIVE rule.** A sweep
#: in which *every* cell is truncated has a truncated median, so nothing is
#: excluded and ``n_cells`` will look healthy. That case is a different failure
#: (the run, not the ensemble) and is visible in the run's own summary; this
#: function cannot distinguish "short because truncated" from "short because
#: fast" without a claim about the biology, and it does not try.
_MIN_CELL_ROW_FRACTION = 0.1

#: ``(group, name) -> units``. The lookup a resolver uses when it holds a card
#: path rather than a parquet column — see ``operands.run_operand``, which stamps
#: units from HERE rather than from the cached node, so that a hand-built or
#: pre-v2 envelope cannot produce an operand that silently declares nothing.
VECTOR_UNITS = {(group, name): units for group, name, units in _VECTOR_COLS.values()}


def extract_vectors(sweep_dir: str, generation_lower_bound: int = 0) -> dict:
    """Return ``{group: {name: {...}}}`` of cell-first aggregated vectors
    (time-mean within cell, then mean across cells).

    Each node carries the ensemble-mean ``vector``, ``n_cells``,
    ``n_cells_excluded_partial``, its ``units``, and ``per_cell`` — the
    n_cells x n_features matrix of per-cell time-mean vectors whose column mean
    IS ``vector``.

    ⭐ **Two things about the PRODUCER are checked here rather than assumed**,
    because both were previously assumptions that every sweep this function had
    seen happened to satisfy:

    1. **Which observable columns exist.** A sweep is extracted for the columns
       it has; a group whose column is absent is OMITTED, never zero-filled.
       Different metabolism processes write different exchange leaves.
    2. **Which rows are a cell.** Grouping by ``(lineage_seed, generation,
       agent_id)`` can yield short birth stubs beside full cell cycles,
       depending on how the producing runner emits daughters at division.
       Cells far below the sweep's median row count are excluded and COUNTED —
       see ``_MIN_CELL_ROW_FRACTION``.

    ⚠ **Both change results only for sweeps that were previously being read
    wrongly.** For a sweep with one full-cycle cell per ``(seed, generation)``
    and all three columns present, the cell set and the column set are
    unchanged, so the extraction is unchanged.

    ⛔ **"Unchanged" here does NOT mean bit-identical, and the difference is the
    function's own, not this change's.** Measured against a real 6-cell sweep:
    re-running the UNMODIFIED function on the same input twice already differs
    by up to ~7e-12 absolute on the proteome vector. DuckDB's scan order varies
    between executions and the per-cell running sums are accumulated in that
    order, so float summation order — and therefore the last bits — is not
    reproducible. This change's output sits inside that same envelope
    (~1.5e-11 absolute, ~1e-15 relative). ⇒ **Do not write an equality
    assertion against a stored vector**; compare with a tolerance, or the test
    will fail intermittently for reasons that have nothing to do with the code
    under test.

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

    # ⛔ SELECT ONLY THE COLUMNS THIS SWEEP ACTUALLY HAS, and omit the rest from
    # the result rather than failing the whole extraction.
    #
    # **Not every metabolism writes every leaf.** ``metabolism.py`` writes
    # ``listeners.fba_results.external_exchange_fluxes``; ``metabolism_redux``
    # does not — it writes ``estimated_exchange_dmdt`` instead. The same
    # divergence is already documented, and already guarded with an explanatory
    # refusal, in ``library/vivarium_ecoli_engine.py`` for the ``gdcw``
    # exchange-flux basis. This function had no equivalent, so a sweep produced
    # by a metabolism that does not write the leaf died inside DuckDB with a
    # binder error naming a column — loud, but it reports the symptom rather
    # than the cause, and it takes the omics groups down with it even though
    # those columns are present and perfectly extractable.
    #
    # ⭐ **Absent means ABSENT — the group is omitted, never emitted as zeros.**
    # A zero-filled exchange vector is indistinguishable from a cell exchanging
    # nothing, which is the failure mode the engine's own guard exists to
    # refuse. A consumer that requires a group it cannot find must say so
    # itself; silently handing it zeros moves an error into a result.
    available = {c.lower() for c in con.sql(f"SELECT * FROM {rel} LIMIT 0").columns}
    present = [
        (col, meta) for col, meta in _VECTOR_COLS.items() if col.lower() in available
    ]
    if not present:
        raise ValueError(
            f"no observable columns found in {sweep_dir!r}. Looked for: "
            + ", ".join(_VECTOR_COLS)
            + ". A sweep with none of them is not gradeable — check that the "
            "run emitted its listeners, and note that the metabolism in use "
            "determines which exchange leaf (if any) is written."
        )
    cols = ", ".join(col for col, _ in present)
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
    # ⚠ Measured 2026-08-21 on an 8x16 basal sweep (50 GB of history parquet) at
    # ``generation_lower_bound=0``: 52 GB resident, 63 GB peak, 42.5 GB of 44 GB
    # swap consumed, the process pinned in uninterruptible I/O wait, and killed
    # rather than failing on its own -- taking the machine with it.
    #
    # ⚠ NOT "it could never have completed": the surviving v1 cache envelope for
    # this same sweep records ``extract_seconds: 680.76`` at ``gen_lb=3`` over
    # the same 104 cells, extracted 2026-07-30 by the ``fetchall`` code. The old
    # path completed this sweep at gen_lb=3 and blew past the machine at
    # gen_lb=0, +21% rows. The defect is that peak scales with the RUN, so the
    # margin is a property of the hardware and not of the code.
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
    col_len = [0] * len(present)          # modal (max) feature length per column
    per_cell_rows: dict[tuple, int] = {}  # rows seen per cell, for the
                                          # ensemble-membership rule below

    while True:
        batch = result.fetchmany(_FETCH_BATCH_ROWS)
        if not batch:
            break
        for r in batch:
            cell = (r[0], r[1], r[2])
            if cell not in seen_cells:
                seen_cells.add(cell)
                cell_order.append(cell)
            # Counted over ROWS, not over any one column's non-null rows, so the
            # membership rule below is a property of the cell rather than of
            # whichever observable happens to be widest.
            per_cell_rows[cell] = per_cell_rows.get(cell, 0) + 1
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

    # ⭐ ENSEMBLE MEMBERSHIP IS DECIDED ONCE, HERE, AND APPLIED TO EVERY COLUMN.
    # Deciding it per column would let the transcriptome and the exchange vector
    # be built from different cell sets — a comparison whose two halves do not
    # describe the same cells, with nothing in the output saying so.
    if cell_order:
        floor = _MIN_CELL_ROW_FRACTION * max(per_cell_rows[c] for c in cell_order)
        included = [c for c in cell_order if per_cell_rows[c] >= floor]
    else:
        included = []
    n_excluded = len(cell_order) - len(included)

    out: dict[str, dict] = {}
    for i, (col, (group, name, units)) in enumerate(present):
        n = col_len[i]
        # per-cell time-mean vector over rows whose array is full-length (drops
        # the [] empties); skip cells with no full-length rows.
        cell_means = []
        for c in included:
            key = (c, i, n)
            count = per_cell_n.get(key)
            if count:
                cell_means.append(per_cell_sum[key] / count)
        per_cell_means = np.array(cell_means)
        ensemble_mean = per_cell_means.mean(axis=0)
        node = {
            "vector": [float(x) for x in ensemble_mean],
            "n_cells": len(cell_means),
            # ⭐ Surfaced, not silent. A provenance panel that prints n_cells
            # without this cannot distinguish an ensemble that never had more
            # cells from one whose extras were dropped as partial.
            "n_cells_excluded_partial": n_excluded,
            "units": units,
            "per_cell": [[float(x) for x in row] for row in per_cell_means],
        }
        out.setdefault(group, {})[name] = node
    return out
