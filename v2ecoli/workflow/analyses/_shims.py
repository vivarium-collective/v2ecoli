"""Shared data shims for native vEcoli analysis ports.

v2ecoli's parquet schema differs from vEcoli in two ways that matter for
the ptools analyses:

    Shim A – bulk count matrix
        vEcoli stores a single ``bulk`` list column (counts in sim_data order).
        v2ecoli stores ``bulk__id`` + ``bulk__count`` (both per-row lists,
        ordering may differ from sim_data).  ``bulk_count_matrix`` re-indexes
        the counts to sim_data column order so all downstream index arithmetic
        is identical to the vEcoli original.

    Shim B – active ribosome scalar
        vEcoli emits ``listeners__unique_molecule_counts__active_ribosome``
        (a scalar per row).  v2ecoli has no such column; the equivalent is the
        sum of ``listeners__ribosome_data__n_ribosomes_per_transcript``
        (a list per row).  Use the SQL snippet ``ACTIVE_RIBOSOME_SQL`` in
        SELECT lists and read the resulting ``active_ribosome`` column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shim B — active-ribosome SQL snippet
# ---------------------------------------------------------------------------

ACTIVE_RIBOSOME_SQL = (
    "list_sum(listeners__ribosome_data__n_ribosomes_per_transcript)"
    " AS active_ribosome"
)
"""Drop this into a SELECT column list to synthesise the active-ribosome
scalar from v2ecoli's per-transcript ribosome count list."""


# ---------------------------------------------------------------------------
# Shim A — bulk count matrix
# ---------------------------------------------------------------------------

def bulk_count_matrix(df: pd.DataFrame, sim_data) -> np.ndarray:
    """Return an (n_tp, n_bulk) count matrix in *sim_data* column order.

    vEcoli provides a single ``bulk`` list column already ordered to match
    ``sim_data.internal_state.bulk_molecules.bulk_data["id"]``.  v2ecoli's
    parquet stores ``bulk__id`` + ``bulk__count`` (per row, but with a stable
    ordering that may differ from sim_data).

    This function:
      1. Reads the parquet ordering from the first row of ``bulk__id``
         (verified constant across rows / files).
      2. Stacks ``bulk__count`` rows into an (n_tp, n_pq) matrix.
      3. Reorders columns to match sim_data order.

    Parameters
    ----------
    df:
        DataFrame returned by ``read_outputs`` containing ``bulk__id`` and
        ``bulk__count`` columns.
    sim_data:
        Loaded sim_data object (provides the canonical bulk ordering).

    Returns
    -------
    np.ndarray of shape (n_tp, n_bulk) with counts in sim_data order.
    """
    sim_ids: list[str] = list(
        sim_data.internal_state.bulk_molecules.bulk_data["id"]
    )
    pq_ids: list[str] = list(df["bulk__id"].iloc[0])
    counts = np.stack(df["bulk__count"].values)          # (n_tp, n_pq)
    pos = {bid: i for i, bid in enumerate(pq_ids)}
    col_idx = [pos[b] for b in sim_ids]
    return counts[:, col_idx]
