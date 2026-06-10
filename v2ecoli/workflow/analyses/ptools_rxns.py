"""Native port of vEcoli ``ecoli/analysis/single/ptools_rxns.py``.

Produces a reaction × timepoint flux TSV (PathwayTools-compatible format).
Registered in ANALYSIS_REGISTRY as ``"ptools_rxns"`` (scale: ``"single"``).

No bulk or ribosome shims required: only
``listeners__fba_results__base_reaction_fluxes`` is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import ptools_heatmap_view
from v2ecoli.workflow.analyses.ptools_rna import consolidate_timepoints


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def build_query(columns, history_sql):
    """Generate SQL query for user-specified parquet columns."""
    query_sql = f"""
        SELECT {",".join(columns)}, global_time AS time
        FROM ({history_sql})
        ORDER BY time
    """
    return query_sql


def read_outputs(
    history_sql: str,
    conn: DuckDBPyConnection,
    columns=None,
):
    """Retrieve specific columns from parquet outputs and return a DataFrame."""
    if columns is None:
        columns = ["listeners__fba_results__base_reaction_fluxes"]
    query_sql = build_query(columns, history_sql)
    outputs_df = conn.sql(query_sql).df()
    outputs_df = outputs_df.groupby("time", as_index=False).sum()
    return outputs_df


# ---------------------------------------------------------------------------
# Analysis subclass
# ---------------------------------------------------------------------------

class PtoolsRxns(Analysis):
    """Reaction × timepoint flux table (PathwayTools-compatible TSV)."""

    name = "ptools_rxns"
    scale = "single"
    config_schema = {"n_tp": "integer", "time_unit": "string"}

    def _do_read_outputs(
        self,
        history_sql: str,
        conn: DuckDBPyConnection,
        columns=None,
    ):
        """Delegate to module-level read_outputs (overridable by mixins)."""
        return read_outputs(history_sql, conn, columns)

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        params = dict(variant_metadata or {})
        params.setdefault("n_tp", 8)
        params.setdefault("time_unit", "minutes")

        if params["time_unit"] not in ("minutes", "seconds"):
            params["time_unit"] = "minutes"

        output_columns = ["listeners__fba_results__base_reaction_fluxes"]
        output_df = self._do_read_outputs(history_sql, conn, output_columns)

        # Drop timestep-0 row where FBA hasn't run yet (flux array is empty at
        # t=0 because metabolism hasn't been called).  v2ecoli deviation: vEcoli
        # keeps t=0 in the raw table; we drop it so the first column is the
        # first real FBA timepoint.
        flux_col = "listeners__fba_results__base_reaction_fluxes"
        output_df = output_df[
            output_df[flux_col].apply(len) > 0
        ].reset_index(drop=True)

        rxn_mtx = np.stack(output_df[flux_col].values)

        rxn_ids_base = sim_data.process.metabolism.base_reaction_ids

        # Sanity-check: v2ecoli's metabolism listener emits
        #   base_reaction_fluxes = reaction_mapping_matrix.dot(...)
        # whose length equals len(base_reaction_ids) in the sim_data that
        # generated this parquet.  They are 1:1 positional — flux column i
        # corresponds to base_reaction_ids[i].  A mismatch means the caller
        # passed a sim_data that does NOT pair with this parquet (e.g.
        # out/kb/simData.cPickle vs out/workflow/simData.cPickle); raise
        # loudly instead of silently mislabeling reactions via truncation.
        n_ids = len(rxn_ids_base)
        flux_width = rxn_mtx.shape[1]
        if n_ids != flux_width:
            raise ValueError(
                f"base_reaction_ids ({n_ids}) != flux width ({flux_width}); "
                "sim_data does not pair with this parquet"
            )

        n_tp = int(params["n_tp"])

        rxn_blocksum, tp_idx = consolidate_timepoints(rxn_mtx, n_tp, normalized=True)

        tp_checkpoints = output_df["time"].values[tp_idx]

        if params["time_unit"] == "minutes":
            tp_checkpoints = tp_checkpoints / 60
            tp_checkpoints = [round(x) for x in tp_checkpoints]

        tp_columns = [str(i) + params["time_unit"][0] for i in tp_checkpoints]

        ptools_rxns_df = pd.DataFrame(
            data=np.abs(rxn_blocksum.transpose()),
            index=rxn_ids_base,
            columns=tp_columns,
        )
        ptools_rxns_df.index.name = "$"

        tsv = ptools_rxns_df.to_csv(
            sep="\t", index=True, header=True, float_format="%.4f"
        )
        # Flux magnitudes are extremely heavy-tailed: a handful of central-
        # carbon reactions carry O(10) flux while most of the ~hundreds of
        # active reactions carry <0.1 (and ~3/4 of all base reactions carry
        # exactly 0).  On a linear color scale the few large reactions saturate
        # the range and the entire matrix renders as a flat ~0 field with no
        # visible band structure.  Render on a log10 color scale and sort
        # reactions by descending magnitude so the decade-spanning structure is
        # legible across all reactions.
        view = ptools_heatmap_view(
            ptools_rxns_df,
            "Reaction fluxes (reaction × timepoint)",
            log_color=True,
            sort_rows=True,
            color_label="|flux| (mmol/gDCW/h)",
        )
        return {"data": {"filename": "ptools_rxns.tsv", "tsv": tsv}, "view": view}
