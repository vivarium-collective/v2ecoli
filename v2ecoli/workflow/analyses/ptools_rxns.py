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
        output_df = read_outputs(history_sql, conn, output_columns)

        # Drop timestep-0 row where FBA hasn't run yet (empty flux array).
        flux_col = "listeners__fba_results__base_reaction_fluxes"
        output_df = output_df[
            output_df[flux_col].apply(len) > 0
        ].reset_index(drop=True)

        rxn_mtx = np.stack(output_df[flux_col].values)

        rxn_ids_base = sim_data.process.metabolism.base_reaction_ids
        # v2ecoli's FBA listener emits one fewer flux than sim_data has reaction
        # IDs (trailing maintenance/boundary entry absent from the output).
        # Truncate to match the actual emitted width.
        n_rxns = rxn_mtx.shape[1]
        rxn_ids_base = rxn_ids_base[:n_rxns]

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
        return {"data": {"filename": "ptools_rxns.tsv", "tsv": tsv}}
