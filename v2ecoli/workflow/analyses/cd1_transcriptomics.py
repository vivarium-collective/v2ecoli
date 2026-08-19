"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_transcriptomics.py``.

Per-cell mean mRNA cistron count for every EcoCyc gene, as a wide TSV: one row
per gene, one column per cell, plus the across-cell mean and standard
deviation.  Registered as ``"cd1_transcriptomics"`` (scale: ``"multiseed"``).

v2ecoli adaptations
-------------------
* ``mrna_ids`` comes from ``sim_data.process.transcription.cistron_data``
  (``["id"][is_mRNA]``) rather than
  ``field_metadata(conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts")``
  — the same substitution the ``subgenerational_expression_table`` port makes.
  Width verified against the parquet list (4345).
* The TSV is returned as ``data["tsv"]`` instead of being written to ``outdir``
  (the runner places it under the sweep's ``ptools/`` dir).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    generation_time_filter_clause,
    read_stacked_columns,
    run_chunked,
    with_cross_cell_stats,
)
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]


class Cd1Transcriptomics(Analysis):
    name = "cd1_transcriptomics"
    scale = "multiseed"
    config_schema = {
        "generation_lower_bound": "integer",
        "time_lower_bound": "float",
    }

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        params = {**(self.config or {}), **(variant_metadata or {})}
        filter_clause = generation_time_filter_clause(params)

        cistron_data = sim_data.process.transcription.cistron_data
        mrna_ids = [
            str(c) for c in cistron_data["id"][cistron_data["is_mRNA"]]
        ]
        history_subquery = read_stacked_columns(
            history_sql,
            ["listeners__rna_counts__mRNA_cistron_counts AS mrna_counts"],
            order_results=False,
        )
        id_cols = ", ".join(_ID_COLS)
        filtered_sql = f"""
            SELECT * FROM ({history_subquery})
            {filter_clause}
        """

        def _batch_sql(cell_filter: str) -> str:
            return f"""
                WITH filtered AS (
                    SELECT * FROM ({filtered_sql}) WHERE {cell_filter}
                ),
                exploded AS (
                    SELECT
                        unnest(mrna_counts) AS mrna_count,
                        generate_subscripts(mrna_counts, 1) AS idx,
                        {id_cols}
                    FROM filtered
                )
                SELECT
                    idx,
                    {id_cols},
                    AVG(mrna_count) AS mrna_avg
                FROM exploded
                GROUP BY idx, {id_cols}
                ORDER BY idx, {id_cols}
                """

        # Chunked one cell at a time (see run_chunked's docstring / item 38):
        # the full-sweep unnest of every cell's mRNA-count array at once is
        # what OOM-kills this analysis. Same AVG math, just per cell.
        transcriptomics = run_chunked(conn, filtered_sql, _batch_sql, id_cols=_ID_COLS)

        if transcriptomics.is_empty():
            empty = pl.DataFrame({"EcoCyc Gene ID": [], "mean": [], "std": []})
            return {"data": {"filename": "transcriptomics.tsv",
                             "tsv": empty.write_csv(separator="\t"),
                             "n_genes": 0, "n_cells": 0}}

        lookup = pl.DataFrame(
            {
                "idx": list(range(1, len(mrna_ids) + 1)),
                # cistron ids are "<gene>_RNA"; drop the suffix for the gene id
                "EcoCyc Gene ID": [m[:-4] for m in mrna_ids],
            }
        )
        tidy = transcriptomics.join(lookup, on="idx", how="left").with_columns(
            pl.format(
                "Cell: {}_{}_{}", pl.col("lineage_seed"), pl.col("generation"),
                pl.col("agent_id")
            ).alias("cell_id")
        )
        output_final = tidy.select(
            ["EcoCyc Gene ID", "cell_id", "mrna_avg"]
        ).pivot(
            index="EcoCyc Gene ID",
            on="cell_id",
            values="mrna_avg",
            sort_columns=True,
        )
        n_cells = output_final.width - 1
        output_final = with_cross_cell_stats(output_final, "EcoCyc Gene ID")
        return {"data": {"filename": "transcriptomics.tsv",
                         "tsv": output_final.write_csv(separator="\t"),
                         "n_genes": output_final.height, "n_cells": n_cells}}
