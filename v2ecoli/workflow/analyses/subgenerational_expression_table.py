"""Native port of vEcoli ``ecoli/analysis/multiseed/subgenerational_expression_table.py``.

Table of subgenerationally-expressed genes (0 < P(expressed per cell cycle) < 1)
with their expression frequency and max mRNA/protein counts.  Returns a TSV.
Registered as ``"subgenerational_expression_table"`` (scale: ``"multiseed"``).

v2ecoli adaptations:
  * Listener field orderings normally read via ``field_metadata`` come from
    sim_data instead — ``listeners__monomer_counts`` ↔
    ``translation.monomer_data["id"]`` (width 4309) and
    ``listeners__rna_counts__mRNA_cistron_counts`` ↔
    ``transcription.cistron_data["id"][is_mRNA]`` (width 4345); both verified
    to match the parquet list widths.
  * ``config_sql`` is unused (the generation count check uses ``num_cells`` over
    ``history_sql``); ``ignore_first_n_gens`` defaults to 0 (vEcoli used 8).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import (
    read_stacked_columns,
    num_cells,
    skip_n_gens,
    ndidx_to_duckdb_expr,
)


class SubgenerationalExpressionTable(Analysis):
    name = "subgenerational_expression_table"
    scale = "multiseed"
    config_schema = {"ignore_first_n_gens": "integer"}

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
        ignore = int(params.get("ignore_first_n_gens", 0))

        history_sql = skip_n_gens(history_sql, ignore)
        if num_cells(conn, history_sql) == 0:
            return {"data": {"note": "Not enough generations run; skipped.",
                             "filename": "subgen.tsv", "tsv": ""}}

        cistron_data = sim_data.process.transcription.cistron_data
        cistron_ids = cistron_data["id"]
        cistron_id_to_protein_id = {
            p["cistron_id"]: p["id"]
            for p in sim_data.process.translation.monomer_data
        }
        mRNA_cistron_ids = [
            cid for cid in cistron_ids if cid in cistron_id_to_protein_id]
        monomer_ids = [cistron_id_to_protein_id[c] for c in mRNA_cistron_ids]
        cistron_id_to_gene_id = {c["id"]: c["gene_id"] for c in cistron_data}
        gene_ids = [cistron_id_to_gene_id[c] for c in mRNA_cistron_ids]

        # vEcoli field_metadata orderings, sourced from sim_data (widths verified)
        mRNA_cistron_ids_table = cistron_data["id"][cistron_data["is_mRNA"]].tolist()
        mRNA_cistron_id_to_index = {
            cid: i + 1 for i, cid in enumerate(mRNA_cistron_ids_table)}
        mRNA_cistron_indexes = [
            mRNA_cistron_id_to_index[c] for c in mRNA_cistron_ids]

        monomer_ids_table = sim_data.process.translation.monomer_data["id"].tolist()
        monomer_id_to_index = {m: i + 1 for i, m in enumerate(monomer_ids_table)}
        monomer_indexes = [monomer_id_to_index[m] for m in monomer_ids]

        monomer_expr = ndidx_to_duckdb_expr(
            "listeners__monomer_counts", [monomer_indexes])
        cistron_expr = ndidx_to_duckdb_expr(
            "listeners__rna_counts__mRNA_cistron_counts", [mRNA_cistron_indexes])
        subquery = read_stacked_columns(
            history_sql, [monomer_expr, cistron_expr], order_results=False)

        out_df = conn.sql(f"""
            WITH unnested_counts AS (
                SELECT lineage_seed, generation, agent_id,
                    unnest(listeners__monomer_counts) AS monomer_counts,
                    unnest(listeners__rna_counts__mRNA_cistron_counts) AS mrna_counts,
                    generate_subscripts(listeners__monomer_counts, 1) AS cistron_idx
                FROM ({subquery})
            ),
            cell_aggregate AS (
                SELECT
                    SUM(mrna_counts) > 0 AS exists,
                    MAX(monomer_counts) AS max_monomer_counts,
                    MAX(mrna_counts) AS max_mRNA_counts,
                    cistron_idx
                FROM unnested_counts
                GROUP BY lineage_seed, generation, agent_id, cistron_idx
            ),
            full_aggregate AS (
                SELECT
                    AVG(exists::INTEGER) AS p_expressed,
                    MAX(max_monomer_counts) AS max_monomer_counts,
                    MAX(max_mRNA_counts) AS max_mRNA_counts,
                    cistron_idx
                FROM cell_aggregate
                GROUP BY cistron_idx
            )
            SELECT * FROM full_aggregate
            WHERE p_expressed > 0 AND p_expressed < 1
            """).pl()

        out_df = out_df.with_columns(
            gene_name=pl.Series(gene_ids)[out_df["cistron_idx"] - 1],
            cistron_name=pl.Series(mRNA_cistron_ids)[out_df["cistron_idx"] - 1],
            protein_name=pl.Series([i[:-3] for i in monomer_ids])[
                out_df["cistron_idx"] - 1],
        )
        tsv = out_df.write_csv(separator="\t")
        return {"data": {"filename": "subgen.tsv", "tsv": tsv,
                         "n_subgen_genes": out_df.height}}
