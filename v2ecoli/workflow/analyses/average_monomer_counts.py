"""Native port of vEcoli ``ecoli/analysis/multivariant/average_monomer_counts.py``.

Per-variant average monomer (protein) counts, saved as unfiltered and filtered
CSV tables comparing each experimental variant to the control variant.
Registered as ``"average_monomer_counts"`` (scale: ``"multivariant"``).

Requires at least two variants (one control + one experimental); with a single
variant the result is an informational note and no tables (matches the source,
whose per-experimental-variant loop is then empty).

v2ecoli note: orderings come from sim_data (no config_sql needed); output is
returned as ``data["files"] = {filename: csv_text}`` instead of written to disk.
"""

from __future__ import annotations

import csv
import io
from typing import Any, cast

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import read_stacked_columns

IGNORE_FIRST_N_GENS = 1


def _csv_text(columns, values) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for i in range(values.shape[0]):
        writer.writerow(values[i, :])
    return buf.getvalue()


class AverageMonomerCounts(Analysis):
    name = "average_monomer_counts"
    scale = "multivariant"
    config_schema = {"filter_num": "integer"}

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        from ecoli.library.parquet_emitter import ndlist_to_ndarray

        params = dict(variant_metadata or {})

        cistron = sim_data.process.transcription.cistron_data.struct_array
        monomer = sim_data.process.translation.monomer_data.struct_array
        new_gene_mRNA_ids = cistron[cistron["is_new_gene"]]["id"].tolist()
        mRNA_monomer_id_dict = dict(zip(monomer["cistron_id"], monomer["id"]))
        new_gene_monomer_ids = [
            cast(str, mRNA_monomer_id_dict.get(m)) for m in new_gene_mRNA_ids]
        all_monomer_ids = monomer["id"]
        original_monomer_ids = all_monomer_ids[
            ~np.isin(all_monomer_ids, new_gene_monomer_ids)]
        monomer_idx_dict = {m: i for i, m in enumerate(all_monomer_ids)}
        original_monomer_idx = [
            cast(int, monomer_idx_dict.get(m)) for m in original_monomer_ids]

        subquery = read_stacked_columns(
            history_sql, ["listeners__monomer_counts"], order_results=False)
        avg_monomer_per_variant = conn.sql(f"""
            WITH unnested_counts AS (
                SELECT unnest(listeners__monomer_counts) AS monomer_counts,
                    generate_subscripts(listeners__monomer_counts, 1)
                        AS monomer_idx, variant
                FROM ({subquery})
            ),
            average_counts AS (
                SELECT avg(monomer_counts) AS avg_counts, variant, monomer_idx
                FROM unnested_counts
                GROUP BY variant, monomer_idx
            )
            SELECT list(avg_counts ORDER BY monomer_idx) AS avg_monomer_counts, variant
            FROM average_counts
            GROUP BY variant
            ORDER BY variant
            """).pl()

        filter_num = int(params.get("filter_num", 0))
        variants = avg_monomer_per_variant["variant"].to_list()
        if len(variants) < 2:
            return {"data": {
                "note": "average_monomer_counts requires >=2 variants "
                        f"(found {len(variants)}); no tables produced.",
                "files": {}}}

        control_variant = variants[0]
        files: dict[str, str] = {}
        for exp_variant in variants[1:]:
            suffix = f"var_{exp_variant}_startGen_{IGNORE_FIRST_N_GENS}.csv"
            variant_pair = avg_monomer_per_variant.filter(
                pl.col("variant").is_in([control_variant, exp_variant])
            ).sort("variant")
            avg_counts = ndlist_to_ndarray(variant_pair["avg_monomer_counts"])

            col_labels = ["all_monomer_ids", "var_0_avg_PCs",
                          f"var_{exp_variant}_avg_PCs"]
            values = np.concatenate(
                (all_monomer_ids[:, np.newaxis], avg_counts.T), axis=1)
            files[f"wcm_full_monomers_{suffix}"] = _csv_text(col_labels, values)

            avg_counts = avg_counts[:, original_monomer_idx]
            v0 = np.nonzero(avg_counts[0] > filter_num)
            v1 = np.nonzero(avg_counts[1] > filter_num)
            shared = np.intersect1d(v0, v1)
            avg_counts = avg_counts[:, shared]
            filtered_ids = original_monomer_ids[shared]
            col_labels = ["filtered_monomer_ids", "var_0_avg_PCs",
                          f"var_{exp_variant}_avg_PCs"]
            values = np.concatenate(
                (filtered_ids[:, np.newaxis], avg_counts.T), axis=1)
            files[f"wcm_filter_monomers_{suffix}"] = _csv_text(col_labels, values)

        return {"data": {"files": files, "n_variants": len(variants)}}
