"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_proteomics.py``.

Per-cell mean protein monomer count for every EcoCyc monomer, as a wide TSV:
one row per monomer, one column per cell, plus the across-cell mean and
standard deviation.  Registered as ``"cd1_proteomics"`` (scale: ``"multiseed"``).

v2ecoli adaptations
-------------------
* ``monomer_ids`` comes from ``sim_data.process.translation.monomer_data["id"]``
  rather than ``field_metadata(conn, config_sql, "listeners__monomer_counts")``
  — the same substitution the ``subgenerational_expression_table`` port makes.
  Width verified against the parquet list (4309).
* The TSV is returned as ``data["tsv"]`` instead of being written to ``outdir``
  (the runner places it under the sweep's ``ptools/`` dir).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    cd1_filter_clause,
    read_stacked_columns,
    with_cross_cell_stats,
)
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]


class Cd1Proteomics(Analysis):
    name = "cd1_proteomics"
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
        filter_clause = cd1_filter_clause(params)

        monomer_ids = [
            str(m) for m in sim_data.process.translation.monomer_data["id"]
        ]
        history_subquery = read_stacked_columns(
            history_sql, ["listeners__monomer_counts"], order_results=False
        )
        id_cols = ", ".join(_ID_COLS)

        proteomics = conn.sql(
            f"""
            WITH history AS ({history_subquery}),
            filtered AS (
                SELECT listeners__monomer_counts AS monomer_counts, {id_cols}
                FROM history
                {filter_clause}
            ),
            exploded AS (
                SELECT
                    unnest(monomer_counts) AS monomer_count,
                    generate_subscripts(monomer_counts, 1) AS idx,
                    {id_cols}
                FROM filtered
            )
            SELECT
                idx,
                {id_cols},
                AVG(monomer_count) AS monomer_mean
            FROM exploded
            GROUP BY idx, {id_cols}
            ORDER BY idx, {id_cols}
            """
        ).pl()

        if proteomics.is_empty():
            empty = pl.DataFrame({"EcoCyc Monomer ID": [], "mean": [], "std": []})
            return {"data": {"filename": "proteomics.tsv",
                             "tsv": empty.write_csv(separator="\t"),
                             "n_monomers": 0, "n_cells": 0}}

        lookup = pl.DataFrame(
            {
                "idx": list(range(1, len(monomer_ids) + 1)),
                # strip the "[c]"-style compartment suffix to get the EcoCyc id
                "EcoCyc Monomer ID": [m[:-3] for m in monomer_ids],
            }
        )
        tidy = proteomics.join(lookup, on="idx", how="left").with_columns(
            pl.format(
                "Cell: {}_{}", pl.col("lineage_seed"), pl.col("agent_id")
            ).alias("cell_id")
        )
        output_final = tidy.select(
            ["EcoCyc Monomer ID", "cell_id", "monomer_mean"]
        ).pivot(
            index="EcoCyc Monomer ID",
            on="cell_id",
            values="monomer_mean",
            sort_columns=True,
        )
        n_cells = output_final.width - 1
        output_final = with_cross_cell_stats(output_final, "EcoCyc Monomer ID")
        return {"data": {"filename": "proteomics.tsv",
                         "tsv": output_final.write_csv(separator="\t"),
                         "n_monomers": output_final.height, "n_cells": n_cells}}
