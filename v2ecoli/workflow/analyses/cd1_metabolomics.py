"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_metabolomics.py``.

Per-cell mean bulk count for every metabolite in the metabolism concentration
dictionary, as a wide TSV: one row per EcoCyc compound, one column per cell,
plus the across-cell mean and standard deviation.  Registered as
``"cd1_metabolomics"`` (scale: ``"multiseed"``).

v2ecoli adaptations
-------------------
* **Shim A (bulk molecules).**  vEcoli emits one ``bulk`` list column ordered to
  match sim_data and reads that ordering via ``field_metadata(conn, config_sql,
  "bulk")``.  v2ecoli emits ``bulk__id`` + ``bulk__count``, so the ordering
  comes from :func:`~v2ecoli.workflow.analyses._helpers.bulk_field_ids` and the
  ``list_select`` reads ``bulk__count``.  See ``_shims`` for the full shim
  catalogue.
* Metabolites absent from the parquet bulk ordering raise rather than being
  silently dropped — matching ``bulk_count_idx_expr``'s contract.  (All 172
  ``conc_dict`` entries are present in the baseline sweep's ordering.)
* ``sim_data`` is injected by the runner, so the port drops the original's
  ``LoadSimData(sim_data_paths[...])`` lookup.
* The TSV is returned as ``data["tsv"]`` instead of being written to ``outdir``
  (the runner places it under the sweep's ``ptools/`` dir).
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    bulk_field_ids,
    cd1_filter_clause,
    read_stacked_columns,
    run_chunked,
    with_cross_cell_stats,
)
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]


class Cd1Metabolomics(Analysis):
    name = "cd1_metabolomics"
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

        mtb_ids = [str(k) for k in sim_data.process.metabolism.conc_dict.keys()]
        # Shim A: parquet bulk ordering, the equivalent of field_metadata("bulk")
        pos = {bid: i for i, bid in enumerate(bulk_field_ids(conn, history_sql))}
        missing = [m for m in mtb_ids if m not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} conc_dict metabolite(s) not in the parquet bulk "
                f"ordering: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        mtb_idxs = [pos[m] + 1 for m in mtb_ids]
        mtb_lookup = pl.DataFrame(
            {
                "idx": list(range(1, len(mtb_ids) + 1)),
                # strip the "[c]"-style compartment suffix to get the EcoCyc id
                "compound_id": [m[:-3] for m in mtb_ids],
            }
        )

        history_subquery = read_stacked_columns(
            history_sql, ["bulk__count"], order_results=False
        )
        id_cols = ", ".join(_ID_COLS)
        idx_list_literal = "[" + ", ".join(str(i) for i in mtb_idxs) + "]"
        filtered_sql = f"""
            SELECT list_select(bulk__count, {idx_list_literal}) AS metabolites,
                {id_cols}
            FROM ({history_subquery})
            {filter_clause}
        """

        def _batch_sql(cell_filter: str) -> str:
            return f"""
                WITH filtered AS (
                    SELECT * FROM ({filtered_sql}) WHERE {cell_filter}
                ),
                exploded AS (
                    SELECT
                        unnest(metabolites) AS metabolite_count,
                        generate_subscripts(metabolites, 1) AS idx,
                        {id_cols}
                    FROM filtered
                )
                SELECT
                    idx,
                    {id_cols},
                    AVG(metabolite_count) AS metabolite_mean
                FROM exploded
                GROUP BY idx, {id_cols}
                ORDER BY idx, {id_cols}
                """

        # Chunked one cell at a time (see run_chunked's docstring / item 38):
        # the full-sweep unnest of every cell's bulk-count array at once is
        # what OOM-kills this analysis. Same AVG math, just per cell.
        metabolite_data = run_chunked(conn, filtered_sql, _batch_sql, id_cols=_ID_COLS)

        if metabolite_data.is_empty():
            empty = pl.DataFrame({"EcoCyc Compound ID": [], "mean": [], "std": []})
            return {"data": {"filename": "metabolomics.tsv",
                             "tsv": empty.write_csv(separator="\t"),
                             "n_compounds": 0, "n_cells": 0}}

        tidy = metabolite_data.join(mtb_lookup, on="idx", how="left").with_columns(
            pl.format(
                "Cell: {}_{}", pl.col("lineage_seed"), pl.col("agent_id")
            ).alias("cell_id")
        )
        output_final = (
            tidy.select(["compound_id", "cell_id", "metabolite_mean"])
            .pivot(
                index="compound_id",
                on="cell_id",
                values="metabolite_mean",
                # Some metabolites share a name once the location suffix is
                # stripped; report their summed means.
                aggregate_function="sum",
                sort_columns=True,
            )
            .rename({"compound_id": "EcoCyc Compound ID"})
        )
        n_cells = output_final.width - 1
        output_final = with_cross_cell_stats(output_final, "EcoCyc Compound ID")
        return {"data": {"filename": "metabolomics.tsv",
                         "tsv": output_final.write_csv(separator="\t"),
                         "n_compounds": output_final.height, "n_cells": n_cells}}
