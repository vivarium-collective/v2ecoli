"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_fluxomics.py``.

Per-cell mean base-reaction flux (mmol/gDW/h) for every EcoCyc reaction, as a
wide TSV: one row per reaction, one column per cell, plus the across-cell mean
and standard deviation.  Registered as ``"cd1_fluxomics"`` (scale:
``"multiseed"``).

v2ecoli adaptations
-------------------
* ``rxn_ids`` comes from ``sim_data.process.metabolism.base_reaction_ids``
  rather than ``field_metadata(conn, config_sql, ...)`` — v2ecoli's parquet
  carries no listener field-metadata table.  Same substitution the
  ``central_carbon_metabolism_scatter`` port makes; the width matches the
  ``listeners__fba_results__base_reaction_fluxes`` list (2820).
* The originals write the TSV to ``outdir``; v2ecoli analyses return it as
  ``data["tsv"]`` and the runner writes it under the sweep's ``ptools/`` dir
  (``analysis_runner.run_analyses``), which the flush then copies into the
  owning study.
* Each cell's FIRST emitted row carries a zero-length flux list (the row is
  emitted before FBA has solved).  ``unnest`` yields no rows for an empty list,
  so those rows drop out of the aggregate on their own — the vEcoli original
  relies on the same behaviour.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection
from ecoli.processes.metabolism import COUNTS_UNITS, MASS_UNITS, TIME_UNITS
from wholecell.utils import units

from v2ecoli.workflow.analyses._helpers import (
    cd1_filter_clause,
    read_stacked_columns,
    run_chunked,
    with_cross_cell_stats,
)
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]


class Cd1Fluxomics(Analysis):
    name = "cd1_fluxomics"
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
        # The runner supplies the per-analysis options both as this Step's
        # config and as ``variant_metadata``; the flush path only sets config.
        # Read both so the params land whichever way the analysis was invoked.
        params = {**(self.config or {}), **(variant_metadata or {})}
        filter_clause = cd1_filter_clause(params)

        rxn_ids = [str(r) for r in sim_data.process.metabolism.base_reaction_ids]
        cell_density = sim_data.constants.cell_density.asNumber(units.g / units.L)

        flux_subquery = read_stacked_columns(
            history_sql,
            [
                "listeners__fba_results__base_reaction_fluxes",
                "listeners__mass__cell_mass",
                "listeners__mass__dry_mass",
            ],
            order_results=False,
        )
        id_cols = ", ".join(_ID_COLS)
        filtered_sql = f"""
            SELECT * FROM ({flux_subquery})
            {filter_clause}
        """

        def _batch_sql(cell_filter: str) -> str:
            return f"""
                WITH cell_batch AS (
                    SELECT * FROM ({filtered_sql}) WHERE {cell_filter}
                ),
                unnest_fluxes AS (
                    SELECT listeners__mass__dry_mass /
                        listeners__mass__cell_mass * {cell_density} AS conversion_coeffs,
                        unnest(listeners__fba_results__base_reaction_fluxes) AS fluxes,
                        generate_subscripts(
                            listeners__fba_results__base_reaction_fluxes, 1) AS idx,
                        {id_cols}
                    FROM cell_batch
                )
                SELECT
                    avg(fluxes / conversion_coeffs) AS "flux-avg",
                    stddev(fluxes / conversion_coeffs) AS "flux-std",
                    {id_cols}, idx
                FROM unnest_fluxes
                GROUP BY idx, {id_cols}
                ORDER BY idx
                """

        # Chunked one cell at a time (see run_chunked's docstring / item 38):
        # the full-sweep unnest of every cell's flux array at once is what
        # OOM-kills this analysis. Same AVG/STDDEV math, same unit
        # conversion, just computed per cell and concatenated.
        flux_data = run_chunked(conn, filtered_sql, _batch_sql, id_cols=_ID_COLS)

        if flux_data.is_empty():
            empty = pl.DataFrame({"EcoCyc Reaction ID": [], "mean": [], "std": []})
            return {"data": {"filename": "cd1_fluxomics_detailed.tsv",
                             "tsv": empty.write_csv(separator="\t"),
                             "n_reactions": 0, "n_cells": 0}}

        reaction_lookup = pl.DataFrame(
            {
                "idx": list(range(1, len(rxn_ids) + 1)),
                "EcoCyc Reaction ID": rxn_ids,
            }
        )
        flux_data = flux_data.join(reaction_lookup, on="idx", how="left")
        flux_data = flux_data.with_columns(
            **{
                # Unum carries the polars expression through the unit algebra:
                # (mmol/g/s) -> mmol/gDW/h, exactly as in the vEcoli original.
                "mean": (
                    (COUNTS_UNITS / MASS_UNITS / TIME_UNITS) * pl.col("flux-avg")
                ).asNumber(units.mmol / units.g / units.h),
                "cell_id": pl.format(
                    "Cell: {}_{}", pl.col("lineage_seed"), pl.col("agent_id")
                ),
            }
        )

        wide_table = flux_data.select(
            ["EcoCyc Reaction ID", "cell_id", "mean"]
        ).pivot(
            index="EcoCyc Reaction ID",
            on="cell_id",
            values="mean",
            sort_columns=True,
        )
        n_cells = wide_table.width - 1
        wide_table = with_cross_cell_stats(wide_table, "EcoCyc Reaction ID")
        return {"data": {"filename": "cd1_fluxomics_detailed.tsv",
                         "tsv": wide_table.write_csv(separator="\t"),
                         "n_reactions": wide_table.height, "n_cells": n_cells}}