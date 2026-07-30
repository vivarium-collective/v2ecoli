"""Native port of vEcoli ``ecoli/analysis/multiseed/cd1_exchange_fluxes.py``.

Per-cell mean external exchange flux for every exchanged metabolite, plus the
mean growth rate, as a wide TSV: one row per compound (and one for
``growth_rate_h``), one column per cell, plus the across-cell mean and standard
deviation.  Registered as ``"cd1_exchange_fluxes"`` (scale: ``"multiseed"``).

The vEcoli original carries a stale ``TODO: Implement`` banner, but its
``plot()`` is complete and is what this ports.

v2ecoli adaptations
-------------------
* **Shim E (exchange flux ordering).**  vEcoli explodes the exchange fluxes
  into one column per molecule and discovers them by ``fnmatch`` on
  ``listeners__fba_results__external_exchange_fluxes__*``.  v2ecoli emits a
  single 87-wide list column with no per-element names, so the port indexes it
  by :func:`~v2ecoli.workflow.analyses._shims.external_exchange_molecule_ids`
  (the sorted external-exchange molecule ids) — see ``_shims`` for why that is
  the emitted order and how it was verified.
* Each cell's FIRST emitted row carries a zero-length flux list (emitted before
  FBA has solved).  Indexing an empty list yields NULL, which ``AVG`` skips, so
  those rows drop out on their own.
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
from v2ecoli.workflow.analyses._shims import external_exchange_molecule_ids
from v2ecoli.workflow.analysis import Analysis

_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]
_FLUX_COL = "listeners__fba_results__external_exchange_fluxes"


class Cd1ExchangeFluxes(Analysis):
    name = "cd1_exchange_fluxes"
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

        # Shim E: name each element of the flux list by its sorted molecule id,
        # then strip the compartment suffix the way the original's column-name
        # split did ("GLC[p]" -> "GLC").
        mol_ids = external_exchange_molecule_ids(sim_data)
        names, seen = [], set()
        selects, avg_fluxes = [], []
        for i, mol in enumerate(mol_ids):
            name = mol.split("[")[0]
            if name in seen:  # keep names unique; compartment-stripping can collide
                name = mol
            seen.add(name)
            names.append(name)
            escaped = name.replace('"', '""')
            selects.append(f'{_FLUX_COL}[{i + 1}] AS "{escaped}"')
            avg_fluxes.append(f'AVG("{escaped}") AS "{escaped}"')

        columns = [
            "listeners__mass__instantaneous_growth_rate * 3600 AS growth_rate_h",
            *selects,
        ]
        flux_subquery = read_stacked_columns(
            history_sql, columns, order_results=False
        )
        id_cols = ", ".join(_ID_COLS)

        flux_data = conn.sql(
            f"""
            SELECT {", ".join(avg_fluxes)},
                avg(growth_rate_h) AS growth_rate_h,
                concat('Cell: ', lineage_seed, '_', agent_id) AS cell_id
            FROM ({flux_subquery})
            {filter_clause}
            GROUP BY {id_cols}
            """
        ).pl()

        if flux_data.is_empty():
            empty = pl.DataFrame({"EcoCyc Compound ID": [], "mean": [], "std": []})
            return {"data": {"filename": "exchange_fluxes.tsv",
                             "tsv": empty.write_csv(separator="\t"),
                             "n_compounds": 0, "n_cells": 0}}

        # Transpose: cell_ids become columns; metabolites + growth_rate_h rows.
        cell_ids = flux_data["cell_id"].to_list()
        wide_table = flux_data.drop("cell_id").transpose(
            include_header=True,
            header_name="EcoCyc Compound ID",
            column_names=cell_ids,
        )
        n_cells = len(cell_ids)
        wide_table = with_cross_cell_stats(wide_table, "EcoCyc Compound ID")
        return {"data": {"filename": "exchange_fluxes.tsv",
                         "tsv": wide_table.write_csv(separator="\t"),
                         "n_compounds": wide_table.height, "n_cells": n_cells}}
