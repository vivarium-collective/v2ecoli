"""Native port of vEcoli ``ecoli/analysis/multigeneration/ribosome_components.py``.

30S and 50S ribosomal-component counts vs time (rRNAs, full complexes, limiting
protein subunits, and active ribosomes).  Returns an Altair HTML view.
Registered as ``"ribosome_components"`` (scale: ``"multigeneration"``).

v2ecoli adaptations:
  * ``field_metadata("bulk")`` → :func:`bulk_field_ids` (parquet bulk order);
    ``named_idx("bulk", …)`` → ``bulk__count`` indexed in parquet order.
  * ``field_metadata("listeners__monomer_counts")`` → sim_data monomer order
    (``sim_data.process.translation.monomer_data["id"]``).
  * ``listeners__unique_molecule_counts__active_ribosome`` (absent) → Shim B.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import (
    read_stacked_columns,
    bulk_field_ids,
    bulk_count_idx_expr,
    chart_to_html,
    named_idx,
    ACTIVE_RIBOSOME_AS_UMC,
)


class RibosomeComponents(Analysis):
    name = "ribosome_components"
    scale = "multigeneration"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        import altair as alt

        s30_protein_ids = list(sim_data.molecule_groups.s30_proteins)
        s30_16s_rRNA_ids = list(sim_data.molecule_groups.s30_16s_rRNA)
        s30_full_complex_id = sim_data.molecule_ids.s30_full_complex
        s50_protein_ids = list(sim_data.molecule_groups.s50_proteins)
        s50_23s_rRNA_ids = list(sim_data.molecule_groups.s50_23s_rRNA)
        s50_5s_rRNA_ids = list(sim_data.molecule_groups.s50_5s_rRNA)
        s50_full_complex_id = sim_data.molecule_ids.s50_full_complex

        complexation = sim_data.process.complexation
        s30_info = complexation.get_monomers(s30_full_complex_id)
        s50_info = complexation.get_monomers(s50_full_complex_id)
        s30_stoich = dict(zip(s30_info["subunitIds"], s30_info["subunitStoich"]))
        s50_stoich = dict(zip(s50_info["subunitIds"], s50_info["subunitStoich"]))

        # Bulk molecules present in the parquet bulk ordering
        bulk_order = set(bulk_field_ids(conn, history_sql))
        s30_16s = [i for i in s30_16s_rRNA_ids if i in bulk_order]
        s50_23s = [i for i in s50_23s_rRNA_ids if i in bulk_order]
        s50_5s = [i for i in s50_5s_rRNA_ids if i in bulk_order]

        # monomer_counts ordering from sim_data (== listener field metadata)
        mono_ids = sim_data.process.translation.monomer_data["id"].tolist()
        mono_index = {mid: idx for idx, mid in enumerate(mono_ids)}
        s30_proteins = [p for p in s30_protein_ids if p in mono_index]
        s50_proteins = [p for p in s50_protein_ids if p in mono_index]

        bulk_cols = [
            bulk_count_idx_expr(conn, history_sql, s30_16s, s30_16s),
            bulk_count_idx_expr(conn, history_sql, s50_23s, s50_23s),
            bulk_count_idx_expr(conn, history_sql, s50_5s, s50_5s),
            bulk_count_idx_expr(conn, history_sql, [s30_full_complex_id],
                                [s30_full_complex_id]),
            bulk_count_idx_expr(conn, history_sql, [s50_full_complex_id],
                                [s50_full_complex_id]),
        ]
        protein_cols = [
            named_idx("listeners__monomer_counts", [pid], [[mono_index[pid]]])
            for pid in s30_proteins + s50_proteins
        ]
        cols = bulk_cols + protein_cols + [ACTIVE_RIBOSOME_AS_UMC]

        data = read_stacked_columns(history_sql, cols, conn=conn)
        df = pl.DataFrame(data).with_columns(Time_min=pl.col("time") / 60)

        s30_16s_sum = pl.sum_horizontal([pl.col(i) for i in s30_16s])
        s50_23s_sum = pl.sum_horizontal([pl.col(i) for i in s50_23s])
        s50_5s_sum = pl.sum_horizontal([pl.col(i) for i in s50_5s])
        s30_complex = pl.col(s30_full_complex_id)
        s50_complex = pl.col(s50_full_complex_id)
        active_ribo = pl.col("listeners__unique_molecule_counts__active_ribosome")

        for pid in s30_proteins:
            df = df.with_columns(**{f"adj_s30_{pid}": pl.col(pid) / s30_stoich[pid]})
        for pid in s50_proteins:
            df = df.with_columns(**{f"adj_s50_{pid}": pl.col(pid) / s50_stoich[pid]})

        s30_lim = pl.min_horizontal([pl.col(f"adj_s30_{pid}") for pid in s30_proteins])
        s50_lim = pl.min_horizontal([pl.col(f"adj_s50_{pid}") for pid in s50_proteins])

        df = df.with_columns(
            s30_16s_total=s30_16s_sum + s30_complex + active_ribo,
            s50_23s_total=s50_23s_sum + s50_complex + active_ribo,
            s50_5s_total=s50_5s_sum + s50_complex + active_ribo,
            s30_limiting=s30_lim,
            s50_limiting=s50_lim,
            s30_total=s30_complex + active_ribo,
            s50_total=s50_complex + active_ribo,
        )

        plot_cols_30 = ["s30_limiting", "s30_16s_total", "s30_total"]
        plot_cols_50 = ["s50_limiting", "s50_23s_total", "s50_5s_total", "s50_total"]
        melt_30 = df.select(["Time_min"] + plot_cols_30).melt(
            id_vars="Time_min", variable_name="component", value_name="count")
        melt_50 = df.select(["Time_min"] + plot_cols_50).melt(
            id_vars="Time_min", variable_name="component", value_name="count")

        chart_30 = (
            alt.Chart(melt_30).mark_line().encode(
                x="Time_min", y="count",
                color=alt.Color("component", title="30S Components"))
            .properties(title="30S Component Counts", width=600))
        chart_50 = (
            alt.Chart(melt_50).mark_line().encode(
                x="Time_min", y="count",
                color=alt.Color("component", title="50S Components"))
            .properties(title="50S Component Counts", width=600))
        combined = (alt.vconcat(chart_30, chart_50)
                    .resolve_scale(color="independent")
                    .resolve_legend(color="independent"))
        return {"view": chart_to_html(combined, title="Ribosome components")}
