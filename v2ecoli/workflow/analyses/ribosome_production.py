"""Native port of vEcoli ``ecoli/analysis/multigeneration/ribosome_production.py``.

Ribosome-production metrics over a lineage: normalized dry mass, cell and rRNA
doubling times, rRNA initiation probabilities, and effective elongation rate,
each as a line + histogram pair.  Returns an Altair HTML view.  Registered as
``"ribosome_production"`` (scale: ``"multigeneration"``).

v2ecoli adaptations:
  * ``bulk`` column → ``bulk__count`` indexed in parquet order
    (:func:`bulk_count_idx_expr`), not sim_data order.
  * ``listeners__unique_molecule_counts__active_ribosome`` (absent) → Shim B
    (``list_sum(n_ribosomes_per_transcript)``) aliased to the original name.
  * The ``WHERE agent_id = 0`` lineage filter is dropped — the multigeneration
    ``history_sql`` already scopes one lineage (its later generations carry
    agent_id ``'00'``, ``'000'``, … not ``'0'``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.schema import bulk_name_to_idx

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import (
    aliased_history,
    bulk_count_idx_expr,
    chart_to_html,
    ACTIVE_RIBOSOME_AS_UMC,
)


def calc_rna_doubling_time(produced_col, count_col, borderline) -> pl.Expr:
    """rRNA doubling time with sanitation (verbatim from vEcoli)."""
    production_rate = pl.col(produced_col) / pl.col("time_step_sec")
    growth_rate = production_rate / pl.col(count_col)
    dt_min = float(np.log(2)) / growth_rate / 60
    valid = (
        (pl.col(produced_col) >= 0)
        & (pl.col(count_col) > 0)
        & (growth_rate > 0)
        & dt_min.is_finite()
        & (dt_min > 0)
        & (dt_min < 2 * borderline)
    )
    return pl.when(valid).then(dt_min).otherwise(None)


class RibosomeProduction(Analysis):
    name = "ribosome_production"
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

        sim_doubling_time = sim_data.doubling_time.asNumber()

        s30_16s = list(sim_data.molecule_groups.s30_16s_rRNA) + [
            sim_data.molecule_ids.s30_full_complex]
        s50_23s = list(sim_data.molecule_groups.s50_23s_rRNA) + [
            sim_data.molecule_ids.s50_full_complex]
        s50_5s = list(sim_data.molecule_groups.s50_5s_rRNA) + [
            sim_data.molecule_ids.s50_full_complex]
        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()

        # sim_data indices used only as stable column-name labels
        idx_16s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(s30_16s, bulk_ids))]
        idx_23s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(s50_23s, bulk_ids))]
        idx_5s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(s50_5s, bulk_ids))]
        names_16s = [f"bulk_{i}" for i in idx_16s]
        names_23s = [f"bulk_{i}" for i in idx_23s]
        names_5s = [f"bulk_{i}" for i in idx_5s]

        required_columns = [
            "time", "variant", "generation", "agent_id",
            "listeners__mass__instantaneous_growth_rate",
            "listeners__mass__dry_mass",
            "listeners__ribosome_data__rRNA16S_initiated",
            "listeners__ribosome_data__rRNA23S_initiated",
            "listeners__ribosome_data__rRNA5S_initiated",
            "listeners__ribosome_data__rRNA16S_init_prob",
            "listeners__ribosome_data__rRNA23S_init_prob",
            "listeners__ribosome_data__rRNA5S_init_prob",
            "listeners__ribosome_data__effective_elongation_rate",
        ]

        base = aliased_history(history_sql)
        bulk_16s_expr = bulk_count_idx_expr(conn, base, s30_16s, names_16s)
        bulk_23s_expr = bulk_count_idx_expr(conn, base, s50_23s, names_23s)
        bulk_5s_expr = bulk_count_idx_expr(conn, base, s50_5s, names_5s)

        sql = f"""
        SELECT {", ".join(required_columns)}, {ACTIVE_RIBOSOME_AS_UMC},
            {bulk_16s_expr}, {bulk_23s_expr}, {bulk_5s_expr}
        FROM ({base})
        ORDER BY generation, time
        """
        df = conn.sql(sql).pl()

        df = df.with_columns((pl.col("time") / 60).alias("time_min"))
        df = df.with_columns(
            pl.col("time").diff().over(["variant", "generation", "agent_id"])
            .alias("time_step_sec")
        )
        df = df.with_columns(
            time_step_sec=pl.when(pl.col("time_step_sec").is_null())
            .then(pl.col("time")).otherwise(pl.col("time_step_sec"))
        )

        if "listeners__mass__instantaneous_growth_rate" in df.columns:
            val = (float(np.log(2))
                   / pl.col("listeners__mass__instantaneous_growth_rate") / 60)
            df = df.with_columns(
                pl.when(val.is_between(0, 2 * sim_doubling_time, closed="both"))
                .then(val).otherwise(None).alias("cell_doubling_time_min")
            )

        df = df.with_columns([
            pl.sum_horizontal([pl.col(n) for n in names_16s]).alias("bulk_16s_count"),
            pl.sum_horizontal([pl.col(n) for n in names_23s]).alias("bulk_23s_count"),
            pl.sum_horizontal([pl.col(n) for n in names_5s]).alias("bulk_5s_count"),
            pl.col("listeners__unique_molecule_counts__active_ribosome")
            .fill_null(0).alias("ribosome_count"),
        ])
        df = df.with_columns([
            (pl.col("bulk_16s_count") + pl.col("ribosome_count")).alias("rrn16s_count"),
            (pl.col("bulk_23s_count") + pl.col("ribosome_count")).alias("rrn23s_count"),
            (pl.col("bulk_5s_count") + pl.col("ribosome_count")).alias("rrn5s_count"),
        ])

        for suffix, prod, cnt in (
            ("16S", "listeners__ribosome_data__rRNA16S_initiated", "rrn16s_count"),
            ("23S", "listeners__ribosome_data__rRNA23S_initiated", "rrn23s_count"),
            ("5S", "listeners__ribosome_data__rRNA5S_initiated", "rrn5s_count"),
        ):
            if prod in df.columns:
                df = df.with_columns(**{
                    f"rrn{suffix}_doubling_time_min": calc_rna_doubling_time(
                        prod, cnt, sim_doubling_time)})

        cond = sim_data.condition
        trans = sim_data.process.transcription
        synth_probs = trans.cistron_tu_mapping_matrix.dot(trans.rna_synth_prob[cond])

        def fit_prob(group_ids):
            cistrons = [rid[:-3] for rid in group_ids]
            idxs = np.where(np.isin(trans.cistron_data["id"], cistrons))[0]
            return synth_probs[idxs].sum() if idxs.size else 0.0

        ref_probs = {
            "16S": fit_prob(sim_data.molecule_groups.s30_16s_rRNA),
            "23S": fit_prob(sim_data.molecule_groups.s50_23s_rRNA),
            "5S": fit_prob(sim_data.molecule_groups.s50_5s_rRNA),
        }

        plot_cols = ["time_min", "variant", "generation"]
        for c in [
            "listeners__mass__dry_mass", "cell_doubling_time_min",
            "rrn16S_doubling_time_min", "rrn23S_doubling_time_min",
            "rrn5S_doubling_time_min", "rrn16S_init_prob", "rrn23S_init_prob",
            "rrn5S_init_prob", "listeners__ribosome_data__effective_elongation_rate",
        ]:
            if c in df.columns:
                plot_cols.append(c)
        plot_df = df.select(plot_cols)

        init_dm = (
            plot_df.filter(pl.col("time_min") == 0)
            .select(["variant", "listeners__mass__dry_mass"])
            .rename({"listeners__mass__dry_mass": "initial_dry_mass"})
        )
        plot_df = plot_df.join(init_dm, on=["variant"], how="left")
        plot_df = plot_df.with_columns(
            (pl.col("listeners__mass__dry_mass") / pl.col("initial_dry_mass"))
            .alias("dry_mass_normalized")
        )

        def create_line_chart(y, title, y_title, ref=None):
            line = (
                alt.Chart(plot_df).mark_line().encode(
                    x=alt.X("time_min:Q", title="Time (min)"),
                    y=alt.Y(f"{y}:Q", title=y_title),
                    color=alt.Color("generation:N",
                                    legend=alt.Legend(title="Simulated Multigeneration Data")),
                ).properties(title=title, width=600, height=120)
            )
            if ref is not None:
                rule = (alt.Chart(pd.DataFrame({"y": [ref]}))
                        .mark_rule(color="red", strokeDash=[5, 5]).encode(y="y:Q"))
                return line + rule
            return line

        def create_histogram(col, title, bins=30, probability=False):
            if probability:
                return (
                    alt.Chart(plot_df)
                    .transform_density(col, as_=[col, "density"], counts=False, steps=bins)
                    .mark_area(opacity=0.6)
                    .encode(x=alt.X(f"{col}:Q", title=f"bin={bins}"),
                            y=alt.Y("density:Q", title="Density"))
                    .properties(width=200, height=120, title=title)
                )
            return (
                alt.Chart(plot_df).mark_bar(opacity=0.6).encode(
                    x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=f"bin={bins}"),
                    y=alt.Y("count():Q", title="Count"), color=alt.value("steelblue"),
                ).properties(width=200, height=120, title=title)
            )

        plots = []
        if "dry_mass_normalized" in plot_df.columns:
            plots.append(alt.hconcat(
                create_line_chart("dry_mass_normalized",
                                  "Normalized Dry Mass Over Time",
                                  "Dry mass (relative to t=0)"),
                create_histogram("dry_mass_normalized",
                                 "Normalized Dry Mass Distribution", probability=True)))
        if "cell_doubling_time_min" in plot_df.columns:
            plots.append(alt.hconcat(
                create_line_chart("cell_doubling_time_min", "Cell Doubling Time",
                                  "Doubling Time (min)", sim_doubling_time),
                create_histogram("cell_doubling_time_min",
                                 "Cell Doubling Time (min) Distribution", probability=True)))
        for suffix in ["16S", "23S", "5S"]:
            col = f"rrn{suffix}_doubling_time_min"
            if col in plot_df.columns:
                plots.append(alt.hconcat(
                    create_line_chart(col, f"{suffix} rRNA Doubling Time",
                                      "Doubling Time (min)", sim_doubling_time),
                    create_histogram(col, f"{suffix} rRNA Doubling Time Distribution",
                                     probability=True)))
        for suffix, ref in ref_probs.items():
            col = f"rrn{suffix}_init_prob"
            if col in plot_df.columns:
                plots.append(alt.hconcat(
                    create_line_chart(col, f"{suffix} rRNA Initiation Probability",
                                      "Probability", ref),
                    create_histogram(col, f"{suffix} rRNA Initiation Probability Distribution",
                                     probability=True)))
        if "listeners__ribosome_data__effective_elongation_rate" in plot_df.columns:
            plots.append(alt.hconcat(
                create_line_chart("listeners__ribosome_data__effective_elongation_rate",
                                  "Ribosome Elongation Rate", "Amino acids/s"),
                create_histogram("listeners__ribosome_data__effective_elongation_rate",
                                 "Ribosome Elongation Rate Distribution", probability=True)))

        if not plots:
            fallback = pl.DataFrame({"message": ["No data available"], "x": [0], "y": [0]})
            plots.append(alt.Chart(fallback).mark_text(size=20, color="red")
                         .encode(x="x:Q", y="y:Q", text="message:N")
                         .properties(width=600, height=400, title="No Data"))

        combined = (alt.vconcat(*plots).resolve_scale(x="shared", y="independent")
                    .properties(title="Ribosome Production Metrics"))
        return {"view": chart_to_html(combined, title="Ribosome production")}
