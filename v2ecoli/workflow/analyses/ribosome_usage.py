"""Native port of vEcoli ``ecoli/analysis/multigeneration/ribosome_usage.py``.

Ribosome-usage statistics over a lineage: cell volume, total/active ribosome
counts and concentrations, molar/mass active fractions, activation/deactivation
counts and per-volume rates, amino acids translated, and effective elongation
rate.  Returns an Altair HTML view.  Registered as ``"ribosome_usage"`` (scale:
``"multigeneration"``).

v2ecoli adaptations:
  * ``bulk`` column → ``bulk__count`` in parquet order (:func:`bulk_count_idx_expr`).
  * ``listeners__unique_molecule_counts__active_ribosome`` (absent) → Shim B.
  * ``listeners__ribosome_data__did_initialize`` is NOT emitted by v2ecoli, so
    only available columns are requested and the activation panels / per-volume
    activation rate are skipped (the deactivation panels use ``did_terminate``,
    which is present).
  * The ``WHERE agent_id = 0`` lineage filter is dropped.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pandas as pd
import numpy as np
from duckdb import DuckDBPyConnection

from ecoli.library.schema import bulk_name_to_idx

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import (
    aliased_history,
    available_columns,
    bulk_count_idx_expr,
    cast_decimals,
    chart_to_html,
    ACTIVE_RIBOSOME_AS_UMC,
)


class RibosomeUsage(Analysis):
    name = "ribosome_usage"
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

        complex_ids_30s = [sim_data.molecule_ids.s30_full_complex]
        complex_ids_50s = [sim_data.molecule_ids.s50_full_complex]
        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
        idx_30s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(complex_ids_30s, bulk_ids))]
        idx_50s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(complex_ids_50s, bulk_ids))]

        n_avogadro = sim_data.constants.n_avogadro
        mw_30s = sim_data.getter.get_masses(complex_ids_30s)
        mw_50s = sim_data.getter.get_masses(complex_ids_50s)
        mw_70s = mw_30s + mw_50s

        candidate_columns = [
            "time", "variant", "generation", "agent_id", "experiment_id",
            "lineage_seed",
            "listeners__mass__instantaneous_growth_rate",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__ribosome_data__did_initialize",
            "listeners__ribosome_data__actual_elongations",
            "listeners__ribosome_data__did_terminate",
            "listeners__ribosome_data__effective_elongation_rate",
        ]
        base = aliased_history(history_sql)
        avail = available_columns(conn, base)
        required_columns = [c for c in candidate_columns if c in avail or c == "time"]

        names_30s = [f"bulk_30s_{i}" for i in idx_30s]
        names_50s = [f"bulk_50s_{i}" for i in idx_50s]
        expr_30s = bulk_count_idx_expr(conn, base, complex_ids_30s, names_30s)
        expr_50s = bulk_count_idx_expr(conn, base, complex_ids_50s, names_50s)

        sql = f"""
        SELECT {", ".join(required_columns)}, {ACTIVE_RIBOSOME_AS_UMC},
            {expr_30s}, {expr_50s}
        FROM ({base})
        ORDER BY generation, time
        """
        df = cast_decimals(conn.sql(sql).pl())

        df = df.with_columns((pl.col("time") / 60).alias("time_min"))
        df = df.with_columns([(pl.col("time") + 1).alias("time_step_sec")])

        cols_30s = [c for c in df.columns if c.startswith("bulk_30s_")]
        cols_50s = [c for c in df.columns if c.startswith("bulk_50s_")]
        df = df.with_columns([
            pl.sum_horizontal(cols_30s).alias("counts_30s"),
            pl.sum_horizontal(cols_50s).alias("counts_50s"),
            pl.col("listeners__unique_molecule_counts__active_ribosome")
            .fill_null(0).alias("active_ribosome_counts"),
        ])
        df = df.with_columns([
            (pl.col("active_ribosome_counts")
             + pl.min_horizontal(pl.col("counts_30s"), pl.col("counts_50s")))
            .alias("total_ribosome_counts"),
            (pl.col("active_ribosome_counts").cast(pl.Float64)
             / (pl.col("active_ribosome_counts")
                + pl.min_horizontal(pl.col("counts_30s"), pl.col("counts_50s"))))
            .alias("molar_fraction_active"),
        ])

        if "listeners__mass__cell_mass" in df.columns:
            cell_density = sim_data.constants.cell_density.asNumber()
            df = df.with_columns(
                (1e-15 * pl.col("listeners__mass__cell_mass") / cell_density)
                .alias("cell_volume"))

        df = df.with_columns([
            (pl.col("total_ribosome_counts") / n_avogadro.asNumber() / pl.col("cell_volume"))
            .alias("total_ribosome_concentration_mM"),
            (pl.col("active_ribosome_counts") / n_avogadro.asNumber() / pl.col("cell_volume"))
            .alias("active_ribosome_concentration_mM"),
        ])

        mw30 = mw_30s.asNumber() if hasattr(mw_30s, "asNumber") else float(mw_30s)
        mw50 = mw_50s.asNumber() if hasattr(mw_50s, "asNumber") else float(mw_50s)
        mw70 = mw_70s.asNumber() if hasattr(mw_70s, "asNumber") else float(mw_70s)
        df = df.with_columns([
            (pl.col("counts_30s") / n_avogadro.asNumber() * mw30).alias("mass_30s"),
            (pl.col("counts_50s") / n_avogadro.asNumber() * mw50).alias("mass_50s"),
            (pl.col("active_ribosome_counts") / n_avogadro.asNumber() * mw70)
            .alias("active_ribosome_mass"),
        ])
        df = df.with_columns([
            (pl.col("active_ribosome_mass") + pl.col("mass_30s") + pl.col("mass_50s"))
            .alias("total_ribosome_mass"),
            (pl.col("active_ribosome_mass")
             / (pl.col("active_ribosome_mass") + pl.col("mass_30s") + pl.col("mass_50s")))
            .alias("mass_fraction_active"),
        ])

        # Per-volume activation/deactivation rates (guarded on column presence)
        if "cell_volume" in df.columns:
            if "listeners__ribosome_data__did_initialize" in df.columns:
                df = df.with_columns(
                    (pl.col("listeners__ribosome_data__did_initialize")
                     / (pl.col("cell_volume") / 1e-15)).alias("activations_per_volume"))
            if "listeners__ribosome_data__did_terminate" in df.columns:
                df = df.with_columns(
                    (pl.col("listeners__ribosome_data__did_terminate")
                     / (pl.col("cell_volume") / 1e-15)).alias("deactivations_per_volume"))

        plot_columns = ["time_min", "variant", "generation"]
        for col in [
            "time_step_sec", "cell_volume", "total_ribosome_counts",
            "total_ribosome_concentration_mM", "active_ribosome_counts",
            "active_ribosome_concentration_mM", "molar_fraction_active",
            "mass_fraction_active", "listeners__ribosome_data__did_initialize",
            "listeners__ribosome_data__did_terminate", "activations_per_volume",
            "deactivations_per_volume", "listeners__ribosome_data__actual_elongations",
            "listeners__ribosome_data__effective_elongation_rate",
        ]:
            if col in df.columns:
                plot_columns.append(col)
        plot_df = df.select(plot_columns)

        def create_line_chart(y_field, title, y_title, skip_first_point=False):
            data = plot_df.to_pandas()
            if skip_first_point:
                filtered = []
                for _, group in data.groupby(["variant", "generation"]):
                    filtered.append(group.iloc[1:] if len(group) > 1 else group)
                data = pd.concat(filtered, ignore_index=True) if filtered else data
            return (
                alt.Chart(data).mark_line().encode(
                    x=alt.X("time_min:Q", title="Time (min)"),
                    y=alt.Y(f"{y_field}:Q", title=y_title),
                    color=alt.Color("generation:N", legend=alt.Legend(title="Generation")),
                ).properties(title=title, width=600, height=120))

        specs = [
            ("time_step_sec", "Length of Time Step", "Length of time step (s)", False),
            ("cell_volume", "Cell Volume", "Cell volume (L)", False),
            ("total_ribosome_counts", "Total Ribosome Count", "Total ribosome count", False),
            ("total_ribosome_concentration_mM", "Total Ribosome Concentration",
             "[Total ribosome] (mM)", False),
            ("active_ribosome_counts", "Active Ribosome Count", "Active ribosome count", True),
            ("active_ribosome_concentration_mM", "Active Ribosome Concentration",
             "[Active ribosome] (mM)", True),
            ("molar_fraction_active", "Molar Fraction Active Ribosomes",
             "Molar fraction active ribosomes", True),
            ("mass_fraction_active", "Mass Fraction Active Ribosomes",
             "Mass fraction active ribosomes", True),
            ("listeners__ribosome_data__did_initialize", "Ribosome Activations",
             "Activations per timestep", False),
            ("listeners__ribosome_data__did_terminate", "Ribosome Deactivations",
             "Deactivations per timestep", False),
            ("activations_per_volume", "Activations per Volume (fL)",
             "Activations per Volume (fL)", False),
            ("deactivations_per_volume", "Deactivations per Volume (fL)",
             "Deactivations per Volume (fL)", False),
            ("listeners__ribosome_data__actual_elongations", "Amino Acids Translated",
             "AA translated", False),
            ("listeners__ribosome_data__effective_elongation_rate",
             "Effective Ribosome Elongation Rate", "Effective elongation rate", False),
        ]
        plots = [create_line_chart(f, t, y, s) for f, t, y, s in specs
                 if f in plot_df.columns]

        if not plots:
            fallback_df = pl.DataFrame(
                {"message": ["No data available for ribosome usage visualization"],
                 "x": [0], "y": [0]})
            plots.append(alt.Chart(fallback_df).mark_text(size=20, color="red")
                         .encode(x="x:Q", y="y:Q", text="message:N")
                         .properties(width=600, height=400,
                                     title="Ribosome Usage Statistics - No Data Available"))

        left_plots = plots[::2]
        right_plots = plots[1::2]
        empty = (alt.Chart(pl.DataFrame({"x": [0], "y": [0]}))
                 .mark_point(opacity=0).encode(x="x:Q", y="y:Q")
                 .properties(width=600, height=120))
        if len(left_plots) > len(right_plots):
            right_plots.append(empty)
        elif len(right_plots) > len(left_plots):
            left_plots.append(empty)

        combined = (alt.hconcat(alt.vconcat(*left_plots), alt.vconcat(*right_plots))
                    .resolve_scale(x="shared", y="independent")
                    .properties(title="Ribosome Usage Statistics"))
        return {"view": chart_to_html(combined, title="Ribosome usage")}
