"""Native port of vEcoli ``ecoli/analysis/multigeneration/replication.py``.

Multi-panel replication visualization over a lineage: DNA-polymerase fork
positions vs time, pairs of replication forks, number of oriC, and dry mass.
Returns an Altair HTML view.  Registered as ``"replication"`` (scale:
``"multigeneration"``).

v2ecoli deviations (recorded in the porting note):
  * Absolute time axis via :func:`cumulative_time_history` (v2ecoli's
    ``global_time`` resets per generation; vEcoli's ``time`` is absolute).
  * ``listeners__replication_data__critical_initiation_mass`` and
    ``...__critical_mass_per_oric`` are NOT emitted by v2ecoli, so the
    "factors of critical initiation mass" and "critical mass per oriC" panels
    are absent (the source already guards each panel with ``if col in
    df.columns``); only available columns are requested.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import (
    read_stacked_columns,
    cumulative_time_history,
    available_columns,
    chart_to_html,
)

CRITICAL_N = [1, 2, 4, 8]


class Replication(Analysis):
    """Replication panels over a lineage (multigeneration)."""

    name = "replication"
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

        genome_length = len(sim_data.process.replication.genome_sequence)

        abs_sql = cumulative_time_history(history_sql)
        avail = available_columns(conn, abs_sql)

        # (underlying listener column or None for always-include, SELECT expr)
        candidates = [
            (None, 'time / 3600 AS "Time (hr)"'),
            ("listeners__replication_data__fork_coordinates",
             "listeners__replication_data__fork_coordinates AS fork_coordinates"),
            ("listeners__replication_data__number_of_oric",
             "listeners__replication_data__number_of_oric AS number_of_oric"),
            ("listeners__mass__cell_mass",
             "listeners__mass__cell_mass AS cell_mass"),
            ("listeners__mass__dry_mass",
             "listeners__mass__dry_mass AS dry_mass"),
            ("listeners__replication_data__critical_initiation_mass",
             "listeners__replication_data__critical_initiation_mass "
             "AS critical_initiation_mass"),
            ("listeners__replication_data__critical_mass_per_oric",
             "listeners__replication_data__critical_mass_per_oric "
             "AS critical_mass_per_oric"),
        ]
        data_columns = [expr for col, expr in candidates
                        if col is None or col in avail]

        plot_data = read_stacked_columns(abs_sql, data_columns, conn=conn)
        df = pl.DataFrame(plot_data)

        if "fork_coordinates" in df.columns:
            df = df.with_columns(
                pairs_of_forks=pl.col("fork_coordinates")
                .list.eval(~pl.element().is_nan())
                .list.sum()
                / 2
            )
        if "cell_mass" in df.columns and "critical_initiation_mass" in df.columns:
            df = df.with_columns(
                critical_mass_equivalents=(
                    pl.col("cell_mass") / pl.col("critical_initiation_mass")
                )
            )

        def create_fork_positions_plot():
            if "fork_coordinates" not in df.columns:
                return None
            fork_df = (
                df.select(["Time (hr)", "fork_coordinates"])
                .explode("fork_coordinates")
                .filter(~pl.col("fork_coordinates").is_nan())
                .rename({"fork_coordinates": "Position"})
            )
            if fork_df.height == 0:
                return None
            return (
                alt.Chart(fork_df)
                .mark_circle(size=5, opacity=0.7)
                .encode(
                    x=alt.X("Time (hr):Q", title="Time (hr)"),
                    y=alt.Y(
                        "Position:Q",
                        scale=alt.Scale(
                            domain=[-genome_length / 2, genome_length / 2]),
                        axis=alt.Axis(
                            values=[-genome_length / 2, 0, genome_length / 2],
                            labelExpr="datum.value == 0 ? 'oriC' : "
                            "(datum.value < 0 ? '-terC' : '+terC')",
                        ),
                        title="DNA polymerase position (nt)",
                    ),
                )
                .properties(
                    title="DNA Polymerase Positions", width=600, height=120)
            )

        def create_pairs_of_forks_plot():
            if "pairs_of_forks" not in df.columns:
                return None
            return (
                alt.Chart(df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("Time (hr):Q", title="Time (hr)"),
                    y=alt.Y("pairs_of_forks:Q",
                            scale=alt.Scale(domain=[0, 6]),
                            title="Pairs of forks"),
                )
                .properties(
                    title="Pairs of Replication Forks", width=600, height=100)
            )

        def create_critical_mass_plot():
            if "critical_mass_equivalents" not in df.columns:
                return None
            base_plot = (
                alt.Chart(df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("Time (hr):Q", title="Time (hr)"),
                    y=alt.Y("critical_mass_equivalents:Q",
                            title="Factors of critical initiation mass"),
                )
            )
            reference_data = pl.DataFrame(
                {"y": CRITICAL_N, "label": [f"N={n}" for n in CRITICAL_N]})
            reference_lines = (
                alt.Chart(reference_data)
                .mark_rule(strokeDash=[5, 5], color="gray", opacity=0.7)
                .encode(y="y:Q")
            )
            reference_labels = (
                alt.Chart(reference_data)
                .mark_text(align="left", dx=5, fontSize=10, color="gray")
                .encode(y="y:Q", text="label:N")
                .transform_calculate(x="0")
                .encode(x=alt.X("x:Q"))
            )
            return (base_plot + reference_lines + reference_labels).properties(
                title="Factors of Critical Initiation Mass", width=600, height=100)

        def create_mass_plot(column_name, title, y_title):
            if column_name not in df.columns:
                return None
            return (
                alt.Chart(df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("Time (hr):Q", title="Time (hr)"),
                    y=alt.Y(f"{column_name}:Q", title=y_title),
                )
                .properties(title=title, width=600, height=100)
            )

        plots = []
        for p in (
            create_fork_positions_plot(),
            create_pairs_of_forks_plot(),
            create_critical_mass_plot(),
            create_mass_plot("dry_mass", "Dry Mass", "Dry mass (fg)"),
            create_mass_plot("number_of_oric", "Number of oriC", "Number of oriC"),
            create_mass_plot("critical_mass_per_oric", "Critical Mass per oriC",
                             "Critical mass per oriC"),
        ):
            if p:
                plots.append(p)

        if plots:
            combined_plot = alt.vconcat(*plots).resolve_scale(x="shared")
        else:
            fallback_data = pl.DataFrame(
                {"x": [0], "y": [0], "text": ["No data available for plotting"]})
            combined_plot = (
                alt.Chart(fallback_data)
                .mark_text(fontSize=20, color="red")
                .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                        text="text:N")
                .properties(width=600, height=400,
                            title="Replication Data Visualization")
            )

        return {"view": chart_to_html(combined_plot, title="Replication")}
