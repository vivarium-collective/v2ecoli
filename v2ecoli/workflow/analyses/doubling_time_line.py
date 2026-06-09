"""Native port of vEcoli ``ecoli/analysis/multivariant/doubling_time_line.py``.

Line plot of doubling time (hr) vs generation for each lineage seed, with a
simulated per-generation average and an experimental reference (1/0.47 hr).
Returns an Altair HTML view.  Registered as ``"doubling_time_line"`` (scale:
``"multivariant"``).

v2ecoli notes: ``success_sql`` is not provided by v2ecoli analyses, so the
death-point overlay (cells absent from the success set) is omitted and all
cells are treated as successful.  Only meaningful for single-daughter lineage
sweeps.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import read_stacked_columns, chart_to_html


class DoublingTimeLine(Analysis):
    """Doubling time vs generation per lineage seed (multivariant)."""

    name = "doubling_time_line"
    scale = "multivariant"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data=None,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        import altair as alt

        doubling_time_sql = read_stacked_columns(
            history_sql, ["time"], order_results=False
        )
        doubling_times = conn.sql(f"""
            SELECT (max(time) - min(time)) / 3600 AS 'Doubling Time (hr)',
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({doubling_time_sql})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """).pl()

        selection = alt.selection_point(fields=["lineage_seed"], bind="legend")
        chart = (
            alt.Chart(doubling_times)
            .mark_line()
            .encode(
                x="generation",
                y="Doubling Time (hr)",
                color=alt.Color("lineage_seed", type="nominal"),
                tooltip=["Doubling Time (hr)", "lineage_seed"],
                opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
            )
            .add_params(selection)
            .interactive()
        )
        exp_avg = alt.Chart().mark_rule(strokeDash=[2, 2]).encode(y=alt.datum(1 / 0.47))
        sim_avg_df = doubling_times.group_by(
            "experiment_id", "variant", "generation"
        ).agg(pl.mean("Doubling Time (hr)"))
        sim_avg = (
            alt.Chart(sim_avg_df)
            .mark_line(strokeDash=[2, 2])
            .encode(x="generation", y="Doubling Time (hr)",
                    tooltip=["Doubling Time (hr)"])
        )
        chart = chart + exp_avg + sim_avg

        return {"view": chart_to_html(chart, title="Doubling time vs generation")}
