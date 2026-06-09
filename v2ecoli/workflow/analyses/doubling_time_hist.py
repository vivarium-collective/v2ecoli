"""Native port of vEcoli ``ecoli/analysis/multivariant/doubling_time_hist.py``.

Histogram of per-cell doubling times (max−min time per cell, in minutes) with
the average marked.  Returns an Altair HTML view.  Registered as
``"doubling_time_hist"`` (scale: ``"multivariant"``).

v2ecoli notes: ``skip_n_gens`` defaults to 0 here (vEcoli used 8 to drop
initialization bias, but typical v2ecoli sweeps have few generations); set it
via the analysis params if desired.  ``success_sql`` is not used by v2ecoli
analyses, so all cells are included.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import read_stacked_columns, skip_n_gens, chart_to_html


class DoublingTimeHist(Analysis):
    """Histogram of doubling times across cells (multivariant)."""

    name = "doubling_time_hist"
    scale = "multivariant"
    config_schema = {"skip_n_gens": "integer"}

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

        params = dict(variant_metadata or {})
        skip = int(params.get("skip_n_gens", 0))

        doubling_time_sql = cast(
            str,
            read_stacked_columns(history_sql, ["time"], order_results=False),
        )
        doubling_time_sql = skip_n_gens(doubling_time_sql, skip)
        doubling_times = conn.sql(f"""
            SELECT (max(time) - min(time)) / 60 AS 'Doubling Time (min)',
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({doubling_time_sql})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """).pl()

        avg_doubling_time = doubling_times["Doubling Time (min)"].mean()
        if avg_doubling_time is None:
            raise ValueError("No doubling times found in the data.")
        avg_rounded = round(cast(float, avg_doubling_time), 2)

        hist = (
            alt.Chart(doubling_times)
            .mark_bar()
            .encode(
                x=alt.X("Doubling Time (min)", bin=alt.Bin(maxbins=40),
                        axis=alt.Axis(title="Doubling Time (min)", labelFlush=False)),
                y=alt.Y("count()", axis=alt.Axis(title="Frequency")),
                tooltip=[alt.Tooltip("Doubling Time (min)", bin=alt.Bin(maxbins=40)),
                         "count()"],
            )
        )
        avg_df = pl.DataFrame({"avg": [avg_doubling_time]})
        rule = (
            alt.Chart(avg_df)
            .mark_rule(color="red", strokeDash=[5, 5], size=2)
            .encode(x=alt.X("avg:Q"),
                    tooltip=[alt.Tooltip("avg", title=f"Average: {avg_rounded} min")])
        )
        text = (
            alt.Chart(avg_df)
            .mark_text(align="left", baseline="middle", dx=7, dy=-20, color="red")
            .encode(x=alt.X("avg:Q"), text=alt.value(f"{avg_rounded} min"),
                    tooltip=[alt.Tooltip("avg", title=f"Average: {avg_rounded} min")])
        )
        chart = (hist + rule + text).properties(
            title="Distribution of Doubling Times").interactive()

        return {"view": chart_to_html(chart, title="Doubling time histogram")}
