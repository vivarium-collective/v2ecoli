"""Cross-variant replication overlay (variant-comparison study, showcase-4).

The dnaA-2x headline: overlays ``listeners.replication_data.number_of_oric``
(and ``critical_mass_per_oric`` when emitted) vs. time as one COLORED LINE PER
VARIANT.  Registered as ``"replication_overlay"`` (scale: ``"multivariant"``).

How it colors by variant
------------------------
Maps the ``variant`` hive-partition column to a display name via the runner's
``variant_metadata`` and encodes it as the Altair ``color`` channel.  oriC count
is averaged across cells at each ``global_time`` within a variant, so a
dnaA-overexpression variant (more frequent initiation -> more oriC) separates
visibly from baseline.

v2ecoli note: ``listeners.replication_data.critical_mass_per_oric`` is not
emitted by every v2ecoli build (see the ``replication`` port note), so that
panel is included only when the column is present.
"""

from __future__ import annotations

from typing import Any

from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401
from v2ecoli.workflow.analyses._helpers import (
    aliased_history,
    available_columns,
    cast_decimals,
    chart_to_html,
    variant_display_expr,
    variant_sort_order,
)


class ReplicationOverlay(Analysis):
    """Number of oriC (+ critical mass/oriC) vs time, one line per variant."""

    name = "replication_overlay"
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

        base = aliased_history(history_sql)
        avail = available_columns(conn, history_sql)
        vexpr = variant_display_expr(variant_metadata)

        metrics = [
            ("listeners__replication_data__number_of_oric", "Number of oriC"),
            ("listeners__replication_data__critical_mass_per_oric",
             "Critical mass per oriC"),
        ]
        present = [(c, lbl) for c, lbl in metrics if c in avail]
        if not present:
            raise ValueError(
                "replication_overlay: number_of_oric not present in history_sql")

        # Safe SQL aliases (no spaces/parens) so vl_convert PNG export does not
        # mis-parse label-as-field-name; human label via axis title.
        safe = [(c, lbl, f"metric_{i}") for i, (c, lbl) in enumerate(present)]
        sel = ", ".join(f"avg({c}) AS {alias}" for c, _, alias in safe)
        df = conn.sql(f"""
            SELECT variant, {vexpr}, time, {sel}
            FROM ({base})
            GROUP BY variant, time
            ORDER BY variant, time
        """).pl()
        if df.is_empty():
            raise ValueError("replication_overlay: no rows after aggregation")
        df = cast_decimals(df)

        order = variant_sort_order(df["variant"].to_list(), variant_metadata)
        panels = []
        for _, lbl, alias in safe:
            panels.append(
                alt.Chart(df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("time:Q", title="Time (s, per-generation clock)"),
                    y=alt.Y(f"{alias}:Q", title=lbl),
                    color=alt.Color("variant_name:N", title="Variant", sort=order),
                    tooltip=["variant_name:N", "time:Q",
                             alt.Tooltip(f"{alias}:Q", title=lbl)],
                )
                .properties(title=lbl, width=560, height=200)
            )

        chart = alt.vconcat(*panels).resolve_scale(color="shared")
        return {
            "view": chart_to_html(chart, title="Replication overlay (per variant)"),
            "data": {"metrics": [lbl for _, lbl in present], "variants": order},
        }
