"""Cross-variant growth overlay (variant-comparison study, showcase-4).

Overlays ``listeners.mass.cell_mass`` and
``listeners.mass.instantaneous_growth_rate`` vs. time as one COLORED LINE PER
VARIANT (not one panel per variant).  Registered as ``"growth_overlay"``
(scale: ``"multivariant"``).

How it colors by variant
------------------------
The multivariant ``history_sql`` spans every variant.  We map the ``variant``
hive-partition column to a display name via :func:`variant_display_expr`
(using the runner's ``variant_metadata`` int->name map) and encode that name as
the Altair ``color`` channel, so baseline + each variant become separate
colored lines on a shared axis.

Time axis: ``global_time`` resets per generation in v2ecoli, so for a clean
overlay across variants we average each metric across cells at each
``global_time`` within a variant (mean over seeds/agents at that time).  This
keeps a single trace per variant even when multiple seeds/generations are
present.
"""

from __future__ import annotations

from typing import Any

import polars as pl
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


class GrowthOverlay(Analysis):
    """Cell mass + instantaneous growth rate vs time, one line per variant."""

    name = "growth_overlay"
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
            ("listeners__mass__cell_mass", "Cell mass (fg)"),
            ("listeners__mass__instantaneous_growth_rate",
             "Instantaneous growth rate (1/s)"),
        ]
        present = [(c, lbl) for c, lbl in metrics if c in avail]
        if not present:
            raise ValueError(
                "growth_overlay: neither cell_mass nor instantaneous_growth_rate "
                "present in history_sql")

        # Mean over cells (seeds/agents/gens) at each (variant, time).
        # Use safe SQL aliases (metric_0, ...) so the field names carry no
        # spaces/parens — Vega-Lite shorthand treats "[" / "(" as access-path
        # syntax and vl_convert PNG export fails on label-as-field-name. The
        # human label is applied via the axis title instead.
        safe = [(c, lbl, f"metric_{i}") for i, (c, lbl) in enumerate(present)]
        sel_metrics = ", ".join(f"avg({c}) AS {alias}" for c, _, alias in safe)
        df = conn.sql(f"""
            SELECT variant, {vexpr}, time, {sel_metrics}
            FROM ({base})
            GROUP BY variant, time
            ORDER BY variant, time
        """).pl()

        if df.is_empty():
            raise ValueError("growth_overlay: no rows after aggregation")
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
                    color=alt.Color("variant_name:N", title="Variant",
                                    sort=order),
                    tooltip=["variant_name:N", "time:Q",
                             alt.Tooltip(f"{alias}:Q", title=lbl)],
                )
                .properties(title=lbl, width=560, height=200)
            )

        chart = alt.vconcat(*panels).resolve_scale(color="shared")
        return {
            "view": chart_to_html(chart, title="Growth overlay (per variant)"),
            "data": {"n_variants": int(df["variant"].n_unique()),
                     "variants": order},
        }
