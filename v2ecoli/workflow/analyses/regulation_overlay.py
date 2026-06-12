"""Cross-variant regulation overlay (variant-comparison study, showcase-4).

The ppGpp-off headline.  Overlays ``listeners.growth_limits.ppgpp_conc`` and
``listeners.growth_limits.fraction_trna_charged`` vs. time as one COLORED LINE
PER VARIANT.  In the ppGpp-off variant the ppGpp trace should collapse toward
zero, visibly separating from baseline.  Registered as
``"regulation_overlay"`` (scale: ``"multivariant"``).

How it colors by variant
------------------------
Maps the ``variant`` hive-partition column to a display name via the runner's
``variant_metadata`` and encodes it as the Altair ``color`` channel.  Each
metric is averaged across cells at each ``global_time`` within a variant.

Note: ``fraction_trna_charged`` is emitted per-tRNA-family (a list); we reduce
it to a scalar mean charged fraction per row before aggregating.
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


class RegulationOverlay(Analysis):
    """ppGpp + tRNA-charged fraction vs time, one line per variant."""

    name = "regulation_overlay"
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

        # fraction_trna_charged may be a scalar or a per-family list; build a
        # scalar mean expression that works for either.
        ftc = "listeners__growth_limits__fraction_trna_charged"
        ppgpp = "listeners__growth_limits__ppgpp_conc"

        # (label, SQL value expr); rendered with safe metric_N aliases so
        # vl_convert PNG export does not mis-parse label-as-field-name.
        specs: list[tuple[str, str]] = []
        if ppgpp in avail:
            specs.append(("ppGpp conc", f"avg({ppgpp})"))
        if ftc in avail:
            # detect list vs scalar from the column type
            typ = conn.sql(
                f"SELECT typeof({ftc}) FROM ({base}) LIMIT 1").fetchone()
            is_list = bool(typ and typ[0] and "[" in str(typ[0]))
            charged_expr = (f"list_avg({ftc})" if is_list else ftc)
            specs.append(("Fraction tRNA charged", f"avg({charged_expr})"))

        if not specs:
            raise ValueError(
                "regulation_overlay: neither ppgpp_conc nor "
                "fraction_trna_charged present in history_sql")

        labels = [lbl for lbl, _ in specs]
        safe = [(lbl, expr, f"metric_{i}") for i, (lbl, expr) in enumerate(specs)]
        sel_parts = [f"{expr} AS {alias}" for _, expr, alias in safe]

        df = conn.sql(f"""
            SELECT variant, {vexpr}, time, {", ".join(sel_parts)}
            FROM ({base})
            GROUP BY variant, time
            ORDER BY variant, time
        """).pl()
        if df.is_empty():
            raise ValueError("regulation_overlay: no rows after aggregation")
        df = cast_decimals(df)

        order = variant_sort_order(df["variant"].to_list(), variant_metadata)
        panels = []
        for lbl, _, alias in safe:
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
            "view": chart_to_html(chart, title="Regulation overlay (per variant)"),
            "data": {"metrics": labels, "variants": order},
        }
