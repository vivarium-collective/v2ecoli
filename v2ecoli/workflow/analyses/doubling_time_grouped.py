"""Cross-variant doubling-time comparison (variant-comparison study, showcase-4).

Grouped box/bar of per-cell doubling time, one group per variant.  Unlike
``doubling_time_hist`` (which collapses the variant axis into a single
histogram), this contrasts the doubling-time DISTRIBUTION across baseline + each
variant side by side.  Registered as ``"doubling_time_grouped"`` (scale:
``"multivariant"``).

Cap-immune measure (matches PR #184)
------------------------------------
A simple ``max(time) - min(time)`` doubling time is biased by the simulation's
run cap: a cell that never divides reports the cap, not its true cycle time.
Following PR #184 we instead derive a **cap-immune** doubling time from the
instantaneous growth rate, ``T_double = ln(2) / mean(mu)``, where ``mu`` is
``listeners.mass.instantaneous_growth_rate`` averaged over the cell's
timeseries (units of 1/s -> minutes).  This is independent of when (or whether)
division was reached.  When the growth-rate listener is unavailable we fall
back to the elapsed-time measure (and say so in the title).

Colors by variant via the ``variant`` hive-partition column mapped to display
names through the runner's ``variant_metadata``.
"""

from __future__ import annotations

import math
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

LN2 = math.log(2.0)


class DoublingTimeGrouped(Analysis):
    """Grouped doubling-time distribution per variant (cap-immune ln2/mu)."""

    name = "doubling_time_grouped"
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
        gr_col = "listeners__mass__instantaneous_growth_rate"
        cap_immune = gr_col in avail

        if cap_immune:
            # Per-cell mean growth rate (1/s) -> doubling time in minutes.
            # Exclude non-positive growth-rate rows from the mean so dead/edge
            # rows don't bias mu toward 0 (which would inflate T_double).
            df = conn.sql(f"""
                WITH per_cell AS (
                    SELECT variant, lineage_seed, generation, agent_id,
                           avg({gr_col}) FILTER (WHERE {gr_col} > 0) AS mu
                    FROM ({base})
                    GROUP BY variant, lineage_seed, generation, agent_id
                )
                SELECT variant, {vexpr},
                       {LN2} / mu / 60 AS doubling_time_min
                FROM per_cell
                WHERE mu IS NOT NULL AND mu > 0
            """).pl()
            measure = "cap-immune ln(2)/mean(mu)"
        else:
            df = conn.sql(f"""
                WITH per_cell AS (
                    SELECT variant, lineage_seed, generation, agent_id,
                           (max(time) - min(time)) / 60 AS dt
                    FROM ({base})
                    GROUP BY variant, lineage_seed, generation, agent_id
                )
                SELECT variant, {vexpr}, dt AS doubling_time_min
                FROM per_cell
                WHERE dt > 0
            """).pl()
            measure = "elapsed max-min time (growth-rate listener unavailable)"

        if df.is_empty():
            raise ValueError("doubling_time_grouped: no doubling times computed")
        df = cast_decimals(df)

        order = variant_sort_order(df["variant"].to_list(), variant_metadata)
        n_per_variant = df.group_by("variant").len().height

        # Box plot per variant, with the per-cell points overlaid (jittered via
        # the boxplot's own outlier marks). With few cells per variant a bar of
        # the mean is clearer, so render both: box (distribution) is primary.
        box = (
            alt.Chart(df)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("variant_name:N", title="Variant", sort=order),
                y=alt.Y("doubling_time_min:Q", title="Doubling time (min)"),
                color=alt.Color("variant_name:N", title="Variant", sort=order),
            )
            .properties(
                title=f"Doubling time per variant — {measure}",
                width=max(360, 90 * max(1, n_per_variant)),
                height=320,
            )
        )

        means = df.group_by("variant_name").agg(
            pl.col("doubling_time_min").mean().alias("pl_mean")
        )
        per_variant_mean = {
            row["variant_name"]: float(row["pl_mean"]) for row in means.iter_rows(named=True)
        }

        return {
            "view": chart_to_html(box, title="Doubling time per variant"),
            "data": {
                "measure": measure,
                "per_variant_mean_min": per_variant_mean,
            },
        }
