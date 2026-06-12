"""Cross-variant mass-fraction comparison (variant-comparison study, showcase-4).

Grouped bars of the time-averaged mass fractions (protein / rRNA / tRNA / DNA /
small-molecule, each over dry mass), grouped BY VARIANT.  Contrasts proteome /
RNA allocation across baseline + each variant on one chart.  Registered as
``"mass_fraction_grouped"`` (scale: ``"multivariant"``).

Colors/groups by variant via the ``variant`` hive-partition column mapped to a
display name through the runner's ``variant_metadata``.  Each mass-fraction
category is an x-axis group; the bars within a group are colored per variant.
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

# (mass column, fraction display label)
_FRACTIONS = [
    ("listeners__mass__protein_mass", "protein"),
    ("listeners__mass__rRna_mass", "rRNA"),
    ("listeners__mass__tRna_mass", "tRNA"),
    ("listeners__mass__dna_mass", "DNA"),
    ("listeners__mass__smallMolecule_mass", "small-mol"),
]


class MassFractionGrouped(Analysis):
    """Time-averaged mass fractions, grouped bars per variant."""

    name = "mass_fraction_grouped"
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
        if "listeners__mass__dry_mass" not in avail:
            raise ValueError("mass_fraction_grouped: dry_mass not in history_sql")
        present = [(c, lbl) for c, lbl in _FRACTIONS if c in avail]
        if not present:
            raise ValueError("mass_fraction_grouped: no mass-component columns present")

        vexpr = variant_display_expr(variant_metadata)
        # Per (variant, time) fraction, then mean across time within a variant.
        # We compute fraction at the row level (dry_mass > 0) and average.
        frac_exprs = ", ".join(
            f"avg({c} / listeners__mass__dry_mass) "
            f"FILTER (WHERE listeners__mass__dry_mass > 0) AS \"{lbl}\""
            for c, lbl in present
        )
        wide = conn.sql(f"""
            SELECT variant, {vexpr}, {frac_exprs}
            FROM ({base})
            GROUP BY variant
            ORDER BY variant
        """).pl()

        if wide.is_empty():
            raise ValueError("mass_fraction_grouped: no rows after aggregation")
        wide = cast_decimals(wide)

        order = variant_sort_order(wide["variant"].to_list(), variant_metadata)
        labels = [lbl for _, lbl in present]
        long = wide.unpivot(
            on=labels,
            index=["variant", "variant_name"],
            variable_name="component",
            value_name="fraction",
        )

        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("component:N", title="Mass component",
                        sort=labels, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("variant_name:N", sort=order),
                y=alt.Y("fraction:Q", title="Mean fraction of dry mass"),
                color=alt.Color("variant_name:N", title="Variant", sort=order),
                tooltip=["variant_name:N", "component:N",
                         alt.Tooltip("fraction:Q", format=".3f")],
            )
            .properties(
                title="Mass fractions per variant (mean over time)",
                width=560, height=320,
            )
        )

        return {
            "view": chart_to_html(chart, title="Mass fractions per variant"),
            "data": {"components": labels, "variants": order},
        }
