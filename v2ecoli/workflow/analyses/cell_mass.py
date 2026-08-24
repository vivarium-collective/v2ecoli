"""Native port of vEcoli ``ecoli/analysis/multivariant/cell_mass.py``.

Per-variant absolute and normalized dry-mass-over-time line plots, coloured by
generation, with a doubling reference line at 2×.  Returns an Altair HTML view.
Registered as ``"cell_mass"`` (scale: ``"multivariant"``).

v2ecoli notes: variant display names come from ``variant_metadata`` (falling
back to ``"Variant {i}"``); ``add_selection`` is updated to ``add_params``
(Altair ≥5 API).
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import chart_to_html


class CellMass(Analysis):
    """Per-variant dry-mass-over-time plots (multivariant)."""

    name = "cell_mass"
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

        required_columns = [
            "global_time AS time",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
            "listeners__mass__dry_mass",
            "listeners__mass__dry_mass_fold_change",
        ]
        sql = f"""
        SELECT {", ".join(required_columns)}
        FROM ({history_sql})
        ORDER BY variant, lineage_seed, generation, time
        """
        df = conn.sql(sql).pl()
        df = df.with_columns([(pl.col("time") / 60).alias("time_min")])

        # Build a per-cell line key.  Each drawn line must be exactly ONE cell's
        # monotonic growth ramp.  In a clean vEcoli run the partition tuple
        # ``(variant, lineage_seed, generation, agent_id)`` already uniquely
        # identifies a cell, so that tuple is the natural detail key.  v2ecoli
        # runs, however, can emit MORE than one cell into a single
        # ``(generation, agent_id)`` partition (an extra division whose daughter
        # was not re-labelled with a new generation/agent_id) — the showcase-2
        # baseline does exactly this, packing a gen-3 daughter into the gen-2
        # partition.  Joining those two cells into one line draws a spurious
        # vertical sawtooth (mass halves at the hidden division) which reads as
        # "extra generations".  Detect the division robustly from the
        # ``dry_mass_fold_change`` reset (it returns to ~1.0 at every birth) and
        # bump a sub-cell counter, so each real cell becomes its own line
        # regardless of how the partitions were labelled.  This also degrades
        # correctly for >2 true generations.
        df = df.with_columns(
            [
                # A division is a sharp drop in fold-change back toward 1.0
                # (mass roughly halves).  Flag any row whose fold-change fell by
                # more than 25% from the previous row within the same partition.
                (
                    pl.col("listeners__mass__dry_mass_fold_change")
                    < 0.75
                    * pl.col("listeners__mass__dry_mass_fold_change").shift(1)
                )
                .fill_null(False)
                .over(["variant", "lineage_seed", "generation", "agent_id"])
                .alias("_is_birth")
            ]
        )
        df = df.with_columns(
            [
                pl.col("_is_birth")
                .cast(pl.Int64)
                .cum_sum()
                .over(["variant", "lineage_seed", "generation", "agent_id"])
                .alias("_subcell")
            ]
        )
        df = df.with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("lineage_seed").cast(pl.Utf8),
                        pl.col("generation").cast(pl.Utf8),
                        pl.col("agent_id").cast(pl.Utf8),
                        pl.col("_subcell").cast(pl.Utf8),
                    ],
                    separator="_",
                ).alias("cell_id")
            ]
        )

        variant_names = dict(variant_metadata or {})
        variants = df.select("variant").unique().to_series().to_list()

        plots = []
        for variant in variants:
            variant_df = df.filter(pl.col("variant") == variant).to_pandas()
            variant_name = variant_names.get(variant, f"Variant {variant}")

            base = alt.Chart(variant_df).add_params(
                alt.selection_interval(bind="scales")
            )
            base_x = alt.X("time_min:Q", title="Time (min)",
                           scale=alt.Scale(nice=False))
            base_color = alt.Color(
                "generation:N",
                legend=alt.Legend(title="Generation"),
                scale=alt.Scale(scheme="category10"),
            )
            tooltip_fields = ["time_min:Q", "generation:N"]

            mass_plot = (
                base.mark_line(strokeWidth=2.5)
                .encode(
                    x=base_x, color=base_color,
                    tooltip=tooltip_fields + ["listeners__mass__dry_mass:Q"],
                    detail="cell_id:N",
                    y=alt.Y("listeners__mass__dry_mass:Q", title="Dry Mass (fg)",
                            scale=alt.Scale(nice=False)),
                )
                .properties(width=400, height=200,
                            title=f"{variant_name} - Absolute Dry Mass")
            )
            norm_mass_plot = (
                base.mark_line(strokeWidth=2.5)
                .encode(
                    x=base_x, color=base_color,
                    tooltip=tooltip_fields
                    + ["listeners__mass__dry_mass_fold_change:Q"],
                    detail="cell_id:N",
                    y=alt.Y("listeners__mass__dry_mass_fold_change:Q",
                            title="Normalized Dry Mass",
                            scale=alt.Scale(nice=False)),
                )
                .properties(width=400, height=200,
                            title=f"{variant_name} - Normalized Dry Mass")
            )
            reference_line = (
                alt.Chart(pd.DataFrame({"y": [2]}))
                .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=1)
                .encode(y="y:Q")
            )
            norm_mass_plot = norm_mass_plot + reference_line

            variant_combined = (
                alt.hconcat(mass_plot, norm_mass_plot)
                .resolve_scale(x="shared")
                .properties(title=f"{variant_name} Cell Mass Analysis")
            )
            plots.append(variant_combined)

        final_plot = plots[0] if len(plots) == 1 else alt.vconcat(*plots)
        final_plot = final_plot.resolve_scale(
            x="independent", y="independent"
        ).properties(title="Multi-Variant Cell Mass Analysis")

        return {"view": chart_to_html(final_plot, title="Cell mass")}
