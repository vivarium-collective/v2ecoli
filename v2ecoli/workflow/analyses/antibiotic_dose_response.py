"""Antibiotic dose-response readout (variant-sweep-phenotype, antibiotic configs).

One COLORED LINE PER VARIANT (dose) for the three things that matter most for the
antibiotic phenotype configs (mecillinam-shape / sulfadiazine / antibiotic-cocktail),
read across the config's concentration sweep:

  1. Target engagement — the drug's mechanism-of-action markers (bulk counts):
       mecillinam[p] (uptake), EG10606-MONOMER[i] (free PBP2),
       mecillinam[p]-EG10606-MONOMER[i] (drug-target complex),
       mecillinam_hydrolyzed[p], and pterin_sulfadiazine[c] (sulfadiazine's
       folate dead-end adduct).
  2. Growth & viability — listeners.mass.cell_mass + instantaneous_growth_rate.
  3. Cell shape — listeners.peptidoglycan_shape.{lysed,resting_radius,
       resting_length} (emitted only when the study declares them as
       `observables` on the vecoli node; absent groups are skipped, not errored).

Registered as ``"antibiotic_dose_response"`` (scale ``"multivariant"``). Only
meaningful on a multi-variant dose sweep. Each metric is averaged over cells
(seeds/agents/generations) at each (variant, time), matching growth_overlay, so
one trace per dose. Missing columns are dropped via ``available_columns`` so the
card renders whatever the run emitted (e.g. bulk-only until the shape observables
are declared).
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

# (flattened history column, axis label, readout group). Ordered target -> growth
# -> shape so the rendered panels read as MoA -> phenotype -> morphology.
_METRICS = [
    ("bulk__mecillinam[p]", "Mecillinam uptake, mecillinam[p] (counts)", "Target engagement"),
    ("bulk__EG10606-MONOMER[i]", "Free PBP2, EG10606-MONOMER[i] (counts)", "Target engagement"),
    ("bulk__mecillinam[p]-EG10606-MONOMER[i]", "Mec-PBP2 complex (counts)", "Target engagement"),
    ("bulk__mecillinam_hydrolyzed[p]", "Hydrolyzed mecillinam (counts)", "Target engagement"),
    ("bulk__pterin_sulfadiazine[c]", "Sulfadiazine folate adduct, pterin_sulfadiazine[c] (counts)", "Target engagement"),
    ("listeners__mass__cell_mass", "Cell mass (fg)", "Growth & viability"),
    ("listeners__mass__instantaneous_growth_rate", "Instantaneous growth rate (1/s)", "Growth & viability"),
    ("listeners__peptidoglycan_shape__lysed", "Lysed (0/1)", "Cell shape"),
    ("listeners__peptidoglycan_shape__resting_radius", "Resting radius (um)", "Cell shape"),
    ("listeners__peptidoglycan_shape__resting_length", "Resting length (um)", "Cell shape"),
]


class AntibioticDoseResponse(Analysis):
    """Target engagement + growth + shape vs time, one line per dose (variant)."""

    name = "antibiotic_dose_response"
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

        present = [(c, lbl, grp) for c, lbl, grp in _METRICS if c in avail]
        if not present:
            raise ValueError(
                "antibiotic_dose_response: none of the antibiotic readout columns "
                "(target-engagement bulk, mass, or peptidoglycan_shape) are present "
                "in history_sql — declare them via observable_bulk_ids/observables.")

        # Safe SQL aliases (metric_0, ...): the bulk ids carry '[' / '(' / '-',
        # which Vega-Lite shorthand and vl_convert PNG export mis-parse as
        # access-path syntax. Human labels are applied via the axis title.
        safe = [(c, lbl, grp, f"metric_{i}") for i, (c, lbl, grp) in enumerate(present)]
        sel = ", ".join(f'avg("{c}") AS {alias}' for c, _, _, alias in safe)
        df = conn.sql(f"""
            SELECT variant, {vexpr}, time, {sel}
            FROM ({base})
            GROUP BY variant, time
            ORDER BY variant, time
        """).pl()
        if df.is_empty():
            raise ValueError("antibiotic_dose_response: no rows after aggregation")
        df = cast_decimals(df)

        order = variant_sort_order(df["variant"].to_list(), variant_metadata)

        # One panel per present metric, grouped by readout dimension via title.
        panels = []
        for _, lbl, grp, alias in safe:
            panels.append(
                alt.Chart(df)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("time:Q", title="Time (s, per-generation clock)"),
                    y=alt.Y(f"{alias}:Q", title=lbl),
                    color=alt.Color("variant_name:N", title="Dose (variant)", sort=order),
                    tooltip=["variant_name:N", "time:Q", alt.Tooltip(f"{alias}:Q", title=lbl)],
                )
                .properties(title=f"{grp} — {lbl}", width=560, height=180)
            )

        chart = alt.vconcat(*panels).resolve_scale(color="shared")
        groups_present = sorted({grp for _, _, grp, _ in safe})
        return {
            "view": chart_to_html(chart, title="Antibiotic dose-response (per dose)"),
            "data": {
                "n_variants": int(df["variant"].n_unique()),
                "variants": order,
                "readout_groups": groups_present,
                "metrics": [c for c, _, _, _ in safe],
            },
        }
