"""Native port of vEcoli ``ecoli/analysis/multivariant/doubling_time_hist.py``,
hardened with a cap-immune instantaneous doubling-time distribution.

The original vEcoli figure histograms one doubling time per cell, derived from
the ``(max_time - min_time)`` span of that cell's emitted rows.  In v2ecoli that
span is artifact-prone: the **final generation of every lineage is truncated at
the 6500-step emit cap before it divides**, so its span reads short (~10-19 min)
and seeds a spurious low-end bin in the histogram.

This port instead histograms the **cap-immune instantaneous doubling time**
``t2 = ln(2)/mu`` from the per-timestep specific growth rate
``listeners.mass.instantaneous_growth_rate`` (mu, 1/s) over the whole ensemble.
This gives *many* samples (one per emitted step) and never references a division
event or the emit cap, so the distribution reflects the true growth rate, not
the truncation artifact.  This is the same measure the runnable
``doubling-time-in-band`` test uses (~52 min).

Returns an Altair HTML view.  Registered as ``"doubling_time_hist"`` (scale:
``"multivariant"``).

v2ecoli notes: ``skip_n_gens`` defaults to 0 here (vEcoli used 8 to drop
initialization bias, but typical v2ecoli sweeps have few generations); set it
via the analysis params if desired.  ``success_sql`` is not used by v2ecoli
analyses, so all cells are included.
"""

from __future__ import annotations

import math
from typing import Any, cast

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import read_stacked_columns, skip_n_gens, chart_to_html

# t2[min] = (ln2/60)/mu  for mu in 1/s.
LN2_PER_MIN = math.log(2) / 60.0


class DoublingTimeHist(Analysis):
    """Distribution of cap-immune instantaneous doubling times (multivariant)."""

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

        inst_sql = cast(
            str,
            read_stacked_columns(
                history_sql,
                ["listeners__mass__instantaneous_growth_rate AS mu"],
                order_results=False,
            ),
        )
        inst_sql = skip_n_gens(inst_sql, skip)
        doubling_times = conn.sql(f"""
            SELECT {LN2_PER_MIN} / mu AS 'Doubling Time (min)',
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({inst_sql})
            WHERE mu IS NOT NULL AND mu > 0
        """).pl()

        if doubling_times.height == 0:
            raise ValueError(
                "no positive instantaneous_growth_rate samples found in "
                "history_sql; cannot build the cap-immune doubling-time "
                "distribution."
            )

        avg_doubling_time = doubling_times["Doubling Time (min)"].mean()
        if avg_doubling_time is None:
            raise ValueError("No doubling times found in the data.")
        avg_rounded = round(cast(float, avg_doubling_time), 2)
        n_samples = doubling_times.height

        hist = (
            alt.Chart(doubling_times)
            .mark_bar()
            .encode(
                x=alt.X("Doubling Time (min)", bin=alt.Bin(maxbins=40),
                        axis=alt.Axis(title="Instantaneous doubling time (min)",
                                      labelFlush=False)),
                y=alt.Y("count()", axis=alt.Axis(title="Frequency (timesteps)")),
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
            title=(f"Cap-immune instantaneous doubling-time distribution "
                   f"(n={n_samples} timesteps, mean {avg_rounded} min)")
        ).interactive()

        return {"view": chart_to_html(chart, title="Doubling time histogram")}
