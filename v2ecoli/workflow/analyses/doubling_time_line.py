"""Native port of vEcoli ``ecoli/analysis/multivariant/doubling_time_line.py``,
hardened with a cap-immune instantaneous doubling-time trace.

The original vEcoli figure plots one doubling-time point per generation, derived
from the ``(max_time - min_time)`` span of that generation's emitted rows.  In
v2ecoli that span is artifact-prone: the **final generation of a lineage is
truncated at the 6500-step emit cap before it divides**, so its raw span reads
short (~10-19 min) and the per-generation line *dips* at the last generation —
an artifact, not a biological decline.

This port therefore renders a layered figure:

  * **Primary: a cap-immune instantaneous doubling-time trace.**  From the
    per-timestep specific growth rate ``listeners.mass.instantaneous_growth_rate``
    (mu, 1/s) the instantaneous doubling time is ``t2 = ln(2)/mu``.  Plotted
    against absolute lineage time, this yields *many* samples (one per emitted
    step, not one per generation) and never references a division event or the
    emit cap, so it is immune to the truncation artifact.  This is the same
    measure the runnable ``doubling-time-in-band`` test uses (~52 min).

  * **Per-generation division-span markers**, kept for continuity with the
    vEcoli figure, but with the *truncated final generation of each lineage
    excluded* (it never divided, so its span is meaningless).  A note records
    how many generations were dropped.

  * The experimental reference (1/0.47 hr).

Returns an Altair HTML view.  Registered as ``"doubling_time_line"`` (scale:
``"multivariant"``).

v2ecoli notes: ``success_sql`` is not provided by v2ecoli analyses, so the
death-point overlay (cells absent from the success set) is omitted and all
cells are treated as successful.  Only meaningful for single-daughter lineage
sweeps.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis
from v2ecoli.workflow.analyses._helpers import read_stacked_columns, chart_to_html

# ln(2) expressed per-hour from a per-second growth rate: t2[hr] = (ln2/3600)/mu.
LN2_PER_HOUR = math.log(2) / 3600.0
EMIT_CAP_STEP_TIME = 6500.0  # absolute global_time (s) of the final emitted step


class DoublingTimeLine(Analysis):
    """Doubling time vs time/generation per lineage seed (multivariant).

    Renders a cap-immune instantaneous doubling-time trace (many samples) plus
    per-generation division-span markers with the truncated final generation
    excluded.
    """

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

        # ---- 1. cap-immune instantaneous trace: t2 = ln(2)/mu ----------------
        # One sample per emitted timestep over the whole lineage. mu>0 guards
        # against the (rare) zero/negative growth rows that would blow up 1/mu.
        inst_sql = read_stacked_columns(
            history_sql,
            ["listeners__mass__instantaneous_growth_rate AS mu"],
            order_results=False,
        )
        inst = conn.sql(f"""
            SELECT time / 3600.0 AS 'Time (hr)',
                {LN2_PER_HOUR} / mu AS 'Doubling Time (hr)',
                lineage_seed, generation
            FROM ({inst_sql})
            WHERE mu IS NOT NULL AND mu > 0
            ORDER BY lineage_seed, generation, time
        """).pl()

        if inst.height == 0:
            raise ValueError(
                "no positive instantaneous_growth_rate samples found in "
                "history_sql; cannot build the cap-immune doubling-time trace"
            )

        n_samples = inst.height
        mean_t2_min = float(
            (inst["Doubling Time (hr)"] * 60).mean()  # type: ignore[operator]
        )

        inst_sel = alt.selection_point(fields=["lineage_seed"], bind="legend")
        trace = (
            alt.Chart(inst)
            .mark_line(opacity=0.85)
            .encode(
                x=alt.X("Time (hr)", title="Lineage time (hr)"),
                y=alt.Y("Doubling Time (hr)",
                        scale=alt.Scale(domainMin=0),
                        title="Doubling Time (hr)"),
                color=alt.Color("lineage_seed", type="nominal"),
                tooltip=["Doubling Time (hr)", "Time (hr)", "lineage_seed",
                         "generation"],
                opacity=alt.when(inst_sel).then(alt.value(0.9))
                .otherwise(alt.value(0.2)),
            )
            .add_params(inst_sel)
        )

        # mean instantaneous t2 reference (the cap-immune headline)
        mean_rule = (
            alt.Chart(pl.DataFrame({"y": [mean_t2_min / 60.0]}))
            .mark_rule(color="#444", strokeDash=[4, 3])
            .encode(y="y:Q", tooltip=[alt.Tooltip(
                "y:Q", title=f"mean t2 = {mean_t2_min:.1f} min")])
        )

        # ---- 2. per-generation division-span markers (truncated gen dropped) -
        span_sql = read_stacked_columns(history_sql, ["time"], order_results=False)
        spans = conn.sql(f"""
            SELECT (max(time) - min(time)) / 3600 AS 'Doubling Time (hr)',
                experiment_id, variant, lineage_seed, generation, agent_id,
                max(time) AS 'last_time'
            FROM ({span_sql})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """).pl()

        # A generation is "truncated" (never divided) iff it is the LAST
        # generation of its lineage AND its final emitted row lands at the emit
        # cap. Drop those from the division-span markers — their span is the
        # time-to-cap, not a doubling time.
        last_gen = spans.group_by("lineage_seed").agg(
            pl.max("generation").alias("_max_gen"))
        spans = spans.join(last_gen, on="lineage_seed", how="left")
        truncated_mask = (
            (pl.col("generation") == pl.col("_max_gen"))
            & (pl.col("last_time") >= EMIT_CAP_STEP_TIME)
        )
        n_truncated = int(spans.filter(truncated_mask).height)
        divided = spans.filter(~truncated_mask)

        markers = (
            alt.Chart(divided)
            .mark_point(filled=True, size=70)
            .encode(
                x=alt.X("generation:O", title="generation"),
                y="Doubling Time (hr)",
                color=alt.Color("lineage_seed", type="nominal"),
                shape=alt.value("circle"),
                tooltip=["Doubling Time (hr)", "lineage_seed", "generation"],
            )
        )

        exp_avg = alt.Chart().mark_rule(strokeDash=[2, 2], color="green").encode(
            y=alt.datum(1 / 0.47))

        title = (
            f"Instantaneous doubling time over the lineage "
            f"(mean t2 = {mean_t2_min:.0f} min, n={n_samples} samples)"
        )
        if n_truncated:
            title += f"; {n_truncated} truncated final-gen marker(s) excluded"

        # Trace + mean + experimental reference share the time x-axis; the
        # per-generation markers use a generation x-axis, so render them as an
        # independent layer concatenated below.
        time_layer = (trace + mean_rule + exp_avg).properties(
            width=560, height=300,
            title="Cap-immune instantaneous doubling-time trace")
        gen_layer = (markers + exp_avg).properties(
            width=560, height=180,
            title="Per-generation division times (truncated final gen excluded)")
        chart = alt.vconcat(time_layer, gen_layer).properties(
            title=title).resolve_scale(color="shared")

        return {"view": chart_to_html(chart, title="Doubling time over the lineage")}
