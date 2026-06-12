"""Cross-variant proteome delta (variant-comparison study, showcase-4).

For each non-baseline variant, scatters its per-monomer average count against
the BASELINE (variant 0) average count (log10-log10), so points off the parity
line are the proteins a variant up-/down-regulates.  Adds a small table of each
variant's Schmidt-2015 and Wisniewski-2014 Pearson r (proteome-quality vs the
literature), so a variant that breaks the proteome shows up as a dropped r.
Registered as ``"proteome_delta"`` (scale: ``"multivariant"``).

Uses the corrected SET ``listeners.monomer_counts`` (PR #185): each row already
holds the cell's instantaneous monomer vector (not an accumulating sum), so a
plain per-position average over rows is the mean proteome.

Colors by variant: each non-baseline variant is one colored series in the
ratio scatter (faceted/colored by variant_name); baseline (variant 0) is the
reference axis, not a series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
from scipy.stats import pearsonr

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401
from v2ecoli.workflow.analyses._helpers import (
    aliased_history,
    available_columns,
    chart_to_html,
    variant_label_map,
)
from v2ecoli.workflow.analyses.protein_counts_validation import (
    _get_simulated_validation_counts,
)


class ProteomeDelta(Analysis):
    """Per-variant monomer-count ratio-vs-baseline + validation Pearson r."""

    name = "proteome_delta"
    scale = "multivariant"

    def _avg_counts_by_variant(self, conn, base) -> dict[int, np.ndarray]:
        """Per-variant mean monomer-count vector (1-D, indexed by position)."""
        df = conn.sql(f"""
            WITH flat AS (
                SELECT variant,
                       unnest(listeners__monomer_counts) AS cnt,
                       generate_subscripts(listeners__monomer_counts, 1) AS idx
                FROM ({base})
                WHERE len(listeners__monomer_counts) > 0
            )
            SELECT variant, idx, avg(cnt) AS avg_cnt
            FROM flat GROUP BY variant, idx ORDER BY variant, idx
        """).pl()
        out: dict[int, np.ndarray] = {}
        for v in df["variant"].unique().to_list():
            sub = df.filter(pl.col("variant") == v).sort("idx")
            out[int(v)] = sub["avg_cnt"].to_numpy()
        return out

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        validation_data=None,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        import altair as alt

        base = aliased_history(history_sql)
        avail = available_columns(conn, history_sql)
        if "listeners__monomer_counts" not in avail:
            raise ValueError("proteome_delta: listeners__monomer_counts not emitted")

        labels = variant_label_map(variant_metadata)
        by_variant = self._avg_counts_by_variant(conn, base)
        if not by_variant:
            raise ValueError("proteome_delta: no monomer_counts rows found")

        variants = sorted(by_variant)
        baseline_v = 0 if 0 in by_variant else variants[0]
        baseline = by_variant[baseline_v]

        sim_monomer_ids = list(sim_data.process.translation.monomer_data["id"])
        if len(baseline) != len(sim_monomer_ids):
            raise ValueError(
                f"proteome_delta: monomer vector width {len(baseline)} != "
                f"sim monomer_ids {len(sim_monomer_ids)}; sim_data does not pair "
                "with this parquet (use out/workflow/simData.cPickle)")

        def vname(v: int) -> str:
            return labels.get(v, f"variant {v}")

        # --- ratio-vs-baseline scatter (log10 baseline vs log10 variant) ------
        rows = []
        for v in variants:
            if v == baseline_v:
                continue
            cur = by_variant[v]
            for b, c in zip(baseline, cur):
                rows.append({
                    "variant_name": vname(v),
                    "baseline_log10": float(np.log10(b + 1)),
                    "variant_log10": float(np.log10(c + 1)),
                })
        if rows:
            sc_df = pl.DataFrame(rows)
            max_val = max(sc_df["baseline_log10"].max(), sc_df["variant_log10"].max())
            scatter = (
                alt.Chart(sc_df)
                .mark_point(opacity=0.25, size=12)
                .encode(
                    x=alt.X("baseline_log10:Q",
                            title=f"log10({vname(baseline_v)} count + 1)"),
                    y=alt.Y("variant_log10:Q", title="log10(variant count + 1)"),
                    color=alt.Color("variant_name:N", title="Variant"),
                    tooltip=["variant_name:N", "baseline_log10:Q", "variant_log10:Q"],
                )
                .properties(width=420, height=420,
                            title="Per-variant proteome vs baseline")
            )
            parity = (
                alt.Chart(pl.DataFrame({"x": np.linspace(0, float(max_val), 50)}))
                .mark_line(color="red", strokeDash=[5, 5])
                .encode(x="x", y="x")
            )
            scatter_chart = (scatter + parity)
        else:
            # Only baseline present (single-variant test data): show baseline
            # self-distribution so the view is non-empty.
            self_df = pl.DataFrame({
                "baseline_log10": np.log10(baseline + 1),
            })
            scatter_chart = (
                alt.Chart(self_df)
                .mark_bar()
                .encode(
                    x=alt.X("baseline_log10:Q", bin=alt.Bin(maxbins=40),
                            title=f"log10({vname(baseline_v)} count + 1)"),
                    y=alt.Y("count()", title="monomers"),
                )
                .properties(width=420, height=320,
                            title="Baseline proteome (only variant present)")
            )

        # --- per-variant validation Pearson r table --------------------------
        r_rows = []
        if validation_data is not None:
            schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
            schmidt_counts = validation_data.protein.schmidt2015Data["glucoseCounts"]
            wis_ids = validation_data.protein.wisniewski2014Data["monomerId"]
            wis_counts = validation_data.protein.wisniewski2014Data["avgCounts"]
            for v in variants:
                avg = by_variant[v]
                sim_s, val_s = _get_simulated_validation_counts(
                    schmidt_counts, avg, schmidt_ids, sim_monomer_ids)
                sim_w, val_w = _get_simulated_validation_counts(
                    wis_counts, avg, wis_ids, sim_monomer_ids)
                r_rows.append({
                    "variant_name": vname(v),
                    "schmidt_r": float(pearsonr(
                        np.log10(sim_s + 1), np.log10(val_s + 1))[0]),
                    "wisniewski_r": float(pearsonr(
                        np.log10(sim_w + 1), np.log10(val_w + 1))[0]),
                })

        if r_rows:
            r_df = pl.DataFrame(r_rows).unpivot(
                on=["schmidt_r", "wisniewski_r"],
                index="variant_name",
                variable_name="dataset", value_name="pearson_r",
            )
            table = (
                alt.Chart(r_df)
                .mark_rect()
                .encode(
                    x=alt.X("dataset:N", title="Validation set"),
                    y=alt.Y("variant_name:N", title="Variant"),
                    color=alt.Color("pearson_r:Q", title="Pearson r",
                                    scale=alt.Scale(scheme="viridis")),
                )
                .properties(width=200, height=30 * max(1, len(r_rows)),
                            title="Proteome validation r per variant")
            )
            text = (
                alt.Chart(r_df)
                .mark_text(baseline="middle")
                .encode(x="dataset:N", y="variant_name:N",
                        text=alt.Text("pearson_r:Q", format=".2f"))
            )
            r_view = (table + text)
            combined = alt.hconcat(scatter_chart, r_view)
        else:
            combined = scatter_chart

        return {
            "view": chart_to_html(combined, title="Proteome delta (per variant)"),
            "data": {
                "baseline_variant": baseline_v,
                "n_variants": len(variants),
                "validation_r": r_rows,
            },
        }
