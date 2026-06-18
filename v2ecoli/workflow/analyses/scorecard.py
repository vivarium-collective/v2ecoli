"""Cross-variant scorecard (variant-comparison study, showcase-4).

One summary heatmap: rows = variants, columns = headline metrics
(doubling time, dry mass, protein fraction, max oriC, mean ppGpp, Toya r,
Schmidt r).  Each cell is colored by its Δ vs the baseline variant (signed,
per-column-normalised), so at a glance you see which variant moves which metric
and in which direction.  The raw value is printed in each cell.  Registered as
``"scorecard"`` (scale: ``"multivariant"``).

Colors by variant: variants are the heatmap rows (keyed by the ``variant``
hive-partition column mapped to a display name via the runner's
``variant_metadata``); the color encodes Δ-vs-baseline per column.

Each metric degrades gracefully: a metric whose source column / validation set
is unavailable is simply omitted from the columns rather than failing the whole
scorecard.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401
from v2ecoli.workflow.analyses._helpers import (
    aliased_history,
    available_columns,
    chart_to_html,
    variant_label_map,
)

LN2 = math.log(2.0)
_DENSITY_UNITS = None  # set lazily to avoid importing units when unused


class Scorecard(Analysis):
    """Per-variant summary heatmap with Δ-vs-baseline coloring."""

    name = "scorecard"
    scale = "multivariant"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data=None,
        validation_data=None,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        import altair as alt

        base = aliased_history(history_sql)
        avail = available_columns(conn, history_sql)
        labels = variant_label_map(variant_metadata)

        # ---- mass/growth/replication/regulation metrics per variant --------
        sel = ["variant"]
        col_specs: list[tuple[str, str]] = []  # (metric label, select expr alias)

        def add(label, expr):
            sel.append(expr)
            col_specs.append((label, label))

        if "listeners__mass__instantaneous_growth_rate" in avail:
            add("doubling time (min)",
                f"{LN2} / avg(listeners__mass__instantaneous_growth_rate) "
                "FILTER (WHERE listeners__mass__instantaneous_growth_rate > 0) "
                "/ 60 AS \"doubling time (min)\"")
        if "listeners__mass__dry_mass" in avail:
            add("dry mass (fg)", "avg(listeners__mass__dry_mass) AS \"dry mass (fg)\"")
        if "listeners__mass__protein_mass" in avail and \
                "listeners__mass__dry_mass" in avail:
            add("protein fraction",
                "avg(listeners__mass__protein_mass / listeners__mass__dry_mass) "
                "FILTER (WHERE listeners__mass__dry_mass > 0) AS \"protein fraction\"")
        if "listeners__replication_data__number_of_oric" in avail:
            add("max oriC",
                "max(listeners__replication_data__number_of_oric) AS \"max oriC\"")
        if "listeners__growth_limits__ppgpp_conc" in avail:
            add("mean ppGpp",
                "avg(listeners__growth_limits__ppgpp_conc) AS \"mean ppGpp\"")

        if len(sel) == 1:
            raise ValueError("scorecard: no scalar metric columns present")

        agg = conn.sql(f"""
            SELECT {", ".join(sel)}
            FROM ({base})
            GROUP BY variant
            ORDER BY variant
        """).df()

        metric_cols = [lbl for lbl, _ in col_specs]
        scorecard = agg.set_index("variant")[metric_cols].copy()

        # ---- validation metrics (Toya r, Schmidt r) per variant ------------
        self._add_validation_metrics(
            conn, base, sim_data, validation_data, scorecard, avail)

        # ---- rename variant rows to display names --------------------------
        scorecard.index = [labels.get(int(v), f"variant {v}")
                           for v in scorecard.index]
        baseline_v = 0 if 0 in agg["variant"].values else int(agg["variant"].min())
        baseline_name = labels.get(baseline_v, f"variant {baseline_v}")

        # ---- build Δ-vs-baseline (per-column normalised) -------------------
        long_rows = []
        for col in scorecard.columns:
            base_val = (scorecard.loc[baseline_name, col]
                        if baseline_name in scorecard.index else np.nan)
            col_vals = scorecard[col].astype(float)
            spread = float(np.nanmax(np.abs(col_vals - base_val))) or 1.0
            for variant_name, val in col_vals.items():
                delta = float(val) - float(base_val) if not np.isnan(base_val) else 0.0
                long_rows.append({
                    "variant_name": variant_name,
                    "metric": col,
                    "value": float(val),
                    "delta_norm": delta / spread,
                })
        long = pd.DataFrame(long_rows)

        row_order = list(scorecard.index)
        col_order = list(scorecard.columns)

        heat = (
            alt.Chart(long)
            .mark_rect(stroke="white")
            .encode(
                x=alt.X("metric:N", title="Metric", sort=col_order,
                        axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("variant_name:N", title="Variant", sort=row_order),
                color=alt.Color(
                    "delta_norm:Q", title="Δ vs baseline (norm.)",
                    scale=alt.Scale(scheme="blueorange", domainMid=0)),
                tooltip=["variant_name:N", "metric:N",
                         alt.Tooltip("value:Q", format=".4g"),
                         alt.Tooltip("delta_norm:Q", format=".2f")],
            )
            .properties(width=90 * max(1, len(col_order)),
                        height=44 * max(1, len(row_order)),
                        title=f"Variant scorecard (Δ vs {baseline_name})")
        )
        text = (
            alt.Chart(long)
            .mark_text(baseline="middle", fontSize=10)
            .encode(x="metric:N", y="variant_name:N",
                    text=alt.Text("value:Q", format=".3g"))
        )

        return {
            "view": chart_to_html(heat + text, title="Variant scorecard"),
            "data": {
                "baseline_variant": baseline_v,
                "metrics": col_order,
                "table": scorecard.reset_index().rename(
                    columns={"index": "variant_name"}).to_dict(orient="records"),
            },
        }

    def _add_validation_metrics(self, conn, base, sim_data, validation_data,
                                scorecard, avail):
        """Append Toya r and Schmidt r columns to ``scorecard`` if available."""
        if validation_data is None or sim_data is None:
            return

        import numpy as np
        from scipy.stats import pearsonr

        variants = list(scorecard.index)

        # --- Schmidt r (proteome) ---
        if "listeners__monomer_counts" in avail and \
                getattr(validation_data, "protein", None) is not None:
            try:
                from v2ecoli.workflow.analyses.protein_counts_validation import (
                    _get_simulated_validation_counts,
                )
                df = conn.sql(f"""
                    WITH flat AS (
                        SELECT variant,
                               unnest(listeners__monomer_counts) AS cnt,
                               generate_subscripts(
                                   listeners__monomer_counts, 1) AS idx
                        FROM ({base})
                        WHERE len(listeners__monomer_counts) > 0
                    )
                    SELECT variant, idx, avg(cnt) AS avg_cnt
                    FROM flat GROUP BY variant, idx ORDER BY variant, idx
                """).df()
                sim_ids = list(sim_data.process.translation.monomer_data["id"])
                schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
                schmidt_counts = \
                    validation_data.protein.schmidt2015Data["glucoseCounts"]
                rmap = {}
                for v in df["variant"].unique():
                    avg = (df[df["variant"] == v].sort_values("idx")["avg_cnt"]
                           .to_numpy())
                    if len(avg) != len(sim_ids):
                        continue
                    sim_s, val_s = _get_simulated_validation_counts(
                        schmidt_counts, avg, schmidt_ids, sim_ids)
                    rmap[int(v)] = float(pearsonr(
                        np.log10(sim_s + 1), np.log10(val_s + 1))[0])
                self._attach(scorecard, "Schmidt r", rmap)
            except Exception:  # noqa: BLE001
                pass

        # --- Toya r (flux) ---
        if "listeners__fba_results__base_reaction_fluxes" in avail and \
                getattr(validation_data, "reactionFlux", None) is not None:
            try:
                from wholecell.utils import units
                density = sim_data.constants.cell_density.asNumber(
                    units.g / units.L)
                base_ids = list(sim_data.process.metabolism.base_reaction_ids)
                rid_to_idx = {r: i for i, r in enumerate(base_ids)}
                mmol = units.mmol / units.g / units.h
                toya = validation_data.reactionFlux.toya2010fluxes
                toya_flux = {r: f.asNumber(mmol) for r, f in zip(
                    toya["reactionID"], toya["reactionFlux"])}
                cc = [r for r in toya["reactionID"] if r in rid_to_idx]

                fdf = conn.sql(f"""
                    SELECT variant,
                           listeners__fba_results__base_reaction_fluxes AS flux,
                           listeners__mass__cell_mass AS cm,
                           listeners__mass__dry_mass AS dm
                    FROM ({base})
                    WHERE len(listeners__fba_results__base_reaction_fluxes) > 0
                """).df()
                rmap = {}
                for v, sub in fdf.groupby("variant"):
                    fl = np.stack(sub["flux"].values)
                    coeff = sub["dm"].values / sub["cm"].values * density
                    conv = (fl / coeff[:, np.newaxis] * 3600).mean(axis=0)
                    model = [conv[rid_to_idx[r]] for r in cc]
                    exp = [toya_flux[r] for r in cc]
                    if len(model) >= 2:
                        rmap[int(v)] = float(pearsonr(model, exp)[0])
                self._attach(scorecard, "Toya r", rmap)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _attach(scorecard, col, rmap_by_variant_int):
        """Attach a per-variant metric keyed by the integer variant id.

        ``scorecard``'s index has already been renamed to display names, so we
        re-key by position: scorecard rows are in ascending variant order, the
        same order produced by the GROUP BY ... ORDER BY variant queries here.
        """
        import numpy as np
        ordered = sorted(rmap_by_variant_int)
        if len(ordered) != len(scorecard.index):
            # partial coverage: fill by position where possible, else NaN
            vals = [rmap_by_variant_int.get(v, np.nan) for v in ordered]
            # pad/truncate to row count
            vals = (vals + [np.nan] * len(scorecard.index))[:len(scorecard.index)]
        else:
            vals = [rmap_by_variant_int[v] for v in ordered]
        scorecard[col] = vals
