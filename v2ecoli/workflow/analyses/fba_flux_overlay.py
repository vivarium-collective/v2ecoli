"""Cross-variant FBA flux overlay (variant-comparison study, showcase-4).

The media headline.  For each variant computes the central-carbon reaction
fluxes from ``listeners.fba_results.base_reaction_fluxes`` and reports:

  * per-variant Toya-2010 Pearson r (flux-quality vs experiment) as a small
    table — present only when ``validation_data.reactionFlux`` is available; and
  * a delta-vs-baseline grouped bar of mean flux for the central-carbon
    reactions (Toya reactions when validation is present, else the top reactions
    by |mean baseline flux|), so a media change that re-routes central carbon
    shows up as bars departing from baseline.

Registered as ``"fba_flux_overlay"`` (scale: ``"multivariant"``).

Colors by variant: each variant is a colored series in the delta bar and a row
in the Toya-r table, keyed by the ``variant`` hive-partition column mapped to a
display name via the runner's ``variant_metadata``.  Flux unit conversion
(mmol/gDW/s -> mmol/gDW/h) follows the ``central_carbon_metabolism_scatter``
port.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection
from scipy.stats import pearsonr

from wholecell.utils import units
from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401
from v2ecoli.workflow.analyses._helpers import (
    aliased_history,
    available_columns,
    chart_to_html,
    variant_label_map,
)

_DENSITY_UNITS = units.g / units.L


class FbaFluxOverlay(Analysis):
    """Per-variant Toya r + central-carbon flux delta-vs-baseline."""

    name = "fba_flux_overlay"
    scale = "multivariant"

    def _variant_mean_fluxes(self, conn, base, sim_data) -> dict[int, np.ndarray]:
        """Per-variant mean converted flux vector over all rows/cells."""
        flux_col = "listeners__fba_results__base_reaction_fluxes"
        df = conn.sql(f"""
            SELECT variant, {flux_col} AS flux,
                   listeners__mass__cell_mass AS cell_mass,
                   listeners__mass__dry_mass AS dry_mass
            FROM ({base})
            WHERE len({flux_col}) > 0
        """).df()
        if df.empty:
            raise ValueError("fba_flux_overlay: no non-empty FBA flux rows")

        density = sim_data.constants.cell_density.asNumber(_DENSITY_UNITS)
        out: dict[int, np.ndarray] = {}
        for v, sub in df.groupby("variant"):
            fluxes = np.stack(sub["flux"].values)
            coeff = sub["dry_mass"].values / sub["cell_mass"].values * density
            converted = fluxes / coeff[:, np.newaxis] * 3600
            out[int(v)] = converted.mean(axis=0)
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
        if "listeners__fba_results__base_reaction_fluxes" not in avail:
            raise ValueError("fba_flux_overlay: base_reaction_fluxes not emitted")

        base_reaction_ids = list(sim_data.process.metabolism.base_reaction_ids)
        rid_to_idx = {r: i for i, r in enumerate(base_reaction_ids)}
        labels = variant_label_map(variant_metadata)
        mean_by_variant = self._variant_mean_fluxes(conn, base, sim_data)

        width = len(next(iter(mean_by_variant.values())))
        if width != len(base_reaction_ids):
            raise ValueError(
                f"fba_flux_overlay: flux width {width} != base_reaction_ids "
                f"{len(base_reaction_ids)}; sim_data does not pair with this "
                "parquet (use out/workflow/simData.cPickle)")

        variants = sorted(mean_by_variant)
        baseline_v = 0 if 0 in mean_by_variant else variants[0]

        def vname(v: int) -> str:
            return labels.get(v, f"variant {v}")

        has_toya = (validation_data is not None
                    and getattr(validation_data, "reactionFlux", None) is not None)

        # --- choose the central-carbon reaction set --------------------------
        if has_toya:
            mmol_per_g_per_h = units.mmol / units.g / units.h
            toya = validation_data.reactionFlux.toya2010fluxes
            toya_ids = list(toya["reactionID"])
            toya_flux = {r: f.asNumber(mmol_per_g_per_h)
                         for r, f in zip(toya_ids, toya["reactionFlux"])}
            cc_rxns = [r for r in toya_ids if r in rid_to_idx]
        else:
            base_mean = mean_by_variant[baseline_v]
            top = np.argsort(np.abs(base_mean))[-20:][::-1]
            cc_rxns = [base_reaction_ids[i] for i in top]
            toya_flux = {}

        if not cc_rxns:
            raise ValueError("fba_flux_overlay: no central-carbon reactions found")

        # --- delta-vs-baseline bar -------------------------------------------
        base_mean = mean_by_variant[baseline_v]
        rows = []
        for v in variants:
            mv = mean_by_variant[v]
            for r in cc_rxns:
                i = rid_to_idx[r]
                rows.append({
                    "variant_name": vname(v),
                    "reaction": r,
                    "flux": float(mv[i]),
                    "delta_vs_baseline": float(mv[i] - base_mean[i]),
                })
        bar_df = pd.DataFrame(rows)

        delta_bar = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                y=alt.Y("reaction:N", title="Central-carbon reaction",
                        sort=cc_rxns),
                x=alt.X("delta_vs_baseline:Q",
                        title="Mean flux Δ vs baseline [mmol/gDW/hr]"),
                yOffset=alt.YOffset("variant_name:N"),
                color=alt.Color("variant_name:N", title="Variant"),
                tooltip=["variant_name:N", "reaction:N",
                         alt.Tooltip("flux:Q", format=".3f"),
                         alt.Tooltip("delta_vs_baseline:Q", format=".3f")],
            )
            .properties(width=420, height=max(300, 22 * len(cc_rxns)),
                        title="Central-carbon flux Δ vs baseline")
        )

        # --- per-variant Toya r table ----------------------------------------
        r_rows = []
        if has_toya:
            for v in variants:
                mv = mean_by_variant[v]
                model, exp = [], []
                for r in cc_rxns:
                    model.append(mv[rid_to_idx[r]])
                    exp.append(toya_flux[r])
                if len(model) >= 2:
                    r_rows.append({"variant_name": vname(v),
                                   "toya_r": float(pearsonr(model, exp)[0])})

        if r_rows:
            r_df = pd.DataFrame(r_rows)
            r_view = (
                alt.Chart(r_df)
                .mark_rect()
                .encode(
                    x=alt.value(0),
                    y=alt.Y("variant_name:N", title="Variant"),
                    color=alt.Color("toya_r:Q", title="Toya r",
                                    scale=alt.Scale(scheme="viridis")),
                )
                .properties(width=120, height=30 * max(1, len(r_rows)),
                            title="Toya-2010 r")
            )
            r_text = (
                alt.Chart(r_df)
                .mark_text()
                .encode(y="variant_name:N",
                        text=alt.Text("toya_r:Q", format=".2f"))
            )
            combined = alt.hconcat(delta_bar, (r_view + r_text))
            title = "FBA flux overlay (Toya r + Δ vs baseline)"
        else:
            combined = delta_bar
            title = ("FBA flux overlay (Δ vs baseline; "
                     "validation_data unavailable — Toya r omitted)")

        return {
            "view": chart_to_html(combined, title=title),
            "data": {
                "baseline_variant": baseline_v,
                "central_carbon_reactions": cc_rxns,
                "toya_r": r_rows,
                "has_validation": has_toya,
            },
        }
