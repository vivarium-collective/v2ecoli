"""Native port of vEcoli ``ecoli/analysis/multiseed/centralCarbonMetabolismScatter.py``.

Scatter plot of simulated central-carbon-metabolism reaction fluxes vs. Toya
2010 experimental measurements.  Registered in ANALYSIS_REGISTRY as
``"central_carbon_metabolism_scatter"`` (scale: ``"multiseed"``).

Key design decisions vs. the vEcoli original
--------------------------------------------
* ``base_reaction_ids`` comes from ``sim_data.process.metabolism.base_reaction_ids``
  (not from ``field_metadata(conn, config_sql, ...)``) — v2ecoli analyses do not
  receive a ``config_sql`` argument.
* 1:1 alignment is asserted loudly: if ``len(base_reaction_ids) != flux_width`` we
  raise a clear ``ValueError`` rather than silently truncating (a mismatch means
  the caller passed the wrong sim_data).  Use ``out/workflow/simData.cPickle``
  (2820 base_reaction_ids) for the parquet emitted by the compare_harness sweep.
* t=0 rows where FBA has not yet run (empty flux array) are dropped before the
  alignment check, matching ``ptools_rxns`` behaviour.
* ``validation_data=None`` guard: the Toya 2010 comparison requires
  ``validation_data.reactionFlux.toya2010fluxes``.  When ``validation_data`` is
  ``None``, the analysis falls back to a bar chart of the top-30 base reactions
  by absolute mean simulated flux so that the view is still informative.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any
from duckdb import DuckDBPyConnection
from scipy.stats import pearsonr

from wholecell.utils import units
from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.render import fig_to_html

# cell_density is a unum quantity [g/L]; asNumber requires a unum unit
# (VOLUME_UNITS/MASS_UNITS from v2ecoli.processes.metabolism are pint units and
# are incompatible with unum's asNumber — use wholecell unum units directly).
_DENSITY_UNITS = units.g / units.L


# ---------------------------------------------------------------------------
# Column list
# ---------------------------------------------------------------------------

_DATA_COLUMNS = [
    "agent_id",
    "generation",
    "lineage_seed",
    "listeners__fba_results__base_reaction_fluxes",
    "listeners__mass__cell_mass",
    "listeners__mass__dry_mass",
]


def _read_ccm_data(history_sql: str, conn: DuckDBPyConnection) -> pd.DataFrame:
    """Query required columns from parquet, ordered by global_time."""
    col_list = ", ".join(_DATA_COLUMNS)
    query = f"""
        SELECT {col_list}, global_time AS time
        FROM ({history_sql})
        ORDER BY time
    """
    return conn.sql(query).df()


# ---------------------------------------------------------------------------
# Analysis subclass
# ---------------------------------------------------------------------------

class CentralCarbonMetabolismScatter(Analysis):
    """Central carbon metabolism flux scatter: model vs. Toya 2010 (multiseed).

    Faithfully ported from vEcoli
    ``ecoli/analysis/multiseed/centralCarbonMetabolismScatter.py``.
    Returns ``{"view": <html with inline SVG>}``.

    When ``validation_data`` is ``None``, falls back to a bar chart of the
    top-30 base reactions by absolute mean simulated flux.
    """

    name = "central_carbon_metabolism_scatter"
    scale = "multiseed"

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
        # --- 1. base_reaction_ids from sim_data ------------------------------
        base_reaction_ids = sim_data.process.metabolism.base_reaction_ids
        base_reaction_id_to_index = {
            rxn_id: i for i, rxn_id in enumerate(base_reaction_ids)
        }

        # --- 2. Query parquet ------------------------------------------------
        result = _read_ccm_data(history_sql, conn)

        # --- 3. Drop t=0 rows where FBA hasn't run (empty flux arrays) -------
        flux_col = "listeners__fba_results__base_reaction_fluxes"
        result = result[result[flux_col].apply(len) > 0].reset_index(drop=True)

        if result.empty:
            raise ValueError("No non-empty FBA flux rows found in history_sql")

        # --- 4. 1:1 alignment assertion (see module docstring) ---------------
        sample_flux = result[flux_col].iloc[0]
        flux_width = len(sample_flux)
        n_ids = len(base_reaction_ids)
        if n_ids != flux_width:
            raise ValueError(
                f"base_reaction_ids ({n_ids}) != flux width ({flux_width}); "
                "sim_data does not pair with this parquet — use "
                "out/workflow/simData.cPickle, not out/kb/simData.cPickle"
            )

        # --- 5. Convert fluxes: raw (mmol/gDW/s) → mmol/gDW/h ---------------
        # coefficient = dryMass/cellMass * density  [g/L]
        # flux_converted = flux / coefficient * 3600  [mmol/g/h]
        cellDensity = sim_data.constants.cell_density
        cell_mass = result["listeners__mass__cell_mass"].values
        dry_mass = result["listeners__mass__dry_mass"].values
        coefficient = dry_mass / cell_mass * cellDensity.asNumber(_DENSITY_UNITS)

        fluxes_full = np.stack(result[flux_col].values)           # (N, n_rxns)
        # broadcast coefficient (N,) across reactions axis
        fluxes_converted = fluxes_full / coefficient[:, np.newaxis] * 3600

        # --- 6. Per-agent mean fluxes ----------------------------------------
        lineage_seeds = result["lineage_seed"].values
        generations = result["generation"].values
        agent_ids = result["agent_id"].values

        # Unique (seed, generation, agent_id) triples
        unique_agents = list(
            {(int(s), int(g), a)
             for s, g, a in zip(lineage_seeds, generations, agent_ids)}
        )

        # --- 7. Branch: with vs. without validation_data ---------------------
        # The Toya comparison needs ``validation_data.reactionFlux``; the minimal
        # v2ecoli validation loader only provides ``.protein`` (schmidt/wisniewski),
        # so guard on the specific attribute, not just non-None — otherwise fall
        # back to the no-validation flux bar chart below.
        if validation_data is not None and getattr(validation_data, "reactionFlux", None) is not None:
            # Full Toya 2010 comparison scatter (faithful port)
            toyaReactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]
            toyaFluxes = validation_data.reactionFlux.toya2010fluxes["reactionFlux"]
            toyaStdev = validation_data.reactionFlux.toya2010fluxes["reactionFluxStdev"]
            toyaFluxesDict = dict(zip(toyaReactions, toyaFluxes))
            toyaStdevDict = dict(zip(toyaReactions, toyaStdev))
            mmol_per_g_per_h = units.mmol / units.g / units.h

            modelFluxes: dict[str, list[float]] = {rxn: [] for rxn in toyaReactions}

            for seed, gen, agent in unique_agents:
                mask = (
                    (lineage_seeds == seed)
                    & (generations == gen)
                    & (agent_ids == agent)
                )
                agent_fluxes = fluxes_converted[mask]  # (n_timepoints_agent, n_rxns)
                for toyaReaction in toyaReactions:
                    if toyaReaction in base_reaction_id_to_index:
                        rxn_idx = base_reaction_id_to_index[toyaReaction]
                        flux_tc = agent_fluxes[:, rxn_idx]
                        modelFluxes[toyaReaction].append(float(np.mean(flux_tc)))

            toyaVsReactionAve = []
            for rxn, toyaFlux in toyaFluxesDict.items():
                if rxn in modelFluxes and len(modelFluxes[rxn]) > 0:
                    toyaVsReactionAve.append((
                        np.mean(modelFluxes[rxn]),
                        toyaFlux.asNumber(mmol_per_g_per_h),
                        np.std(modelFluxes[rxn]),
                        toyaStdevDict[rxn].asNumber(mmol_per_g_per_h),
                    ))

            if not toyaVsReactionAve:
                raise ValueError(
                    "No Toya 2010 reactions found in base_reaction_ids; "
                    "cannot render CCM scatter with validation_data"
                )

            toyaVsReactionAve = np.array(toyaVsReactionAve)
            rWithAll = pearsonr(toyaVsReactionAve[:, 0], toyaVsReactionAve[:, 1])

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(
                "Central Carbon Metabolism Flux, Pearson R = %.4f, p = %s\n"
                % (rWithAll[0], rWithAll[1])
            )
            ax.set_xlabel("Toya 2010 Reaction Flux [mmol/g/hr]")
            ax.set_ylabel("Mean WCM Reaction Flux [mmol/g/hr]")
            ax.errorbar(
                toyaVsReactionAve[:, 1],
                toyaVsReactionAve[:, 0],
                xerr=toyaVsReactionAve[:, 3],
                yerr=toyaVsReactionAve[:, 2],
                fmt=".",
                ecolor="k",
                alpha=0.5,
                linewidth=0.5,
            )
            ylim = ax.get_ylim()
            ax.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], color="k")
            ax.plot(toyaVsReactionAve[:, 1], toyaVsReactionAve[:, 1], color="black")
            ax.plot(
                toyaVsReactionAve[:, 1],
                toyaVsReactionAve[:, 0],
                "ob",
                markeredgewidth=0.1,
                alpha=0.9,
            )
            ax.set_xlim([-20, 30])
            ax.set_ylim([-20, 70])
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_yticks(list(range(int(ylim[0]), int(ylim[1]) + 1, 10)))
            ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, 10)))

        else:
            # validation_data is None — render simulated flux distribution.
            # Compute mean flux across all timepoints and agents for every
            # base reaction, then show the top 30 by absolute mean flux.
            # NOTE: the full Toya 2010 comparison (scatter vs. experimental
            # fluxes) requires validation_data.reactionFlux.toya2010fluxes
            # and is unavailable here.
            mean_fluxes = np.mean(fluxes_converted, axis=0)  # (n_rxns,)

            top_n = min(30, len(base_reaction_ids))
            top_idx = np.argsort(np.abs(mean_fluxes))[-top_n:][::-1]
            top_ids = [base_reaction_ids[i] for i in top_idx]
            top_fluxes = mean_fluxes[top_idx]

            colors = ["#2196F3" if f >= 0 else "#F44336" for f in top_fluxes]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(
                "Central Carbon Metabolism — Simulated Flux Distribution\n"
                f"(validation_data not provided; n_seeds={len(set(lineage_seeds))}, "
                f"top {top_n} of {len(base_reaction_ids)} reactions by |mean flux|)"
            )
            ax.barh(range(top_n), top_fluxes, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_ids, fontsize=7)
            ax.set_xlabel("Mean Reaction Flux [mmol/gDW/hr]")
            ax.axvline(0, color="k", linewidth=0.5)
            ax.invert_yaxis()
            fig.tight_layout()

        html = fig_to_html(fig, title="Central carbon metabolism")
        plt.close(fig)
        return {"view": html}
