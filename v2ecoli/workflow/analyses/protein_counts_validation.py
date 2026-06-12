"""Native port of vEcoli ``ecoli/analysis/multiseed/protein_counts_validation.py``.

Plots average simulated monomer counts vs. Schmidt 2015 and Wisniewski 2014
proteomics data.  Scatter plots on a log10 scale with Pearson r in the title.

Registered as ``"protein_counts_validation"`` (scale: ``"multiseed"``).

v2ecoli adaptations:
* ``listeners__monomer_counts`` is averaged directly in DuckDB over the full
  multiseed history (no per-cell intermediate step needed).
* ID ordering comes from ``sim_data.process.translation.monomer_data["id"]``.
* validation_data is the object produced by
  ``v2ecoli.library.validation_data.build_validation_data``.
* Chart is returned as an HTML string via ``chart_to_html`` rather than
  written to a file.

Fail-loud rules enforced:
* If ``validation_data`` is None, raises ``RuntimeError`` (not silently
  skipped).
* If the ID overlap between sim and either validation set is zero, raises
  ``ValueError``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
from scipy.stats import pearsonr

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import (
    read_stacked_columns,
    chart_to_html,
)


def _get_simulated_validation_counts(
    validation_counts: np.ndarray,
    avg_sim_counts: np.ndarray,
    validation_ids: np.ndarray,
    simulation_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match sim counts to validation counts by monomer ID intersection.

    Replicates ``wholecell.utils.protein_counts.get_simulated_validation_counts``
    without the cross-cell mean step (``avg_sim_counts`` is already a 1-D
    vector of per-monomer averages).

    Raises
    ------
    ValueError
        If the intersection of sim and validation IDs is empty.
    """
    sim_ids_list = list(simulation_ids)
    val_ids_list = list(validation_ids)
    overlapping = set(sim_ids_list) & set(val_ids_list)
    if not overlapping:
        raise ValueError(
            f"Zero overlap between {len(sim_ids_list)} sim monomer IDs and "
            f"{len(val_ids_list)} validation IDs — check that monomer ID "
            "formats match (e.g. 'MONOMER[c]')."
        )

    sim_idx = {sid: i for i, sid in enumerate(sim_ids_list) if sid in overlapping}
    val_idx = {vid: i for i, vid in enumerate(val_ids_list) if vid in overlapping}
    ids = list(overlapping)
    sim_filtered = avg_sim_counts[[sim_idx[m] for m in ids]]
    val_filtered = validation_counts[[val_idx[m] for m in ids]]
    return sim_filtered, val_filtered


class ProteinCountsValidation(Analysis):
    """Compare average simulated monomer counts to Schmidt 2015 + Wisniewski 2014."""

    name = "protein_counts_validation"
    scale = "multiseed"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        validation_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        import altair as alt

        if validation_data is None:
            raise RuntimeError(
                "protein_counts_validation requires validation_data but received None. "
                "Ensure the Schmidt/Wisniewski flat files are installed and "
                "resolve_validation_data() succeeds in analysis_runner.py."
            )

        # ------------------------------------------------------------------
        # 1. Compute per-monomer average counts across all seeds/cells/times
        # ------------------------------------------------------------------
        # read_stacked_columns gives us a subquery over history_sql with the
        # standard cell-id columns + time.  We then unnest the per-row monomer
        # count lists and average each position across ALL rows (cells × times).
        subquery = read_stacked_columns(
            history_sql, ["listeners__monomer_counts"], order_results=False
        )
        avg_counts_df = conn.sql(f"""
            WITH flat AS (
                SELECT
                    unnest(listeners__monomer_counts) AS cnt,
                    generate_subscripts(listeners__monomer_counts, 1) AS idx
                FROM ({subquery})
            )
            SELECT idx, avg(cnt) AS avg_cnt
            FROM flat
            GROUP BY idx
            ORDER BY idx
        """).pl()

        if avg_counts_df.is_empty():
            raise ValueError(
                "protein_counts_validation: no monomer_counts rows found in "
                "history_sql — check that listeners__monomer_counts is emitted."
            )

        # Build average count vector (1-D, indexed by monomer position)
        avg_sim_counts = avg_counts_df["avg_cnt"].to_numpy()

        # ------------------------------------------------------------------
        # 2. Sim monomer IDs (same order as the monomer_counts list)
        # ------------------------------------------------------------------
        sim_monomer_ids = sim_data.process.translation.monomer_data["id"]

        if len(avg_sim_counts) != len(sim_monomer_ids):
            raise ValueError(
                f"protein_counts_validation: avg_sim_counts length "
                f"({len(avg_sim_counts)}) != sim monomer_ids length "
                f"({len(sim_monomer_ids)}).  Check that listeners__monomer_counts "
                "is ordered to match sim_data.process.translation.monomer_data."
            )

        # ------------------------------------------------------------------
        # 3. Load validation IDs / counts
        # ------------------------------------------------------------------
        wisniewski_ids = validation_data.protein.wisniewski2014Data["monomerId"]
        schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
        wisniewski_counts = validation_data.protein.wisniewski2014Data["avgCounts"]
        schmidt_counts = validation_data.protein.schmidt2015Data["glucoseCounts"]

        # ------------------------------------------------------------------
        # 4. Match sim ↔ validation
        # ------------------------------------------------------------------
        sim_schmidt, val_schmidt = _get_simulated_validation_counts(
            schmidt_counts, avg_sim_counts, schmidt_ids, sim_monomer_ids
        )
        sim_wisniewski, val_wisniewski = _get_simulated_validation_counts(
            wisniewski_counts, avg_sim_counts, wisniewski_ids, sim_monomer_ids
        )

        # ------------------------------------------------------------------
        # 5. Build Altair scatter charts (log10, with parity line)
        # ------------------------------------------------------------------
        schmidt_chart = (
            alt.Chart(
                pl.DataFrame({
                    "schmidt": np.log10(val_schmidt + 1),
                    "sim": np.log10(sim_schmidt + 1),
                })
            )
            .mark_point(opacity=0.5)
            .encode(
                x=alt.X("schmidt", title="log10(Schmidt 2015 Counts + 1)"),
                y=alt.Y("sim", title="log10(Simulation Average Counts + 1)"),
            )
            .properties(
                title="Schmidt 2015 — Pearson r: %0.2f"
                % pearsonr(
                    np.log10(sim_schmidt + 1), np.log10(val_schmidt + 1)
                )[0],
                width=350,
                height=350,
            )
        )

        wisniewski_chart = (
            alt.Chart(
                pl.DataFrame({
                    "wisniewski": np.log10(val_wisniewski + 1),
                    "sim": np.log10(sim_wisniewski + 1),
                })
            )
            .mark_point(opacity=0.5)
            .encode(
                x=alt.X("wisniewski", title="log10(Wisniewski 2014 Counts + 1)"),
                y=alt.Y("sim", title="log10(Simulation Average Counts + 1)"),
            )
            .properties(
                title="Wisniewski 2014 — Pearson r: %0.2f"
                % pearsonr(
                    np.log10(sim_wisniewski + 1), np.log10(val_wisniewski + 1)
                )[0],
                width=350,
                height=350,
            )
        )

        max_val = max(
            float(np.log10(val_schmidt + 1).max()),
            float(np.log10(val_wisniewski + 1).max()),
            float(np.log10(sim_schmidt + 1).max()),
            float(np.log10(sim_wisniewski + 1).max()),
        )
        parity = (
            alt.Chart(pl.DataFrame({"x": np.linspace(0, max_val, 50)}))
            .mark_line(color="red", strokeDash=[5, 5])
            .encode(x="x", y="x")
        )

        chart = (schmidt_chart + parity) | (wisniewski_chart + parity)

        # ------------------------------------------------------------------
        # 6. Summary data (for test assertions / downstream inspection)
        # ------------------------------------------------------------------
        data = {
            "n_sim_monomers": int(len(sim_monomer_ids)),
            "n_schmidt_overlap": int(len(sim_schmidt)),
            "n_wisniewski_overlap": int(len(sim_wisniewski)),
            "schmidt_pearsonr": float(
                pearsonr(np.log10(sim_schmidt + 1), np.log10(val_schmidt + 1))[0]
            ),
            "wisniewski_pearsonr": float(
                pearsonr(np.log10(sim_wisniewski + 1), np.log10(val_wisniewski + 1))[0]
            ),
        }

        return {
            "view": chart_to_html(chart, title="Protein Counts Validation"),
            "data": data,
        }
