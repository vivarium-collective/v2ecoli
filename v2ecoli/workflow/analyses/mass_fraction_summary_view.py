"""Native port of vEcoli ``ecoli/analysis/single/mass_fraction_summary.py``.

Line plot of biomass-component masses (protein, tRNA, rRNA, mRNA, DNA, small
molecule, dry) over time, each normalised to its t=0 value, with the mean
fraction of dry mass annotated in the legend.  Returns an Altair HTML view.

Registered as ``"mass_fraction_summary_view"`` (scale: ``"single"``).  The bare
name ``mass_fraction_summary`` is already taken by the record-based
``AnalysisStep`` example in ``analysis.py``; this distinct name avoids
overwriting that registry entry (the two are different analyses).
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._helpers import (
    read_stacked_columns,
    num_cells,
    chart_to_html,
)

COLORS_256 = [  # colorbrewer2.org qualitative 8-class set 1
    [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
]
COLORS = ["#%02x%02x%02x" % tuple(c) for c in COLORS_256]


class MassFractionSummaryView(Analysis):
    """Biomass-component fold-change line plot over a single cell's timeseries."""

    name = "mass_fraction_summary_view"
    scale = "single"

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

        assert num_cells(conn, history_sql) == 1, (
            "Mass fraction summary plot requires single-cell data."
        )

        mass_columns = {
            "Protein": "listeners__mass__protein_mass",
            "tRNA": "listeners__mass__tRna_mass",
            "rRNA": "listeners__mass__rRna_mass",
            "mRNA": "listeners__mass__mRna_mass",
            "DNA": "listeners__mass__dna_mass",
            "Small Mol": "listeners__mass__smallMolecule_mass",
            "Dry": "listeners__mass__dry_mass",
        }
        mass_data = read_stacked_columns(
            history_sql, list(mass_columns.values()), conn=conn
        )

        fractions = {
            k: cast(
                float,
                (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean(),
            )
            for k, v in mass_columns.items()
        }
        new_columns = {
            "Time (min)": (mass_data["time"] - mass_data["time"].min()) / 60,
            **{
                f"{k} ({fractions[k]:.3f})": mass_data[v] / mass_data[v][0]
                for k, v in mass_columns.items()
            },
        }
        mass_fold_change_df = pl.DataFrame(new_columns)

        # Altair long form (no periods in column names).
        melted_df = mass_fold_change_df.melt(
            id_vars="Time (min)",
            variable_name="Submass",
            value_name="Mass (normalized by t = 0 min)",
        )

        chart = (
            alt.Chart(melted_df)
            .mark_line()
            .encode(
                x=alt.X("Time (min):Q", title="Time (min)"),
                y=alt.Y("Mass (normalized by t = 0 min):Q"),
                color=alt.Color("Submass:N", scale=alt.Scale(range=COLORS)),
            )
            .properties(
                title="Biomass components (average fraction of total dry mass "
                "in parentheses)"
            )
        )
        return {"view": chart_to_html(chart, title="Mass fraction summary")}
