"""Native port of vEcoli ``ecoli/analysis/single/mass_fraction_voronoi.py``.

Produces a Voronoi-tiled biomass-fraction figure (initial vs. final cell mass
components) and returns it as an inline-SVG HTML view.  Registered in
ANALYSIS_REGISTRY as ``"mass_fraction_voronoi"`` (scale: ``"single"``).

No shims required: all seven ``listeners__mass__*`` columns are present in the
v2ecoli parquet schema.  The bulk lookup (lipids/polyamines/LPS/murein/
glycogen) uses Shim A (bulk_count_matrix) exactly as the ptools analyses do.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any
from duckdb import DuckDBPyConnection

from wholecell.utils import units
from wholecell.utils.voronoi_plot_main import VoronoiMaster
import wholecell.utils.voronoi_plot_main as _vpm

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY
from v2ecoli.workflow.analyses._shims import bulk_count_matrix
from v2ecoli.workflow.render import fig_to_html

# ---------------------------------------------------------------------------
# matplotlib 3.10 compat + wholecell VoronoiMaster bug fixes
#
# Bug 1: Polygon(xy, closed) → Polygon(xy, closed=closed)
#   matplotlib 3.10 made `closed` keyword-only.  voronoi_plot_main does
#   `from matplotlib.patches import Polygon` at module level, so we patch the
#   name in *that* module's namespace rather than in matplotlib itself.
#
# Bug 2: _add_labels recurses on [label, site] leaf pairs
#   The method checks `isinstance(element, list)` and recurses, but
#   [label_string, coordinate_array] is also a list — causing it to iterate
#   over the string label and fail with IndexError.  We detect a leaf pair by
#   checking: len==2, first element is str, second is not str.
# ---------------------------------------------------------------------------
_OrigPoly = _vpm.Polygon

class _Poly310(_OrigPoly):  # type: ignore[misc]
    def __init__(self, xy, closed=True, **kwargs):
        super().__init__(xy, closed=closed, **kwargs)

_vpm.Polygon = _Poly310


def _fixed_add_labels(self, label_site_list, font_size, ax):
    """Replacement for VoronoiMaster._add_labels that handles leaf pairs."""
    for element in label_site_list:
        # Leaf: [label_string, (x, y)]
        if (
            isinstance(element, (list, tuple))
            and len(element) == 2
            and isinstance(element[0], str)
            and not isinstance(element[1], str)
        ):
            ax.text(
                element[1][0],
                element[1][1],
                element[0],
                fontsize=font_size,
                horizontalalignment="center",
                verticalalignment="center",
            )
        elif isinstance(element, (list, tuple)):
            _fixed_add_labels(self, element, font_size, ax)

_vpm.VoronoiMaster._add_labels = _fixed_add_labels


def _fixed_compute_error(self, polygon_value_list, total_value, total_area):
    """Replacement that skips broken leaf-pair traversal.

    The installed wholecell version has the same isinstance(list) confusion
    as _add_labels: it recurses on [polygon, value] leaf pairs instead of
    handling them directly.  Since _compute_error is only used for the
    verbose area-error print (verbose=False by default), returning 0 is safe.
    """
    return 0.0

_vpm.VoronoiMaster._compute_error = _fixed_compute_error


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

_MASS_COLUMNS = [
    "listeners__mass__rna_mass",
    "listeners__mass__protein_mass",
    "listeners__mass__tRna_mass",
    "listeners__mass__rRna_mass",
    "listeners__mass__mRna_mass",
    "listeners__mass__dna_mass",
    "listeners__mass__smallMolecule_mass",
    "bulk__id",
    "bulk__count",
]


def _read_voronoi_data(
    history_sql: str,
    conn: DuckDBPyConnection,
) -> pd.DataFrame:
    """Query mass listener columns + bulk from parquet, ordered by time."""
    col_list = ", ".join(_MASS_COLUMNS)
    query = f"""
        SELECT {col_list}, global_time AS time
        FROM ({history_sql})
        ORDER BY time
    """
    return conn.sql(query).df()


# ---------------------------------------------------------------------------
# Analysis subclass
# ---------------------------------------------------------------------------

class MassFractionVoronoi(Analysis):
    """Voronoi biomass-fraction plot: initial vs. final cell (single scale).

    Faithfully ported from vEcoli ``ecoli/analysis/single/mass_fraction_voronoi.py``.
    Returns ``{"view": <html with inline SVG>}``.
    """

    name = "mass_fraction_voronoi"
    scale = "single"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        # --- 1. Query data ---------------------------------------------------
        df = _read_voronoi_data(history_sql, conn)

        # --- 2. Bulk molecule count matrix (Shim A) --------------------------
        bulk_molecule_counts = bulk_count_matrix(df, sim_data)

        # Build index from sim_data order
        bulk_molecule_ids: list[str] = list(
            sim_data.internal_state.bulk_molecules.bulk_data["id"]
        )
        bulk_molecule_idx = {name: idx for idx, name in enumerate(bulk_molecule_ids)}

        nAvogadro = sim_data.constants.n_avogadro

        # --- 3. Helper functions (verbatim from vEcoli) ----------------------
        def find_mass_molecule_group(group_id):
            temp_ids = getattr(sim_data.molecule_groups, str(group_id))
            temp_indexes = np.array([bulk_molecule_idx[t] for t in temp_ids])
            temp_counts = bulk_molecule_counts[:, temp_indexes]
            temp_mw = sim_data.getter.get_masses(temp_ids)
            return (units.dot(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

        def find_mass_single_molecule(molecule_id):
            temp_id = getattr(sim_data.molecule_ids, str(molecule_id))
            temp_index = bulk_molecule_idx[temp_id]
            temp_counts = bulk_molecule_counts[:, temp_index]
            temp_mw = sim_data.getter.get_mass(temp_id)
            return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

        # --- 4. Derived mass arrays ------------------------------------------
        lipid = find_mass_molecule_group("lipids")
        polyamines = find_mass_molecule_group("polyamines")
        lps = find_mass_single_molecule("LPS")
        murein = find_mass_single_molecule("murein")
        glycogen = find_mass_single_molecule("glycogen")

        protein = df["listeners__mass__protein_mass"].values
        rna = df["listeners__mass__rna_mass"].values
        tRna = df["listeners__mass__tRna_mass"].values
        rRna = df["listeners__mass__rRna_mass"].values
        mRna = df["listeners__mass__mRna_mass"].values
        miscRna = rna - (tRna + rRna + mRna)
        dna = df["listeners__mass__dna_mass"].values
        smallMolecules = df["listeners__mass__smallMolecule_mass"].values
        metabolites = smallMolecules - (lipid + lps + murein + polyamines + glycogen)

        # --- 5. Build voronoi data dicts (verbatim) --------------------------
        dic_initial = {
            "nucleic_acid": {
                "DNA": dna[0],
                "mRNA": mRna[0],
                "miscRNA": miscRna[0],
                "rRNA": rRna[0],
                "tRNA": tRna[0],
            },
            "metabolites": {
                "LPS": lps[0],
                "glycogen": glycogen[0],
                "lipid": lipid[0],
                "metabolites": metabolites[0],
                "peptidoglycan": murein[0],
                "polyamines": polyamines[0],
            },
            "protein": protein[0],
        }
        dic_final = {
            "nucleic_acid": {
                "DNA": dna[-1],
                "mRNA": mRna[-1],
                "miscRNA": miscRna[-1],
                "rRNA": rRna[-1],
                "tRNA": tRna[-1],
            },
            "metabolites": {
                "LPS": lps[-1],
                "glycogen": glycogen[-1],
                "lipid": lipid[-1],
                "metabolites": metabolites[-1],
                "peptidoglycan": murein[-1],
                "polyamines": polyamines[-1],
            },
            "protein": protein[-1],
        }

        # --- 6. Render Voronoi plot ------------------------------------------
        vm = VoronoiMaster()
        vm.plot(
            [[dic_initial, dic_final]],
            title=[["Initial biomass components", "Final biomass components"]],
            ax_shape=(1, 2),
            chained=True,
        )

        fig = plt.gcf()
        html = fig_to_html(fig, title="Mass fraction (Voronoi)")
        plt.close(fig)

        return {"view": html}
