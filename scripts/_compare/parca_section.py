"""ParCa / sim_data comparison rows for the harness report.

Per-step diffing is delegated to the existing scripts/parca_compare.py;
this module adds a final-sim_data field-by-field diff. It REUSES that
module's extraction helpers (`_reach`, `_as_array`) so it correctly handles
both the v2 dict-of-subsystems checkpoint form (which has no top-level
`process` key) and the vEcoli SimulationDataEcoli object form, plus Unum
quantities and structured arrays.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts._compare.stats import compare_series
from scripts.parca_compare import _reach, _as_array

# Attr paths mirror scripts/parca_compare.py SCALARS + DISTRIBUTIONS.
_SCALARS = [
    ("mass.avg_cell_dry_mass_init", ("mass", "avg_cell_dry_mass_init")),
    ("mass.avg_cell_dry_mass", ("mass", "avg_cell_dry_mass")),
    ("mass.avg_cell_water_mass_init", ("mass", "avg_cell_water_mass_init")),
    ("mass.fitAvgSolubleTargetMolMass", ("mass", "fitAvgSolubleTargetMolMass")),
    ("constants.darkATP", ("constants", "darkATP")),
]
_DISTRIBUTIONS = [
    ("RNA expression — basal",
     ("process", "transcription", "rna_expression", "basal")),
    ("RNA synthesis prob — basal",
     ("process", "transcription", "rna_synth_prob", "basal")),
    ("RNA deg rates",
     ("process", "transcription", "rna_data", "deg_rate")),
    ("Cistron deg rates",
     ("process", "transcription", "cistron_data", "deg_rate")),
    ("Protein deg rates",
     ("process", "translation", "monomer_data", "deg_rate")),
    ("Translation efficiencies",
     ("process", "translation", "translation_efficiencies_by_monomer")),
    ("Km endoRNase (transcribed)",
     ("process", "transcription", "rna_data", "Km_endoRNase")),
    ("Km endoRNase (mature)",
     ("process", "transcription", "mature_rna_data", "Km_endoRNase")),
]


def _row(label: str, left, right, rel_tol: float) -> dict[str, Any]:
    la, ra = _as_array(left), _as_array(right)
    if la is None or ra is None:
        return {"label": label, "left": "n/a", "right": "n/a",
                "verdict": "not_compared",
                "reason": "field missing or non-numeric on one side"}
    r = compare_series(la, ra, rel_tol=rel_tol)
    return {"label": label,
            "left": np.array2string(np.atleast_1d(la), threshold=4),
            "right": np.array2string(np.atleast_1d(ra), threshold=4),
            **r}


def final_sim_data_diff(left, right, *, rel_tol: float) -> list[dict[str, Any]]:
    """Diff curated scalar + distribution fields of two sim_data objects.

    ``left`` / ``right`` may each be either a SimulationDataEcoli object
    (vEcoli) or the v2 dict-of-subsystems checkpoint — `_reach` normalizes
    both.
    """
    rows = []
    for label, path in _SCALARS:
        row = _row(label, _reach(left, path), _reach(right, path), rel_tol)
        row["group"] = "scalars"
        rows.append(row)
    for label, path in _DISTRIBUTIONS:
        row = _row(label, _reach(left, path), _reach(right, path), rel_tol)
        row["group"] = "distributions"
        rows.append(row)
    return rows
