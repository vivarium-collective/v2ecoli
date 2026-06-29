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

# The named growth conditions whose per-condition fits we diff explicitly.
# Condition-specific fields (rna_expression, rna_synth_prob, doubling time) are
# checked for EACH so a drift confined to one medium (e.g. an acetate-only
# transcription-program regression) can't hide behind a basal-only check.
_CONDITIONS = ("basal", "with_aa", "succinate", "no_oxygen", "acetate")

# Attr paths mirror scripts/parca_compare.py SCALARS + DISTRIBUTIONS.
_SCALARS = [
    ("mass.avg_cell_dry_mass_init", ("mass", "avg_cell_dry_mass_init")),
    ("mass.avg_cell_dry_mass", ("mass", "avg_cell_dry_mass")),
    ("mass.avg_cell_water_mass_init", ("mass", "avg_cell_water_mass_init")),
    ("mass.fitAvgSolubleTargetMolMass", ("mass", "fitAvgSolubleTargetMolMass")),
    ("constants.darkATP", ("constants", "darkATP")),
    ("constants.growth_associated_maintenance",
     ("constants", "growth_associated_maintenance")),
    ("constants.non_growth_associated_maintenance",
     ("constants", "non_growth_associated_maintenance")),
    ("constants.c_period", ("constants", "c_period")),
    ("constants.d_period", ("constants", "d_period")),
] + [
    (f"doubling time — {c}", ("condition_to_doubling_time", c))
    for c in _CONDITIONS
]
_DISTRIBUTIONS = [
    # Per-condition transcription program — the condition-specific fit that
    # drives a medium's initial state. Checked for every named condition.
    *[(f"RNA expression — {c}",
       ("process", "transcription", "rna_expression", c)) for c in _CONDITIONS],
    *[(f"RNA synthesis prob — {c}",
       ("process", "transcription", "rna_synth_prob", c)) for c in _CONDITIONS],
    # ppGpp-dependent expression endpoints (drive growth-rate regulation).
    ("RNA expression — free (no ppGpp)",
     ("process", "transcription", "exp_free")),
    ("RNA expression — ppGpp-bound",
     ("process", "transcription", "exp_ppgpp")),
    # Degradation + translation: the inputs that convert expression to counts.
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
