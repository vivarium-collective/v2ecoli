"""Canonical DnaA nucleotide-state observables — the SINGLE source of truth.

Every dnaa render script MUST compute the DnaA-ATP / DnaA-ADP fraction and the
total DnaA pool through this module. Do NOT re-derive them inline.

THE DEFINITION (and why):
    DnaA-ATP fraction = (free DnaA-ATP + Σ bound-ATP) / total DnaA
    total DnaA        = apo + free-ATP + free-ADP + Σ bound-ATP + Σ bound-ADP

The fraction is over the WHOLE pool — free PLUS the DnaA bound at the chromosomal /
oriC / promoter boxes. It is NOT the free-only fraction (free-ATP / free pools).

Why this matters: under the active Langmuir box-binding (dnaa-3 onward), binding
pulls DnaA-ATP OUT of the free bulk pool every cycle, so the FREE-only fraction
swings wildly 0->1 while the TOTAL fraction sits stably in the Boesen [0.2,0.5]
band. Two render scripts once computed this two different ways from the SAME run
(dnaa-3 used free-only, dnaa-4 used total), producing contradictory charts that a
reviewer (Rashmi, 2026-06-20) caught. Centralizing the definition here makes that
class of divergence structurally impossible: there is exactly one fraction.

See docs/conventions/dnaa-observable-definitions.md.
"""
from __future__ import annotations

import numpy as np

# --- listener / bulk column identifiers (the schema the runs emit) ---
L = "listeners__replication_data__"
MASS_COL = "listeners__mass__cell_mass"
ORIC_COL = L + "number_of_oric"
BOXES_COL = L + "total_DnaA_boxes"

# free bulk pools (monomer ids in bulk__id / bulk__count)
APO_ID, ATP_ID, ADP_ID = "PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"

# bound pools (per-tick listener columns). High-affinity chromosomal + oriC-high +
# promoter pools carry both ATP and ADP; oriC-low is ATP-only.
BOUND_ATP_COLS = [L + c for c in (
    "chromosomal_high_bound_atp", "oriC_high_bound_atp",
    "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP_COLS = [L + c for c in (
    "chromosomal_high_bound_adp", "oriC_high_bound_adp",
    "promoter_high_bound_adp")]


def _sum(arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    if not arrays:
        return 0.0
    out = np.zeros_like(arrays[0])
    for a in arrays:
        out = out + a
    return out


def decompose(apo, free_atp, free_adp, bound_atp_arrays, bound_adp_arrays):
    """Canonical DnaA pool decomposition + fractions.

    Args are per-tick arrays: ``apo`` / ``free_atp`` / ``free_adp`` are the bulk
    counts (PD03831 / MONOMER0-160 / MONOMER0-4565); ``bound_*_arrays`` are lists
    of the per-pool bound-count listener arrays (see ``BOUND_*_COLS``).

    Returns a dict of arrays: total_dnaa, atp_fraction, adp_fraction, bound_atp,
    bound_adp, free_atp, free_adp, apo. The fraction is (free+bound)/total.
    """
    apo = np.asarray(apo, dtype=float)
    free_atp = np.asarray(free_atp, dtype=float)
    free_adp = np.asarray(free_adp, dtype=float)
    bound_atp = _sum(bound_atp_arrays)
    bound_adp = _sum(bound_adp_arrays)
    total = apo + free_atp + free_adp + bound_atp + bound_adp
    denom = np.maximum(total, 1.0)
    return {
        "total_dnaa": total,
        "atp_fraction": (free_atp + bound_atp) / denom,
        "adp_fraction": (free_adp + bound_adp) / denom,
        "bound_atp": bound_atp,
        "bound_adp": bound_adp,
        "free_atp": free_atp,
        "free_adp": free_adp,
        "apo": apo,
    }


def decompose_from_frame(df):
    """Convenience: decompose directly from a polars DataFrame that already has the
    bulk pools as columns ``apo``/``atp``/``adp`` and the ``BOUND_*_COLS`` listener
    columns. Missing bound columns are treated as zero (and noted by the caller)."""
    cols = set(df.columns)
    def col(name):
        return df[name].to_numpy() if name in cols else None
    bound_atp = [df[c].to_numpy() for c in BOUND_ATP_COLS if c in cols]
    bound_adp = [df[c].to_numpy() for c in BOUND_ADP_COLS if c in cols]
    return decompose(df["apo"].to_numpy(), df["atp"].to_numpy(),
                     df["adp"].to_numpy(), bound_atp, bound_adp)
