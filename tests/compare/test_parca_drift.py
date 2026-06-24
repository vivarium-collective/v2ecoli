"""ParCa drift guard: v2ecoli's calibrated fit must stay faithful to vEcoli's.

The vEcoli<->v2ecoli comparison repeatedly showed that *static* ParCa fidelity
is the foundation: when it silently drifts (e.g. the
``balance_translation_efficiencies`` compartment-tag bug, which shifted 27
ribosomal-protein efficiencies), downstream sim behavior diverges in ways that
are hard to attribute. This module locks the fit down by diffing v2ecoli's
hydrated SimulationDataEcoli against vEcoli's, field by field, across all named
growth conditions — the same comparison the harness report's dedicated
"ParCa / sim_data" section renders.

It is a two-engine comparison, so it runs only where both fits are present
(skips otherwise): v2ecoli's ParCa state and vEcoli's ``simData.cPickle``. Point
it at non-default locations with ``V2_PARCA_STATE`` / ``VECOLI_SIMDATA``.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from scripts._compare.parca_section import final_sim_data_diff, _CONDITIONS
from scripts.parca_compare import _reach, _as_array

# v2ecoli ParCa state — must be built with CURRENT code (not the shipped
# pre-fix fixture). Prefer an explicit env path, then the fixed full-mode state,
# then a freshly-built cache.
_V2_CANDIDATES = [
    os.environ.get("V2_PARCA_STATE", ""),
    "out/parca_fixed_full/parca_state.pkl",
    "out/cache/sim_data_cache.dill",
]
_VECOLI = os.environ.get(
    "VECOLI_SIMDATA",
    os.path.expanduser("~/code/vEcoli/out/kb/simData.cPickle"))


def _v2_state_path():
    for p in _V2_CANDIDATES:
        if p and Path(p).exists():
            return p
    return None


pytestmark = pytest.mark.skipif(
    _v2_state_path() is None or not Path(_VECOLI).exists(),
    reason="needs both a current v2ecoli ParCa state and vEcoli simData "
           "(set V2_PARCA_STATE / VECOLI_SIMDATA); skipped where absent")


def _load_pair():
    import pickle
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)
    p = _v2_state_path()
    try:
        state = load_parca_state(p)
    except Exception:
        import dill
        with open(p, "rb") as f:
            state = dill.load(f)
    v2 = hydrate_sim_data_from_state(state)
    with open(_VECOLI, "rb") as f:
        ve = pickle.load(f)
    return v2, ve


def test_no_parca_quantity_mismatches_vecoli():
    """No curated ParCa quantity may MISMATCH vEcoli (drift is tolerated, a
    mismatch is a regression). Covers masses, maintenance, c/d period, per-
    condition expression + synthesis prob, deg rates, translation efficiencies,
    and endoRNase Km — 30+ quantities across all named conditions."""
    v2, ve = _load_pair()
    rows = final_sim_data_diff(v2, ve, rel_tol=0.05)
    assert not [r for r in rows if r["verdict"] == "not_compared"], \
        "some ParCa fields stopped resolving on one side"
    mism = [r["label"] for r in rows if r["verdict"] == "mismatch"]
    assert not mism, f"ParCa drift vs vEcoli (mismatch): {mism}"


def test_translation_efficiencies_match_exactly():
    """Regression guard for the balance_translation_efficiencies compartment-tag
    bug: per-monomer translation efficiencies must equal vEcoli's to ~machine
    precision (the bug left 27 r-protein efficiencies up to 0.24 off)."""
    v2, ve = _load_pair()
    path = ("process", "translation", "translation_efficiencies_by_monomer")
    a = _as_array(_reach(v2, path))
    b = _as_array(_reach(ve, path))
    assert a is not None and b is not None and a.shape == b.shape
    assert np.abs(a - b).max() < 1e-6, \
        f"translation efficiencies drifted from vEcoli (max|Δ|={np.abs(a-b).max():.3g})"


def test_per_condition_expression_matches():
    """The condition-specific transcription program must match vEcoli for EVERY
    named medium — a per-medium drift (e.g. acetate-only) can't hide behind a
    basal-only check."""
    v2, ve = _load_pair()
    for c in _CONDITIONS:
        path = ("process", "transcription", "rna_expression", c)
        a = _as_array(_reach(v2, path))
        b = _as_array(_reach(ve, path))
        assert a is not None and b is not None, f"rna_expression[{c}] missing"
        tot = float(np.abs(a - b).sum())
        assert tot < 1e-3, f"rna_expression[{c}] drifted from vEcoli (|Δ|={tot:.3g})"
