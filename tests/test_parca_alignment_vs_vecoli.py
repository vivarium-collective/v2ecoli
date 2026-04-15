"""Alignment tests — v2ecoli-ParCa vs the original vEcoli ``fitSimData_1``.

Diffs the v2ecoli-ParCa step checkpoints against the vEcoli reference
pickles in ``out/original_intermediates/sim_data_<stub>.cPickle`` (dumped
by vivarium-ecoli's ``runscripts/parca.py --save-intermediates``).

Only the steps for which both sides have a checkpoint are compared; today
that's steps 1-4.  Steps 5-9 are skipped until vEcoli intermediates for
those stages are regenerated.

These tests exist so any future drift in v2ecoli-ParCa vs the reference
implementation fails CI loudly.  They're the ParCa analogue of
``tests/test_model_behavior.py``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest


# Intermediate pickles from vEcoli's ``--save-intermediates`` run, shipped
# in the repo to keep this test self-contained.  If they're absent the
# tests skip rather than fail — local dev doesn't need the reference side.
REPO_ROOT = Path(__file__).resolve().parent.parent
VECOLI_DIR = REPO_ROOT / 'out' / 'original_intermediates'
V2PARCA_CKPT_DIR = REPO_ROOT / 'out' / 'sim_data'


STEPS_WITH_REFERENCE = [
    ('initialize',        1),
    ('input_adjustments', 2),
    ('basal_specs',       3),
    ('tf_condition_specs', 4),
]


def _load_pickle(p: Path):
    from v2ecoli.processes.parca.data_loader import _install_legacy_pickle_aliases
    _install_legacy_pickle_aliases()
    with open(p, 'rb') as f:
        return pickle.load(f)


def _reach(obj, path):
    """Follow an attribute/key path like ('mass', 'avg_cell_dry_mass')."""
    for seg in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(seg)
        else:
            obj = getattr(obj, seg, None)
    return obj


def _as_array(x):
    if x is None:
        return None
    if hasattr(x, 'asNumber'):
        x = x.asNumber()
    try:
        return np.asarray(x)
    except Exception:
        return None


# Scalar fields that should match across the two implementations at
# better than 1e-3 rel Δ.  Pulled from the comparison report's SCALARS
# table — these are the cross-cut sanity checks.
ALIGNMENT_SCALARS = [
    ('mass', 'avg_cell_dry_mass_init'),
    ('mass', 'avg_cell_dry_mass'),
    ('mass', 'avg_cell_water_mass_init'),
    ('constants', 'darkATP'),
]


@pytest.mark.parametrize('stub,step_n', STEPS_WITH_REFERENCE)
def test_scalar_alignment_vs_vecoli(stub, step_n):
    v2_ckpt = V2PARCA_CKPT_DIR / f'checkpoint_step_{step_n}.pkl'
    vecoli_sd = VECOLI_DIR / f'sim_data_{stub}.cPickle'
    if not v2_ckpt.exists() or not vecoli_sd.exists():
        pytest.skip(f'missing checkpoint(s): v2={v2_ckpt.exists()} '
                    f'vecoli={vecoli_sd.exists()} — run parca_run.py + '
                    f'vivarium-ecoli parca.py --save-intermediates')
    v2_state = _load_pickle(v2_ckpt)
    vecoli = _load_pickle(vecoli_sd)

    checked = 0
    for path in ALIGNMENT_SCALARS:
        a = _reach(v2_state, path)
        b = _reach(vecoli, path)
        if a is None or b is None:
            continue
        a_num = float(a.asNumber()) if hasattr(a, 'asNumber') else float(a)
        b_num = float(b.asNumber()) if hasattr(b, 'asNumber') else float(b)
        # rel Δ with safe denominator
        denom = max(abs(a_num), abs(b_num), 1e-30)
        rel = abs(a_num - b_num) / denom
        assert rel < 1e-3, (
            f'step {step_n} ({stub}) {".".join(path)} drift: '
            f'v2ecoli-parca={a_num:.6g} vecoli={b_num:.6g} rel Δ={rel:.3e}')
        checked += 1
    assert checked > 0, f'no common scalar fields found for step {step_n}'
