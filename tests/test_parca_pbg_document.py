"""Drift check for ``models/parca.pbg``.

The file is the structural document for the 9-Step ParCa pipeline —
addresses + port wiring, no fitted data. It ships in-tree so other
tooling (visualization, comparisons) can read the pipeline shape
without importing v2ecoli at all.

When STORE_PATH or any step's INPUT_PORTS / OUTPUT_PORTS changes, the
file must be regenerated (workflow_report's `## Update .pbg model
files` block does this automatically). This test fails fast when the
committed file falls out of sync.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v2ecoli.pbg import save_pbg_doc
from v2ecoli.processes.parca.composite import (
    STEP_ORDER, build_parca_document,
)


pytestmark = pytest.mark.fast


REPO_ROOT = Path(__file__).resolve().parent.parent
PBG_PATH = REPO_ROOT / 'models' / 'parca.pbg'


def test_parca_pbg_exists():
    assert PBG_PATH.exists(), (
        f"{PBG_PATH} is missing. Run `python reports/workflow_report.py` "
        "or regenerate via "
        "`save_pbg_doc(build_parca_document(), 'models/parca.pbg')`.")


def test_parca_pbg_matches_live_spec(tmp_path):
    """The committed .pbg must round-trip to the live spec."""
    fresh_path = tmp_path / 'parca.pbg'
    save_pbg_doc(build_parca_document(), str(fresh_path))

    with open(PBG_PATH) as f:
        committed = json.load(f)
    with open(fresh_path) as f:
        fresh = json.load(f)

    assert committed == fresh, (
        f"{PBG_PATH} is out of date with the live spec. Regenerate via "
        "`save_pbg_doc(build_parca_document(), 'models/parca.pbg')`.")


def test_parca_pbg_has_all_steps():
    with open(PBG_PATH) as f:
        doc = json.load(f)
    for step in STEP_ORDER:
        assert step in doc, f"missing step entry: {step}"
        entry = doc[step]
        assert entry.get('_type') == 'step'
        assert entry.get('address', '').startswith('local:')
        assert isinstance(entry.get('inputs'), dict)
        assert isinstance(entry.get('outputs'), dict)
