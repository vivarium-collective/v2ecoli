"""Proves the Task B0 report-card fixture (tests/fixtures/redux_cards/) is
readable end-to-end: make_card_state() returns a CARD_INPUTS-shaped dict,
and the local zarr stores it points at yield real trajectory data via the
SAME local-path reader (read_pbg_local) the new report cards will use.
"""
import os

from scripts._compare.report_cards import CARD_INPUTS
from scripts.compare_matched_trajectories import read_pbg_local

# Bare sibling import (matches this dir's `_card_helpers` convention), NOT
# `from tests.conftest import ...`: this venv has a third-party "tests"
# top-level package (nose, in v2ecoli/.venv/site-packages) that shadows the
# repo's tests/ directory for dotted imports — Python's regular-package
# search prefers the installed package over the repo's implicit namespace
# package once it finds an __init__.py later on sys.path. pytest's default
# "prepend" import mode already puts this directory on sys.path (tests/ has
# no __init__.py of its own), so the bare name resolves correctly.
from conftest import make_card_state


def test_make_card_state_has_all_card_inputs_keys():
    state = make_card_state()
    assert set(CARD_INPUTS.keys()) <= set(state.keys())
    assert state["name"] == "metabolism_redux_basal"
    assert state["condition"] == "basal"
    assert state["v2_dir"] == state["ve_dir"]
    assert os.path.isdir(state["v2_dir"])


def test_fixture_zarr_stores_read_nonempty_cell_mass():
    state = make_card_state()
    v2_path = os.path.join(state["v2_dir"], "v2ecoli_seed00.zarr")
    ve_path = os.path.join(state["ve_dir"], "vecoli_seed00.zarr")

    v2 = read_pbg_local(v2_path, ["cell_mass"])
    ve = read_pbg_local(ve_path, ["cell_mass"])

    assert "cell_mass" in v2 and "cell_mass" in ve
    v2_t, v2_v = v2["cell_mass"]
    ve_t, ve_v = ve["cell_mass"]
    assert len(v2_t) > 0 and len(v2_v) > 0
    assert len(ve_t) > 0 and len(ve_v) > 0
