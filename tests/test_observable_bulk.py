"""Bulk-observable comparison hook: both engines surface declared bulk molecule
counts under listeners.observable_bulk.<id> so the two-arm report can grade them.

Candidate (v2ecoli): run_multigen_xarray injects the selection from the bulk
record. Reference (vecoli engine): VivariumEcoliProcess emits them directly. This
tests the candidate selection helper (pure) + that the reference process declares
the unified listeners.observable_bulk path in its outputs schema.
"""

from __future__ import annotations

import numpy as np


def test_with_observable_bulk_selects_by_id_and_is_pure():
    from v2ecoli.library.xarray_run import _with_observable_bulk
    rec = np.array([("VIOLACEIN[c]", 199000), ("GLC[p]", 5),
                    ("mecillinam[p]-EG10606-MONOMER[i]", 96)],
                   dtype=[("id", "U60"), ("count", "<i8")])
    agent = {"bulk": rec, "listeners": {"mass": {"cell_mass": 1200.0}}}
    out = _with_observable_bulk(
        agent, ["VIOLACEIN[c]", "mecillinam[p]-EG10606-MONOMER[i]", "ABSENT[c]"])
    ob = out["listeners"]["observable_bulk"]
    assert ob["VIOLACEIN[c]"] == 199000.0
    assert ob["mecillinam[p]-EG10606-MONOMER[i]"] == 96.0
    assert ob["ABSENT[c]"] == 0.0          # missing id -> 0.0 (continuous trace)
    assert out["listeners"]["mass"]["cell_mass"] == 1200.0  # existing preserved
    assert "observable_bulk" not in agent["listeners"]      # live state untouched


def test_with_observable_bulk_noop_without_ids_or_bulk():
    from v2ecoli.library.xarray_run import _with_observable_bulk
    agent = {"listeners": {}}
    assert _with_observable_bulk(agent, []) is agent          # no ids -> same object
    assert _with_observable_bulk({"listeners": {}}, ["X"]) == {"listeners": {}}  # no bulk


def test_reference_process_declares_unified_observable_bulk_path(monkeypatch):
    """VivariumEcoliProcess.outputs() must put declared bulk ids under
    listeners.observable_bulk.<id> (NOT a separate bulk root), so it matches the
    candidate's injected path."""
    from v2ecoli.library import vivarium_ecoli_engine as ve
    # avoid building a real EcoliSim: stub the handle
    monkeypatch.setattr(ve.VivariumEcoliProcess, "_PENDING_HANDLE",
                        type("H", (), {"engine": object()})())
    from bigraph_schema import allocate_core
    proc = ve.VivariumEcoliProcess(
        config={"sim_data_path": "x", "observable_bulk_ids": ["VIOLACEIN[c]"]},
        core=allocate_core())
    out = proc.outputs()
    assert "observable_bulk" in out["listeners"]
    assert "VIOLACEIN[c]" in out["listeners"]["observable_bulk"]
    assert "bulk" not in out  # no separate bulk root — unified under listeners
