"""EcoliPackStep wiring for precise RNAP placement + replication state: the
new input ports exist and update() threads their live values through to
pack_from_state. Fast (no sim, no cache) — mirrors tests/test_pack_step.py's
monkeypatch style.
"""
from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.structural import pack_step as ps


def _step(monkeypatch, snapshots, calls):
    def fake_pack_from_state(out_dir, name, counts, volume_fl, **k):
        calls.append((name, k.get("rnaps"), k.get("n_chromosomes"), k.get("fork_fraction")))
        return {"placements": [1]}

    monkeypatch.setattr(ps, "pack_from_state", fake_pack_from_state)
    monkeypatch.setattr(ps, "bulk_to_counts", lambda bulk: {"X": 1})
    return ps.EcoliPackStep(config={"snapshots": snapshots, "study": "s",
                                     "out_dir": "/tmp/o", "epsilon_s": 1.0})


def _fc(rows):
    return np.array(rows, dtype=[("_entryState", "i1"), ("division_time", "f8"),
                                 ("domain_index", "i4")])


def _rnap(rows):
    return np.array(rows, dtype=[("_entryState", "i1"), ("domain_index", "i4"),
                                 ("coordinates", "i8"), ("is_forward", "?")])


def _replisome(rows):
    return np.array(rows, dtype=[("_entryState", "i1"), ("coordinates", "i8")])


def _bulk():
    """A minimal real bulk_array (structured, not a bare list) — bulk_to_locations
    (called unconditionally by EcoliPackStep.update()) indexes it by field name."""
    return np.array([], dtype=[("id", "U1"), ("count", "i8")])


@pytest.mark.fast
def test_inputs_declares_rnap_and_replication_ports():
    step = ps.EcoliPackStep(config={"snapshots": {}, "study": "s", "out_dir": "/tmp/o"})
    ports = step.inputs()
    assert ports["active_RNAP"] == "active_RNAP"
    assert ports["active_replisome"] == "active_replisome"
    assert ports["chromosome_domain"] == "chromosome_domain"
    assert ports["full_chromosome"] == "full_chromosome"   # unchanged


@pytest.mark.fast
def test_update_threads_live_rnap_and_replication_state_into_pack(monkeypatch):
    calls = []
    step = _step(monkeypatch, {"initial": 10.0}, calls)

    state = {
        "bulk": _bulk(), "shape": {"volume_fl": 2.0}, "global_time": 10.0,
        "full_chromosome": _fc([(1, 0.0, 0), (1, 0.0, 10)]),   # 2 active chromosomes
        "active_RNAP": _rnap([(1, 0, 12345, True)]),
        "active_replisome": _replisome([(1, 200_000)]),
        "chromosome_domain": np.array([], dtype=[("_entryState", "i1"),
                                                 ("domain_index", "i4"),
                                                 ("child_domains", "i4", (2,))]),
    }
    step.update(state)

    assert len(calls) == 1
    name, rnaps, n_chromosomes, fork_fraction = calls[0]
    assert name == "initial"
    assert n_chromosomes == 2
    assert fork_fraction == pytest.approx(200_000 / 2_320_826)
    assert rnaps == [{"coordinates": 12345, "domain_index": 0, "is_forward": True,
                      "chromosome_index": 0, "is_daughter": False}]


@pytest.mark.fast
def test_update_missing_rnap_ports_defaults_to_empty_not_fabricated(monkeypatch):
    """A state dict without the new ports (e.g. an older wiring) still packs —
    rnaps=[], n_chromosomes=0, fork_fraction=0.0 — never invented."""
    calls = []
    step = _step(monkeypatch, {"initial": 10.0}, calls)

    state = {"bulk": _bulk(), "shape": {"volume_fl": 2.0}, "global_time": 10.0,
             "full_chromosome": _fc([])}
    step.update(state)

    assert len(calls) == 1
    _, rnaps, n_chromosomes, fork_fraction = calls[0]
    assert rnaps == []
    assert n_chromosomes == 0
    assert fork_fraction == 0.0
