import numpy as np

from v2ecoli.structural import pack_step as ps


def _step(monkeypatch, snapshots, calls):
    # These tests isolate the snapshot-FIRING logic; stub every state-extraction
    # helper update() calls (bulk counts/locations + live RNAP/replication) so a
    # minimal fake state suffices and only the timing path is exercised.
    monkeypatch.setattr(ps, "pack_from_state",
        lambda out_dir, name, counts, volume_fl, **k: calls.append((name, volume_fl)) or {"placements": [1]})
    monkeypatch.setattr(ps, "bulk_to_counts", lambda bulk: {"X": 1})
    monkeypatch.setattr(ps, "bulk_to_locations", lambda bulk: {})
    monkeypatch.setattr(ps, "chromosome_state_from_live", lambda fc, rep=None: (1, 0.0))
    monkeypatch.setattr(ps, "rnaps_from_live", lambda *a, **k: [])
    return ps.EcoliPackStep(config={"snapshots": snapshots, "study": "s",
                                     "out_dir": "/tmp/o", "epsilon_s": 1.0})


def _state(t, division_time=None):
    # Real store shape: full_chromosome is a numpy structured array, one row
    # per chromosome copy, with a per-row division_time field (0/unset until
    # scheduled). division_time=None -> no rows yet (not even unscheduled
    # ones) so the array is empty, matching "not scheduled yet".
    if division_time is None:
        fc = np.array([], dtype=[("division_time", "f8")])
    else:
        fc = np.array([(division_time,)], dtype=[("division_time", "f8")])
    return {"bulk": [], "shape": {"volume_fl": 2.0}, "global_time": t, "full_chromosome": fc}


def test_fixed_time_snapshot_fires_once(monkeypatch):
    calls = []
    step = _step(monkeypatch, {"initial": 10.0}, calls)
    step.update(_state(5.0));  assert calls == []          # before the time
    step.update(_state(10.0)); assert [c[0] for c in calls] == ["initial"]  # at/after
    step.update(_state(20.0)); assert [c[0] for c in calls] == ["initial"]  # not re-fired


def test_pre_division_uses_division_time(monkeypatch):
    calls = []
    step = _step(monkeypatch, {"pre-division": "division_time"}, calls)
    step.update(_state(30.0, division_time=None));  assert calls == []   # not scheduled yet
    step.update(_state(30.0, division_time=100.0)); assert calls == []   # scheduled, not near
    step.update(_state(99.5, division_time=100.0)); assert [c[0] for c in calls] == ["pre-division"]  # within epsilon


def test_zero_volume_skips_without_marking_fired(monkeypatch):
    """If a snapshot is due but 'shape' hasn't been populated yet (volume_fl
    <= 0), update() must skip packing this tick (no pack_from_state call)
    WITHOUT marking the snapshot fired, so it retries once shape is ready."""
    calls = []
    step = _step(monkeypatch, {"initial": 10.0}, calls)

    state = _state(10.0)
    state["shape"] = {"volume_fl": 0.0}
    step.update(state)
    assert calls == []          # skipped: not packed
    assert "initial" not in step._fired   # not marked fired -> will retry

    # Once volume_fl is populated, the same (or a later) tick fires normally.
    step.update(_state(10.0))   # default _state() has volume_fl=2.0
    assert [c[0] for c in calls] == ["initial"]
