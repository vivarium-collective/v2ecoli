from v2ecoli.structural import pack_step as ps


def _step(monkeypatch, snapshots, calls):
    monkeypatch.setattr(ps, "pack_from_state",
        lambda out_dir, name, counts, volume_fl, **k: calls.append((name, volume_fl)) or {"placements": [1]})
    monkeypatch.setattr(ps, "bulk_to_counts", lambda bulk: {"X": 1})
    return ps.EcoliPackStep(config={"snapshots": snapshots, "study": "s",
                                     "out_dir": "/tmp/o", "epsilon_s": 1.0})


def _state(t, division_time=None):
    fc = {"division_time": division_time} if division_time is not None else {}
    return {"bulk": [], "shape": {"volume_fl": 2.0}, "global_time": t, "full_chromosomes": fc}


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
