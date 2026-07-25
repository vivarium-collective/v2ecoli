def test_run_mother_machine_smoke():
    from v2ecoli.colony_bench.devices import run_device
    out = run_device("mother_machine", n_steps=4, dt=30.0, config={"n_channels": 3})
    assert out["n_final"] >= 1
    assert len(out["trajectory"]) == 5  # initial + 4 steps
    assert "n_division_events" in out["phenotypes"]
    assert out["meta"]["env_size"] > 0
    assert out["meta"]["barriers"]  # channel walls present
    # geometry captured for the gif
    frame = out["trajectory"][0]["cells"]
    first = next(iter(frame.values()))
    assert first["location"] is not None and "radius" in first


def test_run_daughter_machine_smoke():
    from v2ecoli.colony_bench.devices import run_device
    out = run_device("daughter_machine", n_steps=4, dt=30.0, config={"env_size": 30})
    assert out["n_final"] >= 1
    assert len(out["trajectory"]) == 5
    assert out["meta"]["env_size"] == 30


def test_unknown_device_raises():
    import pytest
    from v2ecoli.colony_bench.devices import build_device
    with pytest.raises(ValueError, match="unknown device"):
        build_device("bogus")
