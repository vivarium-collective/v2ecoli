from v2ecoli.library.phenotype_sweep import collect_sweep, sweep_endpoints


RUNS = [
    {"label": "v0", "series": {"bulk.X": [0.0, 0.0, 0.0], "growth": [1.0, 1.0, 1.0]}},
    {"label": "v1", "series": {"bulk.X": [0.0, 5.0, 9.0], "growth": [1.0, 0.8, 0.5]}},
]


def test_collect_groups_by_path_then_label():
    out = collect_sweep(RUNS, ["bulk.X", "growth"])
    assert out["bulk.X"] == {"v0": [0.0, 0.0, 0.0], "v1": [0.0, 5.0, 9.0]}
    assert out["growth"]["v1"] == [1.0, 0.8, 0.5]


def test_missing_path_skipped_not_crash():
    out = collect_sweep(RUNS, ["bulk.X", "absent"])
    assert "absent" not in out
    assert set(out) == {"bulk.X"}


def test_endpoints_take_last_value():
    ep = sweep_endpoints(collect_sweep(RUNS, ["bulk.X", "growth"]))
    assert ep["bulk.X"] == {"v0": 0.0, "v1": 9.0}
    assert ep["growth"] == {"v0": 1.0, "v1": 0.5}
