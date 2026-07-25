def _frame(t, cells):
    return {"time": float(t), "cells": cells}


def test_single_division_stats():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # mother "m" grows 2->4 um over t=0..2, divides at t=3 into d1,d2
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m": {"mass": 150.0, "length": 3.0}}),
        _frame(2, {"m": {"mass": 200.0, "length": 4.0}}),
        _frame(3, {"d1": {"mass": 100.0, "length": 2.0},
                   "d2": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 1
    assert out["size_at_division"]["length"] == [4.0]
    assert out["lineage"] == {"d1": "m", "d2": "m"}
    added = out["added_length"][0]
    assert added["birth_length"] == 2.0 and added["delta_length"] == 2.0


def test_interdivision_time_between_generations():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(10, {"a": {"mass": 100.0, "length": 2.0},
                    "b": {"mass": 100.0, "length": 2.0}}),
        _frame(25, {"a": {"mass": 100.0, "length": 2.0},
                    "c": {"mass": 100.0, "length": 2.0},
                    "d": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    # "b" was born at t=10, divided at t=25 -> interdivision 15
    assert 15.0 in out["interdivision_time"]


def test_multi_mother_division_same_frame_has_no_lineage():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # both m1 and m2 divide in the same sampled frame (t=2) -> per-mother
    # stats are still recorded, but lineage attribution is ambiguous
    # (no spatial info) so lineage must stay empty.
    traj = [
        _frame(0, {"m1": {"mass": 100.0, "length": 2.0},
                   "m2": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m1": {"mass": 150.0, "length": 3.0},
                   "m2": {"mass": 150.0, "length": 3.0}}),
        _frame(2, {"m1a": {"mass": 100.0, "length": 2.0},
                   "m1b": {"mass": 100.0, "length": 2.0},
                   "m2a": {"mass": 100.0, "length": 2.0},
                   "m2b": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 2
    assert sorted(out["size_at_division"]["length"]) == [3.0, 3.0]
    assert out["lineage"] == {}


def test_no_divisions_is_empty_panel():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m": {"mass": 110.0, "length": 2.2}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 0
    assert out["size_at_division"]["length"] == []
    assert out["exchange"] is None  # no exchange present in frames
