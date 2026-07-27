def _frame(t, cells):
    return {"time": float(t), "cells": cells}


def test_single_division_stats():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # mother "m" grows 2->4 um over t=0..2, divides at t=3 into m_0, m_1
    # (the "{mother}_{i}" convention grow_divide/EcoliWCM guarantee).
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m": {"mass": 150.0, "length": 3.0}}),
        _frame(2, {"m": {"mass": 200.0, "length": 4.0}}),
        _frame(3, {"m_0": {"mass": 100.0, "length": 2.0},
                   "m_1": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 1
    assert out["size_at_division"]["length"] == [4.0]
    assert out["lineage"] == {"m_0": "m", "m_1": "m"}
    added = out["added_length"][0]
    assert added["birth_length"] == 2.0 and added["delta_length"] == 2.0


def test_interdivision_time_between_generations():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # "m" divides at t=10 into m_0, m_1; "m_1" divides again at t=25 into
    # m_1_0, m_1_1 while "m_0" persists unchanged.
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(10, {"m_0": {"mass": 100.0, "length": 2.0},
                    "m_1": {"mass": 100.0, "length": 2.0}}),
        _frame(25, {"m_0": {"mass": 100.0, "length": 2.0},
                    "m_1_0": {"mass": 100.0, "length": 2.0},
                    "m_1_1": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    # "m_1" was born at t=10, divided at t=25 -> interdivision 15
    assert 15.0 in out["interdivision_time"]


def test_multi_mother_division_attributed_by_id_prefix():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # both m1 and m2 divide in the same sampled frame (t=2) -> the
    # "{mother}_{i}" id-prefix convention gives exact per-mother
    # attribution, so lineage is fully (and correctly) populated.
    traj = [
        _frame(0, {"m1": {"mass": 100.0, "length": 2.0},
                   "m2": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m1": {"mass": 150.0, "length": 3.0},
                   "m2": {"mass": 150.0, "length": 3.0}}),
        _frame(2, {"m1_0": {"mass": 100.0, "length": 2.0},
                   "m1_1": {"mass": 100.0, "length": 2.0},
                   "m2_0": {"mass": 100.0, "length": 2.0},
                   "m2_1": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 2
    assert sorted(out["size_at_division"]["length"]) == [3.0, 3.0]
    assert out["lineage"] == {"m1_0": "m1", "m1_1": "m1",
                               "m2_0": "m2", "m2_1": "m2"}


def test_removal_not_counted_as_division():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # "m" divides at t=2 into m_0, m_1 in the SAME frame that "w" washes
    # out (e.g. remove_crossing in a mother/daughter machine) with no
    # daughter id. The wash-out must not be counted as a division.
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0},
                   "w": {"mass": 400.0, "length": 8.0}}),
        _frame(1, {"m": {"mass": 150.0, "length": 4.0},
                   "w": {"mass": 400.0, "length": 8.0}}),
        _frame(2, {"m_0": {"mass": 100.0, "length": 2.0},
                   "m_1": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 1
    assert out["size_at_division"]["length"] == [4.0]
    assert 8.0 not in out["size_at_division"]["length"]
    assert out["lineage"] == {"m_0": "m", "m_1": "m"}


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
