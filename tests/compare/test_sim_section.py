import numpy as np

from scripts._compare.sim_section import compare_observables, OBSERVABLES


def test_observables_cover_four_families():
    families = {o["family"] for o in OBSERVABLES}
    assert families == {"mass_growth", "molecule_counts",
                        "listeners", "division_lineage"}


def test_compare_observables_builds_rows_with_verdicts():
    left = {"dry_mass": np.array([1.0, 2.0, 4.0]),
            "growth_rate": np.array([0.1, 0.1, 0.1])}
    right = {"dry_mass": np.array([1.0, 2.0, 4.0]),
             "growth_rate": np.array([0.2, 0.2, 0.2])}
    rows = compare_observables(left, right,
                               keys=["dry_mass", "growth_rate"],
                               rel_tol=1e-3)
    by = {r["label"]: r for r in rows}
    assert by["dry_mass"]["verdict"] == "within_tol"
    assert by["growth_rate"]["verdict"] == "mismatch"


def test_compare_observables_missing_key_not_compared():
    rows = compare_observables({"a": np.array([1.0])}, {},
                               keys=["a"], rel_tol=1e-3)
    assert rows[0]["verdict"] == "not_compared"
