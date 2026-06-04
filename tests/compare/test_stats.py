import numpy as np
from scripts._compare.stats import compare_series


def test_identical_series_within_tol():
    x = np.array([1.0, 2.0, 3.0])
    r = compare_series(x, x.copy(), rel_tol=1e-6)
    assert r["verdict"] == "within_tol"
    assert r["max_rel"] == 0.0


def test_small_drift_flagged_as_drift():
    x = np.array([1.0, 2.0, 3.0])
    y = x * 1.01  # 1% off, above a 1e-3 tol but not wildly different
    r = compare_series(x, y, rel_tol=1e-3)
    assert r["verdict"] == "drift"
    assert 0.009 < r["max_rel"] < 0.011


def test_large_difference_flagged_as_mismatch():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    r = compare_series(x, y, rel_tol=1e-3, mismatch_rel=0.5)
    assert r["verdict"] == "mismatch"


def test_shape_mismatch_returns_not_compared():
    r = compare_series(np.array([1.0, 2.0]), np.array([1.0]), rel_tol=1e-3)
    assert r["verdict"] == "not_compared"
    assert "shape" in r["reason"]
