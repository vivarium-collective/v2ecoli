import numpy as np

from v2ecoli.library.vivarium_ecoli_engine import _select_bulk_observables


def _bulk_array(rows):
    """A tiny stand-in for the REAL vEcoli ``bulk`` store: a numpy structured
    array with the same field names/dtypes the fork builds (``id`` string,
    ``count``, plus a submass column) — see ecoli.library.initial_conditions."""
    return np.array(
        [(i, c, 0.0) for (i, c) in rows],
        dtype=[("id", "U16"), ("count", np.int64), ("protein_submass", np.float64)],
    )


def test_selects_requested_ids_as_floats():
    obs_bulk = {"A[c]": 10, "B[p]": 3, "C[m]": 0}
    out = _select_bulk_observables(obs_bulk, ["A[c]", "C[m]"])
    assert out == {"A[c]": 10.0, "C[m]": 0.0}


def test_missing_id_defaults_to_zero_not_crash():
    out = _select_bulk_observables({"A[c]": 5}, ["A[c]", "MISSING[x]"])
    assert out == {"A[c]": 5.0, "MISSING[x]": 0.0}


def test_empty_ids_returns_empty():
    assert _select_bulk_observables({"A[c]": 5}, []) == {}


# --- numpy structured-array (real vEcoli bulk store) cases ---

def test_structured_array_selects_counts_as_floats():
    arr = _bulk_array([("A[c]", 10), ("B[p]", 3), ("C[m]", 0)])
    out = _select_bulk_observables(arr, ["A[c]", "C[m]"])
    assert out == {"A[c]": 10.0, "C[m]": 0.0}
    assert all(isinstance(v, float) for v in out.values())


def test_structured_array_missing_id_defaults_to_zero():
    arr = _bulk_array([("A[c]", 5), ("B[p]", 7)])
    out = _select_bulk_observables(arr, ["A[c]", "MISSING[x]", "B[p]"])
    assert out == {"A[c]": 5.0, "MISSING[x]": 0.0, "B[p]": 7.0}


def test_structured_array_empty_ids_returns_empty_no_truthiness_crash():
    # A multi-element structured array would raise "truth value ambiguous" if the
    # function ever did ``obs_bulk or {}`` — it must not.
    arr = _bulk_array([("A[c]", 5), ("B[p]", 7)])
    assert _select_bulk_observables(arr, []) == {}


def test_none_falls_back_to_zeros_no_crash():
    out = _select_bulk_observables(None, ["A[c]", "B[p]"])
    assert out == {"A[c]": 0.0, "B[p]": 0.0}
