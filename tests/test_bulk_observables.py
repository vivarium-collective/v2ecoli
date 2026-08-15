from v2ecoli.library.vivarium_ecoli_engine import _select_bulk_observables


def test_selects_requested_ids_as_floats():
    obs_bulk = {"A[c]": 10, "B[p]": 3, "C[m]": 0}
    out = _select_bulk_observables(obs_bulk, ["A[c]", "C[m]"])
    assert out == {"A[c]": 10.0, "C[m]": 0.0}


def test_missing_id_defaults_to_zero_not_crash():
    out = _select_bulk_observables({"A[c]": 5}, ["A[c]", "MISSING[x]"])
    assert out == {"A[c]": 5.0, "MISSING[x]": 0.0}


def test_empty_ids_returns_empty():
    assert _select_bulk_observables({"A[c]": 5}, []) == {}
