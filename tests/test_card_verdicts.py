from _card_helpers import _state, _run_card

# One within-tol observable; the rest are absent -> 'not_compared' -> 'ungraded'.
_PER_OBS = {"rna_mass": [
    {"median_rel": 0.02, "max_rel": 0.05, "init_ve": 100.0, "init_v2": 101.0,
     "init_t": 60.0, "ve_mean": 100.0, "v2_mean": 101.0},
    {"median_rel": 0.01, "max_rel": 0.04, "init_ve": 99.5, "init_v2": 100.5,
     "init_t": 60.0, "ve_mean": 99.5, "v2_mean": 100.5},
    {"median_rel": 0.02, "max_rel": 0.05, "init_ve": 100.2, "init_v2": 101.2,
     "init_t": 60.0, "ve_mean": 100.2, "v2_mean": 101.2},
    {"median_rel": 0.01, "max_rel": 0.03, "init_ve": 100.8, "init_v2": 101.8,
     "init_t": 60.0, "ve_mean": 100.8, "v2_mean": 101.8},
    {"median_rel": 0.01, "max_rel": 0.04, "init_ve": 99.9, "init_v2": 100.9,
     "init_t": 60.0, "ve_mean": 99.9, "v2_mean": 100.9},
]}


def test_parca_card_emits_verdict_and_axes():
    out = _run_card("parca", _state(_PER_OBS))
    # the t~0 rna_mass init (~1%) is graded -> a parca.* axis, within tolerance
    assert any(a["id"].startswith("parca.") for a in out["axes"])
    assert out["verdict"] == "within_tol"


def test_standard_card_emits_verdict_and_axes():
    out = _run_card("standard", _state(_PER_OBS))
    assert out["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    ids = {a["id"] for a in out["axes"]}
    assert any(i.startswith("standard.") for i in ids)
    # not_compared rows map to 'ungraded', never the literal 'not_compared'.
    assert all(a["verdict"] != "not_compared" for a in out["axes"])
    # the one matched observable is within_tol -> card overall within_tol.
    assert out["verdict"] == "within_tol"


def test_statistical_card_emits_verdict_and_axes():
    out = _run_card("statistical", _state(_PER_OBS))
    assert isinstance(out["axes"], list)
    assert out["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    # Fixture has 5 rna_mass seeds -> at least one graded axis.
    assert len(out["axes"]) > 0
    assert any(a["verdict"] != "ungraded" for a in out["axes"])
    # ~1% rna_mass difference (100 vs 101 mean) -> within_tol.
    assert out["verdict"] == "within_tol"
    # Every verdict axis must carry exactly the required 6 keys.
    assert all({"id", "label", "verdict", "value", "meter", "detail"} <= set(a)
               for a in out["axes"])
