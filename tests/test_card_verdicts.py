from scripts._compare.report_cards import CardContext, render

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


def _ctx():
    return CardContext(config_name="basal", variant=0, v2_dir="", ve_dir="",
                       seeds=1, gens=1, per_obs=_PER_OBS)


def _verdict_section(sections):
    hits = [s for s in sections if "verdict_axes" in s]
    assert len(hits) == 1, "exactly one section must carry the verdict"
    return hits[0]


def test_standard_card_emits_verdict_and_axes():
    sec = _verdict_section(render("standard", _ctx()))
    assert sec["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    ids = {a["id"] for a in sec["verdict_axes"]}
    assert any(i.startswith("standard.") for i in ids)
    # not_compared rows map to 'ungraded', never the literal 'not_compared'.
    assert all(a["verdict"] != "not_compared" for a in sec["verdict_axes"])
    # the one matched observable is within_tol -> card overall within_tol.
    assert sec["verdict"] == "within_tol"


def test_statistical_card_emits_verdict_and_axes():
    sec = render("statistical", _ctx())[0]
    assert "verdict_axes" in sec and isinstance(sec["verdict_axes"], list)
    assert sec["verdict"] in {"within_tol", "drift", "mismatch", "ungraded"}
    # Fixture has 5 rna_mass seeds -> at least one graded axis.
    assert len(sec["verdict_axes"]) > 0
    assert any(a["verdict"] != "ungraded" for a in sec["verdict_axes"])
    # ~1% rna_mass difference (100 vs 101 mean) -> within_tol.
    assert sec["verdict"] == "within_tol"
    # Every verdict axis must carry exactly the required 6 keys.
    assert all({"id", "label", "verdict", "value", "meter", "detail"} <= set(a)
               for a in sec["verdict_axes"])
