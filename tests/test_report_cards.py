from scripts._compare.report_cards import REPORT_CARD_STEPS
from _card_helpers import _state, _run_card


def test_builtin_cards_registered():
    for name in ("standard", "statistical", "parca", "config_diff", "config"):
        assert f"{name}_report_card" in REPORT_CARD_STEPS


def test_statistical_card_returns_graded_section():
    per_obs = {"cell_mass": [{"ve_mean": 1.0, "v2_mean": 1.0, "median_rel": 0.0,
                               "max_rel": 0.0, "init_ve": 1.0, "init_v2": 1.0, "init_t": 0.0},
                              {"ve_mean": 1.1, "v2_mean": 1.05, "median_rel": 0.05,
                               "max_rel": 0.05, "init_ve": 1.1, "init_v2": 1.05, "init_t": 0.0}],
               "growth_rate": [{"ve_mean": 2e-4, "v2_mean": 2e-4, "median_rel": 0.0,
                                 "max_rel": 0.0, "init_ve": 2e-4, "init_v2": 2e-4, "init_t": 0.0},
                                {"ve_mean": 2.1e-4, "v2_mean": 2.05e-4, "median_rel": 0.05,
                                 "max_rel": 0.05, "init_ve": 2.1e-4, "init_v2": 2.05e-4, "init_t": 0.0}]}
    out = _run_card("statistical", _state(per_obs, name="basal", seeds=2))
    assert out["card_html"] and out["verdict"] in (
        "within_tol", "drift", "mismatch", "ungraded")


def test_assemble_sections_from_studies(monkeypatch, tmp_path):
    from scripts import comparison_report_card as crc
    from scripts._compare.study_spec import StudySpec
    # one study, cards ["parca","standard"] -> overview + parca + (runs+eval).
    # per_obs uses the REAL shape build() produces: obs -> list of per-seed dicts
    # (each a full _matched() stat with init_*/*_mean/median_rel/max_rel).
    seed_stat = {"seed": 0, "init_t": 0.0, "init_v2": 1.0, "init_ve": 1.0,
                 "init_rel": 0.0, "v2_mean": 1.0, "ve_mean": 1.0,
                 "median_rel": 0.0, "max_rel": 0.0, "n": 1}
    cond_data = {"basal": ({"cell_mass": [seed_stat]}, {}, {})}
    spec = StudySpec(name="basal", condition="basal", seeds=1, gens=4,
                     cards=["parca", "standard"], invest_name="inv",
                     v2_cache="c", ve_cache="c", study_path="/x")
    secs = crc.assemble_from_studies(
        [spec], cond_data, conds={"basal": ("v2dir", "vedir")},
        verdict_root=str(tmp_verdict := __import__("tempfile").mkdtemp()),
        studies_root=str(tmp_path / "studies"))
    titles = [s["title"] for s in secs]
    assert titles[0].startswith("Overview")
    assert any("ParCa" in t or "parca" in t.lower() for t in titles)
