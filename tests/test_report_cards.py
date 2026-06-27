import pytest
from scripts._compare import report_cards as rc


def test_register_and_get():
    @rc.report_card("dummy_card_xyz")
    def _c(ctx):
        return {"title": "T", "kind": "content", "html": "<p>x</p>", "anchor": "a"}
    assert "dummy_card_xyz" in rc.all_names()
    fn = rc.get("dummy_card_xyz")
    out = rc.render("dummy_card_xyz", _ctx())
    assert out[0]["title"] == "T" and out[0]["html"]


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        rc.get("does_not_exist")


def _ctx():
    return rc.CardContext(config_name="basal", variant=0, v2_dir="", ve_dir="",
                          seeds=1, gens=1, per_obs={}, plot_trajs={}, v2_bounds={},
                          config={})


def test_builtin_cards_registered():
    for name in ("standard", "statistical", "parca", "config_diff"):
        assert name in rc.all_names()


def test_statistical_card_returns_graded_section():
    ctx = rc.CardContext(config_name="basal", variant=0, v2_dir="", ve_dir="",
                         seeds=2, gens=1,
                         per_obs={"cell_mass": [{"ve_mean": 1.0, "v2_mean": 1.0},
                                                {"ve_mean": 1.1, "v2_mean": 1.05}],
                                  "growth_rate": [{"ve_mean": 2e-4, "v2_mean": 2e-4},
                                                  {"ve_mean": 2.1e-4, "v2_mean": 2.05e-4}]},
                         plot_trajs={}, v2_bounds={}, config={})
    secs = rc.render("statistical", ctx)
    assert secs[0]["html"] and secs[0]["verdict"] in (
        "within_tol", "drift", "mismatch", "ungraded")


def test_assemble_sections_mirrors_manifest(monkeypatch):
    from scripts import comparison_report_card as crc
    # one config, cards ["parca","standard"] -> overview + parca + (runs+eval)
    # per_obs uses the REAL shape build() produces: obs -> list of per-seed dicts
    # (each a full _matched() stat with init_*/*_mean/median_rel/max_rel).
    seed_stat = {"seed": 0, "init_t": 0.0, "init_v2": 1.0, "init_ve": 1.0,
                 "init_rel": 0.0, "v2_mean": 1.0, "ve_mean": 1.0,
                 "median_rel": 0.0, "max_rel": 0.0, "n": 1}
    cond_data = {"basal": ({"cell_mass": [seed_stat]}, {}, {})}
    manifest = {"configs": [{"config": "configs/cond_basal.json",
                             "cards": ["parca", "standard"]}]}
    secs = crc.assemble_from_manifest(
        manifest, cond_data,
        conds={"basal": ("v2dir", "vedir")},
        config_names={"configs/cond_basal.json": "basal"})
    titles = [s["title"] for s in secs]
    assert titles[0].startswith("Overview")
    assert any("ParCa" in t or "parca" in t.lower() for t in titles)
