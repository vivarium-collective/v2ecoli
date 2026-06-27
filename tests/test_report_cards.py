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
