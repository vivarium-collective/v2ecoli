from v2ecoli.library.report_card import render_verdict_html


def _vj():
    return {
        "schema": "report_card_verdict/v1", "overall": "drift",
        "reference_model": "vEcoli @ basal", "model_ref": "v2ecoli @ basal",
        "groups": {
            "standard": {"verdict": "drift", "axes": [
                {"id": "physiology.cell_mass", "label": "Cell mass",
                 "verdict": "within_tol", "value": 1.2, "meter": "Δ=+1%"},
                {"id": "physiology.growth_rate", "label": "Growth rate",
                 "verdict": "drift", "value": 0.9, "meter": "Δ=+7%"}]},
            "config": {"verdict": "within_tol", "axes": [
                {"id": "config.seeds", "label": "Seeds",
                 "verdict": "within_tol", "value": 4, "meter": ""}]},
        },
    }


def test_render_is_self_contained_with_groups_and_axes():
    html = render_verdict_html(_vj(), title="vEcoli ↔ v2ecoli (basal)")
    assert "<img" not in html and "src=" not in html        # no external assets
    assert "Cell mass" in html and "Growth rate" in html
    assert "Standard" in html and "Config" in html          # group headers, title-cased
    assert "vEcoli ↔ v2ecoli (basal)" in html               # title
    assert "overall" in html.lower()


def test_render_tolerates_missing_value_and_meter():
    vj = {"schema": "report_card_verdict/v1", "overall": "ungraded",
          "groups": {"tests": {"verdict": "ungraded", "axes": [
              {"id": "tests.t1", "label": "t1", "verdict": "ungraded"}]}}}
    html = render_verdict_html(vj)
    assert "t1" in html
