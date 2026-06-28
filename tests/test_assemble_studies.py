import json

from scripts._compare.study_spec import StudySpec


def _spec(name, condition, cards):
    return StudySpec(name=name, condition=condition, seeds=1, gens=4, cards=list(cards),
                     invest_name="v2ecoli-vecoli-comparison", v2_cache="c", ve_cache="c",
                     fork="", study_path="/x")


def test_assemble_from_studies_writes_per_study_verdict(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    from scripts._compare import report_cards as rc
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    monkeypatch.setattr(rc, "render", lambda name, ctx: [
        {"title": name, "kind": "content", "html": "",
         "verdict": "drift", "verdict_axes": [{"id": "x", "verdict": "drift"}]}])
    specs = [_spec("basal", "basal", ["config", "parca", "standard"]),
             _spec("basal_4x4", "basal", ["config", "parca", "statistical"])]
    cond_data = {"basal": ({}, {}, []), "basal_4x4": ({}, {}, [])}
    conds = {"basal": ("v2", "ve"), "basal_4x4": ("v2", "ve")}
    crc.assemble_from_studies(specs, cond_data, conds, verdict_root=str(tmp_path))
    for name, graded in (("basal", "standard"), ("basal_4x4", "statistical")):
        doc = json.loads((tmp_path / name / "report_card_verdict.json").read_text(encoding="utf-8"))
        # config + parca render (ungraded via stub returning verdict_axes -> drift here),
        # the graded card group is present and the overall is worst-of.
        assert graded in doc["groups"]
        assert doc["overall"] == "drift"


def test_assemble_from_studies_config_card_sees_study_spec(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    from scripts._compare import report_cards as rc
    seen = {}
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})

    def _capture(name, ctx):
        seen["config"] = ctx.config
        return [{"title": name, "kind": "content", "html": ""}]
    monkeypatch.setattr(rc, "render", _capture)
    specs = [_spec("basal_4x4", "basal", ["config"])]
    crc.assemble_from_studies(specs, {"basal_4x4": ({}, {}, [])},
                              {"basal_4x4": ("v2", "ve")}, verdict_root=str(tmp_path))
    # the config card receives the study's run spec (no manifest config file)
    assert seen["config"] == {"condition": "basal", "seeds": 1,
                              "generations": 4, "cards": ["config"]}


def test_assemble_from_studies_writes_viz_cards(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    from scripts._compare import report_cards as rc
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    monkeypatch.setattr(rc, "render", lambda name, ctx: [
        {"title": f"{ctx.config_name}-{name}", "kind": "content", "html": "<b>card</b>",
         "verdict": "drift", "verdict_axes": [{"id": "x", "verdict": "drift"}]}])
    spec = _spec("basal", "basal", ["standard"])
    crc.assemble_from_studies([spec], {"basal": ({}, {}, [])},
                              {"basal": ("v2", "ve")}, verdict_root=str(tmp_path / "vr"),
                              studies_root=str(tmp_path / "ws/investigations"))
    card = (tmp_path / "ws/investigations/v2ecoli-vecoli-comparison/studies/basal"
            / "viz/report_card/standard.html")
    assert card.is_file() and "<b>card</b>" in card.read_text(encoding="utf-8")
    import json
    vd = json.loads(card.with_name("standard.verdict.json").read_text(encoding="utf-8"))
    assert vd["overall"] == "drift"
