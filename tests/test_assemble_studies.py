import json

from scripts._compare.study_spec import StudySpec


def _spec(name, condition, cards):
    return StudySpec(name=name, condition=condition, seeds=1, gens=4, cards=list(cards),
                     invest_name="whole-cell-model-comparison", v2_cache="c", ve_cache="c",
                     study_path="/x")


def _stub_core(monkeypatch, crc, captured=None):
    """Monkeypatch build_core with a fake core whose link_registry resolves any
    '<card>_report_card' key to a stub Step that returns drift verdict + card html."""
    class _Stub:
        def __init__(self, *a, **k): pass
        def update(self, state):
            if captured is not None:
                captured["state"] = state
            return {"card_html": "<b>card</b>", "verdict": "drift",
                    "axes": [{"id": "x", "verdict": "drift"}]}

    class _Reg(dict):
        def __getitem__(self, k): return _Stub
    class _Core:
        link_registry = _Reg()
        def register_links(self, d): pass
    monkeypatch.setattr(crc, "build_core", lambda: _Core())


def test_assemble_from_studies_writes_per_study_verdict(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    _stub_core(monkeypatch, crc)
    specs = [_spec("basal", "basal", ["config", "parca", "standard"]),
             _spec("basal_4x4", "basal", ["config", "parca", "statistical"])]
    cond_data = {"basal": ({}, {}, []), "basal_4x4": ({}, {}, [])}
    conds = {"basal": ("v2", "ve"), "basal_4x4": ("v2", "ve")}
    crc.assemble_from_studies(specs, cond_data, conds, verdict_root=str(tmp_path),
                              studies_root=str(tmp_path / "studies"))
    for name, graded in (("basal", "standard"), ("basal_4x4", "statistical")):
        doc = json.loads((tmp_path / name / "report_card_verdict.json").read_text(encoding="utf-8"))
        # config + parca + graded card all get verdict "drift" from stub;
        # the graded card group is present and the overall is worst-of.
        assert graded in doc["groups"]
        assert doc["overall"] == "drift"


def test_assemble_from_studies_config_card_sees_study_spec(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    captured = {}
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    _stub_core(monkeypatch, crc, captured=captured)
    specs = [_spec("basal_4x4", "basal", ["config"])]
    crc.assemble_from_studies(specs, {"basal_4x4": ({}, {}, [])},
                              {"basal_4x4": ("v2", "ve")}, verdict_root=str(tmp_path),
                              studies_root=str(tmp_path / "studies"))
    # the config card receives the study's run spec via state["config"]
    assert captured["state"]["config"] == {"condition": "basal", "seeds": 1,
                                           "generations": 4, "cards": ["config"]}


def test_assemble_from_studies_writes_viz_cards(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})

    class _Stub:
        def __init__(self, *a, **k): pass
        def update(self, state):
            return {"card_html": "<b>card</b>", "verdict": "drift",
                    "axes": [{"id": "x", "verdict": "drift"}]}

    class _Reg(dict):
        def __getitem__(self, k): return _Stub
    class _Core:
        link_registry = _Reg()
        def register_links(self, d): pass
    monkeypatch.setattr(crc, "build_core", lambda: _Core())
    spec = _spec("basal", "basal", ["standard"])
    crc.assemble_from_studies([spec], {"basal": ({}, {}, [])},
                              {"basal": ("v2", "ve")}, verdict_root=str(tmp_path / "vr"),
                              studies_root=str(tmp_path / "ws/studies"))
    # per-study cards land in the TOP-LEVEL study registry (studies_root/name),
    # not the old nested workspace/investigations/<inv>/studies/<name> layout —
    # every reader (aggregate.py, study.yaml report_cards, dashboard embeds)
    # resolves cards from the top-level dir.
    card = tmp_path / "ws/studies/basal/viz/report_card/standard.html"
    assert card.is_file() and "<b>card</b>" in card.read_text(encoding="utf-8")
    import json
    assert json.loads(card.with_name("standard.verdict.json").read_text())["overall"] == "drift"
