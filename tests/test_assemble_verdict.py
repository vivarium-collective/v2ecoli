import json


def test_assemble_writes_condition_verdict(tmp_path, monkeypatch):
    import scripts.comparison_report_card as crc
    from scripts._compare import report_cards as rc
    # Stub the heavy overview + card rendering so we test only the wiring.
    monkeypatch.setattr(crc, "overview_section",
                        lambda cond_data: {"title": "o", "kind": "content", "html": ""})
    monkeypatch.setattr(rc, "render", lambda name, ctx: [
        {"title": name, "kind": "content", "html": "",
         "verdict": "drift", "verdict_axes": [{"id": "x", "verdict": "drift"}]}])
    manifest = {"defaults": {"cards": ["standard"]},
                "configs": [{"config": "configs/cond_basal_1x4.json",
                             "cards": ["standard"]}]}
    cond_data = {"basal": ({}, {}, [])}
    conds = {"basal": ("v2", "ve")}
    config_names = {"configs/cond_basal_1x4.json": "basal"}
    crc.assemble_from_manifest(manifest, cond_data, conds, config_names,
                               verdict_root=str(tmp_path))
    doc = json.loads((tmp_path / "basal" / "report_card_verdict.json").read_text(
        encoding="utf-8"))
    assert doc["groups"]["standard"]["verdict"] == "drift"
    assert doc["overall"] == "drift"
