from scripts._compare.report import (
    config_panel_section, converted_processes_section, render_report)


def test_config_panel_is_navigable_json():
    cfg = {"experiment_id": "x", "generations": 2,
           "add_processes": ["example-secretion"],
           "_dropped_vecoli_keys": {"emitter": "parquet"}}
    sec = config_panel_section(cfg)
    assert sec["kind"] == "content"
    assert "add_processes" in sec["html"] and "example-secretion" in sec["html"]
    assert "emitter" in sec["html"]            # dropped keys shown in details
    assert "desc" in sec and sec["desc"]


def test_converted_panel_is_cards_with_ran_status():
    specs = [{"name": "example-secretion", "module": "ecoli.processes",
              "qualname": "ExampleSecretion", "kind": "vivarium_1",
              "config": {"rate": 2.0}, "interval": 1.0,
              "topology": {"counts": ["bulk"]}}]
    sec = converted_processes_section(specs, {"example-secretion": True})
    assert sec["kind"] == "content"
    assert "example-secretion" in sec["html"]
    assert "bulk" in sec["html"]               # port wiring shown
    assert "ran in both" in sec["html"]


def test_render_report_appends_embedded_html():
    html = render_report([{"title": "T", "rows": []}], title="x",
                         embedded_html=["<div id='card'>CARD</div>"])
    assert "CARD" in html


def test_render_grouped_report_separates_configs():
    from scripts._compare.report import render_grouped_report
    g = [{"config": "cfgA", "config_id": "cfga",
          "sections": [{"title": "S", "rows": [
              {"label": "m", "left": "1", "right": "1", "verdict": "within_tol"}]}],
          "embedded_html": []},
         {"config": "cfgB", "config_id": "cfgb", "sections": [], "embedded_html": []}]
    html = render_grouped_report(g, title="t")
    assert "cfg-cfga" in html and "cfg-cfgb" in html
    assert "Run: cfgA" in html and "Run: cfgB" in html
