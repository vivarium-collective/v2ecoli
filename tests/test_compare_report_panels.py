from scripts._compare.report import (
    config_panel_section, converted_processes_section, render_report)


def test_config_panel_lists_added_processes():
    cfg = {"experiment_id": "x", "generations": 2,
           "add_processes": ["example-secretion"],
           "_dropped_vecoli_keys": {"emitter": "parquet"}}
    sec = config_panel_section(cfg)
    labels = [r["label"] for r in sec["rows"]]
    assert "add_processes" in labels
    assert any("emitter" in r["label"] for r in sec["rows"])


def test_converted_panel_marks_ran_in_both():
    specs = [{"name": "example-secretion", "module": "ecoli.processes",
              "qualname": "ExampleSecretion", "kind": "vivarium_1",
              "topology": {"counts": ["bulk"]}}]
    sec = converted_processes_section(specs, {"example-secretion": True})
    row = sec["rows"][0]
    assert row["label"] == "example-secretion"
    assert row["verdict"] == "within_tol"


def test_render_report_appends_embedded_html():
    html = render_report([{"title": "T", "rows": []}], title="x",
                         embedded_html=["<div id='card'>CARD</div>"])
    assert "CARD" in html
