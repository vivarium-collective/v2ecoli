from scripts._compare.report import render_report


def _section(title, rows):
    return {"title": title, "rows": rows}


def test_render_report_two_columns_and_badges():
    sections = [
        _section("Config & schema diff", [
            {"label": "emitter", "left": "parquet", "right": "(dropped)",
             "verdict": "drift"},
        ]),
        _section("ParCa / sim_data", [
            {"label": "mass.avg_cell_dry_mass", "left": "2.5e-13",
             "right": "2.5e-13", "verdict": "within_tol"},
        ]),
    ]
    html = render_report(sections, title="vEcoli vs v2ecoli")

    assert "<html" in html.lower()
    # two column headers present
    assert "vEcoli" in html and "v2ecoli" in html
    # each section title rendered
    assert "Config &amp; schema diff" in html or "Config & schema diff" in html
    assert "ParCa / sim_data" in html
    # verdict drives a CSS class
    assert "verdict-within_tol" in html
    assert "verdict-drift" in html
    # self-contained: no external http(s) asset links
    assert "http://" not in html and "https://" not in html


def test_render_report_handles_not_compared_rows():
    sections = [{"title": "Sim", "rows": [
        {"label": "ribosome", "left": "n/a", "right": "n/a",
         "verdict": "not_compared", "reason": "missing on v2 side"}]}]
    html = render_report(sections, title="t")
    assert "missing on v2 side" in html
    assert "verdict-not_compared" in html
