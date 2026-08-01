from scripts._compare.report_cards.summary import build_summary_html

VERDICT = {"overall": "drift", "groups": {"statistical": {"verdict": "drift", "axes": [
    {"label": "cell", "verdict": "within_tol", "detail": {"median_rel": 0.014}},
    {"label": "growth", "verdict": "drift", "detail": {"median_rel": 0.104}},
]}}}


def test_summary_lists_each_observable_with_status_and_value():
    html = build_summary_html(VERDICT, seeds=4)
    assert "cell" in html and "growth" in html
    assert "1.4%" in html and "10.4%" in html       # median_rel as percent
    assert "4 seeds" in html
    # status conveyed by glyph+label, not color alone
    assert "within_tol" in html and "drift" in html


def test_summary_shows_gate_status():
    html = build_summary_html(VERDICT, seeds=4)
    assert "gate" in html.lower()
