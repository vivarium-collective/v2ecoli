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


# Regression guard: the real axis producers do NOT use `median_rel` --
# parca (scripts/_compare/report_cards/parca.py) packs `detail.init_rel`,
# and statistical/ttest (v2ecoli/library/card_criteria.py) packs
# `detail.delta_rel` (which can be negative). The hand-crafted VERDICT
# fixture above used `median_rel` and so missed this; a real report's heat
# cells were rendering "--" for every observable.
REALISTIC_VERDICT = {"overall": "drift", "groups": {
    "parca": {"verdict": "within_tol", "axes": [
        {"label": "mass", "verdict": "within_tol", "detail": {"init_rel": 0.002}},
    ]},
    "statistical": {"verdict": "drift", "axes": [
        {"label": "growth", "verdict": "drift", "detail": {"delta_rel": -0.104, "p": 0.03}},
    ]},
}}


def test_summary_reads_real_axis_detail_keys_not_just_median_rel():
    html = build_summary_html(REALISTIC_VERDICT, seeds=4)
    assert "0.2%" in html   # parca: detail.init_rel=0.002
    assert "10.4%" in html  # statistical: detail.delta_rel=-0.104, abs()'d
    # the two real observables must NOT fall back to the "no value" placeholder
    assert html.count("--") == 0
