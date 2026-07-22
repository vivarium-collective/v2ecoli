import json
from scripts._compare.viz_cards import write_report_cards


def test_writes_html_and_verdict_per_card(tmp_path):
    cards = [
        {"name": "standard", "verdict": "drift",
         "axes": [{"id": "standard.rna", "verdict": "drift"}],
         "html": "<!DOCTYPE html><b>standard card</b>"},
        {"name": "config", "verdict": "ungraded", "axes": [],
         "html": "<b>config card</b>"},
    ]
    paths = write_report_cards(tmp_path, cards)
    rc = tmp_path / "viz" / "report_card"
    assert (rc / "standard.html").is_file() and (rc / "standard.verdict.json").is_file()
    assert (rc / "config.html").is_file()
    html = (rc / "standard.html").read_text(encoding="utf-8")
    assert html == "<!DOCTYPE html><b>standard card</b>"
    vd = json.loads((rc / "standard.verdict.json").read_text(encoding="utf-8"))
    assert vd["overall"] == "drift"
    assert vd["groups"]["standard"]["verdict"] == "drift"
    assert {p.name for p in paths} >= {"standard.html", "config.html"}
    assert (rc / "config.verdict.json").is_file()
    cvd = json.loads((rc / "config.verdict.json").read_text(encoding="utf-8"))
    assert cvd["overall"] == "ungraded"
