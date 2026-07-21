from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "workspace"
SLUG = "v2ecoli-vecoli-comparison"


def test_aggregate_discovers_studies_in_dag_order():
    from reports._summary.aggregate import aggregate

    summ = aggregate(SLUG, WS)
    assert summ["slug"] == SLUG
    assert summ["question"].startswith("Does v2ecoli reproduce vEcoli")
    slugs = [s["slug"] for s in summ["studies"]]
    assert slugs == [
        "parca", "basal", "with_aa", "succinate",
        "no_oxygen", "acetate", "statistical",
    ]


def test_aggregate_per_study_metadata_and_rollup():
    from reports._summary.aggregate import aggregate

    summ = aggregate(SLUG, WS)
    by = {s["slug"]: s for s in summ["studies"]}
    assert by["acetate"]["result"] == "FAIL"
    assert by["parca"]["result"] == "PASS"
    assert by["parca"]["prerequisites"] == []
    assert by["acetate"]["prerequisites"] == ["parca"]
    assert by["statistical"]["prerequisites"] == ["basal"]
    assert "RNA mass" in (by["acetate"]["finding"] or "")
    # config + standard cards discovered for acetate; parca card for parca
    assert {c["name"] for c in by["acetate"]["cards"]} == {"config", "standard"}
    assert {c["name"] for c in by["parca"]["cards"]} == {"parca"}
    # config card is ungraded, standard is graded
    acards = {c["name"]: c for c in by["acetate"]["cards"]}
    assert acards["config"]["graded"] is False
    assert acards["standard"]["graded"] is True
    assert acards["standard"]["overall"] == "mismatch"
    assert summ["rollup"] == {"PASS": 2, "PARTIAL": 3, "FAIL": 2}


def test_matrix_columns_lead_with_standard_observables():
    from reports._summary.aggregate import aggregate

    m = aggregate(SLUG, WS)["matrix"]
    # standard-card observables appear first, in card order
    assert m["columns"][:5] == [
        "cell mass (fg)", "dry mass (fg)", "protein mass (fg)",
        "RNA mass (fg)", "growth rate (1/s)",
    ]


def test_matrix_cell_verdicts_match_source_json():
    import json
    from reports._summary.aggregate import aggregate

    m = aggregate(SLUG, WS)["matrix"]
    rows = {r["study"]: r["cells"] for r in m["rows"]}
    # acetate growth rate is a mismatch in the source verdict.json
    src = json.loads(
        (WS / "investigations" / SLUG / "studies" / "acetate"
         / "viz" / "report_card" / "standard.verdict.json").read_text()
    )
    axis = {a["label"]: a["verdict"] for a in src["groups"]["standard"]["axes"]}
    assert rows["acetate"]["growth rate (1/s)"] == axis["growth rate (1/s)"] == "mismatch"
    assert rows["acetate"]["RNA mass (fg)"] == "drift"
    # parca has cell mass within tolerance, no growth-rate column value
    assert rows["parca"]["cell mass (fg)"] == "within_tol"
    assert rows["parca"].get("growth rate (1/s)") is None


def test_card_html_inlined_and_full_doc_flagged():
    from reports._summary.aggregate import aggregate

    by = {s["slug"]: s for s in aggregate(SLUG, WS)["studies"]}
    scards = {c["name"]: c for c in by["statistical"]["cards"]}
    # statistical.html is a full <html> document -> flagged for iframe embedding
    assert scards["statistical"]["is_full_doc"] is True
    assert "<html" in scards["statistical"]["html"].lower()
    # standard.html is a fragment (no <html>) -> inlined directly
    acards = {c["name"]: c for c in by["acetate"]["cards"]}
    assert acards["standard"]["is_full_doc"] is False
    assert acards["standard"]["html"].strip() != ""
    assert "<html" not in acards["standard"]["html"].lower()


def test_missing_card_marked_not_crashing(tmp_path):
    from reports._summary.aggregate import _card_stub

    # a study dir with a declared card whose files don't exist
    stub = _card_stub(tmp_path, "viz/report_card/ghost.html")
    assert stub["missing"] is True
    assert stub["html"] == ""
    assert stub["graded"] is False


def test_render_overview_and_matrix():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS), style_css=":root{--x:1}")
    assert "<!doctype html>" in html.lower()
    assert "Does v2ecoli reproduce vEcoli" in html
    # rollup counts present
    assert "2 FAIL" in html and "3 PARTIAL" in html and "2 PASS" in html
    # matrix header + a verdict-colored cell class
    assert "growth rate (1/s)" in html
    assert "verdict-mismatch" in html
    # self-contained: no external stylesheet or script src
    assert "<link " not in html.lower()
    assert "script src" not in html.lower()


def test_render_per_study_sections_and_embedding():
    from reports._summary.aggregate import aggregate
    from reports._summary.render import render

    html = render(aggregate(SLUG, WS))
    # every study appears as a details section
    for slug in ("parca", "acetate", "statistical"):
        assert f'id="study-{slug}"' in html
    # fragment card (acetate standard) inlined: its <h3 ... simulation runs heading present
    assert "simulation runs" in html
    # full-doc card (statistical) embedded via iframe srcdoc
    assert "iframe" in html and "srcdoc=" in html
    # auto-height shim present exactly once
    assert html.count("scrollHeight") >= 1
