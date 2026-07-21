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
