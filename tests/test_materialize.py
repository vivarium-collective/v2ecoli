import textwrap

import yaml

from scripts._compare.study_spec import StudySpec
from scripts._compare.materialize import materialized_fields, materialize_study

CARD_ROOT = "docs/report_cards/v2ecoli-vecoli-comparison"   # for the test's invest_name


def _spec(study_path, name="basal_4x4", cards=("config", "parca", "statistical")):
    return StudySpec(name=name, condition="basal", seeds=4, gens=4, cards=list(cards),
                     invest_name="v2ecoli-vecoli-comparison", v2_cache="out/cache_full",
                     ve_cache="out/compare_harness/vecoli_parca", fork="",
                     study_path=str(study_path))


def test_materialized_fields_one_test_per_graded_card():
    spec = _spec("/x", cards=["config", "parca", "standard"])
    f = materialized_fields(spec)
    # all three cards produce a report_card module
    assert len(f["tests"]) == 3
    assert {t["kind"] for t in f["tests"]} == {"report_card"}
    # only graded cards carry a measure
    graded = [t for t in f["tests"] if "measure" in t]
    groups = [t["measure"]["group"] for t in graded]
    assert groups == ["standard"]                      # config/parca not graded
    t = graded[0]
    assert t["measure"]["kind"] == "report_card_axis"
    assert t["measure"]["card"] == f"{CARD_ROOT}/basal_4x4"
    assert f["report_cards"] == [
        "viz/report_card/config.html",
        "viz/report_card/parca.html",
        "viz/report_card/standard.html",
    ]


def test_materialized_fields_statistical_card():
    f = materialized_fields(_spec("/x"))
    graded = [t for t in f["tests"] if "measure" in t]
    assert [t["measure"]["group"] for t in graded] == ["statistical"]


def test_materialize_declares_report_card_test_modules(tmp_path):
    sp = tmp_path / "study.yaml"
    sp.write_text(
        "name: basal\ninvestigation: v2ecoli-vecoli-comparison\ncondition: basal\n"
        "comparison: {seeds: 1, generations: 4, cards: [config, parca, standard]}\n",
        encoding="utf-8")
    spec = _spec(sp, name="basal", cards=["config", "parca", "standard"])
    import scripts._compare.materialize as M
    M.materialize_study(spec)
    import yaml
    data = yaml.safe_load(sp.read_text(encoding="utf-8"))
    tests = {t["name"]: t for t in data["tests"]}
    # one report_card module per card; all kind: report_card
    assert {t["kind"] for t in data["tests"]} == {"report_card"}
    assert tests["config-vs-vecoli"]["card"] == "config"
    assert "measure" not in tests["config-vs-vecoli"]            # informational
    assert tests["standard-vs-vecoli"]["measure"]["kind"] == "report_card_axis"
    assert "behavior_tests" not in data                          # replaced by tests
    assert data["baseline"][0]["name"] == "v2ecoli-baseline"


def test_materialize_preserves_narrative_and_is_idempotent(tmp_path):
    sp = tmp_path / "study.yaml"
    sp.write_text(textwrap.dedent("""
        name: basal_4x4
        investigation: v2ecoli-vecoli-comparison
        condition: basal
        comparison: {seeds: 4, generations: 4, cards: [config, parca, statistical]}
        claim: v2ecoli reproduces vEcoli on basal across 4 seeds
        question: does it hold across seeds?
    """), encoding="utf-8")
    spec = _spec(sp)
    materialize_study(spec)
    first = sp.read_text(encoding="utf-8")
    data = yaml.safe_load(first)
    # narrative + comparison preserved
    assert data["claim"].startswith("v2ecoli reproduces")
    assert data["comparison"]["seeds"] == 4
    # gating materialized — only graded cards carry a measure
    graded = [t for t in data["tests"] if "measure" in t]
    assert [t["measure"]["group"] for t in graded] == ["statistical"]
    assert "behavior_tests" not in data
    assert data["pipeline_gate"] == {"prerequisites": [], "enables": []}
    # idempotent
    materialize_study(spec)
    assert sp.read_text(encoding="utf-8") == first
