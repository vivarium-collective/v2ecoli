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
    groups = [t["measure"]["group"] for t in f["behavior_tests"]]
    assert groups == ["standard"]                      # config/parca not graded
    t = f["behavior_tests"][0]
    assert t["measure"]["kind"] == "report_card_axis"
    assert t["measure"]["card"] == f"{CARD_ROOT}/basal_4x4"
    assert f["report_cards"] == [f"{CARD_ROOT}/basal_4x4/index.html"]


def test_materialized_fields_statistical_card():
    f = materialized_fields(_spec("/x"))
    assert [t["measure"]["group"] for t in f["behavior_tests"]] == ["statistical"]


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
    # gating materialized
    assert [t["measure"]["group"] for t in data["behavior_tests"]] == ["statistical"]
    assert data["pipeline_gate"] == {"prerequisites": [], "enables": []}
    # idempotent
    materialize_study(spec)
    assert sp.read_text(encoding="utf-8") == first
