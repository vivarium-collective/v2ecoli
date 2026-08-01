from scripts._compare.study_spec import load_investigation

# load_investigation() now supports both investigation schemas: when
# `comparison.configs[]` is present (config-is-the-unit model, post-Task-6)
# it builds specs via specs_from_configs() directly, so it's the realistic
# entry point again (this is also what scripts/comparison_report_card.py's
# render path calls) -- no need to reach for specs_from_configs()/_context()
# by hand here anymore.

_FULL_CARDS = ["summary", "parca", "statistical", "standard",
               "trajectory", "distribution", "metabolism", "composition"]


def test_load_investigation_returns_all_ten_configs_specs():
    _ctx, specs = load_investigation("whole-cell-model-comparison")
    assert len(specs) == 10
    names = {s.name for s in specs}
    assert names == {
        "basal", "with_aa", "acetate", "succinate", "no_oxygen",
        "metabolism_redux_basal", "metabolism_redux_with_aa",
        "metabolism_redux_acetate", "metabolism_redux_succinate",
        "metabolism_redux_no_oxygen",
    }


def test_five_redux_studies_load_with_configs():
    _ctx, specs = load_investigation("whole-cell-model-comparison")
    by = {s.name: s for s in specs}
    for c in ["basal", "with_aa", "succinate", "no_oxygen", "acetate"]:
        s = by[f"metabolism_redux_{c}"]
        assert s.seeds == 4 and s.gens == 1
        assert s.condition == c
        assert s.config == f"configs/metabolism_redux_{c}.json"
        for card in _FULL_CARDS:
            assert card in s.cards
