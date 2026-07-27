from scripts._compare.study_spec import load_investigation


def test_five_redux_studies_load_with_6x1_shape():
    _ctx, specs = load_investigation("v2ecoli-vecoli-comparison")
    by = {s.name: s for s in specs}
    for c in ["basal", "with_aa", "succinate", "no_oxygen", "acetate"]:
        s = by[f"metabolism_redux_{c}"]
        assert s.seeds == 6 and s.gens == 1
        assert s.condition == c
        assert s.from_vecoli_config == f"configs/metabolism_redux_{c}.json"
        for card in ["config", "parca", "standard", "statistical",
                     "trajectory", "distribution", "metabolism", "composition"]:
            assert card in s.cards
