from scripts._compare.study_spec import _context, _invest_dir, specs_from_configs

# NOTE: load_investigation() resolves studies via the legacy `members:`/
# `studies:` list (see study_spec._context) -- the Task 6 migration removed
# `members:` from whole-cell-model-comparison's investigation.yaml in favor
# of `comparison.configs[]`, so load_investigation() now returns an EMPTY
# spec list for it (verified: ctx["members"] == []). specs_from_configs() is
# the real integration entry point for a configs[]-only investigation -- the
# same one run/init/scaffold all use (see the Task 6 fix-round scaffold fix)
# -- so this test exercises that path instead.


def test_five_redux_studies_load_with_configs():
    ctx = _context(_invest_dir("whole-cell-model-comparison"))
    specs = specs_from_configs(ctx)
    by = {s.name: s for s in specs}
    for c in ["basal", "with_aa", "succinate", "no_oxygen", "acetate"]:
        s = by[f"metabolism_redux_{c}"]
        assert s.seeds == 4 and s.gens == 1
        assert s.condition == c
        assert s.config == f"configs/metabolism_redux_{c}.json"
        for card in ["summary", "parca", "statistical", "standard",
                     "trajectory", "distribution", "metabolism", "composition"]:
            assert card in s.cards
