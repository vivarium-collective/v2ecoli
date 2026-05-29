import pytest

from v2ecoli.workflow.variants import parse_variant_params, expand_branches, BranchSpec


def test_single_param_value():
    params = parse_variant_params({"kcat": {"target": "ecoli-metabolism.kcat",
                                             "value": [1, 2, 3]}})
    assert params == [
        {"ecoli-metabolism.kcat": 1},
        {"ecoli-metabolism.kcat": 2},
        {"ecoli-metabolism.kcat": 3},
    ]


def test_linspace_param():
    params = parse_variant_params({"d": {"target": "p.q",
                                         "linspace": {"start": 0.0, "stop": 1.0, "num": 3}}})
    assert [round(list(p.values())[0], 3) for p in params] == [0.0, 0.5, 1.0]


def test_prod_of_two_params():
    params = parse_variant_params({
        "a": {"target": "p.a", "value": [1, 2]},
        "b": {"target": "p.b", "value": [10, 20]},
        "op": "prod",
    })
    assert {"p.a": 1, "p.b": 10} in params
    assert {"p.a": 2, "p.b": 20} in params
    assert len(params) == 4
    assert {"p.a": 1, "p.b": 20} in params
    assert {"p.a": 2, "p.b": 10} in params


def test_expand_branches_grid():
    config = {
        "n_init_sims": 2,
        "lineage_seed": 0,
        "variants": {"kcat": {"target": "ecoli-metabolism.kcat", "value": [1, 2]}},
        "skip_baseline": True,
    }
    branches = expand_branches(config)
    # 2 variants × 2 seeds = 4 branches
    assert len(branches) == 4
    assert all(isinstance(b, BranchSpec) for b in branches)
    seeds = sorted({b.seed for b in branches})
    assert seeds == [0, 1]
    overrides = {tuple(sorted(b.overrides.items())) for b in branches}
    assert (("ecoli-metabolism.kcat", 1),) in overrides
    assert (("ecoli-metabolism.kcat", 2),) in overrides


def test_expand_branches_baseline_included_by_default():
    config = {"n_init_sims": 1, "variants": {}}
    branches = expand_branches(config)
    assert len(branches) == 1
    assert branches[0].variant_name == "baseline"
    assert branches[0].overrides == {}


def test_different_seeds_per_variant_non_overlapping():
    config = {
        "n_init_sims": 2,
        "lineage_seed": 0,
        "different_seeds_per_variant": True,
        "variants": {"kcat": {"target": "m.k", "value": [1, 2]}},
        "skip_baseline": True,
    }
    branches = expand_branches(config)
    by_variant = {}
    for b in branches:
        by_variant.setdefault(b.variant_index, set()).add(b.seed)
    seed_sets = list(by_variant.values())
    # No seed shared between the two variants
    assert seed_sets[0].isdisjoint(seed_sets[1])


def test_add_op_concatenates_single_key_dicts():
    params = parse_variant_params({
        "a": {"target": "p.a", "value": [1, 2]},
        "b": {"target": "p.b", "value": [10]},
        "op": "add",
    })
    # add = concatenate each param's values as separate single-key override dicts
    assert {"p.a": 1} in params
    assert {"p.a": 2} in params
    assert {"p.b": 10} in params
    assert len(params) == 3


def test_zip_length_mismatch_raises():
    with pytest.raises(ValueError):
        parse_variant_params({
            "a": {"target": "p.a", "value": [1, 2]},
            "b": {"target": "p.b", "value": [10]},
            "op": "zip",
        })


def test_empty_variant_block_returns_empty():
    assert parse_variant_params({}) == []
