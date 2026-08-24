import os, yaml

STUDY = os.path.join("workspace", "studies", "variant-sweep-phenotype-demo", "study.yaml")


def test_study_exists_and_is_schema_v4():
    with open(STUDY) as f:
        doc = yaml.safe_load(f)
    assert doc["schema_version"] == 4
    assert doc["name"] == "variant-sweep-phenotype-demo"


def test_variants_sweep_a_variant_axis_over_the_vecoli_composite():
    with open(STUDY) as f:
        doc = yaml.safe_load(f)
    variants = doc["conditions"]["variants"]
    assert len(variants) >= 2
    idxs = [v["params"]["variant"] for v in variants]
    assert idxs == sorted(set(idxs)) and idxs[0] >= 1     # distinct, 1-based
    for v in variants:
        assert v["composite"] == "v2ecoli.composites.vecoli.vecoli"


def test_demo_study_is_structurally_neutral():
    # Public template carries no model content: whole_config unset, no observable
    # ids, only the generic composite — every specific is filled in downstream.
    with open(STUDY) as f:
        doc = yaml.safe_load(f)
    base = doc["conditions"]["baseline"]["params"]
    assert base["whole_config"] == ""
    assert base["observable_bulk_ids"] == []
    for v in doc["conditions"]["variants"]:
        assert v["params"]["whole_config"] == ""
        assert v["params"]["observable_bulk_ids"] == []
