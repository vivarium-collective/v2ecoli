from scripts._compare.config_adapter import schema_diff


def test_schema_diff_partitions_keys():
    vecoli = {"experiment_id": "x", "generations": 2, "emitter": "parquet",
              "analysis_options": {"single": {}}}
    v2 = {"experiment_id": "x", "generations": 2, "cache_dir": "out/cache",
          "analysis_options": {"multiseed": {}}}
    d = schema_diff(vecoli, v2)
    assert d["only_in_vecoli"] == ["emitter"]
    assert d["only_in_v2"] == ["cache_dir"]
    # shared key with differing value is reported with both values
    assert d["different"]["analysis_options"] == (
        {"single": {}}, {"multiseed": {}})
    # shared key with equal value is NOT reported as different
    assert "experiment_id" not in d["different"]
    assert "generations" not in d["different"]
