import json
from scripts._compare.config_adapter import resolve_vecoli_config, schema_diff, translate_vecoli_config


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


def test_translate_maps_known_keys_and_drops_vecoli_only():
    vecoli = {
        "experiment_id": "two_generations",
        "generations": 2,
        "n_init_sims": 2,
        "single_daughters": True,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": "out"},
        "parca_options": {"cpus": 3, "memory_gb": 6},
        "fail_at_max_duration": True,
        "sim_data_path": None,
        "analysis_options": {"single": {"mass_fraction_summary": {}}},
    }
    v2 = translate_vecoli_config(vecoli)
    # shared keys carried through unchanged
    assert v2["experiment_id"] == "two_generations"
    assert v2["generations"] == 2
    assert v2["n_init_sims"] == 2
    assert v2["single_daughters"] is True
    assert v2["analysis_options"] == {"single": {"mass_fraction_summary": {}}}
    # vEcoli-only keys are dropped from the v2 config body
    for dropped in ("emitter", "emitter_arg", "parca_options",
                    "fail_at_max_duration", "sim_data_path"):
        assert dropped not in v2
    # the mapping is recorded for the report
    assert "emitter" in v2["_dropped_vecoli_keys"]
    assert v2["_dropped_vecoli_keys"]["parca_options"] == {"cpus": 3,
                                                            "memory_gb": 6}


def test_translate_sets_lineage_seed_default_when_absent():
    v2 = translate_vecoli_config({"experiment_id": "x", "generations": 1})
    assert v2["lineage_seed"] == 0


def test_resolve_vecoli_config_invokes_vecoli_loader(monkeypatch, tmp_path):
    captured = {}

    def fake_check_output(cmd, cwd=None, text=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return json.dumps({"experiment_id": "resolved", "generations": 2})

    monkeypatch.setattr(
        "scripts._compare.config_adapter.subprocess.check_output",
        fake_check_output,
    )
    cfg = resolve_vecoli_config("/some/two_generations.json")
    assert cfg == {"experiment_id": "resolved", "generations": 2}
    # runs vEcoli's python from the vEcoli repo
    assert captured["cwd"].endswith("/vEcoli")
    assert captured["cmd"][0].endswith("/vEcoli/.venv/bin/python")
