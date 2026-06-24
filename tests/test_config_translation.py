"""Lock in the vEcoli->v2ecoli config translation contract.

Capability #3: "load a vEcoli config and have it automatically translate to an
equivalent v2ecoli config." These tests assert the verified behavior so it can't
silently drift:
  - run-shape knobs carry over by name (v2ecoli reads config['generations'],
    ['seed'], ['lineage_seed'], ['time_step'] under the SAME names),
  - vEcoli's per-generation cap max_duration -> v2ecoli's max_duration_per_gen,
  - vEcoli-only / workflow-infra keys are dropped from the body but recorded,
  - v2 defaults fill in when the vEcoli config omits them.

The representative config below mirrors the real 63-key resolved vEcoli
cond_basal.json (subset of keys, real values).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._compare.config_adapter import (  # noqa: E402
    translate_vecoli_config,
    schema_diff,
)


# Subset of a real resolved vEcoli config (keys + values observed from
# resolve_vecoli_config("configs/cond_basal.json")).
VECOLI_CONFIG = {
    # run-shape (shared names with v2ecoli)
    "condition": "basal",
    "seed": 0,
    "lineage_seed": 0,
    "generations": 16,
    "time_step": 1.0,
    "max_duration": 4000.0,
    # vEcoli-only / workflow-infra (must be dropped from the v2 body)
    "emitter": "timeseries",
    "emitter_arg": {"out_dir": "out"},
    "parca_options": {"cpus": 4},
    "fail_at_max_duration": False,
    "suffix_time": False,
    "sim_data_path": "out/kb/simData.cPickle",
    "inherit_from": ["default.json"],
}


def test_run_shape_keys_carry_over_by_name():
    v2 = translate_vecoli_config(VECOLI_CONFIG)
    # v2ecoli consumes these under the identical names (lineage.py /
    # meta_composite.py read config['generations'], ['seed'], ...).
    assert v2["condition"] == "basal"
    assert v2["seed"] == 0
    assert v2["lineage_seed"] == 0
    assert v2["generations"] == 16
    assert v2["time_step"] == 1.0


def test_max_duration_maps_to_per_gen():
    v2 = translate_vecoli_config(VECOLI_CONFIG)
    # vEcoli's per-generation cap -> v2ecoli's separate key (else v2 uses its
    # 3600s default and force-divides slow-growth media early).
    assert v2["max_duration_per_gen"] == 4000.0
    # The original vEcoli key is not carried verbatim.
    assert "max_duration" not in v2


def test_max_duration_per_gen_not_clobbered_if_present():
    cfg = dict(VECOLI_CONFIG, max_duration_per_gen=1234.0)
    v2 = translate_vecoli_config(cfg)
    assert v2["max_duration_per_gen"] == 1234.0  # setdefault: explicit wins


def test_vecoli_only_keys_dropped_and_recorded():
    v2 = translate_vecoli_config(VECOLI_CONFIG)
    dropped = v2["_dropped_vecoli_keys"]
    for k in ("emitter", "emitter_arg", "parca_options", "fail_at_max_duration",
              "suffix_time", "sim_data_path", "inherit_from"):
        assert k not in v2, f"{k} leaked into v2 config body"
        assert k in dropped, f"{k} not recorded under _dropped_vecoli_keys"
    # values preserved for the report (explicit, not silent)
    assert dropped["sim_data_path"] == "out/kb/simData.cPickle"


def test_v2_defaults_applied_when_missing():
    minimal = {"condition": "basal"}
    v2 = translate_vecoli_config(minimal)
    assert v2["lineage_seed"] == 0
    assert v2["single_daughters"] is True


def test_v2_defaults_do_not_override_present_values():
    cfg = {"condition": "basal", "lineage_seed": 7, "single_daughters": False}
    v2 = translate_vecoli_config(cfg)
    assert v2["lineage_seed"] == 7
    assert v2["single_daughters"] is False


def test_schema_diff_partitions_keys():
    d = schema_diff({"a": 1, "b": 2, "shared": 1}, {"c": 3, "shared": 9})
    assert d["only_in_vecoli"] == ["a", "b"]
    assert d["only_in_v2"] == ["c"]
    assert d["different"] == {"shared": (1, 9)}
