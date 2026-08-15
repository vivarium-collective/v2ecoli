"""build_run_config threads a scalar cache fingerprint (reproducible-rerun-
spine Task 2, v2ecoli side): run_config["cache_fingerprint"] must be the
plain short-hash string (not the detailed dict) so it lands directly on
manifest["env"]["cache_fingerprint"] wherever a downstream manifest builder
(vivarium_workbench.lib.composite_runs.build_run_manifest) reads
params["cache_fingerprint"]. The full diagnostic dict (path/exists/size/mtime)
is preserved under "cache_fingerprint_detail" so nothing already reading it
loses information.
"""
import argparse

from scripts.run_condition_multigen_parquet import build_run_config, cache_fingerprint


def _args(cache_dir):
    return argparse.Namespace(
        experiment_id="exp1", cache_dir=cache_dir, seed=0, generations=1,
        start_gen=1, max_min=1.0, resume_dill=None, out_dir="out/exp1",
    )


def test_build_run_config_cache_fingerprint_is_scalar_string(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dill_path = cache_dir / "sim_data_cache.dill"
    dill_path.write_bytes(b"fake cache contents")

    run_config = build_run_config(
        _args(str(cache_dir)), perturbations={}, applied_record={}, cache={})

    detail = cache_fingerprint(str(cache_dir))
    assert run_config["cache_fingerprint"] == detail["fingerprint"]
    assert isinstance(run_config["cache_fingerprint"], str)
    assert run_config["cache_fingerprint_detail"] == detail


def test_build_run_config_cache_fingerprint_none_when_cache_missing(tmp_path):
    missing_dir = tmp_path / "no_such_cache"

    run_config = build_run_config(
        _args(str(missing_dir)), perturbations={}, applied_record={}, cache={})

    assert run_config["cache_fingerprint"] is None
    assert run_config["cache_fingerprint_detail"]["exists"] is False
