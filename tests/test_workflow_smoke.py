import os
import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")


def test_tiny_sweep_runs_to_completion(tmp_path):
    from v2ecoli.workflow.run import run_workflow

    config = {
        "experiment_id": "smoke",
        "n_init_sims": 1,
        "generations": 1,
        "single_daughters": True,
        "cache_dir": CACHE,
        "out_dir": str(tmp_path / "parquet"),
        "variants": {},
        # Cap so the test ends fast even though a real division is ~2500 s.
        "max_duration_per_gen": 5.0,
        "time_step": 1.0,
    }
    result = run_workflow(config, max_total_time=20.0, pbg_out=str(tmp_path / "sweep.pbg"))
    assert result["complete"] is True
    assert os.path.exists(str(tmp_path / "sweep.pbg"))
    # one branch, completed
    assert len(result["branches"]) == 1
    assert all(b["complete"] for b in result["branches"].values())
