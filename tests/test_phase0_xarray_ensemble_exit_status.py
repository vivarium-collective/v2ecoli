"""scripts/run_phase0_xarray_ensemble.py must exit non-zero when every seed
fails -- this process's exit code is the ONLY signal AWS Batch (the GovCloud
dispatch backend) uses to report SUCCEEDED/FAILED. A 0/N-seed crash was
confirmed live to land as sms-api status "completed" with no error, because
main() always fell through to a normal (exit-0) return regardless of how many
seeds actually succeeded."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_phase0_xarray_ensemble import main, run_one


def _fake_parallel_result(results, mode="sequential"):
    from v2ecoli.library.parallel_seeds import ParallelResult
    return ParallelResult(results=results, wall_s=1.0, mode=mode, n_seeds=len(results))


def test_main_exits_nonzero_when_all_seeds_fail(monkeypatch, tmp_path):
    import scripts.run_phase0_xarray_ensemble as mod

    monkeypatch.setattr(mod.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(
        mod, "run_seeds_parallel",
        lambda *a, **k: _fake_parallel_result([{"seed": 0, "error": "boom", "type": "RuntimeError"}]),
    )
    monkeypatch.setattr(sys, "argv", ["run_phase0_xarray_ensemble.py", "--n-seeds", "1", "--n-steps", "1"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    assert "all 1 seeds failed" in str(exc_info.value.code)


def test_main_exits_zero_when_at_least_one_seed_succeeds(monkeypatch, tmp_path):
    import scripts.run_phase0_xarray_ensemble as mod

    monkeypatch.setattr(mod.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(
        mod, "run_seeds_parallel",
        lambda *a, **k: _fake_parallel_result([
            {"seed": 0, "dry_mass_fg": 220.0, "xarray_steps": 1},
            {"seed": 1, "error": "boom", "type": "RuntimeError"},
        ]),
    )
    monkeypatch.setattr(sys, "argv", ["run_phase0_xarray_ensemble.py", "--n-seeds", "2", "--n-steps", "1"])

    main()  # must not raise

    summary = (tmp_path / "summary.json").read_text()
    assert '"n_seeds_successful": 1' in summary


@pytest.mark.sim
def test_run_one_writes_store_at_ray_layout_convention(monkeypatch, tmp_path):
    """CROSS-REPO CONTRACT: viva-api's GET .../observables/index resolves a seed's
    store via RayLayout.seed_store_uri(experiment_id, seed) -- "v2ecoli_seed{NN:02d}.zarr"
    directly under the experiment prefix, the SAME convention chain-dispatch and the
    multi-node/colony path already write and read correctly. Regression guard against
    reintroducing the nested seed_NN/store.zarr shape: the S3 sync (RAY_OUT_DIR ->
    RAY_OUT_S3) faithfully preserves whatever local layout this writes, so a nested
    store lands in S3 at a path RayLayout never looks for -- the exact bug that 500'd
    every observables read for a single-generation ("phase0") dispatch, the default
    path for any request that doesn't ask for generations > 1 or a multi-node
    composite (cplong, smsvpctest, 2026-08-26)."""
    import os
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    import scripts.run_phase0_xarray_ensemble as mod

    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path)
    summary = run_one(seed=0, n_steps=2, chunk=1)

    assert (tmp_path / "v2ecoli_seed00.zarr").exists()
    assert not (tmp_path / "seed_00" / "store.zarr").exists()
    # summary.json's own nested location is UNCHANGED -- run_standalone_analysis.py's
    # build_multiseed_rows reads it from exactly here and never looks at store.zarr's
    # path at all, so that consumer must keep working unmodified.
    assert (tmp_path / "seed_00" / "summary.json").exists()
    assert summary["xarray_store"] == str(tmp_path / "v2ecoli_seed00.zarr")
