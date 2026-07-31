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

from scripts.run_phase0_xarray_ensemble import main


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
