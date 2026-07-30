"""scripts/run_batch_baseline_ray.py wraps v2ecoli's LineageProcess/
batch_baseline_runner pipeline for real multi-generation dispatch. Neither
dispatch_batch() nor the underlying run_workflow() signals failure via a
non-zero process exit on its own (confirmed: v2ecoli.workflow.run.main() only
warnings.warn()s on an incomplete result) -- this script's own exit-code
check is the fix, and its own restructuring is what makes the output land in
the seed_NN/store.zarr shape the existing landing/analysis code expects."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_batch_baseline_ray import main, restructure_seed_stores


def _make_seed_store(tmp_path: Path, name: str) -> Path:
    store = tmp_path / name
    store.mkdir()
    (store / ".zgroup").write_text('{"zarr_format":2}')
    return store


def test_restructure_moves_stores_and_writes_summaries(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    store0 = _make_seed_store(tmp_path, "batch_baseline_v0_s0.zarr")
    result = {
        "seeds": {
            "00": {"store_path": str(store0), "complete": True, "generations_reached": 3},
            "01": {"error": "run produced no result"},
        }
    }

    entries = restructure_seed_stores(result, out_root)

    assert (out_root / "seed_00" / "store.zarr" / ".zgroup").is_file()
    assert not store0.exists()  # moved, not copied
    summary0 = (out_root / "seed_00" / "summary.json").read_text()
    assert '"complete": true' in summary0
    assert '"generations_reached": 3' in summary0
    summary1 = (out_root / "seed_01" / "summary.json").read_text()
    assert "run produced no result" in summary1
    assert len(entries) == 2


def test_main_exits_nonzero_when_all_seeds_fail(monkeypatch, tmp_path):
    import scripts.run_batch_baseline_ray as mod

    monkeypatch.setattr(mod.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "dispatch_batch", lambda **kw: {
        "seeds": {"00": {"error": "boom"}}, "wall_s": 1.0,
    })
    monkeypatch.setattr(sys, "argv", ["run_batch_baseline_ray.py", "--n-seeds", "1", "--n-generations", "2"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    assert "all 1 seeds failed" in str(exc_info.value.code)


def test_main_exits_zero_when_at_least_one_seed_completes(monkeypatch, tmp_path):
    import scripts.run_batch_baseline_ray as mod

    store0 = _make_seed_store(tmp_path, "store0.zarr")
    monkeypatch.setattr(mod.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "dispatch_batch", lambda **kw: {
        "seeds": {
            "00": {"store_path": str(store0), "complete": True, "generations_reached": 3},
            "01": {"error": "run produced no result"},
        },
        "wall_s": 42.0,
    })
    monkeypatch.setattr(sys, "argv", ["run_batch_baseline_ray.py", "--n-seeds", "2", "--n-generations", "3"])

    main()  # must not raise

    summary = (tmp_path / "summary.json").read_text()
    assert '"n_seeds_successful": 1' in summary
    assert '"n_generations_requested": 3' in summary


def test_main_exits_nonzero_when_dispatch_raises(monkeypatch, tmp_path):
    import scripts.run_batch_baseline_ray as mod

    monkeypatch.setattr(mod.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(mod, "OUT_ROOT", tmp_path)

    def _raise(**kw):
        raise RuntimeError("ray cluster unreachable")

    monkeypatch.setattr(mod, "dispatch_batch", _raise)
    monkeypatch.setattr(sys, "argv", ["run_batch_baseline_ray.py", "--n-seeds", "1"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "ray cluster unreachable" in str(exc_info.value.code)
