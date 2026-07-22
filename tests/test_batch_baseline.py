"""Tests for the batch_baseline composite + BatchBaselineRunner Step.

Fast: the parallel-dispatch logic is exercised with a STUB ``run_one`` (no
ParCa, no real simulation); document construction calls the generator directly.
The real per-seed run (build_composite("baseline") -> run_multigen_xarray) is
covered by the baseline/xarray-run suites and is not re-run here.
"""
from __future__ import annotations

import pytest

from v2ecoli.steps import batch_baseline_runner as bbr
from v2ecoli.steps.batch_baseline_runner import BatchBaselineRunner, dispatch_batch


def _stub_run_one(seed, *, n_generations, cache_dir, max_steps_per_gen, out_root):
    """Lightweight stand-in for a real per-seed baseline run."""
    return {
        "seed": int(seed),
        "store_path": f"{out_root}/seed_{int(seed):02d}/store.zarr",
        "summary": {
            "generations_reached": int(n_generations),
            "steps": 42,
            "final_cell_mass_fg": 100.0 + seed,
        },
    }


def test_dispatch_batch_assembles_per_seed_results():
    batch = dispatch_batch(
        n_seeds=3, n_generations=2, base_seed=0, cache_dir="out/cache",
        max_steps_per_gen=10, out_root="out/batch_baseline", parallel=None,
        run_one=_stub_run_one,
    )
    assert batch["completed"] is True
    assert batch["n_seeds"] == 3
    assert batch["n_generations"] == 2
    assert batch["mode"] == "sequential"          # parallel=None => safe fallback
    assert sorted(batch["seeds"].keys()) == ["00", "01", "02"]
    s0 = batch["seeds"]["00"]
    assert s0["store_path"].endswith("seed_00/store.zarr")
    assert s0["generations_reached"] == 2         # summary flattened into the row
    assert s0["final_cell_mass_fg"] == 100.0


def test_dispatch_batch_respects_base_seed():
    seen = []

    def _capture(seed, **kw):
        seen.append(seed)
        return _stub_run_one(seed, **kw)

    batch = dispatch_batch(
        n_seeds=2, n_generations=1, base_seed=5, cache_dir="c",
        max_steps_per_gen=10, out_root="out/x", parallel=None, run_one=_capture,
    )
    assert seen == [5, 6]                          # base_seed offset applied
    assert sorted(batch["seeds"].keys()) == ["05", "06"]


def test_dispatch_batch_records_missing_result():
    def _bad(seed, **kw):
        return None                                # a worker that produced nothing

    batch = dispatch_batch(
        n_seeds=1, n_generations=1, base_seed=0, cache_dir="c",
        max_steps_per_gen=10, out_root="out/x", parallel=None, run_one=_bad,
    )
    assert batch["seeds"]["00"] == {"error": "run produced no result"}


def test_dispatch_batch_defaults_to_module_run_one(monkeypatch):
    """When no run_one is passed, dispatch_batch resolves the module-level
    _default_run_one at CALL time (so it stays monkeypatch-friendly)."""
    calls = {"n": 0}

    def _fake_default(seed, **kw):
        calls["n"] += 1
        return _stub_run_one(seed, **kw)

    monkeypatch.setattr(bbr, "_default_run_one", _fake_default)
    batch = dispatch_batch(
        n_seeds=2, n_generations=1, base_seed=0, cache_dir="c",
        max_steps_per_gen=10, out_root="out/x", parallel=None,  # no run_one arg
    )
    assert calls["n"] == 2
    assert batch["n_seeds"] == 2


def test_runner_update_dispatches_once_then_is_idempotent(monkeypatch):
    monkeypatch.setattr(bbr, "_default_run_one", _stub_run_one)

    from v2ecoli.core import build_core
    from v2ecoli.composites._helpers import _make_instance

    runner = _make_instance(
        BatchBaselineRunner,
        {"n_seeds": 2, "n_generations": 1, "parallel": ""},  # "" => sequential
        build_core(),
    )

    # First fire: empty batch store -> dispatch the whole workflow.
    out1 = runner.update({"batch": {}})
    assert out1["batch"]["completed"] is True
    assert sorted(out1["batch"]["seeds"].keys()) == ["00", "01"]

    # Second fire with the completed store present -> no-op (fires exactly once).
    out2 = runner.update({"batch": out1["batch"]})
    assert out2 == {}


def test_build_batch_baseline_document_is_cheap_and_well_formed():
    """The generator builds a doc WITHOUT running any baseline (no ParCa)."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.batch_baseline import batch_baseline, BATCH_RUNNER_STEP_NAME

    doc = batch_baseline(core=build_core(), n_seeds=2, n_generations=3)
    state = doc["state"]
    assert state["batch"] == {}                    # empty until run
    node = state[BATCH_RUNNER_STEP_NAME]
    assert node["_type"] == "step"
    assert "BatchBaselineRunner" in node["address"]
    assert node["config"]["n_seeds"] == 2
    assert node["config"]["n_generations"] == 3


def test_batch_baseline_registered_for_build_composite():
    """build_composite resolves the new architecture by name."""
    from pbg_superpowers.composite_generator import _REGISTRY
    import v2ecoli.composites  # noqa: F401 — fires the @composite_generator

    names = {e.name for e in _REGISTRY.values()}
    assert "batch_baseline" in names
