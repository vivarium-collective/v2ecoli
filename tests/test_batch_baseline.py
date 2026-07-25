"""Tests for the batch_baseline composite + BatchBaselineRunner Step.

Fast: the dispatch logic is exercised with a STUB ``run_workflow`` (no ParCa, no
simulations); document construction calls the generator directly. The real
workflow run (LineageProcess -> baseline per generation, parquet + zarr emission,
post-sim flush) is covered by the workflow/lineage suites and is not re-run here.
"""
from __future__ import annotations

import pytest

from v2ecoli.steps import batch_baseline_runner as bbr
from v2ecoli.steps.batch_baseline_runner import (
    BatchBaselineRunner,
    applicable_analysis_scales,
    build_analysis_options,
    build_workflow_config,
    dispatch_batch,
)


def _stub_workflow(seeds=(0, 1, 2), *, complete=True, placed=(), variant=0):
    """A stand-in for run_workflow returning the branch shape it really returns
    (branch keys ``variant=<v>/seed=<s>``, per-generation summaries)."""
    def _run(config):
        _stub_workflow.last_config = dict(config)
        return {
            "complete": complete,
            "elapsed": 12.0,
            "branches": {
                f"variant={variant}/seed={s}": {
                    "complete": complete,
                    "summary": {"generations": [{"gen": g} for g in
                                                range(int(config["generations"]))]},
                }
                for s in seeds
            },
            "parallel": {"mode": "ray", "n_workers": len(seeds), "wall_s": 12.0},
            "flush": {"placed": list(placed), "skipped": []},
        }
    return _run


# --- parameter -> workflow config mapping ------------------------------------

def test_build_workflow_config_maps_batch_names_to_vecoli_keys():
    cfg = build_workflow_config(
        n_seeds=8, n_generations=4, base_seed=5, single_daughters=False,
        time_step=2.0, max_duration=4000.0, cache_dir="c", out_dir="o",
        experiment_id="exp", emitter="both", parallel="ray", analyses="none")
    assert cfg["n_init_sims"] == 8            # n_seeds
    assert cfg["generations"] == 4            # n_generations
    assert cfg["lineage_seed"] == 5           # base_seed
    assert cfg["max_duration_per_gen"] == 4000.0   # max_duration (per generation)
    assert cfg["single_daughters"] is False
    assert cfg["time_step"] == 2.0
    assert cfg["emitter"] == "both"
    assert cfg["parallel"] == "ray"
    assert cfg["analysis_options"] == {}


def test_build_workflow_config_sequential_and_study():
    cfg = build_workflow_config(parallel="", study="my-study", analyses="none")
    assert cfg["parallel"] is None            # "" => sequential, not the string
    assert cfg["study"] == "my-study"
    assert "study" not in build_workflow_config(analyses="none", study="")


# --- analysis selection ------------------------------------------------------

def test_applicable_scales_track_what_the_batch_actually_produced():
    assert applicable_analysis_scales(n_seeds=1, n_generations=1) == ["single"]
    assert applicable_analysis_scales(n_seeds=4, n_generations=1) == [
        "single", "multiseed"]
    assert applicable_analysis_scales(n_seeds=1, n_generations=3) == [
        "single", "multigeneration"]
    assert applicable_analysis_scales(
        n_seeds=4, n_generations=3, single_daughters=False,
        variants={"v": {}}) == [
        "single", "multigeneration", "multiseed", "multidaughter", "multivariant"]


def test_applicable_analyses_are_real_registered_analyses_at_those_scales():
    import v2ecoli.workflow.analyses  # noqa: F401 — registration side effects
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    opts = build_analysis_options("applicable", n_seeds=4, n_generations=3)
    assert set(opts) == {"single", "multigeneration", "multiseed"}
    for scale, names in opts.items():
        assert names, f"{scale} selected but empty"
        for name in names:
            assert ANALYSIS_REGISTRY[name].scale == scale
    # A one-cell batch must not ask for groupings it has no cells for.
    assert set(build_analysis_options("applicable", n_seeds=1,
                                      n_generations=1)) == {"single"}


def test_analysis_denylist_excludes_the_test_fixture_analysis():
    opts = build_analysis_options(
        "applicable", n_seeds=2, n_generations=2, variants={"v": {}})
    assert "dummy" not in opts.get("multivariant", {})


def test_analyses_none_and_explicit_mapping():
    assert build_analysis_options("none", n_seeds=4, n_generations=2) == {}
    assert build_analysis_options("", n_seeds=4, n_generations=2) == {}
    explicit = {"single": {"cell_mass": {"k": 1}}, "multiseed": {}}
    assert build_analysis_options(explicit, n_seeds=4, n_generations=2) == {
        "single": {"cell_mass": {"k": 1}}}          # empty scales dropped


def test_unknown_analyses_choice_raises():
    with pytest.raises(ValueError, match="applicable"):
        build_analysis_options("everything", n_seeds=1, n_generations=1)


# --- dispatch ----------------------------------------------------------------

def test_dispatch_batch_assembles_per_seed_results():
    batch = dispatch_batch(
        n_seeds=3, n_generations=2, base_seed=0, out_dir="out/batch_baseline",
        experiment_id="batch_baseline", analyses="none",
        run_workflow_fn=_stub_workflow((0, 1, 2)))
    assert batch["completed"] is True
    assert batch["n_seeds"] == 3 and batch["n_generations"] == 2
    assert batch["complete"] is True
    assert batch["mode"] == "ray"
    assert sorted(batch["seeds"]) == ["00", "01", "02"]
    s0 = batch["seeds"]["00"]
    assert s0["generations_reached"] == 2
    assert s0["branch"] == "variant=0/seed=0"
    # emitter "both" writes a per-lineage zarr store; report where it landed.
    assert s0["store_path"] == "out/batch_baseline/batch_baseline_v0_s0.zarr"


def test_dispatch_batch_parquet_only_reports_no_store_path():
    batch = dispatch_batch(n_seeds=1, n_generations=1, emitter="parquet",
                           analyses="none",
                           run_workflow_fn=_stub_workflow((0,)))
    assert "store_path" not in batch["seeds"]["00"]


def test_dispatch_batch_respects_base_seed():
    batch = dispatch_batch(n_seeds=2, n_generations=1, base_seed=5,
                           analyses="none",
                           run_workflow_fn=_stub_workflow((5, 6)))
    assert sorted(batch["seeds"]) == ["05", "06"]
    cfg = _stub_workflow.last_config
    assert cfg["lineage_seed"] == 5 and cfg["n_init_sims"] == 2


def test_dispatch_batch_records_a_seed_the_workflow_never_reported():
    batch = dispatch_batch(n_seeds=2, n_generations=1, analyses="none",
                           run_workflow_fn=_stub_workflow((0,)))  # seed 1 missing
    assert batch["seeds"]["01"] == {"error": "run produced no result"}


def test_dispatch_batch_reports_flush_outputs():
    placed = ["studies/s/viz/cell_mass.html", "studies/s/viz/doubling_time.html"]
    batch = dispatch_batch(n_seeds=2, n_generations=2,
                           run_workflow_fn=_stub_workflow((0, 1), placed=placed))
    assert batch["outputs"]["placed"] == placed
    assert batch["outputs"]["skipped"] == []
    # the scales it asked the flush for are recorded alongside
    assert batch["analysis_scales"] == ["multigeneration", "multiseed", "single"]


def test_dispatch_batch_defaults_to_the_real_run_workflow(monkeypatch):
    """With no run_workflow_fn, dispatch resolves v2ecoli.workflow.run.run_workflow
    at CALL time (so it stays monkeypatch-friendly and never imports eagerly)."""
    import v2ecoli.workflow.run as wr
    calls = {"n": 0}

    def _fake(config, **kw):
        calls["n"] += 1
        return _stub_workflow((0, 1))(config)

    monkeypatch.setattr(wr, "run_workflow", _fake)
    batch = dispatch_batch(n_seeds=2, n_generations=1, analyses="none")
    assert calls["n"] == 1 and batch["n_seeds"] == 2


# --- Step + composite --------------------------------------------------------

def test_runner_update_dispatches_once_then_is_idempotent(monkeypatch):
    import v2ecoli.workflow.run as wr
    monkeypatch.setattr(wr, "run_workflow", _stub_workflow((0, 1)))

    from v2ecoli.core import build_core
    from v2ecoli.composites._helpers import _make_instance

    runner = _make_instance(
        BatchBaselineRunner,
        {"n_seeds": 2, "n_generations": 1, "parallel": "", "analyses": "none"},
        build_core(),
    )

    # First fire: empty batch store -> dispatch the whole workflow.
    out1 = runner.update({"batch": {}})
    assert out1["batch"]["completed"] is True
    assert sorted(out1["batch"]["seeds"]) == ["00", "01"]

    # Second fire with the completed store present -> no-op (fires exactly once).
    out2 = runner.update({"batch": out1["batch"]})
    assert out2 == {}


def test_build_batch_baseline_document_is_cheap_and_well_formed():
    """The generator builds a doc WITHOUT running any baseline (no ParCa)."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.batch_baseline import batch_baseline, BATCH_RUNNER_STEP_NAME

    doc = batch_baseline(core=build_core(), n_seeds=2, n_generations=3,
                         max_duration=1234.0)
    state = doc["state"]
    assert state["batch"] == {}                    # empty until run
    node = state[BATCH_RUNNER_STEP_NAME]
    assert node["_type"] == "step"
    assert "BatchBaselineRunner" in node["address"]
    assert node["config"]["n_seeds"] == 2
    assert node["config"]["n_generations"] == 3
    assert node["config"]["max_duration"] == 1234.0
    assert node["config"]["emitter"] == "both"


def test_batch_baseline_registered_for_build_composite():
    """build_composite resolves the new architecture by name."""
    from viva_superpowers.composite_generator import _REGISTRY
    import v2ecoli.composites  # noqa: F401 — fires the @composite_generator

    names = {e.name for e in _REGISTRY.values()}
    assert "batch_baseline" in names


def test_batch_baseline_exposes_the_vecoli_workflow_knobs():
    """The Setup & Run form renders these; they are the batch's whole contract."""
    from viva_superpowers.composite_generator import _REGISTRY
    import v2ecoli.composites  # noqa: F401

    entry = next(e for e in _REGISTRY.values() if e.name == "batch_baseline")
    assert {"n_seeds", "n_generations", "base_seed", "single_daughters",
            "time_step", "max_duration", "variants", "emitter", "analyses",
            "study", "cache_dir", "out_dir", "experiment_id", "parallel"} <= set(
                entry.parameters)


def test_batch_baseline_declares_the_multi_scale_visualizations():
    """A seeds x generations batch ships the single-cell gallery PLUS the
    multigeneration/multiseed panels vEcoli's workflow emits."""
    from viva_superpowers.composite_generator import _REGISTRY
    import v2ecoli.composites  # noqa: F401

    entry = next(e for e in _REGISTRY.values() if e.name == "batch_baseline")
    names = {v["name"] for v in entry.visualizations}
    assert {"cell_mass", "mass_fraction_summary"} <= names          # single-cell
    assert {"ribosome_usage", "new_gene_counts"} <= names           # multigeneration
    assert {"doubling_time_distribution"} <= names                  # multiseed


# --- sim_data pairing --------------------------------------------------------

def test_link_sim_data_pairs_the_sweep_with_its_parca_cache(tmp_path):
    """Analyses resolve sim_data from a sweep-LOCAL pickle first; a batch runs
    every lineage from cache_dir, so that cache's pickle is the exact pairing."""
    from v2ecoli.steps.batch_baseline_runner import link_sim_data

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "simData.cPickle").write_bytes(b"parca")
    out = tmp_path / "sweep"

    dest = link_sim_data(str(out), str(cache))
    assert dest == str(out / "simData.cPickle")
    assert (out / "simData.cPickle").read_bytes() == b"parca"


def test_link_sim_data_is_a_noop_without_a_cache_pickle(tmp_path):
    from v2ecoli.steps.batch_baseline_runner import link_sim_data
    assert link_sim_data(str(tmp_path / "sweep"), str(tmp_path / "empty")) is None


def test_link_sim_data_never_clobbers_a_sweep_local_pickle(tmp_path):
    from v2ecoli.steps.batch_baseline_runner import link_sim_data

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "simData.cPickle").write_bytes(b"cache")
    out = tmp_path / "sweep"
    out.mkdir()
    (out / "simData.cPickle").write_bytes(b"the sweep's own")

    link_sim_data(str(out), str(cache))
    assert (out / "simData.cPickle").read_bytes() == b"the sweep's own"


def test_dispatch_skips_sim_data_pairing_when_no_analyses_run(tmp_path):
    """analyses="none" means no DuckDB reads, so don't touch the sweep dir."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "simData.cPickle").write_bytes(b"parca")
    out = tmp_path / "sweep"

    dispatch_batch(n_seeds=1, n_generations=1, cache_dir=str(cache),
                   out_dir=str(out), analyses="none",
                   run_workflow_fn=_stub_workflow((0,)))
    assert not (out / "simData.cPickle").exists()


# --- run-scoped output dir ---------------------------------------------------

def test_out_dir_defaults_to_the_workbench_run_sweep_dir(monkeypatch):
    """A workbench run points ParquetAnalysisView at <run_dir>/parquet/<run_id>;
    writing the sweep anywhere else leaves every declared visualization saying
    'no parquet history under the run's sweep dir yet' (the bug this fixes), and
    lets consecutive runs overwrite one another."""
    from v2ecoli.steps.batch_baseline_runner import (
        DEFAULT_OUT_DIR, resolve_out_dir)

    monkeypatch.delenv("VIVARIUM_WORKBENCH_SWEEP_DIR", raising=False)
    assert resolve_out_dir() == DEFAULT_OUT_DIR
    assert resolve_out_dir("") == DEFAULT_OUT_DIR

    monkeypatch.setenv("VIVARIUM_WORKBENCH_SWEEP_DIR", "/ws/.pbg/runs/r1/parquet/r1")
    assert resolve_out_dir() == "/ws/.pbg/runs/r1/parquet/r1"
    # An explicit out_dir always wins over the run's.
    assert resolve_out_dir("out/mine") == "out/mine"


def test_dispatch_uses_the_run_sweep_dir_end_to_end(monkeypatch):
    monkeypatch.setenv("VIVARIUM_WORKBENCH_SWEEP_DIR", "/ws/.pbg/runs/r1/parquet/r1")
    batch = dispatch_batch(n_seeds=1, n_generations=1, analyses="none",
                           run_workflow_fn=_stub_workflow((0,)))
    assert _stub_workflow.last_config["out_dir"] == "/ws/.pbg/runs/r1/parquet/r1"
    assert batch["out_dir"] == "/ws/.pbg/runs/r1/parquet/r1"
    assert batch["outputs"]["viz_dir"] == "/ws/.pbg/runs/r1/parquet/r1/viz"
    assert batch["seeds"]["00"]["store_path"].startswith("/ws/.pbg/runs/r1/parquet/r1/")


# --- emitter wiring ----------------------------------------------------------

def test_emitter_captures_the_completed_batch(monkeypatch):
    """The run's Results tab reads what the emitter recorded, so the emitter
    Step must run AFTER the batch runner.

    It didn't: `batch` was declared as both an input and an output of the
    runner, and build_step_network deliberately skips self-loops ("self-loops
    can't trigger"), so NOTHING was registered as producing `batch`. The
    emitter's dependency on it counted as satisfied before the runner had run,
    both landed in the same layer, and every run emitted the empty pre-dispatch
    store. triggers() makes `batch` a silent input, restoring the edge.
    """
    from process_bigraph import Composite

    from v2ecoli.composites.batch_baseline import batch_baseline
    from v2ecoli.core import build_core

    monkeypatch.setattr(
        bbr, "dispatch_batch",
        lambda **kw: {"completed": True, "n_seeds": 1, "mode": "stub"})

    core = build_core()
    state = batch_baseline(core=core, n_seeds=1, analyses="none")["state"]
    # The emitter the workbench injects for the run's selected paths, inlined
    # (mirrors vivarium_workbench.lib.composite_runs.inject_emitter_for_paths)
    # so this test doesn't depend on the workbench being installed.
    state["user_emitter"] = {
        "_type": "step",
        "address": "local:RAMEmitter",
        "config": {"emit": {"batch": "node", "global_time": "node"}},
        "inputs": {"batch": ["batch"], "global_time": ["global_time"]},
    }
    composite = Composite({"state": state}, core=core)
    composite.run(1)

    rows = composite.state["user_emitter"]["instance"].history
    assert len(rows) == 1
    emitted = rows[0]["batch"]
    assert emitted.get("completed") is True, (
        "the emitter recorded the batch store BEFORE the runner wrote it — "
        "the run's Results tab would show an empty batch")
    assert emitted.get("mode") == "stub"
    # What the emitter recorded IS the run's final state, not a snapshot of it.
    live = composite.state["batch"]
    assert {k: v for k, v in emitted.items() if k != "_value"} == \
           {k: v for k, v in live.items() if k != "_value"}


def test_runner_declares_no_scheduling_triggers():
    """Guard the contract directly: batch must not be a scheduling input."""
    from v2ecoli.composites._helpers import _make_instance
    from v2ecoli.core import build_core

    runner = _make_instance(BatchBaselineRunner, {"n_seeds": 1}, build_core())
    assert runner.triggers() == {}
    assert "batch" in runner.inputs()      # still received, just silent


def test_batch_ports_use_registered_type_name_for_pbg_roundtrip():
    """The `batch` port must be declared by its registered type NAME string, not
    an ``InPlaceDict()`` instance.

    An instance serializes to its repr (``"InPlaceDict(_default=None, ...)"``) in
    ``to_document``, which is not a parseable type expression — so when the
    composite is round-tripped as a ``.pbg`` through remote dispatch (workbench
    export -> sms-api compose -> ``run_pbg`` on Batch), ``Composite()`` dies with
    ``bigraph_schema`` ``IncompleteParseError``. The registered name
    ``"inplace_dict"`` (ECOLI_TYPES) round-trips cleanly and resolves back to the
    ``InPlaceDict`` type via the core.
    """
    from v2ecoli.composites._helpers import _make_instance
    from v2ecoli.core import build_core

    runner = _make_instance(BatchBaselineRunner, {"n_seeds": 1}, build_core())
    assert runner.inputs()["batch"] == "inplace_dict"
    assert runner.outputs()["batch"] == "inplace_dict"
    assert isinstance(runner.inputs()["batch"], str)


def test_inplace_dict_store_serializes_its_merged_result_keys():
    """serialize_state() must capture an inplace_dict store's merged data.

    run_pbg's SOLE output artifact on the remote-dispatch path is
    Composite.serialize_state(); if it drops the batch runner's merged result to
    {"_value": {}}, a successful GovCloud run produces an empty batch. Guard the
    serialize dispatch that emits the merged (non-schema) keys.
    """
    from process_bigraph import Composite
    from v2ecoli.core import build_core

    core = build_core()
    comp = Composite({"state": {"batch": {"_type": "inplace_dict"}}}, core=core)
    # mimic InPlaceDict.apply deep-merging a Step's result onto the store node
    comp.state["batch"].update({"completed": True, "n_seeds": 1, "seeds": {0: {"path": "x"}}})

    out = comp.serialize_state()["batch"]
    assert out.get("completed") is True
    assert out.get("n_seeds") == 1
    assert out.get("seeds") == {0: {"path": "x"}}
    assert "_value" not in out


def test_resolve_out_dir_falls_back_to_compose_results_dir(monkeypatch):
    """Under sms-api run_pbg, land the sweep in PBG_RESULTS_DIR (S3-synced), not
    the unsynced workspace default."""
    from v2ecoli.steps.batch_baseline_runner import resolve_out_dir, DEFAULT_OUT_DIR
    monkeypatch.delenv("VIVARIUM_WORKBENCH_SWEEP_DIR", raising=False)
    monkeypatch.setenv("PBG_RESULTS_DIR", "/tmp/pbg_out")
    assert resolve_out_dir() == "/tmp/pbg_out/batch_baseline"
    assert resolve_out_dir("explicit") == "explicit"          # explicit always wins
    monkeypatch.setenv("VIVARIUM_WORKBENCH_SWEEP_DIR", "/ws/sweep")
    assert resolve_out_dir() == "/ws/sweep"                    # workbench dir wins over compose
    monkeypatch.delenv("VIVARIUM_WORKBENCH_SWEEP_DIR", raising=False)
    monkeypatch.delenv("PBG_RESULTS_DIR", raising=False)
    assert resolve_out_dir() == DEFAULT_OUT_DIR                # neither set -> default
