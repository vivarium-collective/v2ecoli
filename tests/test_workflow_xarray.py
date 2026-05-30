"""Workflow xarray-emitter wiring — unit + a cache-gated smoke."""
import os

import pytest

from v2ecoli.workflow.lineage import LineageProcess, DEFAULT_XARRAY_VIEW
from v2ecoli.workflow.meta_composite import build_meta_composite

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")


def test_default_xarray_view_is_scalar_mass():
    entry = DEFAULT_XARRAY_VIEW[0]
    assert entry["root"] == ("listeners", "mass")
    for gauge in ("dry_mass", "cell_mass", "protein_mass"):
        assert gauge in entry["variables"]
        assert entry["variables"][gauge][0]["path"] == gauge


def test_is_xarray_reflects_config():
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {}                       # default
    assert lp._is_xarray() is False
    lp.config = {"emitter": "parquet"}
    assert lp._is_xarray() is False
    lp.config = {"emitter": "xarray"}
    assert lp._is_xarray() is True


def test_meta_composite_threads_emitter_config():
    config = {
        "experiment_id": "x", "n_init_sims": 1, "generations": 1, "variants": {},
        "emitter": "xarray",
        "emitter_arg": {"out_dir": "out/x/zarr",
                        "view": [{"root": ["listeners", "mass"],
                                  "variables": {"dry_mass": [{"path": "dry_mass", "dtype": "<f4"}]}}]},
    }
    node = build_meta_composite(config)["state"]["branches"]["variant=0/seed=0"]["lineage"]
    assert node["config"]["emitter"] == "xarray"
    assert node["config"]["emitter_arg"]["out_dir"] == "out/x/zarr"
    # default is parquet when unspecified
    plain = build_meta_composite({"experiment_id": "p", "n_init_sims": 1,
                                  "generations": 1, "variants": {}})
    assert plain["state"]["branches"]["variant=0/seed=0"]["lineage"]["config"]["emitter"] == "parquet"


def test_ported_xarray_config_loads():
    from v2ecoli.workflow.config import load_config_with_inheritance
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "v2ecoli", "configs")
    cfg = load_config_with_inheritance(os.path.join(cfg_dir, "two_generations_xarray.json"))
    assert cfg["emitter"] == "xarray"
    assert cfg["generations"] == 2                      # inherited from two_generations.json
    assert cfg["emitter_arg"]["view"][0]["root"] == ["listeners", "mass"]


@pytest.mark.skipif(not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")
def test_xarray_sweep_emits_zarr(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import xarray as xr
    from v2ecoli.workflow.run import run_workflow

    out = str(tmp_path / "zarr")
    config = {
        "experiment_id": "xsmoke", "n_init_sims": 1, "generations": 1,
        "single_daughters": True, "cache_dir": CACHE, "out_dir": str(tmp_path),
        "variants": {}, "max_duration_per_gen": 5.0, "time_step": 1.0,
        "emitter": "xarray", "emitter_arg": {"out_dir": out},   # default mass view
    }
    result = run_workflow(config, max_sim_time=20.0)
    assert result["complete"] is True

    store = os.path.join(out, "xsmoke_v0_s0.zarr")
    assert os.path.isdir(store), f"no zarr store written at {store}"
    dt = xr.open_datatree(store, engine="zarr")
    groups = list(dt.groups)
    assert any("dry_mass" in g for g in groups), f"dry_mass not in {groups}"


def test_emitter_config_threads_writer_predicate_and_codecs():
    """Follow-up: emitter_arg.writer / transducer.predicate and per-leaf
    unit/codecs are honored by the config builder."""
    from v2ecoli.library.xarray_run import build_emitter_config
    view = [{"root": ("listeners", "mass"),
             "variables": {"dry_mass": [{"path": "dry_mass", "dtype": "<f4",
                                         "unit": "[fg]",
                                         "codecs": {"compressors_v3": []}}]}}]
    cfg = build_emitter_config(
        store_path="/tmp/x.zarr", view=view, metadata_base={"experiment_id": "t"},
        generation=2, agent_id="00", buffer_size=7,
        writer={"buffers_per_chunk": 5, "backend_config": {"async.concurrency": 3}},
        predicate=[[{"subsample": {"interval": 3}}]])
    w = cfg["writer"]
    assert w["buffers_per_chunk"] == 5                 # user setting honored
    assert w["backend_config"]["async.concurrency"] == 3
    assert w["backend_config"]["format"] == 3          # default preserved on merge
    assert w["store"] == "/tmp/x.zarr"                 # store always forced
    assert cfg["transducer"]["predicate"] == [[{"subsample": {"interval": 3}}]]
    assert cfg["transducer"]["buffer"]["size"] == 7
    leaf = cfg["view"][0]["variables"]["dry_mass"][0]
    assert leaf["unit"] == "[fg]" and "codecs" in leaf  # pass through verbatim
    assert cfg["metadata"]["generation"] == 2 and cfg["metadata"]["agent_id"] == "00"


@pytest.mark.skipif(not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")
def test_null_emitter_captures_only_global_time():
    """Follow-up: on xarray runs the internal emitter is minimised so it
    doesn't waste memory capturing full state."""
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    from v2ecoli.composites._helpers import set_null_emitter_override

    core = build_core()
    set_null_emitter_override(True)
    try:
        doc = baseline(core=core, seed=0, cache_dir=CACHE)
    finally:
        set_null_emitter_override(False)
    comp = Composite(doc, core=core)
    comp.run(1.0)
    em = comp.state["agents"]["0"]["emitter"]["instance"]
    hist = getattr(em, "history", None)
    assert hist, "null emitter captured nothing"
    keys = set(hist[-1].keys())
    assert "global_time" in keys
    assert "bulk" not in keys and "listeners" not in keys  # heavy capture suppressed


@pytest.mark.skipif(not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")
def test_xarray_sweep_emits_vector_observable(tmp_path):
    """Follow-up: a vector observable (monomer_counts) emits with its coord
    dimension via the view filter + output_metadata discovery."""
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import xarray as xr
    from v2ecoli.workflow.run import run_workflow

    out = str(tmp_path / "zarr")
    config = {
        "experiment_id": "xvec", "n_init_sims": 1, "generations": 1,
        "single_daughters": True, "cache_dir": CACHE, "out_dir": str(tmp_path),
        "variants": {}, "max_duration_per_gen": 5.0, "time_step": 1.0,
        "emitter": "xarray",
        "emitter_arg": {"out_dir": out, "view": [
            {"root": ["listeners"],
             "variables": {"monomer_counts": [{"path": "monomer_counts", "dtype": "<i8"}]}}]},
    }
    result = run_workflow(config, max_sim_time=20.0)
    assert result["complete"] is True

    store = os.path.join(out, "xvec_v0_s0.zarr")
    assert os.path.isdir(store), f"no zarr store at {store}"
    dt = xr.open_datatree(store, engine="zarr")
    groups = [g for g in dt.groups if g.endswith("monomer_counts")]
    assert groups, f"monomer_counts not in {list(dt.groups)}"
    node = dt[groups[0]].to_dataset()
    # at least one variable carries a non-time dimension > 1 (the vector coord)
    assert any(any(s > 1 for s in v.shape) for v in node.data_vars.values()), \
        {k: v.shape for k, v in node.data_vars.items()}
