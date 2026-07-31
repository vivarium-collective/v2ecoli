"""build_emitter_config threads run provenance into the XArrayEmitter config so
the emitter can write it to the zarr store's root attrs (self-describing runs)."""
from pathlib import Path
from v2ecoli.library.xarray_run import build_emitter_config

_MD = {"experiment_id": "e", "variant": 0, "lineage_seed": 0,
       "time_step": 1.0, "max_duration": 10.0}


def test_provenance_in_config():
    cfg = build_emitter_config(
        store_path=Path("/tmp/x.zarr"), view=[], metadata_base=_MD,
        generation=1, agent_id="0",
        provenance={"composite": "v2ecoli.composites.ecoli_baseline.ecoli_baseline",
                    "config": {"seed": 0}, "run_id": "r1"},
    )
    assert cfg["provenance"] == {
        "composite": "v2ecoli.composites.ecoli_baseline.ecoli_baseline",
        "config": {"seed": 0}, "run_id": "r1"}


def test_no_provenance_defaults_empty():
    cfg = build_emitter_config(
        store_path=Path("/tmp/x.zarr"), view=[], metadata_base=_MD,
        generation=1, agent_id="0")
    assert cfg["provenance"] == {}


def test_config_always_strips_the_agents_envelope():
    """Regression: build_emitter_config must always set strategy="colony" and
    emit_root=["agents", agent_id] — every caller emits payloads wrapped in an
    {"agents": {agent_id: ...}} envelope (see run_multigen_xarray, run_one in
    scripts/run_phase0_xarray_ensemble.py). A caller that hand-rolls this config
    instead of using build_emitter_config and omits these two keys will see the
    transducer raise KeyError("Unexpected emit path: ('agents', <id>, ...)") on
    the very first update — this happened for real in scripts/run_phase0_xarray_
    ensemble.py's run_one(), which crashed every GovCloud-dispatched seed."""
    cfg = build_emitter_config(
        store_path=Path("/tmp/x.zarr"), view=[], metadata_base=_MD,
        generation=1, agent_id="7")
    assert cfg["strategy"] == "colony"
    assert cfg["emit_root"] == ["agents", "7"]
