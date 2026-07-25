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
        provenance={"composite": "v2ecoli.composites.baseline.baseline",
                    "config": {"seed": 0}, "run_id": "r1"},
    )
    assert cfg["provenance"] == {
        "composite": "v2ecoli.composites.baseline.baseline",
        "config": {"seed": 0}, "run_id": "r1"}


def test_no_provenance_defaults_empty():
    cfg = build_emitter_config(
        store_path=Path("/tmp/x.zarr"), view=[], metadata_base=_MD,
        generation=1, agent_id="0")
    assert cfg["provenance"] == {}
