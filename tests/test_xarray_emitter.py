"""Unit tests for v2ecoli.library.xarray_emitter."""

import pytest

# Skip the entire file if the [xarray] extra isn't installed.
pytest.importorskip("xarray")
pytest.importorskip("zarr")


@pytest.mark.fast
def test_base_module_imports():
    from v2ecoli.library.xarray_emitter._base import (
        BufferedEmitter, StoragePartition, BlockingExecutor,
    )
    assert BufferedEmitter is not None
    assert StoragePartition is not None
    assert BlockingExecutor is not None


@pytest.mark.fast
def test_storage_partition_dataclass():
    from v2ecoli.library.xarray_emitter._base import StoragePartition

    p = StoragePartition(
        experiment_id="exp1", variant=2, lineage_seed=7, agent_id="01"
    )
    assert p.generation == 2  # len(agent_id)
    assert p.parent.agent_id == "0"


@pytest.mark.fast
def test_buffered_emitter_inherits_pbg_emitter():
    from v2ecoli.library.xarray_emitter._base import BufferedEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(BufferedEmitter, Emitter)


@pytest.mark.fast
def test_all_submodules_import():
    """Smoke check: every vendored sub-module imports without error."""
    for name in (
        "transducer", "view", "storage", "writer", "zarr_writer",
        "emit_path", "emit_predicate", "utils",
    ):
        __import__(f"v2ecoli.library.xarray_emitter.{name}")


@pytest.fixture
def core():
    from bigraph_schema import allocate_core
    return allocate_core()


@pytest.mark.fast
def test_xarray_emitter_imports():
    from v2ecoli.library.xarray_emitter import XArrayEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(XArrayEmitter, Emitter)


@pytest.mark.fast
def test_xarray_emitter_config_schema_has_expected_keys():
    from v2ecoli.library.xarray_emitter import XArrayEmitter
    for key in (
        "emit", "out_uri", "transducer", "view", "writer",
        "metadata", "metadata_keys", "metadata_validators",
        "output_metadata", "debug",
    ):
        assert key in XArrayEmitter.config_schema, f"missing: {key}"


@pytest.mark.fast
def test_metadata_validators_failure_raises(tmp_path, core):
    from v2ecoli.library.xarray_emitter import XArrayEmitter

    cfg = {
        "emit": {},
        "out_uri": str(tmp_path / "x.zarr"),
        "transducer": {},
        "view": [],
        "writer": {"type": "async", "out_uri": str(tmp_path / "x.zarr")},
        "metadata": {"single_daughters": False},
        "metadata_validators": {"single_daughters": True},
    }
    with pytest.raises(ValueError, match="single_daughters"):
        XArrayEmitter(config=cfg, core=core)


@pytest.mark.fast
def test_empty_metadata_validators_no_op(tmp_path, core):
    """Empty validators dict => no validation error from the validator path."""
    from v2ecoli.library.xarray_emitter import XArrayEmitter

    cfg = {
        "emit": {},
        "out_uri": str(tmp_path / "x.zarr"),
        "transducer": {},
        "view": [],
        "writer": {"type": "async", "out_uri": str(tmp_path / "x.zarr")},
        "metadata": {"anything": "goes"},
        "metadata_validators": {},
    }
    # Should not raise a validator-related ValueError. Other failures from
    # transducer/writer internals are out-of-scope here.
    try:
        XArrayEmitter(config=cfg, core=core)
    except ValueError as e:
        if "single_daughters" in str(e) or "validator" in str(e).lower():
            pytest.fail(f"unexpected validator error: {e}")
