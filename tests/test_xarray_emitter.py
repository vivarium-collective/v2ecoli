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
