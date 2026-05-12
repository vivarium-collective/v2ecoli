"""Unit tests for v2ecoli.library.parquet_emitter.ParquetEmitter."""

import os
from pathlib import Path

import numpy as np
import pytest

# Skip the entire file if the [parquet] extra isn't installed.
pytest.importorskip("duckdb")
pytest.importorskip("polars")


@pytest.fixture
def core():
    from bigraph_schema import allocate_core
    return allocate_core()


@pytest.mark.fast
def test_import_succeeds_with_parquet_extra():
    from v2ecoli.library.parquet_emitter import ParquetEmitter
    assert ParquetEmitter is not None


@pytest.mark.fast
def test_class_inherits_process_bigraph_emitter():
    from v2ecoli.library.parquet_emitter import ParquetEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(ParquetEmitter, Emitter)


@pytest.mark.fast
def test_class_has_expected_config_schema_keys():
    from v2ecoli.library.parquet_emitter import ParquetEmitter
    schema = ParquetEmitter.config_schema
    for key in (
        "emit", "out_dir", "out_uri", "batch_size", "threaded",
        "flatten_separator", "partitioning_keys", "dtype_overrides",
        "metadata",
    ):
        assert key in schema, f"missing config_schema key: {key}"


@pytest.mark.fast
def test_roundtrip_no_partitioning(tmp_path, core):
    """Write 5 ticks of synthetic state, query back, verify rows and dtypes."""
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    emitter = ParquetEmitter(
        config={
            "emit": {"x": "node", "y": "node"},
            "out_dir": str(tmp_path / "out"),
            "batch_size": 2,
            "threaded": False,
        },
        core=core,
    )
    for i in range(5):
        emitter.update({"x": float(i), "y": int(i * 10)})
    emitter.close()

    df = emitter.query()
    assert len(df) == 5
    assert set(df.columns) >= {"x", "y"}
    assert df["x"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert df["y"].to_list() == [0, 10, 20, 30, 40]


@pytest.mark.fast
def test_dtype_overrides_exact_name(tmp_path, core):
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    emitter = ParquetEmitter(
        config={
            "emit": {"counts": "node"},
            "out_dir": str(tmp_path / "out"),
            "batch_size": 1,
            "threaded": False,
            "dtype_overrides": {"counts": "UInt16"},
        },
        core=core,
    )
    emitter.update({"counts": 42})
    emitter.close()

    df = emitter.query()
    assert str(df.schema["counts"]) == "UInt16"


@pytest.mark.fast
def test_dtype_overrides_fnmatch_glob(tmp_path, core):
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    emitter = ParquetEmitter(
        config={
            "emit": {"listeners__rna_synth_prob__foo": "node"},
            "out_dir": str(tmp_path / "out"),
            "batch_size": 1,
            "threaded": False,
            "dtype_overrides": {"listeners__rna_synth_prob__*": "UInt32"},
        },
        core=core,
    )
    emitter.update({"listeners__rna_synth_prob__foo": 1000})
    emitter.close()

    df = emitter.query()
    assert str(df.schema["listeners__rna_synth_prob__foo"]) == "UInt32"


@pytest.mark.fast
def test_partitioning_keys_hive_layout(tmp_path, core):
    """With partitioning_keys, output should land at the expected hive path."""
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    out_dir = tmp_path / "out"
    emitter = ParquetEmitter(
        config={
            "emit": {"v": "node"},
            "out_dir": str(out_dir),
            "batch_size": 1,
            "threaded": False,
            "partitioning_keys": ["experiment_id", "variant"],
            "metadata": {"experiment_id": "exp1", "variant": 2},
        },
        core=core,
    )
    emitter.update({"v": 1.0})
    emitter.close()

    expected_history = out_dir / "exp1" / "history" / "experiment_id=exp1" / "variant=2"
    assert expected_history.exists(), f"missing hive dir: {expected_history}"
    parquet_files = list(expected_history.glob("*.pq"))
    assert len(parquet_files) >= 1


@pytest.mark.fast
def test_partitioning_missing_key_raises_keyerror(tmp_path, core):
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    with pytest.raises(KeyError, match="experiment_id"):
        ParquetEmitter(
            config={
                "emit": {"v": "node"},
                "out_dir": str(tmp_path / "out"),
                "partitioning_keys": ["experiment_id"],
                "metadata": {"variant": 0},  # missing experiment_id
            },
            core=core,
        )


@pytest.mark.fast
def test_close_idempotent(tmp_path, core):
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    emitter = ParquetEmitter(
        config={
            "emit": {"v": "node"},
            "out_dir": str(tmp_path / "out"),
            "batch_size": 4,
            "threaded": False,
        },
        core=core,
    )
    emitter.update({"v": 1})
    emitter.close()
    # Second close is a no-op
    emitter.close()


@pytest.mark.fast
def test_close_with_success_writes_sentinel_when_partitioned(tmp_path, core):
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    out_dir = tmp_path / "out"
    emitter = ParquetEmitter(
        config={
            "emit": {"v": "node"},
            "out_dir": str(out_dir),
            "batch_size": 1,
            "threaded": False,
            "partitioning_keys": ["experiment_id"],
            "metadata": {"experiment_id": "exp1"},
        },
        core=core,
    )
    emitter.update({"v": 1})
    emitter.close(success=True)

    sentinel = out_dir / "exp1" / "success" / "experiment_id=exp1" / "s.pq"
    assert sentinel.exists(), f"missing success sentinel: {sentinel}"


@pytest.mark.fast
def test_close_with_success_no_sentinel_when_not_partitioned(tmp_path, core):
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    out_dir = tmp_path / "out"
    emitter = ParquetEmitter(
        config={
            "emit": {"v": "node"},
            "out_dir": str(out_dir),
            "batch_size": 1,
            "threaded": False,
        },
        core=core,
    )
    emitter.update({"v": 1})
    emitter.close(success=True)
    # No partitioning => no sentinel anywhere
    assert not list(out_dir.rglob("s.pq"))
