"""Unit tests for v2ecoli.library.parquet_emitter.ParquetEmitter."""

import os
from pathlib import Path

import pytest

# Skip the entire file if the [parquet] extra isn't installed.
pytest.importorskip("duckdb")
pytest.importorskip("polars")


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
