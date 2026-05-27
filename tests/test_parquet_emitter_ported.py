"""Tests for v2ecoli.library.parquet_emitter — ported from vEcoli's
ecoli/library/test_parquet_emitter.py (PR #414 head, ~1530 lines).

Helper-function tests (TestHelperFunctions) are near-verbatim ports. Integration
tests (TestParquetEmitter, TestParquetEmitterEdgeCases) are rewritten against
v2ecoli's process-bigraph API:

    vEcoli:  emit({"table": "configuration", "data": {"experiment_id": ..., "agents": {...}}})
             emit({"table": "simulation",    "data": {"time": ..., "agents": {agent_id: state}}})
             finalize()
    v2ecoli: ParquetEmitter(config={"metadata": {...}, "partitioning_keys": [...]}, core)
             emitter.update(state)
             emitter.close(success=...)

Scenario coverage (variable shapes, ragged, extreme dtypes, nested nullable,
multithreaded buffer clearing, etc.) is preserved. vEcoli's
``test_multiple_agents`` is not portable — v2ecoli's ``update(state)`` takes
a single composite tick directly; the multi-agent gating belongs at the
composite/runner layer.
"""

from __future__ import annotations

import datetime
import math
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from unittest.mock import Mock, patch

import duckdb
import numpy as np
import polars as pl
import pytest

from v2ecoli.library.emitter_presets import VECOLI_PARQUET_DTYPE_OVERRIDES
from v2ecoli.library.parquet_emitter import (
    ParquetEmitter,
    create_duckdb_conn,
    flatten_dict,
    json_to_parquet,
    list_columns,
    named_idx,
    ndidx_to_duckdb_expr,
    np_dtype,
    quote_columns,
    union_pl_dtypes,
)


# ============================================================================
# Helper-function tests (TestHelperFunctions) — near-verbatim from vEcoli.
# ============================================================================


class TestHelperFunctions:
    @pytest.fixture
    def query_conn(self):
        conn = duckdb.connect(":memory:")
        df = pl.DataFrame(  # noqa: F841
            {
                "a": [[0.1, 0.0, 0.3], [0.4, 0.5, 0.0], [None, 0.8, 0.9]],
                "b": [
                    [[0.1, 0.2], [0.3, None]],
                    [[0.5, 0.6], [0.0, 0.8]],
                    [[0.9, 0.0], [1.1, 1.2]],
                ],
                "c": [[[0.1, 0.2], [0.3]], [[0.5], [0.0, 0.8]], [[0.9], [1.1]]],
            }
        )
        conn.sql("CREATE OR REPLACE TABLE test_table AS SELECT * FROM df")
        yield conn

    def test_named_idx(self, query_conn):
        col_expr = named_idx("a", ["col1", "col2", "col3"], [[0, 1, 2]])
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {"col1": [0.1, 0.4, None], "col2": [0.0, 0.5, 0.8], "col3": [0.3, 0.0, 0.9]}
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "a", ["col1", "col2", "col3"], [[0, 1, 2]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.4, None],
                "col2": [None, 0.5, 0.8],
                "col3": [0.3, None, 0.9],
            }
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "b", ["col1", "col2", "col3", "col4"], [[0, 1], [0, 1]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.5, 0.9],
                "col2": [0.2, 0.6, None],
                "col3": [0.3, None, 1.1],
                "col4": [None, 0.8, 1.2],
            }
        )
        assert result.equals(expected)

    def test_ndidx_to_duckdb_expr(self, query_conn):
        expr = ndidx_to_duckdb_expr("b", [0, 1])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.2]], [[0.6]], [[0.0]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("b", [":", [True, False]])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.1], [0.3]], [[0.5], [0.0]], [[0.9], [1.1]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("c", [[0], ":"])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"c": [[[0.1, 0.2]], [[0.5]], [[0.9]]]})
        assert result.equals(expected)

    def test_flatten_dict(self):
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        assert flatten_dict({"a": {"b": 1, "c": 2}, "d": 3}) == {
            "a__b": 1,
            "a__c": 2,
            "d": 3,
        }
        assert flatten_dict({"a": {"b": {"c": {"d": 1}}}, "e": 2}) == {
            "a__b__c__d": 1,
            "e": 2,
        }
        assert flatten_dict({}) == {}
        nested = flatten_dict({"a": [1, 2, 3], "b": {"c": np.array([4, 5, 6])}})
        assert nested["a"] == [1, 2, 3]
        np.testing.assert_array_equal(nested["b__c"], np.array([4, 5, 6]))

    def test_np_dtype(self):
        # Basic types
        assert np_dtype(1.0, "float_field") == np.float64
        assert np_dtype(True, "bool_field") == np.bool_
        assert np_dtype("text", "string_field") == np.dtypes.StringDType
        assert np_dtype(42, "int_field") == np.int64

        # Override path (v2ecoli equivalent of vEcoli's hard-coded USE_UINT16 / USE_UINT32)
        overrides = VECOLI_PARQUET_DTYPE_OVERRIDES
        assert (
            np_dtype(42, "listeners__ribosome_data__mRNA_TU_index", overrides)
            == np.uint16
        )
        assert np_dtype(42, "listeners__monomer_counts", overrides) == np.uint32

        # Arrays with various dimensions
        assert np_dtype(np.array([1, 2, 3]), "array1d_field") == np.int64
        assert np_dtype(np.array([[1, 2], [3, 4]]), "array2d_field") == np.int64
        # Empty arrays still have a dtype
        assert np_dtype(np.array([]), "empty_array_field") == np.float64

        # Raise to trigger Polars fallback in the emitter
        with pytest.raises(ValueError, match="empty_list_field has unsupported"):
            np_dtype([[], [], None], "empty_list_field")
        with pytest.raises(ValueError, match="none_field has unsupported"):
            np_dtype(None, "none_field")
        with pytest.raises(ValueError, match="complex_field has unsupported type"):
            np_dtype(complex(1, 2), "complex_field")

    def test_union_pl_dtypes(self):
        # Basic types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Int32, pl.Int64, "fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Float32, pl.String, "fail")

        # Nested types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.Int16), pl.List(pl.Int64), "nest")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.UInt16), pl.List(pl.String), "nest_fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.List(pl.UInt16)), pl.List(pl.String), "nest_fail"
            )
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.UInt16), pl.List(pl.Array(pl.String, (1,))), "nest_fail"
            )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)

        # Forced types
        assert union_pl_dtypes(pl.Int16, pl.UInt8, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(pl.UInt16, pl.Int64, "force_u32", pl.UInt32) == pl.UInt32
        assert (
            union_pl_dtypes(pl.UInt16, pl.String, "force_u32", pl.UInt32) == pl.UInt32
        )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.String), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.Array(pl.UInt16, (1, 1)),
            pl.List(pl.List(pl.Int64)),
            "force_u16",
            pl.UInt16,
        ) == pl.List(pl.List(pl.UInt16))

        # Null merge
        assert union_pl_dtypes(pl.Null, pl.Int64, "null_merge") == pl.Int64
        assert union_pl_dtypes(pl.Null, pl.Float64, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(
            pl.Null, pl.List(pl.Int64), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Float32)), "null_merge"
        ) == pl.List(pl.List(pl.Float32))
        assert union_pl_dtypes(
            pl.Array(pl.Null, (1, 1, 1)),
            pl.List(pl.Array(pl.Float32, (1, 1))),
            "null_merge",
        ) == pl.List(pl.List(pl.List(pl.Float32)))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.String), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Int32)), "force_u32", pl.UInt32
        ) == pl.List(pl.List(pl.UInt32))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.List(pl.Int32))), "null_merge"
        ) == pl.List(pl.List(pl.List(pl.Int32)))
        assert union_pl_dtypes(
            pl.List(pl.Null),
            pl.List(pl.List(pl.List(pl.Int32))),
            "force_u32",
            pl.UInt32,
        ) == pl.List(pl.List(pl.List(pl.UInt32)))

    def test_quote_columns(self):
        # Singles
        assert quote_columns("simple") == '"simple"'
        assert quote_columns("with spaces") == '"with spaces"'
        assert quote_columns("with-hyphens") == '"with-hyphens"'
        assert quote_columns("with[brackets]") == '"with[brackets]"'
        assert quote_columns("with/slashes") == '"with/slashes"'
        # Pre-quoted (must be escaped)
        assert quote_columns('already"quoted') == '"already""quoted"'
        assert quote_columns('"fully"quoted"') == '"""fully""quoted"""'
        # Lists
        assert quote_columns(["col1", "col2", "col3"]) == ['"col1"', '"col2"', '"col3"']
        assert quote_columns(["with spaces", "with-hyphens"]) == [
            '"with spaces"',
            '"with-hyphens"',
        ]
        assert quote_columns(["normal", "space here", "hyphen-here", 'quote"here']) == [
            '"normal"',
            '"space here"',
            '"hyphen-here"',
            '"quote""here"',
        ]
        # Empty cases
        assert quote_columns("") == '""'
        assert quote_columns([]) == []

        # End-to-end with DuckDB
        with tempfile.TemporaryDirectory() as tmp_path:
            test_file = os.path.join(tmp_path, "weird_cols.parquet")
            test_data = pl.DataFrame(
                {
                    "simple": [1, 2, 3],
                    "with spaces": [4, 5, 6],
                    "with-hyphens": [7, 8, 9],
                    "with[brackets]": [10, 11, 12],
                    "with/slashes": [13, 14, 15],
                    'has"quote': [16, 17, 18],
                    "dot.name": [19, 20, 21],
                    "colon:name": [22, 23, 24],
                }
            )
            test_data.write_parquet(test_file, statistics=False)
            conn = create_duckdb_conn()
            for col in test_data.columns:
                quoted_col = quote_columns(col)
                result = conn.sql(f"SELECT {quoted_col} FROM '{test_file}'").pl()
                assert result.shape == (3, 1)
                assert result.columns[0] == col
                assert result[col].to_list() == test_data[col].to_list()
            weird_cols = ["with spaces", "with-hyphens", "with[brackets]", 'has"quote']
            quoted_cols = ", ".join(quote_columns(weird_cols))
            result = conn.sql(f"SELECT {quoted_cols} FROM '{test_file}'").pl()
            assert result.shape == (3, 4)
            for col in weird_cols:
                assert col in result.columns
                assert result[col].to_list() == test_data[col].to_list()

    def test_list_columns(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            test_file = os.path.join(tmp_path, "test.parquet")
            test_data = pl.DataFrame(
                {
                    "col_a": [1, 2, 3],
                    "col_b": [4.0, 5.0, 6.0],
                    "listeners__mass__cell_mass": [7.0, 8.0, 9.0],
                    "listeners__mass__dry_mass": [10.0, 11.0, 12.0],
                    "listeners__growth__instantaneous_growth_rate": [0.1, 0.2, 0.3],
                    "bulk": [[1, 2], [3, 4], [5, 6]],
                }
            )
            test_data.write_parquet(test_file, statistics=False)
            conn = create_duckdb_conn()
            subquery = f"SELECT * FROM '{test_file}'"
            all_cols = list_columns(conn, subquery)
            assert len(all_cols) == 6
            assert "col_a" in all_cols
            assert "col_b" in all_cols
            assert "listeners__mass__cell_mass" in all_cols
            listener_cols = list_columns(conn, subquery, "listeners__*")
            assert len(listener_cols) == 3
            assert all(col.startswith("listeners__") for col in listener_cols)
            mass_cols = list_columns(conn, subquery, "listeners__mass__*")
            assert len(mass_cols) == 2
            assert "listeners__mass__cell_mass" in mass_cols
            assert "listeners__mass__dry_mass" in mass_cols
            no_match = list_columns(conn, subquery, "nonexistent__*")
            assert len(no_match) == 0
            col_pattern = list_columns(conn, subquery, "col_?")
            assert len(col_pattern) == 2
            assert "col_a" in col_pattern
            assert "col_b" in col_pattern
            exact = list_columns(conn, subquery, "bulk")
            assert exact == ["bulk"]


def compare_nested(a, b) -> bool:
    """Compare two values for equality, with NaN-aware semantics."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(compare_nested(a[i], b[i]) for i in range(len(a)))
    if a != b:
        try:
            return math.isnan(a) and math.isnan(b)
        except TypeError:
            return False
    return True


# ============================================================================
# Integration tests (TestParquetEmitter) — rewritten against v2ecoli API.
# ============================================================================


def _vecoli_metadata(**overrides):
    """vEcoli-style metadata block for partitioning into experiment/variant/etc."""
    md = {
        "experiment_id": "test_exp",
        "variant": 1,
        "lineage_seed": 1,
        "generation": 1,
        "agent_id": "1",
    }
    md.update(overrides)
    return md


VECOLI_PARTITION_KEYS = [
    "experiment_id", "variant", "lineage_seed", "generation", "agent_id"
]


class TestParquetEmitter:
    @pytest.fixture
    def temp_dir(self):
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)

    def test_initialization(self, temp_dir, core):
        """ParquetEmitter accepts out_dir, out_uri, and batch_size."""
        emitter = ParquetEmitter(
            config={"out_dir": temp_dir, "threaded": False}, core=core,
        )
        assert emitter.out_uri == os.path.abspath(temp_dir)
        assert emitter.batch_size == 400

        emitter2 = ParquetEmitter(
            config={
                "out_uri": "gs://bucket/path",
                "batch_size": 100,
                "threaded": False,
            },
            core=core,
        )
        assert emitter2.out_uri == "gs://bucket/path"
        assert emitter2.batch_size == 100

    def test_emit_configuration(self, temp_dir, core):
        """Metadata in config drives the one-shot configuration emit."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(
                    nested={"value": 42},
                    meta1="value1",
                    meta2="value2",
                ),
            },
            core=core,
        )
        # Verify partitioning path
        assert emitter.experiment_id == "test_exp"
        assert "experiment_id=test_exp" in emitter.partitioning_path
        assert "variant=1" in emitter.partitioning_path

        # Configuration parquet is written under the partition path.
        emitter.last_batch_future.result()
        config_path = os.path.join(
            emitter.out_uri,
            emitter.experiment_id,
            "configuration",
            emitter.partitioning_path,
            "config.pq",
        )
        assert os.path.exists(config_path), f"missing config.pq: {config_path}"
        cfg = pl.read_parquet(config_path)
        # Nested metadata is flattened
        assert "nested__value" in cfg.columns
        assert cfg["nested__value"].to_list() == [42]
        assert cfg["meta1"].to_list() == ["value1"]

    def test_emit_simulation_data(self, temp_dir, core):
        """Simulation rows of mixed types round-trip through Parquet."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 2,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        state = {
            "time": 1.0,
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "string_field": "hello",
            "array_field": np.array([1, 2, 3]),
            "nested": {"value": 100},
        }

        emitter.update(state)
        assert emitter.num_emits == 1
        # First row: buffered as fixed-shape ndarray
        assert "int_field" in emitter.buffered_emits
        assert emitter.buffered_emits["int_field"][0] == 42
        assert emitter.buffered_emits["float_field"][0] == 3.14
        assert emitter.buffered_emits["bool_field"][0]
        assert emitter.buffered_emits["string_field"][0] == "hello"
        np.testing.assert_array_equal(
            emitter.buffered_emits["array_field"][0], np.array([1, 2, 3])
        )
        assert emitter.buffered_emits["nested__value"][0] == 100

        # Second emit triggers flush
        emitter.update(state)
        assert emitter.num_emits == 2
        emitter.last_batch_future.result()

        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "*.pq",
            )
        )
        assert t["int_field"].to_list() == [42] * 2
        assert t["float_field"].to_list() == [3.14] * 2
        assert t["bool_field"].to_list() == [True] * 2
        assert all(t["string_field"] == ["hello"] * 2)
        np.testing.assert_array_equal(t["array_field"].to_list(), [[1, 2, 3]] * 2)
        assert all(t["nested__value"] == [100] * 2)

    def test_variable_length_arrays(self, temp_dir, core):
        """Arrays with shifting shapes fall back to the Polars list path."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 3,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({
            "time": 1.0,
            "dynamic_array": np.array([1, 2, 3]),
            "ragged_nd": [[1, 2, 3], [1, 2], [1]],
        })

        assert "dynamic_array" in emitter.buffered_emits
        assert emitter.buffered_emits["dynamic_array"].shape[1:] == (3,)
        assert "ragged_nd" in emitter.buffered_emits
        assert all(
            emitter.buffered_emits["ragged_nd"][0]
            == pl.Series([[1, 2, 3], [1, 2], [1]])
        )

        # Shape changes → Polars fallback for both fields
        emitter.update({
            "time": 2.0,
            "dynamic_array": np.array([4, 5, 6, 7]),
            "ragged_nd": [[1], [1, 2], [1, 2, 3]],
        })
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)
        assert emitter.buffered_emits["dynamic_array"][0] == [1, 2, 3]
        assert emitter.buffered_emits["dynamic_array"][1].to_list() == [4, 5, 6, 7]
        assert all(
            emitter.buffered_emits["ragged_nd"][1]
            == pl.Series([[1], [1, 2], [1, 2, 3]])
        )

        # Third emit fills the batch — write triggers
        emitter.update({
            "time": 3.0,
            "dynamic_array": np.array([4, 5, 6, 7]),
            "ragged_nd": [[1], [1, 2], [1, 2, 3]],
        })
        emitter.last_batch_future.result()

        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "*.pq",
            )
        )
        assert t["dynamic_array"].to_list() == [
            [1, 2, 3],
            [4, 5, 6, 7],
            [4, 5, 6, 7],
        ]
        assert t["ragged_nd"].to_list() == [
            [[1, 2, 3], [1, 2], [1]],
            [[1], [1, 2], [1, 2, 3]],
            [[1], [1, 2], [1, 2, 3]],
        ]

    def test_extreme_data_types(self, temp_dir, core):
        """Extreme integers, NaN/Inf, deep nesting, ragged nullable, datetime, bytes."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 2,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({
            "time": 1.0,
            "max_int": np.iinfo(np.int64).max,
            "min_int": np.iinfo(np.int64).min,
            "max_float": np.finfo(np.float64).max,
            "tiny_float": 1e-100,
            "nan_value": np.nan,
            "inf_value": np.inf,
            "empty_array": np.array([]),
            "zero_dim_array": np.array(42),
            "unicode_string": "Unicode: 日本語",
            "very_long_string": "x" * 10000,
            "deep_nesting": {"level1": {"level2": {"level3": [1, 2, 3]}}},
            "ragged_nullable": [None, [1, 2], [None, 1, 2, 3]],
            "datetime_list": [
                datetime.datetime(2000, 12, 25),
                datetime.datetime(2001, 4, 1, 12),
                datetime.datetime(2002, 1, 1, 0, 1),
                datetime.datetime(2003, 2, 14, 5, 5, 5),
                datetime.datetime(2003, 7, 4, 7, 8, 9, 10),
            ],
            "time_list": [
                datetime.time(1),
                datetime.time(2, 3),
                datetime.time(4, 5, 6),
                datetime.time(7, 8, 9, 10),
            ],
            "datetime": datetime.datetime(1776, 7, 4),
            "npbytes": np.array([b"test bytes"])[0],
            "pybytes": b"test bytes",
            "npbytes_list": np.array([b"test1", b"test2"]),
            "pybytes_list": [b"test1", b"test2"],
        })
        assert emitter.num_emits == 1

        # Spot-check buffered values
        assert emitter.buffered_emits["max_int"][0] == np.iinfo(np.int64).max
        assert emitter.buffered_emits["min_int"][0] == np.iinfo(np.int64).min
        assert emitter.buffered_emits["max_float"][0] == np.finfo(np.float64).max
        assert emitter.buffered_emits["tiny_float"][0] == 1e-100
        assert np.isnan(emitter.buffered_emits["nan_value"][0])
        assert emitter.buffered_emits["unicode_string"][0] == "Unicode: 日本語"
        assert np.array_equal(
            emitter.buffered_emits["deep_nesting__level1__level2__level3"][0],
            np.array([1, 2, 3], dtype=int),
        )

        # Second emit varies shapes / introduces a new mid-batch field
        emitter.update({
            "time": 2.0,
            "max_int": np.iinfo(np.int64).min,
            "min_int": np.iinfo(np.int64).max,
            "max_float": np.finfo(np.float64).min,
            "tiny_float": 1e100,
            "nan_value": np.inf,
            "inf_value": np.nan,
            "empty_array": np.array([np.nan]),
            "zero_dim_array": np.array(0),
            "unicode_string": "Unicode: 日本語 再び",
            "very_long_string": "x" * 100000,
            "deep_nesting": {
                "level1": {
                    "level2": {"level3": [1, 2, 3, 4], "level4": [5, 6, 7]}
                }
            },
            "ragged_nullable": [
                [1, 3, 4],
                [None, None, 1],
                None,
                [1, 2, None],
            ],
            "datetime_list": [datetime.datetime(2000, 12, 25)],
            "time_list": [datetime.time(1)],
            "datetime": datetime.datetime(2000, 12, 25),
            "npbytes": np.array([b"short"])[0],
            "pybytes": b"short",
            "npbytes_list": np.array([b"much longer bytestring", b"1"]),
            "pybytes_list": [b"much longer bytestring", b"1"],
        })
        emitter.last_batch_future.result()

        out_path = os.path.join(
            emitter.out_uri,
            emitter.experiment_id,
            "history",
            emitter.partitioning_path,
            f"{emitter.num_emits}.pq",
        )
        output_pl = pl.read_parquet(out_path)

        expected = {
            "max_int": [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
            "min_int": [np.iinfo(np.int64).min, np.iinfo(np.int64).max],
            "max_float": [np.finfo(np.float64).max, np.finfo(np.float64).min],
            "tiny_float": [1e-100, 1e100],
            "nan_value": [np.nan, np.inf],
            "inf_value": [np.inf, np.nan],
            "empty_array": [[], [np.nan]],
            "zero_dim_array": [42, 0],
            "unicode_string": ["Unicode: 日本語", "Unicode: 日本語 再び"],
            "very_long_string": ["x" * 10000, "x" * 100000],
            "deep_nesting__level1__level2__level3": [[1, 2, 3], [1, 2, 3, 4]],
            "deep_nesting__level1__level2__level4": [None, [5, 6, 7]],
            "ragged_nullable": [
                [None, [1, 2], [None, 1, 2, 3]],
                [[1, 3, 4], [None, None, 1], None, [1, 2, None]],
            ],
            "datetime_list": [
                [
                    datetime.datetime(2000, 12, 25),
                    datetime.datetime(2001, 4, 1, 12),
                    datetime.datetime(2002, 1, 1, 0, 1),
                    datetime.datetime(2003, 2, 14, 5, 5, 5),
                    datetime.datetime(2003, 7, 4, 7, 8, 9, 10),
                ],
                [datetime.datetime(2000, 12, 25)],
            ],
            "time_list": [
                [
                    datetime.time(1),
                    datetime.time(2, 3),
                    datetime.time(4, 5, 6),
                    datetime.time(7, 8, 9, 10),
                ],
                [datetime.time(1)],
            ],
            "datetime": [
                datetime.datetime(1776, 7, 4),
                datetime.datetime(2000, 12, 25),
            ],
            "npbytes": [b"test bytes", b"short"],
            "pybytes": [b"test bytes", b"short"],
            # Note: NumPy bytes arrays truncate to the first-encountered length.
            "npbytes_list": [[b"test1", b"test2"], [b"much\x20", b"1"]],
            "pybytes_list": [[b"test1", b"test2"], [b"much longer bytestring", b"1"]],
        }
        for key, value in expected.items():
            assert compare_nested(output_pl[key].to_list(), value), (
                f"Mismatch in field {key}"
            )

    def test_close_writes_partial_batch(self, temp_dir, core):
        """close() flushes a partial batch to a truncated parquet."""
        # Replaces vEcoli's test_finalize (which mocked json_to_parquet).
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 4,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({"time": 0.0, "field1": 10, "field2": 20.5})
        emitter.close()

        history_path = os.path.join(
            emitter.out_uri, emitter.experiment_id, "history",
            emitter.partitioning_path, "1.pq",
        )
        assert os.path.exists(history_path)
        t = pl.read_parquet(history_path)
        # Only the rows that were emitted should appear.
        assert len(t) == 1
        assert t["field1"].to_list() == [10]
        assert t["field2"].to_list() == [20.5]

    def test_close_with_success_writes_sentinel(self, temp_dir, core):
        """Success sentinel ``s.pq`` is written under the partition path."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 4,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()
        emitter.update({"time": 0.0, "field1": 10})
        emitter.close(success=True)

        sentinel = os.path.join(
            emitter.out_uri, emitter.experiment_id, "success",
            emitter.partitioning_path, "s.pq",
        )
        assert os.path.exists(sentinel), f"missing sentinel: {sentinel}"

    def test_batch_processing(self, temp_dir, core):
        """Multiple emits trigger one parquet per batch_size rows."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 3,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        for i in range(4):
            emitter.update({"time": float(i), "value": i * 10})
            emitter.last_batch_future.result()

        assert emitter.num_emits == 4
        # batch_size=3 → one parquet at row 3; one row left in buffer
        assert len(emitter.buffered_emits["value"]) == emitter.batch_size
        # In the non-threaded path the buffer is NOT zeroed after flush — it's
        # reused for the next batch — so index 0 of the new batch holds row 3.
        assert emitter.buffered_emits["value"][0] == 30


# ============================================================================
# Edge cases (TestParquetEmitterEdgeCases) — rewritten against v2ecoli API.
# ============================================================================


class TestParquetEmitterEdgeCases:
    @pytest.fixture
    def temp_dir(self):
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)

    # Patch the executor where ParquetEmitter looks it up (upstream
    # pbg_emitters module). v2ecoli.library.parquet_emitter is now a
    # re-export shim and patching its attribute would not intercept
    # the running emitter.
    @patch("pbg_emitters.parquet_emitter.ThreadPoolExecutor")
    def test_multithreaded_buffer_clearing(self, mock_executor_class, temp_dir, core):
        """Clearing the buffer in the main thread doesn't race with the writer."""
        real_executor = ThreadPoolExecutor(max_workers=1)
        data_capture_queue: Queue = Queue()

        def capture_submit(func, *args, **kwargs):
            # Deep-ish copy what's being handed off to json_to_parquet.
            emit_dict_copy = {
                k: v.copy() if hasattr(v, "copy") else v for k, v in args[0].items()
            }
            pl_types_copy = args[2].copy()
            data_capture_queue.put((emit_dict_copy, pl_types_copy))

            future: Future = Future()

            def delayed_execution():
                time.sleep(0.1)
                result = func(*args, **kwargs)
                future.set_result(result)
                return result

            real_executor.submit(delayed_execution)
            return future

        mock_executor = Mock()
        mock_executor.submit.side_effect = capture_submit
        mock_executor_class.return_value = mock_executor

        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 2,
                "threaded": True,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )

        # First two updates fill the batch and trigger a submit.
        emitter.update({
            "time": 1.0,
            "field1": np.array([1, 2, 3]),
            "field2": 42,
        })
        emitter.update({
            "time": 2.0,
            "field1": np.array([4, 5, 6]),
            "field2": 43,
        })

        # Immediately add many new fields after the batch flush — must not
        # mutate the buffers that the worker thread is still writing.
        emitter.update({
            "time": 3.0,
            **{f"field{i}": np.array([7, 8, 9, 10]) for i in range(1, 10)},
        })

        # The first captured payload is the configuration write.
        captured_data, captured_types = data_capture_queue.get(timeout=2)
        assert captured_data["experiment_id"] == "test_exp"
        assert captured_data["variant"] == 1
        assert captured_data["lineage_seed"] == 1
        # v2ecoli's configuration emit comes from config["metadata"] verbatim,
        # so the captured polars types match the metadata dict only.
        assert set(captured_types) == {
            "experiment_id", "variant", "lineage_seed", "generation", "agent_id",
        }

        # The second captured payload is the history flush.
        captured_data, captured_types = data_capture_queue.get(timeout=2)
        assert len(captured_data["field1"]) == emitter.batch_size
        assert captured_data["field1"][0].tolist() == [1, 2, 3]
        assert captured_data["field1"][1].tolist() == [4, 5, 6]
        assert captured_types == {
            "time": pl.Float64,
            "field1": pl.List(pl.Int64),
            "field2": pl.Int64,
        }

        real_executor.shutdown()

    def test_variable_shape_detection_at_boundaries(self, temp_dir, core):
        """Shape fallback fires within a batch but resets at batch boundaries.

        The "reset at boundary" behavior relies on ``threaded=True`` clearing
        the buffer after each flush — otherwise the previous batch's buffer
        carries over and subtle_array's shape change at emit 4 also falls back
        to the Polars list path.
        """
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 3,
                "threaded": True,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        # First emit — both fields treated as fixed-shape ndarrays.
        emitter.update({
            "time": 1.0,
            "dynamic_array": np.array([1, 2, 3]),
            "subtle_array": np.array([[1], [2], [3]]),
        })
        emitter.last_batch_future.result()
        assert isinstance(emitter.buffered_emits["dynamic_array"], np.ndarray)
        assert emitter.buffered_emits["dynamic_array"].shape[1:] == (3,)

        # Shape changes mid-batch → fallback to list.
        emitter.update({
            "time": 2.0,
            "dynamic_array": np.array([1, 2, 3, 4, 5]),
            "subtle_array": np.array([[1], [2], [3]]),
        })
        emitter.last_batch_future.result()
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)

        # Third emit fills the batch — triggers the background flush AND clears
        # the buffer (threaded path).
        emitter.update({
            "time": 3.0,
            "dynamic_array": np.array([1, 2, 3, 4, 5, 6, 7]),
            "subtle_array": np.array([[1], [2], [3]]),
        })
        emitter.last_batch_future.result()

        # subtle_array changed shape but at a fresh batch boundary — treated
        # as fixed-shape again because the buffer was cleared on flush.
        emitter.update({
            "time": 4.0,
            "dynamic_array": np.array([1]),
            "subtle_array": np.array([[1], [2], [3], [4], [5]]),
        })
        emitter.last_batch_future.result()
        # dynamic_array remembers it's variable from last batch via pl_serialized.
        assert isinstance(emitter.buffered_emits["dynamic_array"], list)
        assert emitter.buffered_emits["dynamic_array"][0].to_list() == [1]
        # subtle_array reset to fixed-shape with the new dimensions.
        assert isinstance(emitter.buffered_emits["subtle_array"], np.ndarray)
        assert emitter.buffered_emits["subtle_array"].shape[1:] == (5, 1)

        emitter.update({
            "time": 5.0,
            "dynamic_array": np.array([1]),
            "subtle_array": np.array([[1], [2], [3], [4], [5]]),
        })
        emitter.update({
            "time": 6.0,
            "dynamic_array": np.array([1]),
            "subtle_array": np.array([[1], [2], [3], [4], [5]]),
        })
        emitter.last_batch_future.result()

        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "*.pq",
            )
        )
        assert t["dynamic_array"].to_list() == [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7],
            [1],
            [1],
            [1],
        ]
        assert t["subtle_array"].to_list() == [
            [[1], [2], [3]],
            [[1], [2], [3]],
            [[1], [2], [3]],
            [[1], [2], [3], [4], [5]],
            [[1], [2], [3], [4], [5]],
            [[1], [2], [3], [4], [5]],
        ]

    def test_expected_failures(self, temp_dir, core):
        """A few cases that are expected to fail are required to fail loudly."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 3,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        # null / empty list / empty ndarray seed types
        emitter.update({
            "time": 1.0,
            "init_null": None,
            "init_empty_list": [],
            "init_empty_array": np.array([]),
            "3d_array": np.random.rand(2, 3, 4),
            "another_3d_array": np.random.rand(2, 3, 4),
        })

        assert isinstance(emitter.buffered_emits["init_null"], list)
        assert emitter.buffered_emits["init_null"][0] is None
        assert emitter.pl_types["init_null"] == pl.Null
        assert isinstance(emitter.buffered_emits["init_empty_list"], list)
        assert emitter.buffered_emits["init_empty_list"][0].dtype == pl.Null
        assert emitter.pl_types["init_empty_list"] == pl.List(pl.Null)
        assert isinstance(emitter.buffered_emits["init_empty_array"], np.ndarray)
        assert emitter.buffered_emits["init_empty_array"].dtype == np.float64
        assert emitter.pl_types["init_empty_array"] == pl.List(pl.Float64)

        # Promoting empty array → 2D ragged list should raise incompatible inner types.
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Incompatible inner types for field init_empty_array: Float64 and List(Null)."
            ),
        ):
            emitter.update({"time": 2.0, "init_empty_array": [[]]})

        # 3D array → 2D non-null list should raise incompatible inner types.
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Incompatible inner types for field 3d_array: List(Float64) and Float64."
            ),
        ):
            emitter.update({"time": 2.0, "3d_array": [[1.0, 2.0]]})

        # NumPy unsupported datetime resolution
        with pytest.raises(ValueError, match="incorrect NumPy datetime resolution"):
            emitter.update({
                "time": 3.0,
                "datetime_arr": np.array(
                    [
                        np.datetime64("2023-01-01T01"),
                        np.datetime64("2023-01-02T02:02"),
                    ]
                ),
            })

        # NumPy void
        with pytest.raises(ValueError):
            emitter.update({
                "time": 4.0,
                "npvoid": np.array([b"test bytes"], dtype=np.void)[0],
            })

        # NumPy datetime64 in a Python list
        with pytest.raises(
            TypeError, match=re.escape("not yet implemented: Nested object types")
        ):
            emitter.update({
                "time": 3.0,
                "mixed_datetime": [
                    np.datetime64("2023-01-01"),
                    np.datetime64("2023-01-02"),
                ],
            })

        # Variable-shape 3D NumPy array — v2ecoli's Polars fallback converts
        # multi-D ndarrays to nested Python lists, so this now succeeds where
        # vEcoli's upstream raises a dtype('O') error. Behavior improvement
        # added so real cell composites (which emit fields like
        # listeners__rna_synth_prob__n_bound_TF_per_TU with shape transitions
        # (0, 0) → (M, N)) don't crash on first non-empty tick.
        emitter.update({
            "time": 3.0,
            "another_3d_array": np.zeros((10, 10, 10)),
        })
        # The 3D field now lives on the Polars list path.
        assert "another_3d_array" in emitter.pl_serialized

    def test_nested_nullable(self, temp_dir, core):
        """Nullable nested types that grow deeper across rows reconcile."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 4,
                "threaded": False,
                "partitioning_keys": VECOLI_PARTITION_KEYS,
                "metadata": _vecoli_metadata(),
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({"time": 0.0, "nullable_nested": None})
        assert isinstance(emitter.buffered_emits["nullable_nested"], list)
        assert emitter.buffered_emits["nullable_nested"][0] is None
        assert emitter.pl_types["nullable_nested"] == pl.Null

        emitter.update({"time": 1.0, "nullable_nested": [None, None]})
        assert isinstance(emitter.buffered_emits["nullable_nested"], list)
        assert emitter.buffered_emits["nullable_nested"][1].dtype == pl.Null
        assert emitter.pl_types["nullable_nested"] == pl.List(pl.Null)

        emitter.update({
            "time": 2.0,
            "nullable_nested": [None, [None], [], [None, None], []],
        })
        assert emitter.buffered_emits["nullable_nested"][2].dtype == pl.List(pl.Null)
        assert emitter.pl_types["nullable_nested"] == pl.List(pl.List(pl.Null))

        emitter.update({
            "time": 3.0,
            "nullable_nested": [
                [],
                [["wow", "this", "is"], [], ["deep"], None],
                None,
                [[], None],
            ],
        })
        emitter.last_batch_future.result()

        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "4.pq",
            )
        )
        assert t["nullable_nested"].to_list() == [
            None,
            [None, None],
            [None, [None], [], [None, None], []],
            [[], [["wow", "this", "is"], [], ["deep"], None], None, [[], None]],
        ]

        # Across the batch boundary the resolved type sticks.
        emitter.update({"time": 4.0, "nullable_nested": None})
        assert isinstance(emitter.buffered_emits["nullable_nested"], list)
        assert emitter.buffered_emits["nullable_nested"][0] is None
        assert emitter.pl_types["nullable_nested"] == pl.List(
            pl.List(pl.List(pl.String))
        )

        for _ in range(3):
            emitter.update({"time": 5.0, "nullable_nested": None})
        emitter.last_batch_future.result()

        t = pl.read_parquet(
            os.path.join(
                emitter.out_uri,
                emitter.experiment_id,
                "history",
                emitter.partitioning_path,
                "8.pq",
            )
        )
        assert t["nullable_nested"].to_list() == [None] * 4
        assert t["nullable_nested"].dtype == pl.List(pl.List(pl.List(pl.String)))
