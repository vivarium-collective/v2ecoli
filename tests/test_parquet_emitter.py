"""Unit tests for v2ecoli.library.parquet_emitter.ParquetEmitter."""

import os
from pathlib import Path

import pytest

# Skip the entire file if the [parquet] extra isn't installed.
# CI's default `uv sync --extra dev` doesn't install [parquet] — duckdb /
# polars may still resolve transitively via vivarium-dashboard, but
# pbg-emitters (the actual ParquetEmitter implementation) won't, so the
# v2ecoli.library.parquet_emitter shim's import would raise ImportError.
# Skip cleanly in that case.
pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("pbg_emitters")


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
def test_query_on_open_emitter_flushes_partial_batch(tmp_path, core):
    """query() on an emitter before close() flushes in-memory rows first."""
    # Regression: pre-fix, query() called a non-existent _flush_batch() and
    # crashed on any open emitter with rows still in the buffer.
    from v2ecoli.library.parquet_emitter import ParquetEmitter

    emitter = ParquetEmitter(
        config={
            "emit": {"x": "node"},
            "out_dir": str(tmp_path / "out"),
            "batch_size": 4,  # > 3 emits below → partial batch in memory
            "threaded": False,
        },
        core=core,
    )
    for i in range(3):
        emitter.update({"x": float(i)})

    df = emitter.query()
    assert len(df) == 3, "open-emitter query must include unflushed rows"
    assert df["x"].to_list() == [0.0, 1.0, 2.0]

    # close() after partial-batch flush must not double-write
    emitter.close()
    df2 = emitter.query()
    assert len(df2) == 3, "row count must not double after close"


@pytest.mark.fast
def test_daughter_emitter_must_not_wipe_parent_partition(tmp_path, core):
    """Regression for the multi-gen full-history loss at division.

    A ParquetEmitter writes its whole per-generation history to
    ``generation=N/agent_id=P``. When the Division step builds daughter
    docs via ``baseline()`` while the parquet override is still active, a
    fresh emitter is constructed. If that emitter inherits the parent's
    partition metadata, its ``_write_configuration`` (run in ``__init__``)
    DELETES the parent's history partition — leaving only the daughter's
    birth rows (the early-cycle-slice symptom).

    The fix re-points the daughter to its OWN slot
    (``generation=N+1/agent_id=P0|P1``); this test pins both halves:
    same-slot construction wipes (the bug), daughter-slot construction
    does not (the fix).
    """
    from v2ecoli.library.parquet_emitter import ParquetEmitter
    from v2ecoli.library.emitter_presets import parquet_vecoli

    out_dir = str(tmp_path / "out")
    emit_schema = {"global_time": "float"}

    def _cfg(agent_id, generation):
        return {"emit": emit_schema, "threaded": False, "batch_size": 4,
                **parquet_vecoli(out_dir=out_dir, experiment_id="exp",
                                 agent_id=agent_id, generation=generation)}

    # Parent writes a full partition (3 batches + partial) for gen 1.
    parent = ParquetEmitter(_cfg("0", 1), core)
    for t in range(10):
        parent.update({"global_time": float(t)})
    parent.close(success=True)
    assert len(parent.query()) == 10

    # --- the bug: same-slot daughter wipes the parent partition ---
    bug = ParquetEmitter(_cfg("0", 1), core)  # parent's slot
    bug.update({"global_time": 0.0})
    bug.close(success=True)
    assert len(bug.query()) == 1, "same-slot construction left >1 row"

    # --- the fix: daughter on its OWN slot leaves the parent intact ---
    # Rewrite the parent (the bug step above wiped it), then build the
    # daughter on generation=2/agent_id=00 and confirm the parent's gen-1
    # full history survives.
    parent2 = ParquetEmitter(_cfg("0", 1), core)
    for t in range(10):
        parent2.update({"global_time": float(t)})
    parent2.close(success=True)

    daughter = ParquetEmitter(_cfg("00", 2), core)  # daughter's own slot
    daughter.update({"global_time": 0.0})
    daughter.close(success=True)

    # Parent's gen-1 partition must still hold all 10 ticks.
    df_parent = parent2.query().filter(
        __import__("polars").col("agent_id") == "0")
    assert len(df_parent) == 10, (
        "daughter on its own slot must not wipe the parent's full history")


@pytest.mark.fast
def test_parquet_emitter_registry_roundtrip():
    """register/get/unregister let Division find the parent emitter to close."""
    from v2ecoli.composites._helpers import (
        register_parquet_emitter, get_parquet_emitter,
        unregister_parquet_emitter)

    sentinel = object()
    assert get_parquet_emitter("000") is None
    register_parquet_emitter("000", sentinel)
    assert get_parquet_emitter("000") is sentinel
    unregister_parquet_emitter("000")
    assert get_parquet_emitter("000") is None


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
