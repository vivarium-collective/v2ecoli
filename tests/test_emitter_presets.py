"""Unit tests for v2ecoli.library.emitter_presets.

These tests cross-check the vEcoli preset constants against the installed
ecoli.library.parquet_emitter module — if upstream changes USE_UINT16 /
USE_UINT32, these tests catch the drift.
"""

import pytest


@pytest.mark.fast
def test_vecoli_parquet_dtype_overrides_matches_upstream():
    from v2ecoli.library.emitter_presets import VECOLI_PARQUET_DTYPE_OVERRIDES
    from ecoli.library.parquet_emitter import USE_UINT16, USE_UINT32

    expected = {name: "UInt16" for name in USE_UINT16} | {
        name: "UInt32" for name in USE_UINT32
    }
    assert VECOLI_PARQUET_DTYPE_OVERRIDES == expected


@pytest.mark.fast
def test_vecoli_xarray_metadata_keys_is_nonempty_list_of_strings():
    from v2ecoli.library.emitter_presets import VECOLI_XARRAY_METADATA_KEYS
    assert isinstance(VECOLI_XARRAY_METADATA_KEYS, list)
    assert len(VECOLI_XARRAY_METADATA_KEYS) > 20
    assert all(isinstance(k, str) for k in VECOLI_XARRAY_METADATA_KEYS)
    for k in ("experiment_id", "lineage_seed", "agent_id", "parca_options"):
        assert k in VECOLI_XARRAY_METADATA_KEYS


@pytest.mark.fast
def test_parquet_vecoli_returns_full_config_dict():
    from v2ecoli.library.emitter_presets import parquet_vecoli

    cfg = parquet_vecoli(
        "/tmp/test_out",
        experiment_id="exp1",
        variant=2,
        lineage_seed=7,
        agent_id="01",
    )

    assert cfg["out_dir"] == "/tmp/test_out"
    assert cfg["partitioning_keys"] == [
        "experiment_id", "variant", "lineage_seed", "generation", "agent_id"
    ]
    assert cfg["flatten_separator"] == "__"
    assert cfg["batch_size"] == 400
    assert cfg["threaded"] is True
    assert cfg["metadata"]["experiment_id"] == "exp1"
    assert cfg["metadata"]["variant"] == 2
    assert cfg["metadata"]["lineage_seed"] == 7
    assert cfg["metadata"]["agent_id"] == "01"
    assert cfg["metadata"]["generation"] == 2  # len("01")
    from ecoli.library.parquet_emitter import USE_UINT16
    a_uint16_key = next(iter(USE_UINT16))
    assert cfg["dtype_overrides"][a_uint16_key] == "UInt16"


@pytest.mark.fast
def test_parquet_vecoli_quote_plus_on_special_chars():
    from v2ecoli.library.emitter_presets import parquet_vecoli

    cfg = parquet_vecoli("/tmp/x", experiment_id="exp with spaces")
    assert cfg["metadata"]["experiment_id"] == "exp+with+spaces"


@pytest.mark.fast
def test_parquet_vecoli_extra_metadata_merges():
    from v2ecoli.library.emitter_presets import parquet_vecoli

    cfg = parquet_vecoli(
        "/tmp/x",
        extra_metadata={"my_field": 42, "experiment_id": "overridden"},
    )
    assert cfg["metadata"]["my_field"] == 42
    assert cfg["metadata"]["experiment_id"] == "overridden"


@pytest.mark.fast
def test_xarray_vecoli_returns_full_config_dict():
    from v2ecoli.library.emitter_presets import (
        xarray_vecoli,
        VECOLI_XARRAY_METADATA_KEYS,
    )

    cfg = xarray_vecoli(
        "/tmp/test_out.zarr",
        transducer={"some": "config"},
        view=["some_view"],
    )

    assert cfg["out_uri"] == "/tmp/test_out.zarr"
    assert cfg["transducer"] == {"some": "config"}
    assert cfg["view"] == ["some_view"]
    assert cfg["metadata_keys"] == VECOLI_XARRAY_METADATA_KEYS
    assert cfg["metadata_validators"] == {}
    assert cfg["debug"] is False
    assert cfg["writer"] == {"type": "async", "out_uri": "/tmp/test_out.zarr"}


@pytest.mark.fast
def test_xarray_vecoli_writer_override():
    from v2ecoli.library.emitter_presets import xarray_vecoli

    custom_writer = {"type": "blocking", "out_uri": "gs://my-bucket/x.zarr"}
    cfg = xarray_vecoli(
        "/tmp/x.zarr",
        transducer={},
        view=[],
        writer=custom_writer,
    )
    assert cfg["writer"] == custom_writer


@pytest.mark.fast
def test_parquet_vecoli_preset_drives_emitter(tmp_path, core):
    """End-to-end: construct ParquetEmitter from parquet_vecoli config, emit
    a state matching one of the listener columns, verify hive layout and
    dtype-override behavior."""
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    from v2ecoli.library.emitter_presets import parquet_vecoli
    from v2ecoli.library.parquet_emitter import ParquetEmitter
    from ecoli.library.parquet_emitter import USE_UINT16

    cfg = parquet_vecoli(
        str(tmp_path / "out"),
        experiment_id="exp1",
        variant=2,
        lineage_seed=7,
        agent_id="01",
        batch_size=1,
        threaded=False,
    )
    cfg["emit"] = {"v": "node"}  # caller wiring

    emitter = ParquetEmitter(config=cfg, core=core)

    # Emit a synthetic state that includes a USE_UINT16 column.
    uint16_col = next(iter(USE_UINT16))
    emitter.update({"v": 1.0, uint16_col: 42})
    emitter.close(success=True)

    # Hive path layout
    base = tmp_path / "out" / "exp1" / "history"
    expected = (
        base
        / "experiment_id=exp1"
        / "variant=2"
        / "lineage_seed=7"
        / "generation=2"
        / "agent_id=01"
    )
    assert expected.exists(), f"missing hive path: {expected}"

    # Dtype override applied
    df = emitter.query()
    assert str(df.schema[uint16_col]) == "UInt16"

    # Success sentinel written
    sentinel_dir = tmp_path / "out" / "exp1" / "success"
    assert list(sentinel_dir.rglob("s.pq"))


@pytest.mark.fast
def test_xarray_vecoli_preset_validates(tmp_path):
    """Construct an xarray_vecoli config and assert it carries the vEcoli
    metadata_keys and empty validators, plus passes XArrayEmitter.validate_config.

    Full XArrayEmitter construction needs a real transducer config (the
    preset takes them as required kwargs), so this test exercises only the
    config-builder side and the static validator."""
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    from v2ecoli.library.emitter_presets import (
        xarray_vecoli, VECOLI_XARRAY_METADATA_KEYS,
    )
    from v2ecoli.library.xarray_emitter import XArrayEmitter

    cfg = xarray_vecoli(
        str(tmp_path / "x.zarr"),
        transducer={"some": "config"},
        view=["some_view"],
        metadata={"experiment_id": "exp1", "variant": 0,
                  "lineage_seed": 0, "agent_id": "1"},
    )
    cfg["emit"] = {}

    assert cfg["metadata_keys"] == VECOLI_XARRAY_METADATA_KEYS
    assert cfg["metadata_validators"] == {}
    XArrayEmitter.validate_config(cfg)  # raises if structure is bad
