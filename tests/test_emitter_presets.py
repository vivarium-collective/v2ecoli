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
