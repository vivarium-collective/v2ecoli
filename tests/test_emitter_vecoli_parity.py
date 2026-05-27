"""@pytest.mark.sim parity tests against installed vEcoli + checked-in
xarray golden fixture.

These compare v2ecoli's ported emitters against PR #414's source to catch
behavior drift.
"""

from pathlib import Path

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("pbg_emitters")


def _synthetic_payload() -> dict:
    """Build a fake history row that includes columns from USE_UINT16/USE_UINT32."""
    from ecoli.library.parquet_emitter import USE_UINT16, USE_UINT32
    payload = {"global_time": 0.0, "scalar_int": 1, "scalar_float": 1.5}
    payload[next(iter(USE_UINT16))] = 7
    payload[next(iter(USE_UINT32))] = 70000
    return payload


@pytest.mark.sim
def test_parquet_parity_against_installed_vecoli(tmp_path, core):
    """Same input through v2ecoli's ParquetEmitter (preset) vs the installed
    ecoli.library.parquet_emitter.ParquetEmitter. Output dtypes and row
    values must match for columns covered by the dtype-override sets.
    """
    import polars as pl

    from ecoli.library.parquet_emitter import ParquetEmitter as VEcoliPq
    from v2ecoli.library.emitter_presets import parquet_vecoli
    from v2ecoli.library.parquet_emitter import ParquetEmitter as V2Pq

    row = _synthetic_payload()
    meta = {
        "experiment_id": "exp1", "variant": 0,
        "lineage_seed": 0, "agent_id": "1",
    }

    # --- v2ecoli port ---
    out_v2 = tmp_path / "v2"
    cfg_v2 = parquet_vecoli(str(out_v2), **meta, batch_size=1, threaded=False)
    cfg_v2["emit"] = {k: "node" for k in row}
    e2 = V2Pq(config=cfg_v2, core=core)
    e2.update(row)
    e2.close(success=True)

    # --- installed vEcoli ParquetEmitter ---
    # The vEcoli ParquetEmitter uses vivarium's two-channel emit() handshake:
    #   1. emit({"table": "configuration", "data": {...}}) to set up partitioning
    #   2. emit({"table": "history", "data": {"agents": {agent_id: data}, "time": t}})
    out_ve = tmp_path / "ve"
    e1 = VEcoliPq({"out_dir": str(out_ve), "batch_size": 1, "threaded": False})
    # Configuration emit: metadata dict is popped out and merged at the top level.
    config_payload = {"metadata": dict(meta), **meta}
    e1.emit({"table": "configuration", "data": config_payload})
    # History emit: agents dict wrapping, with time at top level.
    e1.emit({"table": "history", "data": {"agents": {"1": dict(row)}, "time": 0.0}})
    e1.success = True
    e1.finalize()

    # Compare the dtype-overridden columns
    from ecoli.library.parquet_emitter import USE_UINT16, USE_UINT32
    uint16_col = next(iter(USE_UINT16))
    uint32_col = next(iter(USE_UINT32))

    df_v2 = e2.query()
    # The vEcoli reference output lands under <out_ve>/exp1/history/...
    ve_history = out_ve / "exp1" / "history"
    pq_files = list(ve_history.rglob("*.pq"))
    assert pq_files, (
        f"no parquet emitted by vEcoli reference: {ve_history}. "
        f"Contents of {out_ve}: {list(out_ve.rglob('*'))}"
    )
    df_ve = pl.read_parquet(pq_files[0])

    assert str(df_v2.schema[uint16_col]) == "UInt16", (
        f"v2ecoli port: {uint16_col} dtype={df_v2.schema[uint16_col]}, expected UInt16"
    )
    assert str(df_ve.schema[uint16_col]) == "UInt16", (
        f"vEcoli ref: {uint16_col} dtype={df_ve.schema[uint16_col]}, expected UInt16"
    )
    assert str(df_v2.schema[uint32_col]) == "UInt32", (
        f"v2ecoli port: {uint32_col} dtype={df_v2.schema[uint32_col]}, expected UInt32"
    )
    assert str(df_ve.schema[uint32_col]) == "UInt32", (
        f"vEcoli ref: {uint32_col} dtype={df_ve.schema[uint32_col]}, expected UInt32"
    )
    assert df_v2[uint16_col].to_list() == df_ve[uint16_col].to_list(), (
        f"{uint16_col}: v2ecoli={df_v2[uint16_col].to_list()}, "
        f"vEcoli={df_ve[uint16_col].to_list()}"
    )
    assert df_v2[uint32_col].to_list() == df_ve[uint32_col].to_list(), (
        f"{uint32_col}: v2ecoli={df_v2[uint32_col].to_list()}, "
        f"vEcoli={df_ve[uint32_col].to_list()}"
    )


@pytest.mark.sim
def test_xarray_parity_against_golden(tmp_path, core):
    """Run XArrayEmitter on the same inputs as the golden fixture; compare
    the resulting Zarr store structurally against the committed golden."""
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import xarray as xr

    from v2ecoli.library.xarray_emitter import XArrayEmitter

    golden_store = Path(__file__).parent / "fixtures" / "xarray_golden" / "store.zarr"
    assert golden_store.exists(), (
        f"golden fixture missing: {golden_store}. "
        f"Regenerate with: uv run python tests/fixtures/xarray_golden/regenerate.py"
    )

    # Reproduce the inputs from regenerate.py
    out_uri = str(tmp_path / "out.zarr")
    cfg = {
        "emit": {"global_time": "node"},
        "out_uri": out_uri,
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": [
            {
                "root": ("listeners",),
                "variables": {
                    "global_time": [{"path": "global_time", "dtype": "<f4"}],
                },
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": out_uri,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": "golden", "variant": 0,
            "lineage_seed": 0, "agent_id": "1",
            "time_step": 1.0, "max_duration": 4.0,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }
    e = XArrayEmitter(config=cfg, core=core)
    for t in range(4):
        # State must use agents/<agent_id> wrapping (vivarium store layout).
        e.update({
            "time": float(t),
            "agents": {"1": {"listeners": {"global_time": float(t)}}},
        })
    e.close(success=True)

    # Compare DataTrees structurally
    golden = xr.open_datatree(str(golden_store), engine="zarr")
    fresh = xr.open_datatree(out_uri, engine="zarr")

    # Walk groups and compare data variables
    def iter_groups(dt):
        """Yield (path, dataset) pairs for all groups in the DataTree."""
        for path in dt.groups:
            yield path, dt[path].ds

    golden_groups = dict(iter_groups(golden))
    fresh_groups = dict(iter_groups(fresh))
    assert set(golden_groups) == set(fresh_groups), (
        f"group paths differ: golden={sorted(golden_groups)}, "
        f"fresh={sorted(fresh_groups)}"
    )
    for path, gds in golden_groups.items():
        fds = fresh_groups[path]
        assert set(gds.data_vars) == set(fds.data_vars), (
            f"{path}: data_vars differ: golden={set(gds.data_vars)}, "
            f"fresh={set(fds.data_vars)}"
        )
        for var in gds.data_vars:
            gv = gds[var].values
            fv = fds[var].values
            assert gv.shape == fv.shape, (
                f"{path}/{var}: shape {gv.shape} vs {fv.shape}"
            )
            assert (gv == fv).all(), (
                f"{path}/{var}: values differ: golden={gv}, fresh={fv}"
            )
