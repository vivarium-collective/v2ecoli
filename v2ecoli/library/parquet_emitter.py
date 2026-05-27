"""Re-export shim: ParquetEmitter + bare reader helpers from pbg-emitters.

The emitter implementation was extracted to a focused library
(https://github.com/vivarium-collective/pbg-emitters) so it can ship
heavy optional deps (duckdb / polars / pyarrow / fsspec) per-emitter
and iterate independently. v2ecoli imports from here go through this
shim so existing call sites (and external users of
``v2ecoli.library.parquet_emitter``) keep working.

v2ecoli-specific layers stay in this package:
  * ``v2ecoli.library.emitter_presets.parquet_vecoli`` — preset config
    with vEcoli's USE_UINT16 / USE_UINT32 dtype overrides.
  * ``v2ecoli.library.parquet_run.run_multigen_parquet`` — lineage walk.

Notes on the shim surface:
  * ``ThreadPoolExecutor`` is re-exported at module scope so existing
    tests using ``unittest.mock.patch("v2ecoli.library.parquet_emitter
    .ThreadPoolExecutor", ...)`` keep finding the attribute. Production
    code inside ParquetEmitter resolves ``ThreadPoolExecutor`` against
    pbg_emitters.parquet_emitter, so patches on this module no longer
    intercept the running emitter — tests that need to swap the executor
    must patch ``pbg_emitters.parquet_emitter.ThreadPoolExecutor``.
"""

# Re-export the public API from pbg-emitters. If the [parquet] extra
# (and therefore duckdb / polars / fsspec / ...) isn't installed,
# ``from pbg_emitters import ParquetEmitter`` raises ImportError; we
# wrap that in a friendlier message that points at the v2ecoli extra.
try:
    from pbg_emitters import (  # noqa: F401
        ParquetEmitter,
        BlockingExecutor,
        METADATA_PREFIX,
        # Bare DuckDB / parquet reader helpers re-exported for downstream code:
        create_duckdb_conn,
        named_idx,
        ndidx_to_duckdb_expr,
        ndlist_to_ndarray,
        list_columns,
        quote_columns,
        union_by_name,
        dataset_sql,
        field_metadata,
        config_value,
        plot_metadata,
        read_stacked_columns,
        num_cells,
        skip_n_gens,
        np_dtype,
        union_pl_dtypes,
        flatten_dict,
        json_to_parquet,
        pl_dtype_from_ndarray,
        open_arbitrary_sim_data,
    )
except ImportError as e:
    raise ImportError(
        f"v2ecoli.library.parquet_emitter requires the [parquet] extra. "
        f"Install with: pip install 'v2ecoli[parquet]'. (missing: {e.name})"
    ) from e

# Re-export ThreadPoolExecutor at module scope so existing tests that
# patch ``v2ecoli.library.parquet_emitter.ThreadPoolExecutor`` still
# find the attribute. See module docstring for the patching caveat.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402, F401
