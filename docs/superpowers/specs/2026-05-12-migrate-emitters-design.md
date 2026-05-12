# Migrate ParquetEmitter and XArrayEmitter into v2ecoli

**Status:** Design approved, awaiting implementation plan.
**Date:** 2026-05-12.
**Upstream snapshot:** `vivarium-collective/vEcoli@b25ca24737ad7b1e4e6f042a1a84c2f03627f6a6` (PR #414 head, branch `xarray-emitter-pr`).

## Goal

Port two emitters from vEcoli into v2ecoli as `process_bigraph.emitter.Emitter` subclasses:

- `ParquetEmitter` — buffered hive-partitioned Parquet writer, currently single-file `ecoli/library/parquet_emitter.py` in vEcoli (~1140 LOC). DuckDB-readable. Designed for flexible-schema sparse selections over a large fraction of the simulator state.
- `XArrayEmitter` — Zarr-via-Xarray writer, currently a 10-module package `ecoli/library/xarray_emitter/` in PR #414 (~3000 LOC across `transducer.py`, `view.py`, `storage.py`, `writer.py`, `zarr_writer.py`, `emit_path.py`, `emit_predicate.py`, `utils.py`, `__init__.py`, `emitter.py`). Designed for static-shaped tensor variables in array-programming downstream.

Both upstream classes are vivarium-Engine-shaped and carry vEcoli-specific values baked in as class attributes or module-level constants. The port makes both classes **fully generic** (no vEcoli-specific constants, no vivarium-Engine handshake) and ships a **preset module** that reproduces exact vEcoli behavior via a populated config dict.

## Non-goals

- Replacing the existing `process_bigraph.emitter.RAMEmitter` / `SQLiteEmitter` / `JSONEmitter` / `ConsoleEmitter`. Those keep working unchanged.
- Wiring the emitters into any v2ecoli composite (`baseline`, `departitioned`, `reconciled`). That's follow-up work.
- Backporting changes into vEcoli. The vendored source diverges from upstream the moment we re-root the base class.
- Performance tuning beyond what the upstream code already does.

## Architecture

```
v2ecoli/library/
├── parquet_emitter.py         # ParquetEmitter(Emitter), generic
├── xarray_emitter/            # package vendored from PR #414, generic
│   ├── __init__.py
│   ├── _base.py               # BufferedEmitter, StoragePartition,
│   │                          #   BlockingExecutor (vendored from
│   │                          #   ecoli/library/emitter.py @ PR #414)
│   ├── emitter.py             # XArrayEmitter(BufferedEmitter)
│   ├── transducer.py
│   ├── view.py
│   ├── storage.py
│   ├── writer.py
│   ├── zarr_writer.py
│   ├── emit_path.py
│   ├── emit_predicate.py
│   └── utils.py
└── emitter_presets.py         # parquet_vecoli(), xarray_vecoli(),
                               # VECOLI_PARQUET_DTYPE_OVERRIDES,
                               # VECOLI_XARRAY_METADATA_KEYS
```

Each emitter class inherits from `process_bigraph.emitter.Emitter` (a `Step` subclass with `config_schema = {'emit': 'schema'}`). The vivarium `Engine.emit(data)` two-channel handshake (`"table": "configuration"` once, `"table": "history"` per tick) collapses into the process-bigraph lifecycle:

| vivarium | v2ecoli |
|---|---|
| `emit({"table": "configuration", "data": {...}})` | `__init__` writes the one-shot configuration artifact from `config["metadata"]` and `config["output_metadata"]` |
| `emit({"table": "history", "data": state})` | `update(state)` buffers the row, flushes per `batch_size` |
| `finalize(success=...)` | `close(success=...)` — flushes final batch, writes success sentinel when applicable. Called explicitly; `__del__` calls it as a safety net |

The vivarium `Emitter` import is severed; nothing in `v2ecoli/library/parquet_emitter.py` or `v2ecoli/library/xarray_emitter/` imports `vivarium.core.emitter`, `vivarium.core.engine`, or `vivarium.core.store` after the port. `vEcoli[dev]>=1.1.0` stays a runtime dep of v2ecoli for other reasons (processes use `ecoli.*` modules), but the emitter modules don't reach into it.

The vendored XArrayEmitter sub-modules (`transducer`, `view`, `storage`, `writer`, `zarr_writer`, `emit_path`, `emit_predicate`, `utils`) are kept structurally intact — the PR's internal layout is preserved. Re-rooting touches only two things:

1. `_base.py` ports PR #414's `ecoli/library/emitter.py` (`BufferedEmitter`, `StoragePartition`, `BlockingExecutor`). `BufferedEmitter` re-roots onto `process_bigraph.emitter.Emitter`. The abstract `reset_emit_flags(*, engine, agent, emit_paths)` method is **dropped** — it's vivarium-Engine-specific (suppress-default-emits behavior) with no process-bigraph analog, since composites already emit only what's wired. `StoragePartition` is kept as a vEcoli-shaped dataclass (`experiment_id`, `variant`, `lineage_seed`, `generation`, `agent_id`); the generic `XArrayEmitter` always rebuilds it from `config["metadata"]` via the inherited `extract_partition()` helper. Callers wanting a different partition shape subclass `XArrayEmitter` and override `extract_partition()` — out of scope for this port. `BlockingExecutor` is unchanged.
2. `emitter.py` re-roots `XArrayEmitter` from `BufferedEmitter` (now in `_base.py`) onto the re-rooted base. Its constructor is rewritten to do the work that `Engine._emit_configuration` used to do at first emit: validate config, build the partition, call `transducer.alloc(partition=..., metadata=..., coords=...)`, open the writer's store. `update(state)` calls `transducer.step(state)` then flushes when the buffer fills, exactly mirroring the `case "history":` branch in PR #414's `emit()`. `_finalize` is called from `close()`.

All `from vivarium ...` imports are removed from the vendored modules. `acceptance criteria` includes `rg "from vivarium" v2ecoli/library/parquet_emitter.py v2ecoli/library/xarray_emitter/` returning zero hits.

## API contract

### Shared config schema (both emitters)

```python
config_schema = {
    **Emitter.config_schema,        # 'emit': 'schema'
    'out_dir':         'string',
    'out_uri':         'maybe[string]',     # mutually exclusive with out_dir
    'metadata':        'maybe[map[any]]',   # written once at __init__
    'output_metadata': 'maybe[map[any]]',   # port-schema dict, written once
}
```

### Lifecycle

- **Construction (`__init__(config, core)`):** validate config, open output filesystem, write the one-shot configuration artifact from `config["metadata"]` and `config["output_metadata"]`, initialize the buffer and background-write executor.
- **Per tick (`update(state)`):** flatten/transduce `state`, append to the buffer, flush when `len(buffer) >= batch_size`. Returns `{}` like all process-bigraph emitters.
- **`query(paths=None)`:** read back what was written. ParquetEmitter routes through DuckDB (`dataset_sql` when partitioned, plain `read_parquet` otherwise) and returns a `polars.DataFrame`. XArrayEmitter routes through `xarray.open_datatree(out_uri)` and returns a `DataTree` (or filtered `Dataset` for `paths`). Heavy analysis stays post-hoc; `query()` exists for tests and small notebooks.
- **`close(success=False)`:** flush remaining buffer, join the background executor, write the success sentinel if `success=True` *and* the emitter has a partitioning layout configured. Idempotent. Called explicitly by sim callers; `__del__` calls it defensively.

### Threading

ParquetEmitter wraps writes in a `ThreadPoolExecutor(1)` (or a `BlockingExecutor` when `threaded=False`); XArrayEmitter uses its own `AsyncBufferWriter`. Both are joined / closed inside `close()`. GC ordering during interpreter shutdown is undefined, so durability-sensitive callers must call `close()` explicitly — same contract as `process_bigraph.emitter.SQLiteEmitter`.

## ParquetEmitter

### Generic config schema (extends shared base)

```python
config_schema = {
    **Emitter.config_schema,
    'out_dir':           'string',
    'out_uri':           'maybe[string]',
    'batch_size':        {'_type': 'integer', '_default': 400},
    'threaded':          {'_type': 'boolean', '_default': True},
    'flatten_separator': {'_type': 'string',  '_default': '__'},
    'partitioning_keys': 'maybe[list[string]]',
    'dtype_overrides':   'maybe[map[string]]',   # column name or fnmatch glob → polars dtype name
    'metadata':          'maybe[map[any]]',
}
```

### Behavior

- `partitioning_keys = []` (default): writes to flat `{out}/history/{n}.pq`. `query()` reads via plain `read_parquet`.
- `partitioning_keys = [...]`: builds hive path from `metadata[k]` for each `k`; raises `KeyError` with a one-line message naming the missing key. `query()` reads via the vendored `dataset_sql` helper.
- `dtype_overrides` accepts exact column names *and* fnmatch globs (e.g. `"listeners__rna_*__*_count"`). Dtypes are specified as Polars dtype strings (`"UInt16"`, `"UInt32"`, `"Int32"`, `"Float32"`, ...); a small private table in the module maps string → `pl.DataType`. Glob matches lose to exact-name matches.
- `close(success=True)` writes the `success/.../s.pq` sentinel only when `partitioning_keys` is non-empty (the sentinel concept is hive-layout-specific). `close(success=False)` or `close()` does not.

### `parquet_vecoli` preset

```python
parquet_vecoli(
    out_dir,
    *,
    experiment_id="default",
    variant=0,
    lineage_seed=0,
    agent_id="1",
    generation=None,            # defaults to len(agent_id)
    batch_size=400,
    threaded=True,
    extra_metadata=None,
) -> dict
```

Returns a config dict that supplies:

- `partitioning_keys = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]`
- `dtype_overrides = VECOLI_PARQUET_DTYPE_OVERRIDES` — the upstream `USE_UINT16` set mapped to `"UInt16"` and `USE_UINT32` set mapped to `"UInt32"`
- `metadata = {"experiment_id": urllib.parse.quote_plus(experiment_id), "variant": variant, "lineage_seed": lineage_seed, "generation": generation or len(agent_id), "agent_id": agent_id, **(extra_metadata or {})}`
- `flatten_separator = "__"`
- `batch_size`, `threaded` as passed through

## XArrayEmitter

### Generic config schema (extends shared base)

```python
config_schema = {
    **Emitter.config_schema,
    'out_uri':            'string',
    'transducer':         'map[any]',           # passed to XarrayTransducer
    'view':               'list[any]',          # passed to ForestView
    'writer':             'map[any]',           # passed to AsyncBufferWriter.dispatch
    'metadata':           'maybe[map[any]]',
    'metadata_keys':      'maybe[list[string]]',
    'metadata_validators':'maybe[map[any]]',
    'output_metadata':    'maybe[map[any]]',
    'debug':              {'_type': 'boolean', '_default': False},
}
```

### What moved from class attribute to config

- `metadata_keys` (class-level list in PR #414): now a config knob. `None` means "retain all of `config['metadata']`".
- `validate_metadata()`'s hardcoded `expected` dict: now read from `metadata_validators`. `{}` means "no validation". The method is preserved.
- Everything else (transducer config, view, writer, metadata extraction, coords extraction from `output_metadata`) was already config-driven in PR #414; left intact.

### `xarray_vecoli` preset

```python
xarray_vecoli(
    out_uri,
    *,
    transducer,                 # required, usage-specific (no preset)
    view,                       # required, usage-specific (no preset)
    writer=None,                # default supplied: AsyncBufferWriter to out_uri
    metadata=None,
    output_metadata=None,
    debug=False,
) -> dict
```

Returns a config dict that supplies:

- `metadata_keys = VECOLI_XARRAY_METADATA_KEYS` — the PR's literal list: `experiment_id`, `description`, `sim_data_path`, `time`, `suffix_time`, `time_step`, `initial_global_time`, `max_duration`, `fail_at_max_duration`, `lineage_seed`, `seed`, `variants`, `n_init_sims`, `generations`, `agent_id`, `parallel`, `skip_baseline`, `log_updates`, `single_daughters`, `daughter_outdir`, `fixed_media`, `condition`, `parca_options`, `mar_regulon`, `amp_lysis`, `divide`, `d_period`, `division_threshold`, `division_variable`, `chromosome_path`.
- `metadata_validators = {}` — **deliberately empty.** The PR's `expected` dict (`single_daughters: True`, `save: False`, `save_times: False`, `emit_config: False`, `emit_topology: False`, `emit_processes: False`, `emit_unique: False`) is vivarium-Engine-specific and no longer meaningful in a process-bigraph composite. The preset docstring lists the dropped checks so the behavior change is greppable.
- `writer` defaults to `{"type": "async", "out_uri": out_uri}` when `None`.
- `transducer`, `view`, `metadata`, `output_metadata`, `debug` pass through.

**Calls not preset-able:** `transducer` and `view` describe *which variables* the user wants emitted and at what shape, which is per-composite. The preset only fills in the vEcoli defaults around them.

## Shared helpers

`emitter_presets.py` exports as public API:

- `parquet_vecoli(...)` — config builder, signature above.
- `xarray_vecoli(...)` — config builder, signature above.
- `VECOLI_PARQUET_DTYPE_OVERRIDES: dict[str, str]` — the literal upstream `USE_UINT16` ∪ `USE_UINT32` mapping, importable so users can extend it (`{**VECOLI_PARQUET_DTYPE_OVERRIDES, "my_col": "Float32"}`).
- `VECOLI_XARRAY_METADATA_KEYS: list[str]` — the metadata key list, same reason.

The module imports neither `polars` nor `xarray`/`zarr`. Dtypes are referenced as strings. The module is therefore importable in any v2ecoli install regardless of which extras are selected.

## Dependency management

### `pyproject.toml` extras

```toml
[project.optional-dependencies]
parquet = [
    "duckdb",
    "polars",
    "fsspec",
    "tqdm",
]
xarray = [
    "xarray>=2026.04",
    "zarr~=3.1.6",
    "zarrs>=0.2",
]
emitters = ["v2ecoli[parquet,xarray]"]
```

Version pins for `xarray`, `zarr`, `zarrs` match PR #414's `pyproject.toml` exactly. The PR carries an explicit `# concurrency control in ecoli.library.xarray_emitter.zarr_writer might require adjustments for zarr>=3.2` comment; v2ecoli inherits that constraint and the same risk. `duckdb`, `polars`, `fsspec`, `tqdm` are unpinned upstream and stay that way here.

### Import guard pattern

Each emitter module starts with:

```python
# v2ecoli/library/parquet_emitter.py
try:
    import duckdb
    import polars as pl
    import fsspec
    import tqdm
except ImportError as e:
    raise ImportError(
        f"v2ecoli.library.parquet_emitter requires the [parquet] extra. "
        f"Install with: pip install 'v2ecoli[parquet]'. (missing: {e.name})"
    ) from e
```

Same pattern for `xarray_emitter/__init__.py`. Users get one clear line, not a deep traceback, when the wrong extra is installed.

### `uv.lock`

Adding extras updates the lockfile; the implementation PR commits the regenerated `uv.lock`.

### PR call-out

Per AGENTS.md, the PR description has an explicit **"Dependency changes"** section listing the seven added packages (`duckdb`, `polars`, `fsspec`, `tqdm`, `xarray`, `zarr`, `zarrs`) and the extras they slot into.

## Testing

### Unit tests

- `tests/test_parquet_emitter.py` — round-trip a synthetic state through `ParquetEmitter` end-to-end, no v2ecoli composite, no vEcoli code. Verify: row count, column names, dtype fidelity, `dtype_overrides` honored (exact-name and fnmatch), `partitioning_keys` produces correct hive paths, `KeyError` on missing partition key, `close()` idempotent and joins executor, `query()` returns expected rows in both partitioned and non-partitioned layouts.
- `tests/test_xarray_emitter.py` — same shape for `XArrayEmitter`. Verify: zarr store round-trips, `query()` returns a `DataTree`, sub-selection by path works, `metadata_validators` failures raise `ValueError` with a useful message, empty validators silently pass.

Neither test is marked `@pytest.mark.sim`. Fast, deterministic, no ParCa cache, no fixtures.

### Preset tests

- `tests/test_emitter_presets.py` — call `parquet_vecoli(...)` / `xarray_vecoli(...)` with representative arguments, assert the returned dicts contain the expected partition keys / dtype overrides / metadata keys. Cross-check `VECOLI_PARQUET_DTYPE_OVERRIDES` against the upstream `USE_UINT16` / `USE_UINT32` sets (`ecoli.library.parquet_emitter.USE_UINT16` is already importable in v2ecoli's venv). Construct an emitter from the preset config, emit a synthetic state matching one of the listener columns, assert the parquet schema's dtype matches and the hive path is `experiment_id=.../variant=.../lineage_seed=.../generation=.../agent_id=...`. Same shape for xarray.

### vEcoli-parity smoke test

- `tests/test_emitter_vecoli_parity.py` — **marked `@pytest.mark.sim`.**
  - **Parquet:** synthesize a fake history payload that looks like vEcoli's flattened state (a few `listeners__*__*` columns including some from `USE_UINT16` / `USE_UINT32`). Write it via the v2ecoli port using the `parquet_vecoli` preset; write the same payload via the installed `ecoli.library.parquet_emitter.ParquetEmitter`. Diff: same column set, same dtypes for the override columns, same row count and values, same hive layout. Allowed differences: file count partitioning, column insertion order, parquet file metadata.
  - **XArray:** no installed counterpart exists. Compare against a checked-in tiny golden Zarr store at `tests/fixtures/xarray_golden/`, generated once from PR #414's `XarrayEmitter` on a synthetic payload. Per AGENTS.md, fixtures are deliberate — this addition is called out in the implementation PR's description. The golden is regenerated only when PR #414's `transducer` / `writer` behavior changes upstream.

### Verified during implementation (not separate tests)

- `ImportError` text fires when extras are missing (manual smoke).
- `uv.lock` regenerates cleanly.
- `rg "from vivarium" v2ecoli/library/parquet_emitter.py v2ecoli/library/xarray_emitter/` returns zero hits.

### Explicitly not tested

- Full sim runs and behavior tests. `tests/test_model_behavior.py` already gates composite correctness; wiring the emitters into composites is a follow-up PR.
- Backend internals (DuckDB query planner, Zarr chunking).
- Performance / throughput.

## Risks

1. **PR #414 is draft and may rebase.** We pin to commit `b25ca24737ad7b1e4e6f042a1a84c2f03627f6a6` and record that hash in this design doc and in a `# vendored from vivarium-collective/vEcoli@b25ca24` header at the top of each `xarray_emitter` file. If upstream ships substantially changed, a follow-up PR re-vendors against the merged commit.
2. **`zarr~=3.1.6` is a narrow pin.** Upstream carries an explicit "may need adjustment for zarr>=3.2" comment. We carry the same constraint; the `emitter_presets.py` docstring repeats the warning so future maintainers don't bump it blindly.
3. **`vEcoli[dev]>=1.1.0` stays a runtime dep** for the parity test and for other v2ecoli code that uses `ecoli.*`. The port severs only `vivarium.core.emitter` imports from the emitter modules.
4. **`fsspec` explicit dep.** `polars` transitively pulls a compatible `fsspec`; making it explicit makes the parquet write path's `url_to_fs` call resilient if `polars` drops it later.
5. **`__del__`-driven `close()`** under interpreter shutdown has undefined ordering. Documented in the emitter docstrings: durability-sensitive callers must call `close()` explicitly. Same contract as `SQLiteEmitter`.
6. **`metadata_validators` empty in `xarray_vecoli` preset.** A user porting a vEcoli script that relied on the validators firing won't see them fire. The preset docstring lists the dropped checks (`single_daughters`, `save`, `save_times`, `emit_config`, `emit_topology`, `emit_processes`, `emit_unique`) so the behavior change is greppable.

## Open items (not resolved in this spec)

- **Composite teardown hook.** `process_bigraph.Composite` has no documented "on end-of-run" hook for Steps. The emitters rely on the caller (or `__del__`) to call `close()`. If upstream adds a teardown hook later, the emitters can adopt it without API change.
- **`dtype_overrides` matcher performance.** The vEcoli sets are ~30 names. Flat dict + fnmatch is fine. If the override list grows to hundreds of patterns we may need a pre-compiled matcher.
- **`[notebooks]` extra** for pandas / pyarrow conveniences — out of scope.

## Sequencing

**Single PR.** Reasoning:

- Both emitters share `emitter_presets.py`, the import-guard pattern, the testing scaffold, and the dependency-changes call-out.
- Both come from the same upstream snapshot (PR #414 head); landing together keeps that snapshot coherent.
- Reviewer load: ~3000 LOC of vendored code (largely unchanged from PR #414) plus ~600 LOC of new generic-class glue, presets, and tests.

**Fallback split** if reviewers push back:
- PR 1: `ParquetEmitter` + shared scaffolding (extras setup, import guards, `emitter_presets.py` skeleton with only `parquet_vecoli`).
- PR 2: `XArrayEmitter` package + `xarray_vecoli` preset.
The codebase tolerates that split — `xarray_emitter/` is a self-contained package.

## Acceptance criteria

- [ ] `pip install -e '.[parquet]'` succeeds; `python -c "from v2ecoli.library.parquet_emitter import ParquetEmitter"` succeeds.
- [ ] `pip install -e '.[xarray]'` succeeds; `python -c "from v2ecoli.library.xarray_emitter import XArrayEmitter"` succeeds.
- [ ] `pip install -e .` (no extras) succeeds; importing either emitter raises the guarded `ImportError` with the documented hint.
- [ ] `pytest -m "not sim" tests/test_parquet_emitter.py tests/test_xarray_emitter.py tests/test_emitter_presets.py` passes.
- [ ] `pytest -m sim tests/test_emitter_vecoli_parity.py` passes.
- [ ] `rg "from vivarium" v2ecoli/library/parquet_emitter.py v2ecoli/library/xarray_emitter/` returns no hits.
- [ ] `uv.lock` regenerated and committed.
- [ ] PR description has a "Dependency changes" section listing the seven added packages and their extras.
