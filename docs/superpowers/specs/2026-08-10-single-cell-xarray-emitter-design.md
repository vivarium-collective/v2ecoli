# Single-cell XArray emitter — design

**Date:** 2026-08-10
**Repo:** v2ecoli
**Status:** approved design, pre-implementation

## Problem

Selecting `emitter="xarray"` for a **single-cell** `ecoli_baseline` run produces a
zarr store containing only `global_time` — no `bulk`, no `listeners`. The run
looks like it succeeded but the observable data is missing, which silently breaks
the Results tab and any analysis that reads the store.

### Root cause (code-cited)

- The single-cell document wraps the cell as `agents/0`, so the real data lives
  at `agents/0/bulk` and `agents/0/listeners/...`
  (`v2ecoli/composites/ecoli_baseline.py:1208-1211`).
- For `emitter=="xarray"`, `baseline()` calls `set_null_emitter_override(True)`
  (`ecoli_baseline.py:1094-1111`), which minimizes the internal `emitter` step to
  `global_time` only (`composites/_helpers.py:352-357`). The inline comment states
  the intent: XArray was built to be emitted *out of band* by the batch/lineage
  runner (`workflow/lineage.py`), so there is deliberately no in-document XArray
  step for a single cell.
- A working in-document XArray branch **does exist**
  (`_build_declared_emitter`, `_helpers.py:445-463`) with agent-relative wiring,
  but it is never reached for `emitter=="xarray"` (the null override pre-empts the
  declared-default path), and it lacks the per-composite `view`/`output_metadata`
  the XArrayEmitter needs (`_helpers.py:448-451`).

By contrast, **parquet works** because its emitter step sits *inside* `cell_state`
with agent-relative topology `{"bulk":("bulk",), "listeners":("listeners",)}`
(`_helpers.py:437-441`); after the cell is wrapped as `agents/0` those relative
wires resolve to `agents/0/bulk`. Any *top-level* emitter misses the nested data
unless told `emit_root=("agents","0")`.

## Goal / non-goals

**Goal:** A single-cell run with `emitter="xarray"` streams real
`global_time` + `bulk` + `listeners` to a zarr store with bounded memory —
matching what the parquet path captures today.

**Non-goals (YAGNI):**
- Do **not** change the default emitter. `ecoli_baseline` stays **parquet** by
  default (already memory-safe after vivarium-workbench #778).
- Do **not** touch the workbench. The dashboard Composites-tab run uses the
  declared parquet emitter; this change is exercised via
  `build_composite(..., emitter="xarray")`, study configs, and the emitter param.
- Do **not** change the batch/lineage xarray path (it already works out-of-band
  via `workflow/lineage.py`).

## Approach — in-document XArrayEmitter (chosen)

Mirror how parquet already works: give the single-cell `emitter="xarray"` path a
real in-document XArrayEmitter step, co-located inside `cell_state` with
agent-relative wiring so it resolves to `agents/0/*`.

### Change 1 — `ecoli_baseline.py` `emitter=="xarray"` branch

Replace `set_null_emitter_override(True)` (+ the "minimised to global_time"
warning) with an emitter override that materializes the in-document XArrayEmitter:
address `local:XArrayEmitter`, agent-relative input wiring
`{"global_time":("global_time",), "bulk":("bulk",), "listeners":("listeners",)}`,
and a `config` carrying the per-composite `view` + `output_metadata` (see Change
2). The existing `_build_declared_emitter` XArray branch (`_helpers.py:445-463`)
is the wiring template; this change makes it reachable and gives it the config it
was missing. Keep the external-override guard (`_any_external`) unchanged — an
explicit caller override still wins.

### Change 2 — per-composite `view` + `output_metadata` from `cell_state`

The XArrayEmitter needs to know the leaf set, shapes, and coordinates it will
write — notably that `bulk` is a **vector** needing count/molecule coords. Compute
these at build time from `cell_state` by reusing the existing helpers in
`v2ecoli/library/xarray_run.py`:

- `view_from_emit_paths(...)` — build the view from `["global_time","bulk","listeners"]`.
- `filter_view_to_existing_leaves(...)` — drop leaves absent from this build.
- `extract_output_metadata_from_state(...)` — coords/shapes (handles the known
  vector leaves, incl. `bulk`; see `_KNOWN_VECTOR_LEAVES`, `xarray_run.py:46-52`).
- `build_emitter_config(...)` — assemble the final `config` dict passed to the step.

These run against `cell_state` (agent-relative), so the resulting view/topology is
agent-relative and resolves correctly after the `agents/0` wrap.

### Bounded memory

Inherent to the XArrayEmitter: the `XarrayTransducer` buffers a small window and
the `AsyncBufferWriter` streams to zarr on buffer-fill
(`pbg_emitters/xarray_emitter/emitter.py:154-197`) — no unbounded accumulation.

## Data flow

```
build_composite("ecoli_baseline", emitter="xarray")
  → baseline():
      cell_state = {..., bulk, listeners, unique, ...}
      view, output_metadata = <xarray_run helpers over cell_state>   # Change 2
      cell_state['emitter'] = XArrayEmitter step (agent-relative wires + config)  # Change 1
      state = {'agents': {'0': cell_state}, 'global_time': 0.0}
  → composite.run(N)
      emitter reads agents/0/{global_time,bulk,listeners} each tick
      → transducer buffer → AsyncBufferWriter → zarr (streaming)
```

## Testing

1. **Unit (fast):** `build_composite("ecoli_baseline", emitter="xarray")`; assert
   `cell_state['emitter']` (i.e. `state['agents']['0']['emitter']`) has address
   ending `XArrayEmitter`, agent-relative input wiring for `global_time/bulk/
   listeners`, and a non-empty `view` covering `bulk` and `listeners`. Assert the
   null-override is NOT applied (internal emitter is the real XArray step).
2. **Integration (one real short run):** run the xarray composite ~5 ticks with an
   `out_uri` under a temp dir; assert the zarr store holds non-empty observable
   data for `bulk` and `listeners` (an observable-leaf check in the spirit of the
   workbench's `_zarr_store_has_observable_data`). Confirms the nested wiring
   actually captures data, not just that the step is present.
3. **Regression:** `emitter="parquet"` (default) build is byte-identical to before
   (no accidental behavior change to the default path).

## Risks & mitigations

- **`bulk` vector coords:** the main correctness risk. Mitigated by reusing
  `extract_output_metadata_from_state` / `_KNOWN_VECTOR_LEAVES`, which already
  handle vector leaves for the lineage xarray path.
- **Empty-store on very short runs:** the async writer only persists a filled
  buffer; a run shorter than the buffer may leave an empty store. Mitigation:
  the integration test uses enough ticks to fill the buffer (or a close-flush),
  and we verify observable data, matching the workbench's known short-run caveat.
- **Divergence from lineage xarray:** we deliberately reuse `xarray_run.py`
  helpers so the single-cell view/metadata are produced by the same code the
  lineage path trusts, avoiding a second, drifting implementation.
