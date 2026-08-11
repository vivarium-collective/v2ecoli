# Single-cell XArray Emitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a single-cell `ecoli_baseline` run with `emitter="xarray"` stream real `global_time` + `bulk` + `listeners` to a zarr store with bounded memory (today it emits only `global_time`).

**Architecture:** Replace the `set_null_emitter_override(True)` in `ecoli_baseline.py`'s `emitter=="xarray"` branch with an in-document `XArrayEmitter` step, materialized via the existing dormant branch in `_helpers._build_declared_emitter` (which already wires `global_time/bulk/listeners` with agent-relative topology). The step's `config` carries a `view` + `output_metadata` computed at build time from the already-seeded `cell_state` (bulk/listeners shapes are present from `cell_state.update(initial_state)` at `ecoli_baseline.py:1081`), using `strategy="flat"` + `emit_root=()` because the in-document step is co-located inside `agents/0` (unlike the lineage path's `strategy="colony"`).

**Tech Stack:** Python 3.12, process-bigraph, pbg_emitters `XArrayEmitter`, zarr v3, existing `v2ecoli/library/xarray_run.py` helpers.

## Global Constraints

- **Parquet stays the default.** Only the `emitter=="xarray"` branch changes. `emitter="parquet"` (default), `"sqlite"`, `"null"`, batch/lineage paths, and any external override (`_any_external`) must be byte-identical to before.
- **v2ecoli only.** No vivarium-workbench changes.
- **Run in the compare-harness venv** (has v2ecoli + pbg_emitters + a valid cache): `~/code/v2ecoli--compare-harness/.venv/bin/python`, with `PYTHONPATH=~/code/v2ecoli--xarray-singlecell` prepended and `cache_dir="/Users/eranagmon/code/v2ecoli/out/cache"` (absolute). Verify the worktree's package wins: `python -c "import v2ecoli; print(v2ecoli.__file__)"`.
- **Bounded memory** is a hard requirement: the config must stream (transducer buffer + async writer), never accumulate a per-tick history list.
- Emitter class → in-document topology is agent-relative (`{"bulk":("bulk",), ...}`), NOT top-level.

---

### Task 1: Spike — establish the minimal working single-cell XArray config

**Rationale:** The `view`/`output_metadata` construction for `bulk` (a vector, dropped by `view_from_emit_paths`) plus the listener set, and the flat in-document emitter's end-to-end behavior, cannot be written with confidence from static reading. This spike produces the exact working config recipe that Tasks 2-4 productionize. It is throwaway (a scratch script), not committed.

**Files:**
- Create (scratch, not committed): `/private/tmp/claude-502/.../scratchpad/xarray_spike.py`
- Read: `v2ecoli/library/xarray_run.py` (`view_from_emit_paths:113`, `extract_output_metadata_from_state:167`, `build_emitter_config:382`, `_view_leaves`, `_KNOWN_VECTOR_LEAVES:46`), `v2ecoli/composites/_helpers.py:445-468` (XArrayEmitter branch), `v2ecoli/library/output_metadata.py` (`output_metadata(state)`).

**Interfaces:**
- Produces: a concrete `config` dict (the "single-cell flat XArray config recipe") that, wired into an in-document XArrayEmitter step, yields a zarr store with non-empty `bulk` and `listeners` after a short run. Record the exact recipe (keys, view shape, strategy/emit_root, transducer buffer size) in the task's completion notes for Task 2.

- [ ] **Step 1: Build ecoli_baseline (parquet default) and capture the seeded cell_state**

```python
# xarray_spike.py
from v2ecoli import build_composite
comp = build_composite("ecoli_baseline", cache_dir="/Users/eranagmon/code/v2ecoli/out/cache")
state = comp.state
cell = state["agents"]["0"]
print("bulk type/shape:", type(cell["bulk"]), getattr(cell["bulk"], "shape", None))
print("listener roots:", list(cell["listeners"].keys())[:20])
```

Run: `PYTHONPATH=~/code/v2ecoli--xarray-singlecell ~/code/v2ecoli--compare-harness/.venv/bin/python xarray_spike.py`
Expected: prints bulk as an array with a shape and the listener sub-keys — confirms the shapes are available at build time.

- [ ] **Step 2: Assemble a candidate flat config and construct an XArrayEmitter directly**

```python
from pathlib import Path
from v2ecoli.library.xarray_run import (
    view_from_emit_paths, extract_output_metadata_from_state)
from v2ecoli.library.output_metadata import output_metadata
from pbg_emitters import XArrayEmitter
from bigraph_schema import allocate_core

# Enumerate concrete listener leaf paths present in this cell (dotted).
def _leaf_paths(d, prefix="listeners"):
    for k, v in d.items():
        p = f"{prefix}.{k}"
        if isinstance(v, dict): yield from _leaf_paths(v, p)
        else: yield p
listener_paths = list(_leaf_paths(cell["listeners"]))
view = view_from_emit_paths(listener_paths)            # listeners only
# Add bulk as an explicit vector leaf root (view_from_emit_paths drops it):
view.append({"root": ("bulk",), "variables": {"bulk": [{"path": (), "dtype": "<i8"}]}})
named = output_metadata(state)
omd = extract_output_metadata_from_state(state, view, named_metadata=named)
store = Path("/private/tmp/.../scratchpad/spike.zarr")
config = {
    "emit": {"global_time": "float", "bulk": "array[integer]", "listeners": "tree"},
    "out_uri": str(store), "strategy": "flat", "emit_root": [],
    "transducer": {"predicate": [[{"subsample": {"interval": 1}}]], "buffer": {"size": 3}},
    "view": view, "output_metadata": omd,
    "writer": {"backend": "zarr", "store": str(store),
               "buffers_per_chunk": 1, "backend_config": {"format": 3}},
    "metadata": {}, "metadata_keys": [], "metadata_validators": {},
    "provenance": {}, "debug": False,
}
core = allocate_core()
em = XArrayEmitter(config, core)   # must not raise
print("constructed XArrayEmitter OK")
```

Run the script. Expected: constructs without raising. If it raises on `view`/`output_metadata` shape, adjust the `view` leaf structure (the `bulk` root spec and listener nesting are the likely culprits) until construction succeeds. **Record the working `view`/config shape.**

- [ ] **Step 3: Wire the emitter in-document and run to prove non-empty bulk/listeners**

Replace `cell["emitter"]` in the built document with a step `{ "_type": "step", "address": "local:XArrayEmitter", "config": config, "inputs": {"global_time":["global_time"],"bulk":["bulk"],"listeners":["listeners"]}, ... }`, rebuild the Composite from the mutated state, and `composite.run(5)`. Then assert the zarr store has non-empty observable data:

```python
import xarray as xr
dt = xr.open_datatree(str(store), engine="zarr")
groups = {g: {v: int(dt[g].ds[v].size) for v in dt[g].ds.data_vars} for g in dt.groups}
print(groups)
assert any(sz > 0 for g in groups.values() for sz in g.values()), "empty store"
print("SPIKE OK: bulk/listeners captured")
```

Run. Expected: `SPIKE OK`. If empty, iterate on `strategy`/`emit_root`/`view` (try `emit_root=[]` vs the step being scoped; try a larger buffer or a close-flush) until observable data lands. **The final working config + wiring is the recipe for Task 2.** If after reasonable iteration the flat in-document path cannot capture `bulk`, STOP and report — the fallback is the workbench-side `emit_root=("agents","0")` approach (Option 2 from the design), which changes the plan.

- [ ] **Step 4: No commit (throwaway spike).** Record the working recipe in the task notes.

---

### Task 2: `_single_cell_xarray_config()` helper

**Files:**
- Modify: `v2ecoli/composites/ecoli_baseline.py` (add a module-level helper near the other build helpers; ~40 lines)
- Test: `tests/test_single_cell_xarray_emitter.py` (create)

**Interfaces:**
- Consumes: the working config recipe from Task 1; `xarray_run.view_from_emit_paths`, `xarray_run.extract_output_metadata_from_state`, `output_metadata.output_metadata`.
- Produces: `_single_cell_xarray_config(cell_state: dict, *, out_uri: str, buffer_size: int = 3) -> dict` — returns the XArrayEmitter `config` dict (view + output_metadata + transducer + writer + `strategy="flat"` + `emit_root=[]`) built from `cell_state`. Pure (no IO). Used by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_single_cell_xarray_emitter.py
from v2ecoli.composites.ecoli_baseline import _single_cell_xarray_config

def _fake_cell():
    return {"global_time": 0.0,
            "bulk": [0, 1, 2, 3],
            "listeners": {"mass": {"cell_mass": 1.0, "dry_mass": 0.3}}}

def test_single_cell_xarray_config_is_flat_and_covers_bulk_and_listeners(tmp_path):
    cfg = _single_cell_xarray_config(_fake_cell(), out_uri=str(tmp_path / "s.zarr"))
    assert cfg["strategy"] == "flat" and cfg["emit_root"] == []
    assert cfg["out_uri"].endswith("s.zarr")
    # streaming, bounded: a small transducer buffer, not an unbounded history
    assert cfg["transducer"]["buffer"]["size"] >= 1
    roots = {tuple(entry["root"]) for entry in cfg["view"]}
    assert ("bulk",) in roots and ("listeners",) in roots
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=~/code/v2ecoli--xarray-singlecell ~/code/v2ecoli--compare-harness/.venv/bin/python -m pytest tests/test_single_cell_xarray_emitter.py::test_single_cell_xarray_config_is_flat_and_covers_bulk_and_listeners -q`
Expected: FAIL — `ImportError: cannot import name '_single_cell_xarray_config'`.

- [ ] **Step 3: Implement `_single_cell_xarray_config`** encoding Task 1's recipe (build listener paths from `cell_state['listeners']`, `view_from_emit_paths` for listeners, append the `bulk` vector root, `extract_output_metadata_from_state`, assemble the flat config). Use the EXACT view/config shapes validated in Task 1.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/composites/ecoli_baseline.py tests/test_single_cell_xarray_emitter.py
git commit -m "feat(baseline): single-cell XArray emitter config builder (flat, agent-relative)"
```

---

### Task 3: Wire the in-document XArray emitter into the `emitter=="xarray"` branch

**Files:**
- Modify: `v2ecoli/composites/ecoli_baseline.py:1202-1219` (the `emitter=="xarray"` branch)
- Test: `tests/test_single_cell_xarray_emitter.py` (extend)

**Interfaces:**
- Consumes: `_single_cell_xarray_config` (Task 2); `_helpers.set_default_emitter_decl`; `_helpers._build_declared_emitter`'s `XArrayEmitter` branch (`_helpers.py:450`).
- Produces: after this task, `build_composite("ecoli_baseline", emitter="xarray")` yields `state["agents"]["0"]["emitter"]` as a live `XArrayEmitter` with agent-relative wiring, and does NOT apply the null override.

- [ ] **Step 1: Write the failing test**

```python
def test_xarray_build_has_in_document_emitter(tmp_path):
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline",
                           cache_dir="/Users/eranagmon/code/v2ecoli/out/cache",
                           emitter="xarray")
    emitter_step = comp.state["agents"]["0"]["emitter"]
    inst = emitter_step["instance"] if isinstance(emitter_step, dict) else emitter_step[0]
    assert type(inst).__name__ == "XArrayEmitter"
    # agent-relative wiring resolves bulk -> agents/0/bulk (not top-level)
    wires = emitter_step["inputs"] if isinstance(emitter_step, dict) else {}
    assert "bulk" in wires and "listeners" in wires
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_single_cell_xarray_emitter.py::test_xarray_build_has_in_document_emitter -q`
Expected: FAIL — the emitter is a minimal RAMEmitter (global_time only) from the null override.

- [ ] **Step 3: Implement the branch change**

Replace the body of `if emitter == "xarray" and not _any_external:` (`ecoli_baseline.py:1202-1219`) — remove the `warnings.warn(...)` + `set_null_emitter_override(True)`; instead resolve an `out_uri` (reuse the sqlite branch's `_find_workspace_root()` → `<ws>/.pbg/xarray-runs/<experiment_id>.zarr`, fallback `out/xarray`), then:

```python
    if emitter == "xarray" and not _any_external:
        _xr_out = _resolve_xarray_out_uri(experiment_id)  # helper: ws/.pbg/xarray-runs/... or out/xarray
        set_default_emitter_decl({
            "address": "local:XArrayEmitter",
            "config": _single_cell_xarray_config(cell_state, out_uri=_xr_out),
            "paths": ["global_time", "bulk", "listeners"],
        })
```

`_get_special_step`/`_build_declared_emitter` (XArrayEmitter branch) then materialize the step with agent-relative topo, merging our `config`. The `finally` block already restores overrides.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Regression — parquet default unchanged**

```python
def test_parquet_default_still_parquet(tmp_path):
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline", cache_dir="/Users/eranagmon/code/v2ecoli/out/cache")
    step = comp.state["agents"]["0"]["emitter"]
    inst = step["instance"] if isinstance(step, dict) else step[0]
    assert "Parquet" in type(inst).__name__ or "RAM" in type(inst).__name__  # unchanged default path
```

Run: `... -m pytest tests/test_single_cell_xarray_emitter.py -q`. Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/composites/ecoli_baseline.py tests/test_single_cell_xarray_emitter.py
git commit -m "feat(baseline): emit real bulk/listeners for single-cell emitter='xarray' (#754 follow-up)"
```

---

### Task 4: Integration — real short run captures non-empty zarr data

**Files:**
- Test: `tests/test_single_cell_xarray_emitter.py` (extend; mark slow)

**Interfaces:**
- Consumes: the wired build from Task 3.

- [ ] **Step 1: Write the failing/slow test**

```python
import pytest

@pytest.mark.slow
def test_xarray_run_captures_bulk_and_listeners(tmp_path, monkeypatch):
    from v2ecoli import build_composite
    monkeypatch.setenv("V2ECOLI_EMITTER_EXPERIMENT_ID", "xr_it")
    # Point the xarray out_uri at tmp_path via the workspace-root resolver, or
    # pass an explicit out_uri override if the branch supports one.
    comp = build_composite("ecoli_baseline",
                           cache_dir="/Users/eranagmon/code/v2ecoli/out/cache",
                           emitter="xarray", out_dir=str(tmp_path))
    comp.run(5)
    # locate the produced .zarr under tmp_path and assert non-empty observables
    import xarray as xr, glob
    stores = glob.glob(str(tmp_path / "**/*.zarr"), recursive=True)
    assert stores, "no zarr store written"
    dt = xr.open_datatree(stores[0], engine="zarr")
    sizes = [int(dt[g].ds[v].size) for g in dt.groups for v in dt[g].ds.data_vars]
    assert any(s > 0 for s in sizes), "zarr store has no observable data"
```

- [ ] **Step 2: Run to verify it fails, then passes after implementation is correct**

Run: `... -m pytest tests/test_single_cell_xarray_emitter.py::test_xarray_run_captures_bulk_and_listeners -q -m slow`
Expected: initially may fail if `out_dir` isn't threaded to the xarray branch — thread `out_dir` into `_resolve_xarray_out_uri` so the test can target `tmp_path`. Iterate until PASS with non-empty data. This is the real proof that the nested wiring captures data end-to-end.

- [ ] **Step 3: Commit**

```bash
git add tests/test_single_cell_xarray_emitter.py v2ecoli/composites/ecoli_baseline.py
git commit -m "test(baseline): integration — single-cell xarray run captures bulk/listeners"
```

---

## Self-Review

- **Spec coverage:** root cause (Task 3 removes null-override), in-document emitter (Tasks 2-3), view/output_metadata via helpers (Tasks 1-2), bounded memory (config transducer/writer — asserted in Task 2, exercised in Task 4), parquet-default-unchanged (Task 3 Step 5), unit + one-real-run (Tasks 2-4). Covered.
- **Placeholder scan:** Task 1 is a spike (throwaway) whose output is the concrete recipe; Tasks 2-4 encode it. `_resolve_xarray_out_uri` and `_single_cell_xarray_config` are named + specified. The `out_dir` threading in Task 4 Step 2 is called out explicitly rather than assumed.
- **Type consistency:** `_single_cell_xarray_config(cell_state, *, out_uri, buffer_size=3) -> dict` used consistently in Tasks 2-3; emitter step access `state["agents"]["0"]["emitter"]` consistent across Tasks 3-4.
- **Risk:** if Task 1 shows the flat in-document path can't capture `bulk`, the fallback (workbench-side `emit_root=("agents","0")`) is a different plan — Task 1 Step 3 says STOP and report in that case.
