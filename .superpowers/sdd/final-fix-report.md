# Final robustness fixes — analysis-flush branch

## Fix 1 — `place_output` moved inside per-step try/except

**File:** `v2ecoli/workflow/flush.py`, `run_flush()` inner loop

`place_output(...)` was called _after_ the `try/except` block, so an `OSError`
(e.g. disk full) or any other placement failure would propagate and abort the
entire flush. Moved the call inside the same `try/except` so a placement failure
appends to `skipped` and continues to the next step.

**New test:** `tests/test_run_flush.py::test_run_flush_skips_step_when_placement_raises` — GREEN

## Fix 2 — `RunExtract.close()` clears `self._ctx`

**File:** `v2ecoli/workflow/flush.py`, `RunExtract.close()`

After closing the DuckDB connection, the `_ctx` dict still held a reference to
the dead connection object. A subsequent `conn_ctx()` call would short-circuit
on `if not self._ctx` (truthy) and return the closed connection. Added
`self._ctx = {}` after `conn.close()`.

**New test:** `tests/test_run_extract.py::test_close_clears_ctx` — GREEN

## Test results

Full flush suite (7 files): **25/25 PASSED**
