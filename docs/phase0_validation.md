# Phase 0 — Local Basal Validation Verdict

*2026-06-24*

## Outcome: mass explosion FIXED; multi-generation blocked on a separate division bug

### What passed
- **v2ecoli basal (1 gen):** `cell_mass` physical, ratio **1.78×** over the generation, divides cleanly. `physical=True`.
- **Upstream-wrapper basal (1 gen):** after the bulk-reconcile fix (commit `077ba750`),
  `cell_mass` grows **1.83×** (1282→2347 fg) and divides at generation 1 in 2581 steps —
  a realistic ~59-min doubling. **Was a 16.8× explosion before the fix.**
- **Engine convergence:** upstream **1.83×** vs v2ecoli **1.78×** from the same 1282 fg
  start — the two engines now agree closely on gen 1, exactly what the harness exists to show.

### Root cause fixed (the mass explosion)
`BulkNumpyUpdate` had no `reconcile` dispatch → multiple same-layer `bulk` writers fell back to
`reconcile(Node)`'s last-wins, dropping all but one writer's deltas. In the wrapper this dropped
PolypeptideInitiation's ribosomal-subunit consumption → runaway ribosome initiation → mass
explosion. Fixed by a concatenating `reconcile` (`v2ecoli/types/bulk_numpy.py`). Regression test
`tests/test_bulk_reconcile.py`. v2ecoli parity golden re-captured (accepted small baseline shift).

### What is still blocked (NEW, separate bug)
The 2-generation gate crashes at the **division boundary** (gen-1→gen-2 handoff, 2581s):

```
ecoli/processes/partition.py:98   process = states["process"][0]
IndexError: tuple index out of range
```

The daughter cell's `process` store (the shared `PartitionedProcess` instances, stored as 1-tuples)
is **empty** after division, so generation 2 cannot run. `_build_full_daughter` builds a fresh cell
(which should seed `process`) and overlays only `bulk/unique/environment/boundary` — so the empty
`process` store is a real, separate root cause still to be traced. The fail-loud guard in
`xarray_run.py` correctly caught this instead of emitting a partial zarr.

### Status of PR #289 / sms-api #147 merge gate
- Gen-1 physical validity (the dynamics validation #289 was missing) is now **demonstrated** for
  both engines, plus the explosion fix on top.
- Multi-generation (needed for the 16-gen goal) is **not** yet green — blocked on the division
  handoff bug above. Recommend resolving that before merging the bundle, or merging the
  explosion-fix + gen-1 validation and tracking division separately.

### Next
1. Trace the daughter empty-`process` division bug (systematic-debugging) → multi-gen green.
2. Then resume the plan: per-process divergence attribution across the 5 conditions → GovCloud 16×16×5.
