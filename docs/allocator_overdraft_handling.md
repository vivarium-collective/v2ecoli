# Allocator over-draft handling (PROTON[c] `NegativeCountsError`)

## Background

Under metabolic-edge conditions a multi-generation run could die with:

```
v2ecoli.steps.allocator.NegativeCountsError: Negative value(s) in counts_unallocated:
PROTON[c] (-916)
```

always preceded by `Warning: GLP_NOFEAS` (FBA infeasibility). Investigation
(see the dnaa-replication handoff note) established two distinct baseline defects
in `v2ecoli/steps/allocator.py`, fixed here.

## Defect 1 — int64 overflow in `calculatePartition`

The proportional split computed the excess fraction as

```python
requests * total_counts / total_requested
```

where `requests * total_counts` is an **int64** intermediate. For high-count
molecules (PROTON[c]-scale pools `> ~3e9`, where `pool * request > 2^63`) this
overflows and wraps to garbage, corrupting the partition (it could allocate
negative or over-allocate). Fuzzing (220k cases) showed the split is otherwise
correct at all normal magnitudes — overflow is the *only* arithmetic failure
mode.

**Fix:** cast to float before the multiply. For every non-overflowing magnitude
the result is bit-identical to the prior int64 path (the division already
produced float), verified at 0 mismatches / 20,000 random normal cases.

## Defect 2 — hard crash on a pre-existing-negative pool

The `-916` itself was *not* an allocator over-allocation. With the partition math
proven correct at normal magnitudes, the only way to get a small, clean
`counts_unallocated` deficit with all-positive allocations is that the pool was
**already negative on entry** (no process even requested it that tick):
`counts_unallocated = original(-916) - 0`. The allocator was merely *reporting* a
pool driven negative upstream — the FBA `GLP_NOFEAS` infeasibility tick — and a
single transient infeasibility was killing an entire lineage.

**Fix:** the over-draft guard now degrades gracefully (`resolve_overdraft`):
clamp any genuine over-allocation down to the available pool, leave an
already-negative pool for the upstream to heal, and emit a **rate-limited
warning** with an event counter instead of raising. Genuine corruption — a
negative *request* or a negative *partition* — is still a hard `NegativeCountsError`.

This is intentionally a robustness band-aid at the allocator layer. The true
root cause (why FBA returns `GLP_NOFEAS` and lets PROTON[c] go negative) is a
separate, harder metabolism fix and is **not** addressed here; it was also not
reproducible on the current investigation branch (the autoregulation work has
since been retuned toward homeostasis, removing the metabolic edge).

## Tests

`tests/test_allocator_robustness.py` covers both defects: overflow-magnitude
splits stay within the pool; `resolve_overdraft` reports a pre-existing negative
pool, clamps a positive-pool over-allocation, and is a no-op within budget;
`Allocator.update` no longer raises on a negative pool but still raises on a
negative request.
