"""Robustness tests for the molecule Allocator.

Covers two baseline defects (see docs/proton_allocator_negativecounts_crash.md):

1. ``calculatePartition`` over-allocated when the int64 intermediate
   ``requests * total_counts`` overflowed (PROTON[c]-scale pools), corrupting the
   partition. The fractional split must use float arithmetic so it never
   overflows, and must never hand out more than ``total_counts``.

2. The over-draft guard hard-raised ``NegativeCountsError`` and killed the whole
   lineage when a molecule pool was already negative on entry (a transient FBA
   infeasibility upstream, e.g. PROTON[c] after GLP_NOFEAS). A single infeasible
   tick must degrade gracefully (clamp + report) instead of crashing.
"""

import numpy as np
import pytest

from v2ecoli.steps.allocator import (
    Allocator,
    NegativeCountsError,
    calculatePartition,
    resolve_overdraft,
)


# ---------------------------------------------------------------------------
# calculatePartition: never over-allocate, even at int64-overflow magnitudes
# ---------------------------------------------------------------------------

def _max_overdraft(partitioned, total_counts):
    """Largest amount by which any molecule's allocation exceeds its pool."""
    return int(np.max(partitioned.sum(axis=1) - total_counts))


def test_partition_no_overdraft_at_overflow_magnitudes():
    """PROTON-scale pools (>3e9) made `requests * total_counts` overflow int64,
    producing a corrupt (over- or under-allocating) partition. The split must
    stay within the pool."""
    rng = np.random.RandomState(0)
    # two processes each requesting ~the full pool -> excess branch; pool large
    # enough that req*pool > int64 max (9.2e18).
    for tot in [3_100_000_000, 4_000_000_000, 4_900_000_000]:
        total_counts = np.array([tot], dtype=np.int64)
        requests = np.array([[tot, tot]], dtype=np.int64)  # total 2*tot > pool
        priorities = np.zeros(2)
        part = calculatePartition(priorities, requests, total_counts.copy(), rng)
        assert np.all(part >= 0), f"negative allocation at tot={tot}: {part}"
        assert _max_overdraft(part, total_counts) <= 0, (
            f"over-draft at tot={tot}: allocated {part.sum()} of {tot}")
        # an excess single-priority split should hand out the whole pool
        assert part.sum() == tot, f"under-allocated at tot={tot}: {part.sum()}/{tot}"


def test_partition_normal_regime_unchanged():
    """Regression guard: in the normal regime the split is fully-allocating and
    bounded (the property the parity baseline depends on)."""
    rng = np.random.RandomState(1)
    for _ in range(2000):
        n_proc = rng.randint(1, 8)
        tot = rng.randint(0, 200, size=3)
        req = rng.randint(0, 80, size=(3, n_proc))
        prio = rng.randint(0, 3, size=n_proc).astype(float)
        part = calculatePartition(prio, req.copy(), tot.copy(), rng)
        assert np.all(part >= 0)
        assert _max_overdraft(part, tot) <= 0


# ---------------------------------------------------------------------------
# resolve_overdraft: graceful clamp instead of hard crash
# ---------------------------------------------------------------------------

def test_resolve_overdraft_reports_preexisting_negative_pool():
    """Pool already negative on entry, nothing allocated (no process requested
    it): report the deficit, leave the (zero) allocation untouched, don't raise."""
    partitioned = np.array([[0, 0], [5, 3]], dtype=np.int64)   # mol0 unallocated
    original = np.array([-916, 100], dtype=np.int64)           # mol0 pre-negative
    clamped, overdrafts = resolve_overdraft(partitioned, original)
    assert [m for m, _ in overdrafts] == [0]
    assert dict(overdrafts)[0] == -916          # reported deficit
    assert np.array_equal(clamped[0], [0, 0])   # nothing to claw back
    assert np.array_equal(clamped[1], [5, 3])   # healthy molecule untouched


def test_resolve_overdraft_clamps_positive_pool_overallocation():
    """If a positive pool is over-allocated, scale the allocation down so the sum
    never exceeds the pool (defense-in-depth)."""
    partitioned = np.array([[60, 60]], dtype=np.int64)  # sums to 120
    original = np.array([100], dtype=np.int64)          # only 100 available
    clamped, overdrafts = resolve_overdraft(partitioned, original)
    assert [m for m, _ in overdrafts] == [0]
    assert clamped.sum() == 100                 # clamped to the pool exactly
    assert np.all(clamped >= 0)


def test_resolve_overdraft_noop_when_within_pool():
    """No over-draft -> returns the partition unchanged and reports nothing."""
    partitioned = np.array([[10, 20], [0, 5]], dtype=np.int64)
    original = np.array([100, 100], dtype=np.int64)
    clamped, overdrafts = resolve_overdraft(partitioned, original)
    assert overdrafts == []
    assert np.array_equal(clamped, partitioned)


# ---------------------------------------------------------------------------
# Allocator.update: a transient infeasible tick must not kill the run
# ---------------------------------------------------------------------------

def _make_allocator(molecule_names, process_names):
    return Allocator(parameters={
        "molecule_names": molecule_names,
        "process_names": process_names,
        "custom_priorities": {},
        "seed": 0,
    })


def _bulk(ids, counts):
    arr = np.zeros(len(ids), dtype=[("id", "U20"), ("count", "i8")])
    arr["id"] = ids
    arr["count"] = counts
    return arr


def test_update_does_not_raise_on_preexisting_negative_pool():
    """The reported crash: PROTON[c] negative on entry, no process requesting it.
    The allocator must report + continue, not raise NegativeCountsError."""
    mols = ["PROTON[c]", "ATP[c]"]
    procs = ["procA", "procB"]
    alloc = _make_allocator(mols, procs)
    states = {
        "bulk": _bulk(["PROTON[c]", "ATP[c]"], [-916, 1000]),
        "request": {
            # only ATP[c] (local idx 1) is requested; PROTON[c] is not
            "procA": {"bulk": [(1, 100)]},
            "procB": {"bulk": [(1, 100)]},
        },
        "allocator_rng": np.random.RandomState(0),
        "listeners": {"atp": {
            "atp_requested": np.zeros(len(procs), dtype=int),
            "atp_allocated_initial": np.zeros(len(procs), dtype=int),
        }},
    }
    update = alloc.update(states)  # must not raise
    # ATP allocated fully (within pool); PROTON allocations are zero/non-negative
    alloc_a = update["allocate"]["procA"]["bulk"]
    alloc_b = update["allocate"]["procB"]["bulk"]
    assert alloc_a[1] + alloc_b[1] <= 1000          # ATP within pool
    assert alloc_a[0] == 0 and alloc_b[0] == 0       # no PROTON handed out
    assert getattr(alloc, "_overdraft_events", 0) >= 1


def test_update_still_raises_on_negative_request():
    """A negative *request* is genuine corruption, not a transient pool dip — it
    must stay a hard error."""
    mols = ["PROTON[c]", "ATP[c]"]
    procs = ["procA"]
    alloc = _make_allocator(mols, procs)
    states = {
        "bulk": _bulk(["PROTON[c]", "ATP[c]"], [100, 1000]),
        "request": {"procA": {"bulk": [(1, -5)]}},
        "allocator_rng": np.random.RandomState(0),
        "listeners": {"atp": {
            "atp_requested": np.zeros(1, dtype=int),
            "atp_allocated_initial": np.zeros(1, dtype=int),
        }},
    }
    with pytest.raises(NegativeCountsError):
        alloc.update(states)
