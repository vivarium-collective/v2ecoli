"""Cell-level aggregation for comparison cards: window by generation, average
within a cell, then aggregate across cells.

WHY THIS MODULE EXISTS. A card that grades a population statistic has to be
handed a population. Two defects motivated this, both found on the bioproduction
card and both invisible in the output:

  1. The card graded ONE SEED. `_collect` gathered every seed and the consumer
     took `traces[0]`, so a study declaring `seeds: 4` was graded on seed 0 and
     the other three were read from disk and dropped.
  2. NO GENERATION WINDOW existed anywhere in `scripts/_compare/`, so a mean
     spanned generations 0..N-1 including the pre-settling ones, even though the
     reference config's own analyses declare `generation_lower_bound: 5`.

⇒ The acceptance criteria called themselves population statistics and were not.

THE AGGREGATION SHAPE, and it is not a free choice. A cell is the unit of
observation, so a statistic is built by TIME-AVERAGING WITHIN A CELL FIRST and
then aggregating ACROSS CELLS -- never as a statistic over pooled raw timepoints.
Pooling timepoints weights a cell by how many emit points it happened to
produce: a cell that lived longer, or was emitted more densely, silently counts
for more. `n` is then a count of emit points rather than of cells, so any
dispersion derived from it describes the emitter's cadence, not the biology.

⚠ THE AVERAGE IS A MEAN, DELIBERATELY, AND A MEDIAN IS NOT A SUBSTITUTE. A
time-average is the quantity that integrates to a total -- a mean secretion rate
times a duration is an amount produced, which is what a yield is built from. A
time-median has no such property. "Robustify against spikes" is a plausible
reason to reach for a median here and it would silently change what is being
measured, so the tests pin the operator with asymmetric values.

⛔ REFUSAL, NOT ZERO. Every function here returns `None` when it cannot compute
an answer -- an empty trace, a window that excludes everything, a generation
label axis that does not line up, or a window that cannot be honoured because no
labels were supplied. It never returns 0.0 as a stand-in. This module is
downstream of a defect where an unpopulated leaf read as exactly 0.0 and graded
as a pass, so "no data" and "measured zero" must stay distinguishable all the way
to the card. A caller that turns a None into a 0.0 reintroduces it.

⚠ ALIGNMENT IS A PRECONDITION THE CALLER MUST ESTABLISH -- IT IS NOT GUARANTEED
BY READING ONE STORE. An earlier version of this docstring claimed `gen_trace`
samples the same time axis as `trace` "by construction, because both come from
one store read". THAT IS FALSE, and it was the load-bearing assumption:
`compare_matched_trajectories.py` writes `_generation` with `setdefault` INSIDE
its per-observable loop, so the labels are captured from whichever observable is
read first and then shared by all of them -- while each observable's time axis is
rebuilt from its OWN per-generation emit count, and an observable contributes
nothing for a generation where it emitted nothing. Any observable that is ragged
relative to the first one read will therefore NOT align.

⇒ A sparse observable -- one that only starts emitting once a pathway is induced
-- is exactly this shape, and is exactly what a bioproduction card grades. When
the axes do not line up these functions refuse, which is safe in isolation but
NOT safe on the grading path, where a refused axis is `ungraded` and scores 0
severity, i.e. it RELAXES a gate rather than failing it. A consumer must pass a
`_generation` trace belonging to the SAME observable, not a shared one.

⚠ GENERATION INDICES ARE THE STORE'S OWN, built by the reader from the zarr group
names (`generation=<g>`), so the window is exact rather than inferred from a
dry-mass halving heuristic. They are not necessarily 0-based across engines:
`lower_bound` means ">= this label", and the caller owns what the label means.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "generation_window",
    "labelled_cell_means",
    "per_cell_means",
    "aggregate_cells",
    "aggregate_seeds",
    "CellStats",
]


class CellStats:
    """Result of aggregating across cells.

    `mean` is the across-cell mean of per-cell time-averages. `n` is the NUMBER
    OF CELLS -- not the number of timepoints. `n_seeds` is how many distinct
    seeds contributed at least one cell, and is carried because it CANNOT be
    recovered from `n`: with unequal generation counts per seed, `n=9` is
    equally consistent with "3 seeds x 3 generations" and with "4 seeds, one of
    which died early". A caller that needs to say how many replicates it has
    must read `n_seeds`.

    ⚠ `sem` IS OPTIMISTIC, AND `n` BEING A CELL COUNT IS THE REASON. Cells within
    one seed are a LINEAGE -- shared ancestry, shared initial state, serially
    correlated -- so they are not independent draws. `std/sqrt(n)` therefore
    understates the true standard error by roughly `sqrt(1 + (m-1)*ICC)` for `m`
    cells per seed. Do NOT treat `sem` as a population standard error for an
    equivalence or tolerance claim; the honest estimator when `n_seeds > 1` is
    the between-seed variance, which this module does not compute.

    ⭐ `cell_keys` is `[(seed_index, generation_label), ...]` aligned row-for-row
    with `per_cell`, and exists so a BETWEEN-SEED vs ACROSS-GENERATION variance
    decomposition remains computable downstream. Those two variances mean
    different things -- replicate spread versus drift along a lineage -- and a
    flat list of values cannot distinguish them. This module deliberately does
    not compute the decomposition; it just declines to destroy the information.
    ⊕ The key mirrors `v2ecoli/library/card_vectors.py`, which keys cells
    `(lineage_seed, generation, agent_id)`; there is no agent id on this path.
    `generation_label` is None when labels were unavailable
    (`single_cell_per_trace`).

    ⚠ `std`/`sem` are `nan` when `n == 1`: the sample standard deviation is not
    estimable from one observation, and substituting 0.0 there would hand a
    zero-width interval to the weakest possible evidence. `nan` is
    self-announcing and propagates arithmetically; 0.0 is also the honest answer
    when `n > 1` and every cell agrees, so it would be ambiguous.
    """

    __slots__ = ("mean", "n", "std", "sem", "per_cell", "n_seeds", "cell_keys")

    def __init__(self, mean: float, n: int, std: float, sem: float,
                 per_cell: list[float], n_seeds: int | None = None,
                 cell_keys: list[tuple] | None = None):
        self.mean = mean
        self.n = n
        self.std = std
        self.sem = sem
        self.per_cell = per_cell
        self.n_seeds = n_seeds
        self.cell_keys = cell_keys

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CellStats(mean={self.mean!r}, n={self.n!r}, "
                f"n_seeds={self.n_seeds!r}, std={self.std!r}, sem={self.sem!r})")


def _as_pair(trace):
    """Coerce a (times, values) trace to two equal-length float arrays, or None.

    ⚠ The coercions are INSIDE the guarded block on purpose: a ragged leaf (this
    store emits `[]` defaults, so ragged columns are a known shape) or a
    non-numeric one raises from `np.asarray(..., dtype=float)`, and this module
    promises refusal rather than exceptions.
    """
    if trace is None:
        return None
    try:
        t_raw, v_raw = trace
        # A bare 2-element sequence (e.g. a values-only array of length 2)
        # unpacks happily into two SCALARS and would otherwise ravel into two
        # size-1 arrays and pass every guard below -- reporting a confident
        # one-point "measurement" instead of refusing.
        if np.ndim(t_raw) == 0 or np.ndim(v_raw) == 0:
            return None
        t = np.asarray(t_raw, dtype=float).ravel()
        v = np.asarray(v_raw, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if t.size == 0 or v.size == 0 or t.size != v.size:
        return None
    return t, v


def _aligned(pair, gpair):
    """True when a generation trace samples the same time axis as its trace."""
    t, _ = pair
    gt, _ = gpair
    return gt.size == t.size and np.allclose(gt, t, rtol=0, atol=1e-6)


def _windowed(trace, gen_trace, lower_bound: int | None):
    """THE single windowing implementation. Returns `(t, v, gv)` or None.

    `gv` is the surviving generation labels, or None when no labels were
    supplied and none were required. Both public windowing entry points go
    through here: a previous version had `generation_window` and the per-cell
    path each masking independently, and the two copies had ALREADY drifted
    apart on what to do when `gen_trace` is None. Two implementations of "which
    window did we apply" is the exact hazard this module exists to remove, so
    there is one.
    """
    pair = _as_pair(trace)
    if pair is None:
        return None
    t, v = pair

    gpair = _as_pair(gen_trace)
    if gpair is None:
        # A window we cannot honour is a refusal, never a silently-unwindowed
        # answer. With no window asked for, the unlabelled trace is fine.
        return None if lower_bound is not None else (t, v, None)

    # ⚠ Labels were SUPPLIED, so they are consulted — and a caller that cannot
    # trust the cell split must not get a cell split. Misalignment refuses here
    # even with no window, because `labelled_cell_means` uses `gv` to decide
    # WHERE ONE CELL ENDS, not only which cells to keep.
    if not _aligned(pair, gpair):
        return None
    _, gv = gpair

    if lower_bound is None:
        return t, v, gv

    mask = gv >= float(lower_bound)
    if not mask.any():
        return None
    return t[mask], v[mask], gv[mask]


def generation_window(trace, gen_trace, *, lower_bound: int | None):
    """Restrict `trace` to timepoints whose generation label is >= `lower_bound`.

    Returns a (times, values) pair, or None if nothing survives.

    `lower_bound=None` is an explicit "no window" and returns the trace
    unchanged -- distinct from a window that happens to admit everything,
    because the caller may want to report which of the two it got.

    ⛔ A window that cannot be honoured -- `lower_bound` set with no generation
    labels -- REFUSES. Silently returning the unwindowed trace would produce a
    plausible number that is not what was asked for.

    ⚠ Refuses rather than aligning a mismatched axis by interpolation: a
    generation label is a step function, and interpolating it would invent
    fractional generations. See the module docstring -- misalignment is a real
    and expected shape, not a defensive hypothetical.
    """
    # ⚠ THE ONE PLACE THIS DIVERGES FROM `_windowed`, AND IT IS NOT AN OVERSIGHT.
    # With `lower_bound=None` this function consults the labels for NOTHING — it
    # returns the trace whole — so their alignment is irrelevant and refusing on
    # it would deny an answer to a question nobody asked. `labelled_cell_means`
    # is the opposite: it needs `gv` to place cell boundaries, so supplied-but-
    # misaligned labels must refuse there even with no window. Same helper, two
    # legitimately different needs; documented rather than unified away.
    # ⇒ On the grading path a refusal is `ungraded`, which scores 0 and RELAXES a
    # gate, so a needless refusal is the UNSAFE direction, not the cautious one.
    if lower_bound is None:
        return _as_pair(trace)
    got = _windowed(trace, gen_trace, lower_bound)
    return (got[0], got[1]) if got else None


def labelled_cell_means(trace, gen_trace, *, lower_bound: int | None = None
                        ) -> list[tuple[float, float]] | None:
    """Time-average `trace` WITHIN each generation, keeping the generation label.

    Returns `[(generation_label, mean), ...]` ordered by label, or None if there
    is nothing to average.

    ⭐ THE LABEL IS CARRIED SO A VARIANCE DECOMPOSITION STAYS POSSIBLE. Between-
    seed and across-generation variance mean different things -- one is replicate
    spread, the other is drift or non-stationarity along a lineage -- and a card
    is intended to present them separately. Flattening cells to bare values
    destroys the only information that separates them, so this module keeps the
    labels even though nothing here computes the decomposition.

    ⚠ This is the step that must not be skipped. Averaging the windowed trace
    directly would pool timepoints across cells and weight each cell by its emit
    count; see the module docstring.
    """
    got = _windowed(trace, gen_trace, lower_bound)
    if got is None:
        return None
    _, v, gv = got

    # No labels (and none required) => the whole trace is the one cell we can
    # identify, and its generation is unknown rather than guessed at 0.
    if gv is None:
        finite = v[np.isfinite(v)]
        return [(None, float(finite.mean()))] if finite.size else None

    cells: list[tuple[float, float]] = []
    for g in np.unique(gv):          # np.unique sorts, so cells are in gen order
        cell = v[gv == g]
        cell = cell[np.isfinite(cell)]
        if cell.size:
            cells.append((float(g), float(cell.mean())))
    return cells or None


def per_cell_means(trace, gen_trace, *, lower_bound: int | None = None
                   ) -> list[float] | None:
    """The per-cell time-averages alone, dropping their generation labels.

    See `labelled_cell_means` for the labelled form. This projection exists
    because most callers want the values; anything computing a variance
    DECOMPOSITION wants the labels and should use the labelled form.
    """
    got = labelled_cell_means(trace, gen_trace, lower_bound=lower_bound)
    return [m for _, m in got] if got else None


def aggregate_cells(values: Iterable[float] | None, *,
                    n_seeds: int | None = None,
                    cell_keys: list[tuple] | None = None) -> CellStats | None:
    """Aggregate per-cell values into a population statistic.

    `std` is the SAMPLE standard deviation (ddof=1). ⚠ At `n == 1` both `std`
    and `sem` are `nan`, not 0.0 -- see `CellStats`. An n=1 result is reported
    rather than refused (a single cell IS a measurement), but its dispersion is
    explicitly not estimable and says so.
    """
    if values is None:
        return None
    try:
        vals = [float(x) for x in values]
    except (TypeError, ValueError):
        return None
    vals = [x for x in vals if np.isfinite(x)]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    if n > 1:
        std = float(arr.std(ddof=1))
        sem = float(std / np.sqrt(n))
    else:
        std = float("nan")
        sem = float("nan")
    return CellStats(mean=mean, n=n, std=std, sem=sem, per_cell=vals,
                     n_seeds=n_seeds,
                     cell_keys=list(cell_keys) if cell_keys is not None else None)


def aggregate_seeds(traces: Sequence, gen_traces: Sequence | None = None, *,
                    lower_bound: int | None = None,
                    single_cell_per_trace: bool = False) -> CellStats | None:
    """Aggregate ACROSS SEEDS AND GENERATIONS -- the entry point a card wants.

    `traces` is one (times, values) per seed, as `_collect` already gathers.
    `gen_traces` is the matching `_generation` trace per seed.

    ⛔ `gen_traces=None` REFUSES unless `single_cell_per_trace=True`. This is the
    module's sharpest edge and it was originally the other way round: making the
    cell-level correction opt-in meant the SHORTEST call --
    `aggregate_seeds(traces)` -- silently reproduced BOTH defects this module
    exists to fix, pooling every generation by emit count and returning a seed
    count in `n` while the docstring promised a cell count. Measured on 4 seeds
    x 8 generations that was a mean wrong by 36% and a `sem` 4.4x too tight. A
    caller that has not wired `_generation` has not consented to pooling; it has
    simply not finished. `single_cell_per_trace=True` is how a caller says "each
    trace really is one cell" and means it.

    Every cell from every seed contributes ONE value, so `n` is the total cell
    count -- e.g. 4 seeds x 3 windowed generations = 12, not 4 and not the
    number of emit points. `n_seeds` reports how many seeds contributed.

    ⚠ POOLING IS FLAT, ACROSS ALL CELLS, and that is deliberate: the estimand is
    "the mean over post-burn-in cells", for which unweighted pooling is the
    unbiased estimator, and it matches the `cell_order` convention already used
    elsewhere in this repo. ⚠ But with UNEQUAL generation counts per seed the
    pooled mean is confounded with lineage depth -- the same variable
    `lower_bound` exists to control. `n_seeds` and `per_cell` are carried so a
    caller can detect that imbalance; this module does not correct it.

    ⛔ Returns None if NO seed yields a usable cell. A partial result is returned
    when SOME seeds do -- a crashed seed should not void the run -- and
    `n_seeds` is the honest replicate count for reporting.
    """
    if traces is None or len(traces) == 0:
        return None
    if gen_traces is None:
        if not single_cell_per_trace:
            return None
        gens: Sequence = [None] * len(traces)
    else:
        gens = list(gen_traces)
        if len(gens) != len(traces):
            return None

    cells: list[float] = []
    keys: list[tuple] = []
    contributing = 0
    for seed_index, (trace, gen_trace) in enumerate(zip(traces, gens)):
        got = labelled_cell_means(trace, gen_trace, lower_bound=lower_bound)
        if got:
            for gen_label, mean in got:
                cells.append(mean)
                keys.append((seed_index, gen_label))
            contributing += 1
    return aggregate_cells(cells, n_seeds=contributing, cell_keys=keys)
