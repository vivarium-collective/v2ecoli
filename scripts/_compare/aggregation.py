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

⛔ REFUSAL, NOT ZERO. Every function here returns `None` when it cannot compute
an answer -- an empty trace, a window that excludes everything, a generation
label axis that does not line up. It never returns 0.0 as a stand-in. This
module is downstream of a defect where an unpopulated leaf read as exactly 0.0
and graded as a pass, so "no data" and "measured zero" must stay distinguishable
all the way to the card. A caller that turns a None into a 0.0 reintroduces it.

⚠ GENERATION INDICES ARE THE STORE'S OWN. `_generation` is built by the reader
from the zarr group names (`generation=<g>`), on the same time axis as every
other observable from that store. So the window is exact rather than inferred
from a dry-mass halving heuristic. It is NOT necessarily 0-based across engines
-- callers get `lower_bound` semantics of ">= this label", and the caller owns
what the label means.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "generation_window",
    "per_cell_means",
    "aggregate_cells",
    "CellStats",
]


class CellStats:
    """Result of aggregating across cells.

    `mean` is the across-cell mean of per-cell time-averages. `n` is the NUMBER
    OF CELLS -- not the number of timepoints -- which is what makes `sem`
    meaningful as a population quantity.
    """

    __slots__ = ("mean", "n", "std", "sem", "per_cell")

    def __init__(self, mean: float, n: int, std: float, sem: float,
                 per_cell: list[float]):
        self.mean = mean
        self.n = n
        self.std = std
        self.sem = sem
        self.per_cell = per_cell

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CellStats(mean={self.mean!r}, n={self.n!r}, "
                f"std={self.std!r}, sem={self.sem!r})")


def _as_pair(trace):
    """Coerce a (times, values) trace to two float arrays, or None."""
    if trace is None:
        return None
    try:
        t, v = trace
    except (TypeError, ValueError):
        return None
    t = np.asarray(t, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if t.size == 0 or v.size == 0 or t.size != v.size:
        return None
    return t, v


def generation_window(trace, gen_trace, *, lower_bound: int | None):
    """Restrict `trace` to timepoints whose generation label is >= `lower_bound`.

    Returns a (times, values) pair, or None if nothing survives.

    `lower_bound=None` is an explicit "no window" and returns the trace
    unchanged -- distinct from a window that happens to admit everything,
    because the caller may want to report which of the two it got.

    ⚠ Requires `gen_trace` to sample the SAME time axis as `trace`; both come
    from one store read, so they do by construction. If they do not line up this
    refuses (None) rather than aligning them by interpolation -- a generation
    label is a step function and interpolating it would invent fractional
    generations.
    """
    pair = _as_pair(trace)
    if pair is None:
        return None
    if lower_bound is None:
        return pair

    gpair = _as_pair(gen_trace)
    if gpair is None:
        return None

    t, v = pair
    gt, gv = gpair
    if gt.size != t.size or not np.allclose(gt, t, rtol=0, atol=1e-6):
        return None

    mask = gv >= float(lower_bound)
    if not mask.any():
        return None
    return t[mask], v[mask]


def per_cell_means(trace, gen_trace, *, lower_bound: int | None = None
                   ) -> list[float] | None:
    """Time-average `trace` WITHIN each generation, after applying the window.

    Returns one float per generation present, ordered by generation label, or
    None if there is nothing to average.

    ⚠ This is the step that must not be skipped. Averaging the windowed trace
    directly would pool timepoints across cells and weight each cell by its emit
    count; see the module docstring.
    """
    pair = _as_pair(trace)
    gpair = _as_pair(gen_trace)
    if pair is None:
        return None

    # Without generation labels there is exactly one cell we can identify -- the
    # whole trace -- and only if no window was asked for. A window we cannot
    # honour is a refusal, not a silently-unwindowed answer.
    if gpair is None:
        if lower_bound is not None:
            return None
        t, v = pair
        finite = v[np.isfinite(v)]
        return [float(finite.mean())] if finite.size else None

    t, v = pair
    gt, gv = gpair
    if gt.size != t.size or not np.allclose(gt, t, rtol=0, atol=1e-6):
        return None

    if lower_bound is not None:
        mask = gv >= float(lower_bound)
        if not mask.any():
            return None
        v = v[mask]
        gv = gv[mask]

    means: list[float] = []
    for g in np.unique(gv):
        cell = v[gv == g]
        cell = cell[np.isfinite(cell)]
        if cell.size:
            means.append(float(cell.mean()))
    return means or None


def aggregate_cells(values: Iterable[float] | None) -> CellStats | None:
    """Aggregate per-cell values into a population statistic.

    `std` is the SAMPLE standard deviation (ddof=1) and is 0.0 for a single
    cell, where `sem` is likewise 0.0. ⚠ An n=1 result is reported rather than
    refused -- a single cell IS a measurement -- but a caller grading a
    population claim should check `n` before believing the dispersion.
    """
    if values is None:
        return None
    vals = [float(x) for x in values if np.isfinite(x)]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    return CellStats(mean=mean, n=n, std=std, sem=sem, per_cell=vals)


def aggregate_seeds(traces: Sequence, gen_traces: Sequence | None = None, *,
                    lower_bound: int | None = None) -> CellStats | None:
    """Aggregate ACROSS SEEDS AND GENERATIONS -- the entry point a card wants.

    `traces` is one (times, values) per seed, as `_collect` already gathers.
    `gen_traces` is the matching `_generation` trace per seed; pass None only
    when no window is required and each trace is to be treated as one cell.

    Every cell from every seed contributes ONE value, so `n` is the total cell
    count -- e.g. 4 seeds x 3 windowed generations = 12, not 4 and not the
    number of emit points.

    ⛔ Returns None if NO seed yields a usable cell. A partial result is
    returned when SOME seeds do: that is deliberate (a crashed seed should not
    void the run) but it means `n` is the honest denominator and a caller
    reporting "4 seeds" without checking `n` would be overclaiming.
    """
    if not traces:
        return None
    gens = list(gen_traces) if gen_traces is not None else [None] * len(traces)
    if len(gens) != len(traces):
        return None

    cells: list[float] = []
    for trace, gen_trace in zip(traces, gens):
        got = per_cell_means(trace, gen_trace, lower_bound=lower_bound)
        if got:
            cells.extend(got)
    return aggregate_cells(cells)
