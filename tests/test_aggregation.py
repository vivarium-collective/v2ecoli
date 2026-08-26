"""Tests for `scripts/_compare/aggregation.py`.

⚠ SEVERAL OF THESE ARE WRITTEN TO DISCRIMINATE, not merely to pass. The defects
this module fixes were both silent, so a test that a naive implementation also
passes is worthless here. The ones that carry weight are marked:

  * `test_pooled_timepoints_would_give_a_different_answer` -- fails if cells are
    pooled instead of averaged within-cell-first. Constructed with UNEQUAL emit
    counts per cell, because with equal counts the two methods agree and the
    test would be vacuous.
  * `test_window_without_generation_labels_refuses` -- fails if a window that
    cannot be honoured is silently ignored, which is the shape that produced a
    plausible wrong number in the first place.
  * `test_*_refuses_rather_than_returning_zero` -- fails if "no data" collapses
    into a measured 0.0, which on this card's grading path reads as a PASS.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts._compare.aggregation import (
    aggregate_cells,
    aggregate_seeds,
    generation_window,
    per_cell_means,
)


def trace(values, *, t0=0.0, dt=60.0):
    v = np.asarray(values, dtype=float)
    t = t0 + (np.arange(v.size) + 1) * dt
    return t, v


def gen_trace_for(times, gens):
    return np.asarray(times, dtype=float), np.asarray(gens, dtype=float)


# --------------------------------------------------------------------------- #
# generation_window
# --------------------------------------------------------------------------- #
def test_window_keeps_only_generations_at_or_above_the_bound():
    t, v = trace([1.0, 2.0, 3.0, 4.0])
    g = gen_trace_for(t, [0, 0, 1, 1])
    wt, wv = generation_window((t, v), g, lower_bound=1)
    assert list(wv) == [3.0, 4.0]
    assert list(wt) == list(t[2:])


def test_window_none_means_no_window_and_returns_everything():
    t, v = trace([1.0, 2.0, 3.0])
    g = gen_trace_for(t, [0, 1, 2])
    wt, wv = generation_window((t, v), g, lower_bound=None)
    assert list(wv) == [1.0, 2.0, 3.0]


def test_window_excluding_everything_refuses_rather_than_returning_zero():
    t, v = trace([1.0, 2.0])
    g = gen_trace_for(t, [0, 0])
    # ⛔ The point of the test: not 0.0, not an empty array -- None.
    assert generation_window((t, v), g, lower_bound=5) is None


def test_window_refuses_when_generation_axis_does_not_line_up():
    t, v = trace([1.0, 2.0, 3.0])
    # Same length, different times -- an interpolation would invent fractional
    # generations, so this must refuse instead.
    g = gen_trace_for(t + 7.0, [0, 1, 1])
    assert generation_window((t, v), g, lower_bound=1) is None


def test_window_refuses_on_length_mismatch():
    t, v = trace([1.0, 2.0, 3.0])
    g = gen_trace_for(t[:2], [0, 1])
    assert generation_window((t, v), g, lower_bound=1) is None


# --------------------------------------------------------------------------- #
# per_cell_means -- the aggregation SHAPE
# --------------------------------------------------------------------------- #
def test_per_cell_means_averages_within_each_generation():
    t, v = trace([1.0, 3.0, 10.0, 20.0])
    g = gen_trace_for(t, [0, 0, 1, 1])
    assert per_cell_means((t, v), g) == [2.0, 15.0]


def test_pooled_timepoints_would_give_a_different_answer():
    """⭐ THE DISCRIMINATING TEST for the aggregation shape.

    Cell 0 emits 4 points averaging 1.0; cell 1 emits 1 point of 6.0.
      cell-first:  mean(1.0, 6.0)               == 3.5
      pooled:      mean(1,1,1,1,6) == 10/5      == 2.0
    An implementation that pools timepoints returns 2.0 and fails here. With
    equal emit counts both methods give the same number, which is exactly why
    the counts are unequal.
    """
    t, v = trace([1.0, 1.0, 1.0, 1.0, 6.0])
    g = gen_trace_for(t, [0, 0, 0, 0, 1])

    assert per_cell_means((t, v), g) == [1.0, 6.0]

    stats = aggregate_cells(per_cell_means((t, v), g))
    assert stats.mean == pytest.approx(3.5)
    assert stats.mean != pytest.approx(2.0)
    # n is a CELL count, not an emit-point count.
    assert stats.n == 2


def test_window_applies_before_averaging():
    t, v = trace([100.0, 100.0, 2.0, 4.0])
    g = gen_trace_for(t, [0, 0, 1, 1])
    # Generation 0's large values must not reach the mean at all.
    assert per_cell_means((t, v), g, lower_bound=1) == [3.0]


def test_window_without_generation_labels_refuses():
    """⭐ DISCRIMINATING: a window we cannot honour must not be silently dropped.

    Silently ignoring it returns the unwindowed mean -- a plausible number that
    is not what was asked for, which is the failure mode this module exists to
    stop.
    """
    t, v = trace([1.0, 2.0, 3.0])
    assert per_cell_means((t, v), None, lower_bound=5) is None
    # ...but with no window asked for, the whole trace is one cell.
    assert per_cell_means((t, v), None) == [2.0]


def test_non_finite_values_are_dropped_not_propagated():
    t, v = trace([1.0, np.nan, 3.0, np.inf])
    g = gen_trace_for(t, [0, 0, 1, 1])
    assert per_cell_means((t, v), g) == [1.0, 3.0]


def test_cell_that_is_entirely_non_finite_is_omitted_not_zeroed():
    t, v = trace([np.nan, np.nan, 4.0])
    g = gen_trace_for(t, [0, 0, 1])
    # Generation 0 contributes NO cell rather than a 0.0 cell.
    assert per_cell_means((t, v), g) == [4.0]


def test_empty_trace_refuses_rather_than_returning_zero():
    assert per_cell_means(None, None) is None
    assert per_cell_means((np.array([]), np.array([])), None) is None


# --------------------------------------------------------------------------- #
# aggregate_cells
# --------------------------------------------------------------------------- #
def test_aggregate_cells_reports_cell_count_and_dispersion():
    stats = aggregate_cells([2.0, 4.0, 6.0])
    assert stats.n == 3
    assert stats.mean == pytest.approx(4.0)
    assert stats.std == pytest.approx(2.0)          # ddof=1
    assert stats.sem == pytest.approx(2.0 / np.sqrt(3))


def test_single_cell_reports_n_one_with_zero_dispersion():
    stats = aggregate_cells([5.0])
    assert stats.n == 1
    assert stats.mean == pytest.approx(5.0)
    assert stats.std == 0.0
    assert stats.sem == 0.0


def test_aggregate_cells_refuses_rather_than_returning_zero():
    assert aggregate_cells([]) is None
    assert aggregate_cells(None) is None
    assert aggregate_cells([np.nan, np.inf]) is None


# --------------------------------------------------------------------------- #
# aggregate_seeds -- the across-seed defect
# --------------------------------------------------------------------------- #
def test_all_seeds_contribute_not_just_the_first():
    """⭐ THE DISCRIMINATING TEST for the seed defect.

    The old consumer took traces[0]. Seed 0 alone means 1.0; all four seeds
    together mean 2.5. An implementation that grades seed 0 returns 1.0.
    """
    traces, gens = [], []
    for val in (1.0, 2.0, 3.0, 4.0):
        t, v = trace([val, val])
        traces.append((t, v))
        gens.append(gen_trace_for(t, [0, 0]))

    stats = aggregate_seeds(traces, gens)
    assert stats.mean == pytest.approx(2.5)
    assert stats.mean != pytest.approx(1.0)
    assert stats.n == 4


def test_n_counts_seeds_times_windowed_generations():
    traces, gens = [], []
    for _ in range(4):
        t, v = trace([1.0, 2.0, 3.0, 4.0])
        traces.append((t, v))
        gens.append(gen_trace_for(t, [0, 1, 2, 3]))

    # Window admits generations 1..3 => 3 cells per seed, 4 seeds => 12.
    stats = aggregate_seeds(traces, gens, lower_bound=1)
    assert stats.n == 12


def test_a_dead_seed_reduces_n_rather_than_voiding_the_run():
    t_ok, v_ok = trace([2.0, 2.0])
    g_ok = gen_trace_for(t_ok, [0, 0])
    traces = [(t_ok, v_ok), None, (t_ok, v_ok)]
    gens = [g_ok, None, g_ok]

    stats = aggregate_seeds(traces, gens)
    assert stats.n == 2                       # the honest denominator
    assert stats.mean == pytest.approx(2.0)


def test_all_seeds_dead_refuses_rather_than_returning_zero():
    assert aggregate_seeds([None, None], [None, None]) is None
    assert aggregate_seeds([]) is None


def test_mismatched_gen_trace_count_refuses():
    t, v = trace([1.0])
    assert aggregate_seeds([(t, v), (t, v)], [gen_trace_for(t, [0])]) is None
