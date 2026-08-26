"""Tests for `scripts/_compare/aggregation.py`.

⚠ THESE ARE WRITTEN TO DISCRIMINATE, not merely to pass. Both defects the module
fixes were silent, so a test a naive implementation also passes is worthless.

⛔ AND THE FIRST VERSION OF THIS FILE FAILED AT EXACTLY THAT, in a way worth
recording because adding more tests would not have found it. Every within-cell
fixture used a symmetric two-point set (`[1,3]->2`, `[10,20]->15`) where the mean
and the median COINCIDE -- so a `np.median` implementation passed all 20 tests
while measuring a different scientific quantity. The operator the module is named
for was not pinned by a single case. That is a FIXTURE-DESIGN failure, not a
coverage gap: the fixtures could not have distinguished the two no matter how
many assertions were hung off them. Hence `test_*_asymmetric_*` below, and the
rule that a central-tendency test uses values whose mean and median differ.

⭐ Mutation-verified. Naive implementations each killed by the marked tests:
pool-timepoints · first-seed-only · silently-ignore-window · zero-instead-of-
refuse · median-within-cell · median-across-cells · seed-weighted-mean ·
dropped-alignment-guards · dropped-length-guard.
⚠ That list bounds what was IMAGINED, not what is wrong.
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
    _, wv = generation_window((t, v), g, lower_bound=None)
    assert list(wv) == [1.0, 2.0, 3.0]


def test_window_excluding_everything_refuses_rather_than_returning_zero():
    t, v = trace([1.0, 2.0])
    g = gen_trace_for(t, [0, 0])
    assert generation_window((t, v), g, lower_bound=5) is None


def test_window_without_labels_refuses_on_the_window_function_too():
    """⭐ DISCRIMINATING. The refusal was previously pinned only on
    `per_cell_means`, so a `generation_window` that silently returned the
    unwindowed trace survived the whole suite."""
    t, v = trace([1.0, 2.0, 3.0])
    assert generation_window((t, v), None, lower_bound=5) is None
    # No window asked for => no labels needed.
    assert generation_window((t, v), None, lower_bound=None) is not None


def test_window_refuses_when_generation_axis_does_not_line_up():
    t, v = trace([1.0, 2.0, 3.0])
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


def test_within_cell_average_is_a_mean_not_a_median_asymmetric():
    """⭐ DISCRIMINATING for the OPERATOR itself.

    cell values [1, 1, 10]: mean 4.0, median 1.0. A time-average integrates to a
    total (rate x duration = amount, which is what a yield is built from); a
    median does not. Symmetric fixtures cannot tell them apart.
    """
    t, v = trace([1.0, 1.0, 10.0])
    g = gen_trace_for(t, [0, 0, 0])
    assert per_cell_means((t, v), g) == [4.0]
    assert per_cell_means((t, v), g) != [1.0]


def test_pooled_timepoints_would_give_a_different_answer():
    """⭐ DISCRIMINATING for the aggregation shape.

    Cell 0 emits 4 points averaging 1.0; cell 1 emits 1 point of 6.0.
      cell-first:  mean(1.0, 6.0)          == 3.5
      pooled:      mean(1,1,1,1,6) == 10/5 == 2.0
    Unequal emit counts are the point -- with equal counts the two agree.
    """
    t, v = trace([1.0, 1.0, 1.0, 1.0, 6.0])
    g = gen_trace_for(t, [0, 0, 0, 0, 1])

    assert per_cell_means((t, v), g) == [1.0, 6.0]

    stats = aggregate_cells(per_cell_means((t, v), g))
    assert stats.mean == pytest.approx(3.5)
    assert stats.mean != pytest.approx(2.0)
    assert stats.n == 2


def test_cells_are_returned_in_generation_order():
    t, v = trace([9.0, 1.0])
    g = gen_trace_for(t, [3, 1])
    assert per_cell_means((t, v), g) == [1.0, 9.0]


def test_window_applies_before_averaging():
    t, v = trace([100.0, 100.0, 2.0, 4.0])
    g = gen_trace_for(t, [0, 0, 1, 1])
    assert per_cell_means((t, v), g, lower_bound=1) == [3.0]


def test_window_without_generation_labels_refuses():
    """⭐ DISCRIMINATING: a window we cannot honour must not be silently dropped."""
    t, v = trace([1.0, 2.0, 3.0])
    assert per_cell_means((t, v), None, lower_bound=5) is None
    assert per_cell_means((t, v), None) == [2.0]


def test_per_cell_means_refuses_on_misaligned_axis():
    """⭐ DISCRIMINATING on the LIVE path.

    `aggregate_seeds` calls `per_cell_means`, not `generation_window`, so the
    alignment guards that matter are these. They were previously pinned only on
    the other function.
    """
    t, v = trace([1.0, 2.0, 3.0])
    assert per_cell_means((t, v), gen_trace_for(t + 7.0, [0, 1, 1])) is None
    assert per_cell_means((t, v), gen_trace_for(t[:2], [0, 1])) is None


def test_non_finite_values_are_dropped_not_propagated():
    t, v = trace([1.0, np.nan, 3.0, np.inf])
    g = gen_trace_for(t, [0, 0, 1, 1])
    assert per_cell_means((t, v), g) == [1.0, 3.0]


def test_cell_that_is_entirely_non_finite_is_omitted_not_zeroed():
    t, v = trace([np.nan, np.nan, 4.0])
    g = gen_trace_for(t, [0, 0, 1])
    assert per_cell_means((t, v), g) == [4.0]


def test_empty_trace_refuses_rather_than_returning_zero():
    assert per_cell_means(None, None) is None
    assert per_cell_means((np.array([]), np.array([])), None) is None


# --------------------------------------------------------------------------- #
# _as_pair contract -- refusal, never an exception, never a silent misread
# --------------------------------------------------------------------------- #
def test_mismatched_times_and_values_lengths_refuse():
    """⭐ DISCRIMINATING: this guard was pinned nowhere."""
    assert per_cell_means((np.arange(3.0), np.arange(2.0)), None) is None


def test_bare_two_element_sequence_is_not_read_as_a_one_point_trace():
    """A values-only array of length 2 unpacks into two scalars and would
    otherwise ravel into two size-1 arrays and pass every guard -- reporting a
    confident measurement instead of refusing."""
    assert per_cell_means(np.array([5.0, 7.0]), None) is None


def test_ragged_and_non_numeric_traces_refuse_rather_than_raise():
    """This store emits `[]` defaults, so ragged columns are a known shape."""
    t = np.arange(2.0)
    assert per_cell_means((t, [[1.0, 2.0], [3.0]]), None) is None
    assert per_cell_means((t, ["a", "b"]), None) is None
    assert per_cell_means((t, [{"a": 1}, {"a": 2}]), None) is None


def test_aggregate_cells_refuses_non_numeric_rather_than_raising():
    assert aggregate_cells([1.0, None]) is None


# --------------------------------------------------------------------------- #
# aggregate_cells
# --------------------------------------------------------------------------- #
def test_aggregate_cells_reports_cell_count_and_dispersion():
    stats = aggregate_cells([2.0, 4.0, 6.0])
    assert stats.n == 3
    assert stats.mean == pytest.approx(4.0)
    assert stats.std == pytest.approx(2.0)          # ddof=1
    assert stats.sem == pytest.approx(2.0 / np.sqrt(3))


def test_across_cell_average_is_a_mean_not_a_median_asymmetric():
    """⭐ DISCRIMINATING for the across-cell operator, same failure as within."""
    stats = aggregate_cells([1.0, 1.0, 10.0])
    assert stats.mean == pytest.approx(4.0)
    assert stats.mean != pytest.approx(1.0)


def test_single_cell_reports_n_one_with_NON_ESTIMABLE_dispersion():
    """⛔ `nan`, not 0.0.

    Substituting 0.0 hands a ZERO-WIDTH interval to the weakest possible
    evidence -- maximum stated confidence at minimum actual confidence -- and is
    ambiguous with the honest 0.0 you get when n>1 and every cell agrees.
    """
    stats = aggregate_cells([5.0])
    assert stats.n == 1
    assert stats.mean == pytest.approx(5.0)
    assert np.isnan(stats.std)
    assert np.isnan(stats.sem)


def test_zero_dispersion_is_reported_when_it_is_genuinely_zero():
    stats = aggregate_cells([5.0, 5.0, 5.0])
    assert stats.n == 3
    assert stats.std == 0.0
    assert stats.sem == 0.0


def test_aggregate_cells_refuses_rather_than_returning_zero():
    assert aggregate_cells([]) is None
    assert aggregate_cells(None) is None
    assert aggregate_cells([np.nan, np.inf]) is None


# --------------------------------------------------------------------------- #
# aggregate_seeds -- the across-seed defect and the opt-in trap
# --------------------------------------------------------------------------- #
def test_all_seeds_contribute_not_just_the_first():
    """⭐ DISCRIMINATING for the seed defect. Seed 0 alone means 1.0; all four
    together mean 2.5."""
    traces, gens = [], []
    for val in (1.0, 2.0, 3.0, 4.0):
        t, v = trace([val, val])
        traces.append((t, v))
        gens.append(gen_trace_for(t, [0, 0]))

    stats = aggregate_seeds(traces, gens)
    assert stats.mean == pytest.approx(2.5)
    assert stats.mean != pytest.approx(1.0)
    assert stats.n == 4
    assert stats.n_seeds == 4


def test_missing_generation_traces_refuse_rather_than_pooling_silently():
    """⭐ THE SHARPEST TEST IN THE FILE.

    `aggregate_seeds(traces)` -- the shortest, most natural call -- previously
    treated each whole multi-generation trace as ONE cell: it pooled every
    generation by emit count and returned a SEED count in `n` while the
    docstring promised a cell count. On 4 seeds x 8 generations that was a mean
    wrong by 36% and a `sem` 4.4x too tight, with no way to tell from the result.
    Not wiring `_generation` is 'unfinished', not 'consent to pool'.
    """
    traces, gens = [], []
    for _ in range(4):
        t, v = trace([1.0, 2.0, 3.0, 4.0])
        traces.append((t, v))
        gens.append(gen_trace_for(t, [0, 1, 2, 3]))

    assert aggregate_seeds(traces) is None
    assert aggregate_seeds(traces, None) is None
    # ...and the correct call still works.
    assert aggregate_seeds(traces, gens).n == 16


def test_single_cell_per_trace_is_an_explicit_opt_in():
    traces = [trace([1.0, 3.0]), trace([5.0, 7.0])]
    stats = aggregate_seeds(traces, None, single_cell_per_trace=True)
    assert stats.n == 2            # one cell per trace, as asked
    assert stats.mean == pytest.approx(4.0)


def test_n_counts_seeds_times_windowed_generations():
    traces, gens = [], []
    for _ in range(4):
        t, v = trace([1.0, 2.0, 3.0, 4.0])
        traces.append((t, v))
        gens.append(gen_trace_for(t, [0, 1, 2, 3]))

    stats = aggregate_seeds(traces, gens, lower_bound=1)
    assert stats.n == 12           # 4 seeds x 3 windowed generations
    assert stats.n_seeds == 4


def test_unequal_cells_per_seed_pool_flat_not_seed_weighted():
    """⭐ DISCRIMINATING for the pooling decision, which no test previously
    exercised because every fixture had equal cells per seed.

    Seed A contributes cells [1, 1, 1]; seed B contributes [5].
      flat pooling:   mean(1,1,1,5) == 2.0
      seed-weighted:  mean(mean(1,1,1), 5) == mean(1,5) == 3.0
    Flat pooling is the unbiased estimator of 'the mean over post-burn-in
    cells', which is the estimand a card grades.
    """
    ta, va = trace([1.0, 1.0, 1.0])
    tb, vb = trace([5.0])
    traces = [(ta, va), (tb, vb)]
    gens = [gen_trace_for(ta, [0, 1, 2]), gen_trace_for(tb, [0])]

    stats = aggregate_seeds(traces, gens)
    assert stats.mean == pytest.approx(2.0)
    assert stats.mean != pytest.approx(3.0)
    assert stats.n == 4            # cells, not seeds
    assert stats.n_seeds == 2      # and the imbalance is detectable


def test_a_dead_seed_reduces_n_seeds_rather_than_voiding_the_run():
    t_ok, v_ok = trace([2.0, 2.0])
    g_ok = gen_trace_for(t_ok, [0, 0])
    stats = aggregate_seeds([(t_ok, v_ok), None, (t_ok, v_ok)],
                            [g_ok, None, g_ok])
    assert stats.n_seeds == 2      # the honest replicate count
    assert stats.mean == pytest.approx(2.0)


def test_all_seeds_dead_refuses_rather_than_returning_zero():
    assert aggregate_seeds([None, None], [None, None]) is None
    assert aggregate_seeds([]) is None


def test_mismatched_gen_trace_count_refuses():
    t, v = trace([1.0])
    assert aggregate_seeds([(t, v), (t, v)], [gen_trace_for(t, [0])]) is None
