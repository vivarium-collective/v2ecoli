"""``generation_time_offsets`` — the narrow per-generation offset scan that lets
``_MultigenMixin`` stream the wide multigeneration read one generation at a time
without re-materialising the whole lineage.

The multigeneration ptools failure (Run 1) was a DuckDB temp-disk spill: the
per-generation streaming filtered ``cumulative_time_history()``, whose recursive
CTE joins the wide ``o.*`` (every ``DOUBLE[]`` list column) back to itself, and
that self-join is not pruned by an outer ``WHERE generation = g`` — so each
"streamed" generation still read the whole lineage's arrays.  The fix computes
the cumulative-time offsets ONCE from a narrow scan (this function) and applies
each as a scalar to a per-generation base scan (which prunes).

For that substitution to be correct, the scalar offsets must reproduce EXACTLY
the ``global_time`` that ``cumulative_time_history`` bakes into its CTE — on both
the absolute-clock (parquet) path (offsets all 0, data untouched) and the
reset-clock (xarray) path (generations stacked end-to-end).  Jim's earlier
"filter first, then rewrite" attempt got this wrong by leaving every generation's
offset at 0 on reset-clock data; these tests are the guard against that class of
error.
"""
import duckdb
import pytest

from v2ecoli.workflow.analyses._helpers import (
    cumulative_time_history,
    generation_time_offsets,
)


def _con(rows):
    con = duckdb.connect()
    con.sql("CREATE TABLE h (generation BIGINT, global_time DOUBLE, val DOUBLE)")
    con.executemany("INSERT INTO h VALUES (?,?,?)", rows)
    return con


# gen0 t=0..2, gen1 t=3..5, gen2 t=6..8 — monotonic (parquet emitter)
_ABSOLUTE = [
    (0, 0.0, 10), (0, 1.0, 11), (0, 2.0, 12),
    (1, 3.0, 20), (1, 4.0, 21), (1, 5.0, 22),
    (2, 6.0, 30), (2, 7.0, 31), (2, 8.0, 32),
]
# each generation restarts at 0 (xarray emitter)
_RESET = [
    (0, 0.0, 10), (0, 1.0, 11), (0, 2.0, 12),
    (1, 0.0, 20), (1, 1.0, 21), (1, 2.0, 22),
    (2, 0.0, 30), (2, 1.0, 31), (2, 2.0, 32),
]


def test_absolute_clock_offsets_are_all_zero():
    """Already-absolute time must be left untouched (offset 0 every generation)."""
    con = _con(_ABSOLUTE)
    assert generation_time_offsets(con, "SELECT * FROM h") == {0: 0.0, 1: 0.0, 2: 0.0}


def test_reset_clock_stacks_generations_end_to_end():
    """Per-generation resets stack with a 1-unit gap: gen1 by 3 (2+1), gen2 by 6."""
    con = _con(_RESET)
    assert generation_time_offsets(con, "SELECT * FROM h") == {0: 0.0, 1: 3.0, 2: 6.0}


@pytest.mark.parametrize("rows", [_ABSOLUTE, _RESET], ids=["absolute", "reset"])
def test_scalar_offsets_reproduce_cumulative_time_history(rows):
    """The scalar offsets, applied per generation, must equal the global_time
    that cumulative_time_history bakes into its recursive CTE — bit for bit."""
    con = _con(rows)
    offs = generation_time_offsets(con, "SELECT * FROM h")
    baked = con.sql(
        f"SELECT generation, global_time FROM ({cumulative_time_history('SELECT * FROM h')}) "
        "ORDER BY generation, global_time"
    ).fetchall()
    case = " ".join(f"WHEN generation = {g} THEN {o}" for g, o in offs.items())
    scalar = con.sql(
        f"SELECT generation, global_time + CASE {case} ELSE 0 END AS gt "
        "FROM h ORDER BY generation, gt"
    ).fetchall()
    assert scalar == baked


def test_no_generation_column_returns_empty():
    con = duckdb.connect()
    con.sql("CREATE TABLE h2 (global_time DOUBLE, val DOUBLE)")
    con.executemany("INSERT INTO h2 VALUES (?,?)", [(0.0, 1), (1.0, 2)])
    assert generation_time_offsets(con, "SELECT * FROM h2") == {}


def test_single_generation_offset_is_zero():
    con = _con([(0, 0.0, 10), (0, 1.0, 11), (0, 2.0, 12)])
    assert generation_time_offsets(con, "SELECT * FROM h") == {0: 0.0}
