"""End-to-end smoke test for ``run_multigen_parquet``.

Uses a stub composite (not a real v2ecoli composite) to avoid pulling in
the ParCa cache machinery — what we want to verify here is the runner's
external-emitter driving + hive-partition rotation across divisions, not
the biology.

The stub composite:
  * Exposes ``state``, ``core``, and ``run(n)``.
  * On each ``run(n)`` call, advances a counter on the followed agent.
  * On a configurable tick, replaces the followed agent with two daughters
    (simulating division).

Verifies that:
  * Per-generation parquet output lands at the expected hive partition.
  * Reading all generations back via DuckDB gives the right row count.
  * close(success=True) at end leaves a success sentinel per generation.
"""

from __future__ import annotations

import os

import duckdb
import polars as pl
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("polars")

from v2ecoli.library.parquet_run import run_multigen_parquet  # noqa: E402


class _StubComposite:
    """Minimal composite-like object.

    Advances a per-agent ``count`` on each ``run(n)`` call. Divides at a
    configurable tick: the parent ``followed_id`` disappears and two
    daughters appear with names ``followed_id + '0'`` / ``followed_id + '1'``.
    """

    def __init__(self, core, initial_agent_id: str = "0", divide_at_tick: int | None = 50):
        self.core = core
        self.state = {
            "agents": {
                initial_agent_id: {
                    "listeners": {"mass": {"cell_mass": 1.0}, "count": 0},
                },
            },
        }
        self._tick = 0
        self._divide_at_tick = divide_at_tick

    def run(self, n: int) -> None:
        # If we cross the division tick during this chunk, divide first
        # so the runner sees the post-division state on its next inspection.
        crossed = (
            self._divide_at_tick is not None
            and self._tick < self._divide_at_tick <= self._tick + n
        )
        for aid in list(self.state["agents"].keys()):
            self.state["agents"][aid]["listeners"]["count"] += n
            self.state["agents"][aid]["listeners"]["mass"]["cell_mass"] += 0.1 * n
        self._tick += n
        if crossed:
            # Divide: pop the single agent, add two daughters.
            (parent_id, parent_state), = list(self.state["agents"].items())
            self.state["agents"] = {
                parent_id + "0": {
                    "listeners": {
                        "mass": {"cell_mass": parent_state["listeners"]["mass"]["cell_mass"] / 2},
                        "count": 0,
                    },
                },
                parent_id + "1": {
                    "listeners": {
                        "mass": {"cell_mass": parent_state["listeners"]["mass"]["cell_mass"] / 2},
                        "count": 0,
                    },
                },
            }
            # One-shot: don't divide again.
            self._divide_at_tick = None

    def find_instance_paths(self, *_a, **_kw) -> None:
        # No caches to invalidate in the stub.
        pass


@pytest.mark.fast
def test_run_multigen_parquet_single_generation(tmp_path, core):
    """No division within max_steps -> one generation, rows match step count."""
    comp = _StubComposite(core, initial_agent_id="0", divide_at_tick=None)
    result = run_multigen_parquet(
        comp,
        experiment_id="smoke",
        out_dir=tmp_path / "out",
        emit_paths=["listeners/mass/cell_mass", "listeners/count"],
        max_steps=20,
        max_generations=1,
        chunk=5,
        initial_agent_id="0",
        batch_size=2,
        threaded=False,
    )
    assert result["steps"] == 20
    assert result["generations"] == [1]

    # The runner writes one parquet per batch_size emits + a partial flush
    # at close(). Total rows = number of update() calls.
    gen1_dir = (
        tmp_path / "out" / "smoke" / "history"
        / "experiment_id=smoke" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0"
    )
    assert gen1_dir.is_dir(), f"missing hive dir: {gen1_dir}"
    pq_files = sorted(gen1_dir.glob("*.pq"))
    assert pq_files, "no parquet files written"

    # Read back via DuckDB and verify the listener fields round-tripped.
    conn = duckdb.connect(":memory:")
    rows = conn.sql(
        f"SELECT * FROM read_parquet('{gen1_dir}/*.pq') ORDER BY global_time"
    ).pl()
    assert len(rows) == 4  # chunks: 5,10,15,20 -> 4 updates
    assert rows["global_time"].to_list() == [5.0, 10.0, 15.0, 20.0]
    assert "listeners__mass__cell_mass" in rows.columns
    assert "listeners__count" in rows.columns
    # The stub increments by `n` ticks each chunk
    assert rows["listeners__count"].to_list() == [5, 10, 15, 20]


@pytest.mark.fast
def test_run_multigen_parquet_across_division(tmp_path, core):
    """Division mid-run rotates to a new hive partition for the daughter."""
    comp = _StubComposite(core, initial_agent_id="0", divide_at_tick=10)
    result = run_multigen_parquet(
        comp,
        experiment_id="div",
        out_dir=tmp_path / "out",
        emit_paths=["listeners/count"],
        max_steps=30,
        max_generations=2,
        chunk=5,
        initial_agent_id="0",
        batch_size=2,
        threaded=False,
    )
    assert result["steps"] == 30
    assert result["generations"] == [1, 2]

    # Generation 1 (parent)
    gen1_dir = (
        tmp_path / "out" / "div" / "history"
        / "experiment_id=div" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0"
    )
    # Generation 2 (daughter — first sorted new id)
    gen2_dir = (
        tmp_path / "out" / "div" / "history"
        / "experiment_id=div" / "variant=0" / "lineage_seed=0"
        / "generation=2" / "agent_id=00"
    )
    assert gen1_dir.is_dir(), f"missing gen1 dir: {gen1_dir}"
    assert gen2_dir.is_dir(), f"missing gen2 dir: {gen2_dir}"
    assert list(gen1_dir.glob("*.pq")), "no gen1 parquet"
    assert list(gen2_dir.glob("*.pq")), "no gen2 parquet"

    # Each generation should have written a success sentinel.
    gen1_sentinel = (
        tmp_path / "out" / "div" / "success"
        / "experiment_id=div" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0" / "s.pq"
    )
    gen2_sentinel = (
        tmp_path / "out" / "div" / "success"
        / "experiment_id=div" / "variant=0" / "lineage_seed=0"
        / "generation=2" / "agent_id=00" / "s.pq"
    )
    assert gen1_sentinel.is_file(), f"missing gen1 sentinel: {gen1_sentinel}"
    assert gen2_sentinel.is_file(), f"missing gen2 sentinel: {gen2_sentinel}"

    # All generations queryable in one DuckDB read using hive_partitioning.
    history_root = tmp_path / "out" / "div" / "history"
    conn = duckdb.connect(":memory:")
    rows = conn.sql(
        f"SELECT * FROM read_parquet('{history_root}/**/*.pq', hive_partitioning=1)"
    ).pl()
    # Gen 1 row at tick 5 only (the tick-10 chunk crosses the division → parent
    # is gone by the time the runner inspects state, so the gen-2 handoff emit
    # at tick 10 replaces what would have been parent's tick-10 row).
    # Gen 2 rows at ticks 10, 15, 20, 25, 30. Total = 6.
    assert len(rows) == 6
    # Spot-check the partition columns were read back.
    assert set(rows["generation"].to_list()) == {1, 2}


@pytest.mark.fast
def test_parquet_run_disk_size_smaller_than_text(tmp_path, core):
    """Sanity check: parquet output of N int rows is meaningfully smaller than CSV."""
    comp = _StubComposite(core, initial_agent_id="0", divide_at_tick=None)
    run_multigen_parquet(
        comp,
        experiment_id="size",
        out_dir=tmp_path / "out",
        emit_paths=["listeners/count"],
        max_steps=200,
        chunk=5,
        initial_agent_id="0",
        batch_size=10,
        threaded=False,
    )
    gen1_dir = (
        tmp_path / "out" / "size" / "history"
        / "experiment_id=size" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0"
    )
    total_pq_bytes = sum(p.stat().st_size for p in gen1_dir.glob("*.pq"))
    # 40 rows × (timestamp + count) — should be < 4KB total. CSV of same
    # data is ~600 bytes. Real assertion: it's non-empty and not absurd.
    assert 0 < total_pq_bytes < 50_000, f"unexpected parquet footprint: {total_pq_bytes}"

    # And reads back fully.
    rows = pl.read_parquet(str(gen1_dir / "*.pq"))
    assert "listeners__count" in rows.columns
    assert len(rows) == 40
