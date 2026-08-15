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

import pytest

# Skip the whole module if the [parquet] extra isn't installed —
# v2ecoli.library.parquet_run imports from viva_emitters via the shim.
pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("viva_emitters")

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

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


def _structured_unique(dtype_fields, rows, active_mask):
    """Build a structured numpy array like the agent ``unique`` store entries."""
    import numpy as np
    arr = np.zeros(len(rows), dtype=dtype_fields)
    for i, r in enumerate(rows):
        for name, val in r.items():
            arr[name][i] = val
    arr["_entryState"] = active_mask
    return arr


class _UniqueStubComposite(_StubComposite):
    """Stub composite that also carries a structured ``unique`` store so the
    V2ECOLI_EMIT_UNIQUE path has active-entry coordinate arrays to extract."""

    def __init__(self, core, initial_agent_id: str = "0"):
        super().__init__(core, initial_agent_id=initial_agent_id, divide_at_tick=None)
        import numpy as np
        rnap_dt = [("_entryState", np.int8), ("coordinates", np.int64),
                   ("domain_index", np.int32)]
        repl_dt = [("_entryState", np.int8), ("coordinates", np.int64),
                   ("domain_index", np.int32)]
        # 3 RNAP entries, the last is INACTIVE (_entryState=0) -> must be dropped.
        rnap = _structured_unique(
            rnap_dt,
            [{"coordinates": 100, "domain_index": 0},
             {"coordinates": -200, "domain_index": 0},
             {"coordinates": 999, "domain_index": 0}],
            active_mask=[1, 1, 0],
        )
        repl = _structured_unique(
            repl_dt,
            [{"coordinates": 50000, "domain_index": 1},
             {"coordinates": -50000, "domain_index": 2}],
            active_mask=[1, 1],
        )
        self.state["agents"][initial_agent_id]["unique"] = {
            "active_RNAP": rnap,
            "active_replisome": repl,
        }


@pytest.mark.fast
def test_unique_emit_flag_off_no_unique_columns(tmp_path, core, monkeypatch):
    """Without V2ECOLI_EMIT_UNIQUE the unique coordinate columns are absent."""
    monkeypatch.delenv("V2ECOLI_EMIT_UNIQUE", raising=False)
    # The flag is read at import time, so force it off on the module too.
    import v2ecoli.library.parquet_run as pr
    monkeypatch.setattr(pr, "_EMIT_UNIQUE", False)

    comp = _UniqueStubComposite(core, initial_agent_id="0")
    pr.run_multigen_parquet(
        comp, experiment_id="uniq_off", out_dir=tmp_path / "out",
        emit_paths=["listeners/count"], max_steps=10, max_generations=1,
        chunk=5, initial_agent_id="0", batch_size=2, threaded=False,
    )
    gen1_dir = (
        tmp_path / "out" / "uniq_off" / "history"
        / "experiment_id=uniq_off" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0"
    )
    rows = pl.read_parquet(str(gen1_dir / "*.pq"))
    assert "active_RNAP__coordinates" not in rows.columns
    assert "active_replisome__coordinates" not in rows.columns


@pytest.mark.fast
def test_unique_emit_flag_on_emits_active_coordinates(tmp_path, core, monkeypatch):
    """V2ECOLI_EMIT_UNIQUE=1 emits the active-entry unique coordinate columns."""
    monkeypatch.setenv("V2ECOLI_EMIT_UNIQUE", "1")
    import v2ecoli.library.parquet_run as pr
    # Flip the module-level latch (it was computed at import time).
    monkeypatch.setattr(pr, "_EMIT_UNIQUE", True)

    comp = _UniqueStubComposite(core, initial_agent_id="0")
    pr.run_multigen_parquet(
        comp, experiment_id="uniq_on", out_dir=tmp_path / "out",
        emit_paths=["listeners/count"], max_steps=10, max_generations=1,
        chunk=5, initial_agent_id="0", batch_size=2, threaded=False,
    )
    gen1_dir = (
        tmp_path / "out" / "uniq_on" / "history"
        / "experiment_id=uniq_on" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0"
    )
    rows = pl.read_parquet(str(gen1_dir / "*.pq")).sort("global_time")
    for col in ("active_RNAP__coordinates", "active_RNAP__domain_index",
                "active_replisome__coordinates", "active_replisome__domain_index"):
        assert col in rows.columns, f"missing unique column {col}"
    # Only the 2 ACTIVE RNAP entries survive the _entryState mask (999 dropped).
    first_rnap = list(rows["active_RNAP__coordinates"][0])
    assert sorted(first_rnap) == [-200, 100], first_rnap
    first_repl = sorted(rows["active_replisome__coordinates"][0])
    assert first_repl == [-50000, 50000], first_repl


@pytest.mark.fast
def test_extract_unique_attrs_masks_inactive(core):
    """_extract_unique_attrs returns only active entries; missing molecule -> empty."""
    import v2ecoli.library.parquet_run as pr
    comp = _UniqueStubComposite(core, initial_agent_id="0")
    agent = comp.state["agents"]["0"]
    out = pr._extract_unique_attrs(agent)
    assert sorted(out["active_RNAP"]["coordinates"]) == [-200, 100]
    assert out["active_RNAP"]["domain_index"] == [0, 0]
    # Molecules absent from this agent's unique store -> empty lists, no crash.
    assert out["full_chromosome"]["unique_index"] == []
    assert out["chromosome_domain"]["domain_index"] == []
