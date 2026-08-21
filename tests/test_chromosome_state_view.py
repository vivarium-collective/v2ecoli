"""Test for the chromosome_state_view native Analysis (chromosome-state GIF)."""

from __future__ import annotations

import duckdb


def test_chromosome_state_view_registered_single_scale():
    from v2ecoli.workflow.analyses import chromosome_state_view  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis

    cls = ANALYSIS_REGISTRY["chromosome_state_view"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def _synthetic_history():
    """A tiny in-memory ``history`` table with a few timesteps of replication
    listener data — fork_coordinates (array of bp positions, may be empty)
    and number_of_oric (int)."""
    conn = duckdb.connect()
    conn.sql("""
        CREATE TABLE history AS
        SELECT * FROM (VALUES
            ('exp1', 0, 1, 0, '0', 0.0,   [1000000, -500000], 2),
            ('exp1', 0, 1, 0, '0', 60.0,  [1200000, -700000, 300000, -100000], 4),
            ('exp1', 0, 1, 0, '0', 120.0, CAST([] AS BIGINT[]), 1)
        ) AS t(experiment_id, variant, lineage_seed, generation, agent_id,
               global_time, listeners__replication_data__fork_coordinates,
               listeners__replication_data__number_of_oric)
    """)
    return conn, "SELECT * FROM history"


def test_chromosome_state_view_returns_gif_data_uri():
    from v2ecoli.workflow.analyses.chromosome_state_view import ChromosomeStateView

    conn, history_sql = _synthetic_history()
    out = ChromosomeStateView.__new__(ChromosomeStateView).analyze(
        conn=conn, history_sql=history_sql
    )
    assert isinstance(out, dict)
    view = out.get("view") or ""
    assert "data:image/gif;base64," in view


def test_chromosome_state_view_degrades_on_missing_column():
    """Missing replication columns must never raise — degrade to a note."""
    from v2ecoli.workflow.analyses.chromosome_state_view import ChromosomeStateView

    conn = duckdb.connect()
    conn.sql("""
        CREATE TABLE history AS
        SELECT * FROM (VALUES
            ('exp1', 0, 1, 0, '0', 0.0, 400.0)
        ) AS t(experiment_id, variant, lineage_seed, generation, agent_id,
               global_time, listeners__mass__dry_mass)
    """)
    out = ChromosomeStateView.__new__(ChromosomeStateView).analyze(
        conn=conn, history_sql="SELECT * FROM history"
    )
    view = out.get("view") or ""
    assert "unavailable" in view
    assert "data:image/gif;base64," not in view
