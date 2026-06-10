"""Tests for Analysis subclasses that return a rendered HTML view."""

from bigraph_schema import allocate_core  # noqa: F401


def test_mass_fraction_voronoi_registered_single_view():
    from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["mass_fraction_voronoi"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


def test_ccm_scatter_registered_multiseed():
    from v2ecoli.workflow.analyses import central_carbon_metabolism_scatter  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["central_carbon_metabolism_scatter"]
    assert issubclass(cls, Analysis) and cls.scale == "multiseed"


def _gen_bounds(conn, abs_sql):
    return {
        int(g): (float(lo), float(hi))
        for g, lo, hi in conn.sql(
            f"SELECT generation, min(global_time), max(global_time) "
            f"FROM ({abs_sql}) GROUP BY generation ORDER BY generation"
        ).fetchall()
    }


def test_cumulative_time_leaves_absolute_clock_untouched():
    """v2ecoli's parquet emitter already carries an absolute clock across
    generations; cumulative_time_history must NOT re-offset it (the old
    version double-offset, inserting a spurious time gap and stretching the
    lineage axis)."""
    import duckdb
    from v2ecoli.workflow.analyses._helpers import cumulative_time_history

    conn = duckdb.connect()
    # gen 1: 0..100, gen 2: 110..200 — already absolute, gen2 starts after gen1.
    raw = (
        "SELECT 1 AS generation, t AS global_time FROM (SELECT unnest([0,50,100]) t) "
        "UNION ALL "
        "SELECT 2, t FROM (SELECT unnest([110,150,200]) t)"
    )
    bounds = _gen_bounds(conn, cumulative_time_history(raw))
    assert bounds == {1: (0.0, 100.0), 2: (110.0, 200.0)}


def test_cumulative_time_stacks_per_generation_reset_clock():
    """When global_time resets to 0 each generation (xarray path), the
    generations must be stacked end-to-end, correctly for >2 generations."""
    import duckdb
    from v2ecoli.workflow.analyses._helpers import cumulative_time_history

    conn = duckdb.connect()
    raw = (
        "SELECT 1 AS generation, t AS global_time FROM (SELECT unnest([0,100,200]) t) "
        "UNION ALL SELECT 2, t FROM (SELECT unnest([0,100,200,300]) t) "
        "UNION ALL SELECT 3, t FROM (SELECT unnest([0,100]) t)"
    )
    bounds = _gen_bounds(conn, cumulative_time_history(raw))
    # gen1 0..200; gen2 starts at 201, ends 501; gen3 starts 502, ends 602.
    assert bounds == {1: (0.0, 200.0), 2: (201.0, 501.0), 3: (502.0, 602.0)}
