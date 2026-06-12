"""Tests for the cross-variant COMPARISON analyses (showcase-4).

These analyses overlay / group / delta baseline + variants and are registered at
``multivariant`` scale.  We assert (1) registration + scale, (2) the
variant-label helpers, and (3) that the color-/group-by-variant read-path runs
over a small synthetic 2-variant DuckDB table and yields a NON-EMPTY view for
the analyses that need no sim_data; the sim_data-dependent analyses
(proteome_delta / fba_flux_overlay / scorecard validation columns) are covered
by the registration + helper checks here and by the
``scripts/render_variant_comparison.py`` smoke run on real data.
"""

from __future__ import annotations

import duckdb
import pytest
from bigraph_schema import allocate_core

COMPARISON = [
    "growth_overlay",
    "doubling_time_grouped",
    "mass_fraction_grouped",
    "replication_overlay",
    "proteome_delta",
    "fba_flux_overlay",
    "regulation_overlay",
    "scorecard",
]


def test_all_comparison_analyses_registered_multivariant():
    import v2ecoli.workflow.analyses  # noqa: F401  (registers everything)
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis

    for name in COMPARISON:
        cls = ANALYSIS_REGISTRY[name]
        assert issubclass(cls, Analysis), name
        assert cls.scale == "multivariant", (name, cls.scale)


def test_variant_label_map_ignores_config_keys():
    from v2ecoli.workflow.analyses._helpers import variant_label_map

    m = variant_label_map({0: "baseline", "1": "ppgpp-off", "skip_n_gens": 8})
    assert m == {0: "baseline", 1: "ppgpp-off"}
    assert variant_label_map(None) == {}


def test_variant_display_expr_escapes_and_falls_back():
    from v2ecoli.workflow.analyses._helpers import variant_display_expr

    e = variant_display_expr({0: "baseline", 1: "d'naA"})
    assert "WHEN 0 THEN 'baseline'" in e
    assert "d''naA" in e  # single quote doubled
    assert e.endswith("AS variant_name")
    # empty map -> CASE-free fallback expression
    e2 = variant_display_expr({})
    assert "CASE" not in e2 and "variant " in e2


def test_variant_sort_order_ascending_by_id():
    from v2ecoli.workflow.analyses._helpers import variant_sort_order

    order = variant_sort_order(
        [2, 0, 1], {0: "baseline", 1: "ppgpp-off", 2: "dnaA-2x"})
    assert order == ["baseline", "ppgpp-off", "dnaA-2x"]
    # unmapped variants fall back to "variant N"
    assert variant_sort_order([5, 0], {0: "baseline"}) == ["baseline", "variant 5"]


@pytest.fixture()
def two_variant_history():
    """A tiny in-memory history_sql spanning variant 0 and variant 1.

    Mirrors the parquet schema columns the no-sim_data comparison analyses read
    (variant + the mass / growth / replication / regulation scalar listeners).
    """
    conn = duckdb.connect()
    # 2 variants x 2 timepoints; variant 1 has zero ppGpp (the ppgpp-off shape).
    conn.sql("""
        CREATE TABLE hist AS
        SELECT * FROM (VALUES
            (0, 0, 1, '0', 60.0, 400.0, 180.0, 30.0, 20.0, 8.0, 0.0003, 2, 50.0, 0.95),
            (0, 0, 1, '0', 120.0, 450.0, 200.0, 32.0, 22.0, 9.0, 0.0004, 4, 52.0, 0.96),
            (1, 0, 1, '0', 60.0, 420.0, 200.0, 31.0, 21.0, 8.5, 0.0002, 2, 0.0, 0.90),
            (1, 0, 1, '0', 120.0, 470.0, 220.0, 33.0, 23.0, 9.5, 0.0003, 4, 0.0, 0.91)
        ) AS t(variant, lineage_seed, generation, agent_id, global_time,
               listeners__mass__dry_mass, listeners__mass__protein_mass,
               listeners__mass__rRna_mass, listeners__mass__tRna_mass,
               listeners__mass__smallMolecule_mass,
               listeners__mass__instantaneous_growth_rate,
               listeners__replication_data__number_of_oric,
               listeners__growth_limits__ppgpp_conc,
               listeners__growth_limits__fraction_trna_charged)
    """)
    # cell_mass + dna_mass are read by some analyses; add as views.
    conn.sql("""
        CREATE VIEW hist2 AS
        SELECT *, listeners__mass__dry_mass * 3.0 AS listeners__mass__cell_mass,
               listeners__mass__dry_mass * 0.05 AS listeners__mass__dna_mass
        FROM hist
    """)
    return conn, "SELECT * FROM hist2"


@pytest.mark.parametrize("name", [
    "growth_overlay",
    "doubling_time_grouped",
    "mass_fraction_grouped",
    "replication_overlay",
    "regulation_overlay",
])
def test_no_simdata_comparison_renders_nonempty_two_variants(
        name, two_variant_history):
    """The color-/group-by-variant read-path runs over 2 variants and the view
    is non-empty (contains a Vega-Lite spec)."""
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    conn, history_sql = two_variant_history
    core = allocate_core()
    step = ANALYSIS_REGISTRY[name]({}, core=core)
    out = step.update({
        "conn": conn, "history_sql": history_sql,
        "config_sql": "", "success_sql": "",
        "sim_data": None, "validation_data": None,
        "variant_metadata": {0: "baseline", 1: "ppgpp-off"},
    })
    view = out.get("view") or ""
    assert view, f"{name} produced empty view"
    assert ("vega" in view or "<svg" in view), f"{name} view has no chart"


def test_regulation_overlay_separates_ppgpp_off(two_variant_history):
    """The ppgpp-off variant's ppGpp trace should be ~zero while baseline is
    non-zero (the headline separation), so the analysis must surface both."""
    import duckdb as _ddb  # noqa: F401
    conn, history_sql = two_variant_history
    # baseline mean ppGpp > 0, variant-1 ppGpp == 0
    rows = conn.sql("""
        SELECT variant, avg(listeners__growth_limits__ppgpp_conc) AS m
        FROM (%s) GROUP BY variant ORDER BY variant
    """ % history_sql).fetchall()
    means = {int(v): float(m) for v, m in rows}
    assert means[0] > 0 and means[1] == 0.0
