"""Regression tests for the ptools_* generation_lower_bound/time_lower_bound
parameterization (v2ecoli#542): porting the burn-in filtering the cd1_*
family already has, without changing ptools_*'s own aggregation.

These are synthetic-DuckDB unit tests of the query construction and
row-admission logic itself — no real sweep parquet or sim_data required
(``test_ptools_analyses.py``'s fidelity tests need both and are skipped
without them). They confirm:

  * with no bound set, output is unchanged from the pre-#542 behaviour
    (``filter_clause=""`` is a no-op, byte-for-byte);
  * a bound actually restricts which rows reach the existing
    ``groupby("time").sum()``/``collapse_cross_seed`` aggregation, not the
    aggregation itself;
  * the ``generation`` column used only for filtering never leaks into the
    returned DataFrame.
"""

from __future__ import annotations

import duckdb
import polars as pl


def _synthetic_history():
    """A tiny synthetic ``history_sql`` table: one row per (generation, time)."""
    conn = duckdb.connect()
    df = pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "generation": [0, 0, 1, 1, 2],
            "global_time": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    conn.register("t", df)
    return conn, "SELECT * FROM t"


# ---------------------------------------------------------------------------
# ptools_rna.read_outputs / build_query (shared shape with ptools_rxns/proteins)
# ---------------------------------------------------------------------------


def test_read_outputs_default_filter_is_a_no_op():
    """No bounds set -> filter_clause="" -> every row survives, as before #542."""
    from v2ecoli.workflow.analyses.ptools_rna import read_outputs

    conn, history_sql = _synthetic_history()
    out = read_outputs(history_sql, conn, columns=["value"])
    assert sorted(out["time"].tolist()) == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert sorted(out["value"].tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert "generation" not in out.columns


def test_read_outputs_generation_lower_bound_restricts_rows_only():
    from v2ecoli.workflow.analyses._helpers import generation_time_filter_clause
    from v2ecoli.workflow.analyses.ptools_rna import read_outputs

    conn, history_sql = _synthetic_history()
    filter_clause = generation_time_filter_clause({"generation_lower_bound": 1})
    out = read_outputs(history_sql, conn, columns=["value"], filter_clause=filter_clause)

    # Only generation >= 1 rows (value 3, 4, 5 at time 20, 30, 40) survive.
    assert sorted(out["time"].tolist()) == [20.0, 30.0, 40.0]
    assert sorted(out["value"].tolist()) == [3.0, 4.0, 5.0]
    assert "generation" not in out.columns


def test_read_outputs_time_lower_bound_restricts_rows_only():
    from v2ecoli.workflow.analyses._helpers import generation_time_filter_clause
    from v2ecoli.workflow.analyses.ptools_rna import read_outputs

    conn, history_sql = _synthetic_history()
    filter_clause = generation_time_filter_clause({"time_lower_bound": 25})
    out = read_outputs(history_sql, conn, columns=["value"], filter_clause=filter_clause)

    assert sorted(out["time"].tolist()) == [30.0, 40.0]
    assert sorted(out["value"].tolist()) == [4.0, 5.0]


def test_ptools_rxns_and_proteins_share_the_same_filtering_shape():
    """ptools_rxns/proteins reuse the identical build_query/read_outputs shape
    (each module defines its own copy, not shared code) — confirm both
    independently honour the same filter, not just ptools_rna."""
    from v2ecoli.workflow.analyses._helpers import generation_time_filter_clause
    from v2ecoli.workflow.analyses import ptools_rxns, ptools_proteins

    filter_clause = generation_time_filter_clause({"generation_lower_bound": 2})

    for module in (ptools_rxns, ptools_proteins):
        conn, history_sql = _synthetic_history()
        out = module.read_outputs(
            history_sql, conn, columns=["value"], filter_clause=filter_clause
        )
        assert out["time"].tolist() == [40.0], module.__name__
        assert out["value"].tolist() == [5.0], module.__name__
        assert "generation" not in out.columns, module.__name__


# ---------------------------------------------------------------------------
# _MultiseedMixin._do_read_outputs — filter applies before collapse_cross_seed
# ---------------------------------------------------------------------------


def test_multiseed_do_read_outputs_filters_before_collapsing_seeds():
    """Two seeds share absolute time values; generation_lower_bound must drop
    the excluded generation from BOTH seeds before collapse_cross_seed sums
    them, not after (an after-the-fact drop would double-count nothing, but a
    filter that missed a seed would silently under/over-collapse)."""
    from v2ecoli.workflow.analyses._helpers import generation_time_filter_clause
    from v2ecoli.workflow.analyses.ptools_multiscale import _MultiseedMixin

    conn = duckdb.connect()
    df = pl.DataFrame(
        {
            "value": [10.0, 10.0, 20.0, 20.0],
            "generation": [0, 1, 0, 1],
            "global_time": [0.0, 10.0, 0.0, 10.0],
        }
    )
    conn.register("t", df)

    filter_clause = generation_time_filter_clause({"generation_lower_bound": 1})
    out = _MultiseedMixin()._do_read_outputs(
        "SELECT * FROM t", conn, columns=["value"], filter_clause=filter_clause
    )

    # Only generation=1 rows survive from each seed (value 10 + value 20 at
    # time=10), collapsed by collapse_cross_seed's scalar-sum path.
    assert out["time"].tolist() == [10.0]
    assert out["value"].tolist() == [30.0]
    assert "generation" not in out.columns


# ---------------------------------------------------------------------------
# config_schema — the new params are declared, matching the cd1_* convention
# ---------------------------------------------------------------------------


def test_ptools_config_schemas_declare_the_new_bounds():
    from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna
    from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns
    from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins

    for cls in (PtoolsRna, PtoolsRxns, PtoolsProteins):
        assert cls.config_schema["generation_lower_bound"] == "integer", cls.__name__
        assert cls.config_schema["time_lower_bound"] == "float", cls.__name__
