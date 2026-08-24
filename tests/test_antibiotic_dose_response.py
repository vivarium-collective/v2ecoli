"""Tests for the antibiotic_dose_response multivariant analysis.

Asserts (1) registration + multivariant scale, (2) the read-path runs over a
small synthetic 2-dose DuckDB history spanning target-engagement (bulk), growth,
and shape columns and yields a non-empty chart with all three readout groups,
and (3) graceful degradation — when only the target-engagement bulk columns are
present (shape not declared/emitted), it still renders and simply omits the
absent readout groups.
"""

from __future__ import annotations

import duckdb
import pytest
from bigraph_schema import allocate_core

_FULL_COLS = [
    "bulk__mecillinam[p]",
    "bulk__EG10606-MONOMER[i]",
    "bulk__mecillinam[p]-EG10606-MONOMER[i]",
    "bulk__mecillinam_hydrolyzed[p]",
    "bulk__pterin_sulfadiazine[c]",
    "listeners__mass__cell_mass",
    "listeners__mass__instantaneous_growth_rate",
    "listeners__peptidoglycan_shape__lysed",
    "listeners__peptidoglycan_shape__resting_radius",
    "listeners__peptidoglycan_shape__resting_length",
]


def _history(cols):
    """A tiny 2-dose x 2-timepoint history exposing the requested readout cols
    (plus the variant/time keys). Dose 1 has higher uptake + more sequestration."""
    conn = duckdb.connect()
    key_cols = ["variant", "lineage_seed", "generation", "agent_id", "global_time"]
    key_defs = ('variant INT, lineage_seed INT, generation INT, agent_id VARCHAR, '
                'global_time DOUBLE')
    metric_defs = ", ".join(f'"{c}" DOUBLE' for c in cols)
    conn.sql(f"CREATE TABLE hist ({key_defs}, {metric_defs})")
    # value per metric: dose 0 = low/none, dose 1 = high (monotonic response)
    def row(variant, t):
        base = {c: (0.0 if variant == 0 else 100.0 * (i + 1))
                for i, c in enumerate(cols)}
        keys = [variant, 0, 1, "'0'", t]
        vals = [str(base[c]) for c in cols]
        return "(" + ", ".join(str(k) for k in keys) + ", " + ", ".join(vals) + ")"
    rows = ", ".join(row(v, t) for v in (0, 1) for t in (60.0, 120.0))
    collist = ", ".join(key_cols + [f'"{c}"' for c in cols])
    conn.sql(f"INSERT INTO hist ({collist}) VALUES {rows}")
    return conn, "SELECT * FROM hist"


def test_registered_multivariant():
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis

    cls = ANALYSIS_REGISTRY["antibiotic_dose_response"]
    assert issubclass(cls, Analysis)
    assert cls.scale == "multivariant"


def _run(conn, history_sql):
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    step = ANALYSIS_REGISTRY["antibiotic_dose_response"]({}, core=allocate_core())
    return step.update({
        "conn": conn, "history_sql": history_sql,
        "config_sql": "", "success_sql": "",
        "sim_data": None, "validation_data": None,
        "variant_metadata": {0: "0 mM", 1: "high"},
    })


def test_full_readouts_render_all_three_groups():
    conn, history_sql = _history(_FULL_COLS)
    out = _run(conn, history_sql)
    view = out.get("view") or ""
    assert view and ("vega" in view or "<svg" in view)
    assert set(out["data"]["readout_groups"]) == {
        "Target engagement", "Growth & viability", "Cell shape"}
    assert out["data"]["n_variants"] == 2


def test_graceful_degradation_bulk_only():
    # only target-engagement bulk columns present (no mass, no shape)
    bulk_only = [c for c in _FULL_COLS if c.startswith("bulk__")]
    conn, history_sql = _history(bulk_only)
    out = _run(conn, history_sql)
    view = out.get("view") or ""
    assert view and ("vega" in view or "<svg" in view)
    assert out["data"]["readout_groups"] == ["Target engagement"]


def test_errors_when_no_readout_columns():
    conn, history_sql = _history([])  # keys only, no readouts
    with pytest.raises(ValueError, match="none of the antibiotic readout columns"):
        _run(conn, history_sql)
