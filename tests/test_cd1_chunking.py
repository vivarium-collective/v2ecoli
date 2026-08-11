"""Regression tests for item 38's cd1_* query-restructuring fix.

The real-AWS re-test (item 38) showed the ``cd1_*`` multiseed analyses
OOM-kill the analysis DAG node: each of ``cd1_proteomics``/``cd1_fluxomics``/
``cd1_metabolomics``/``cd1_transcriptomics`` explodes every cell's full
per-timepoint entity-count list into individual rows (``unnest`` +
``generate_subscripts``) BEFORE its ``GROUP BY`` collapses them back down —
materializing the whole sweep's arrays in memory at once. The fix
(``_helpers.run_chunked``/``distinct_cell_filters``) restructures each
module's query to process one cell at a time and concatenate results,
bounding peak memory to a single cell's arrays. Every module's ``GROUP BY``
already partitions by the same cell-identity columns (never combines rows
across cells), so this is provably lossless — these tests confirm it
numerically, not just structurally.

Nothing about the aggregation math, filters, unit conversions, or biological
assumptions changes — only how the query executes.
"""

from __future__ import annotations

import types

import duckdb
import numpy as np
import polars as pl
import pytest
from bigraph_schema import allocate_core


# ---------------------------------------------------------------------------
# Generic chunking-primitive tests (_helpers.distinct_cell_filters/run_chunked)
# ---------------------------------------------------------------------------


def _synthetic_conn():
    conn = duckdb.connect()
    df = pl.DataFrame(
        {
            "cell": ["a", "a", "b", "b", "c", "c"],
            "value": [1.0, 3.0, 10.0, 20.0, 100.0, 300.0],
        }
    )
    conn.register("t", df)
    return conn


def test_distinct_cell_filters_chunk_size_one_yields_one_fragment_per_cell():
    from v2ecoli.workflow.analyses._helpers import distinct_cell_filters

    conn = _synthetic_conn()
    fragments = distinct_cell_filters(conn, "SELECT * FROM t", id_cols=["cell"], chunk_size=1)
    assert len(fragments) == 3
    assert all("(cell) IN (" in f for f in fragments)


def test_distinct_cell_filters_batches_by_chunk_size():
    from v2ecoli.workflow.analyses._helpers import distinct_cell_filters

    conn = _synthetic_conn()
    fragments = distinct_cell_filters(conn, "SELECT * FROM t", id_cols=["cell"], chunk_size=2)
    assert len(fragments) == 2  # 3 cells, batches of 2 -> [2, 1]


def test_run_chunked_matches_unchunked_equivalent():
    """The chunked path must produce the exact same AVG/GROUP BY result as a
    single unchunked query over the whole table — the core correctness claim
    every cd1_* module's fix depends on."""
    from v2ecoli.workflow.analyses._helpers import run_chunked

    conn = _synthetic_conn()

    unchunked = conn.sql(
        "SELECT cell, AVG(value) AS avg_value FROM t GROUP BY cell ORDER BY cell"
    ).pl()

    def _batch_sql(cell_filter: str) -> str:
        return f"""
            SELECT cell, AVG(value) AS avg_value FROM t WHERE {cell_filter}
            GROUP BY cell
            """

    chunked = run_chunked(conn, "SELECT * FROM t", _batch_sql, id_cols=["cell"], chunk_size=1)
    chunked = chunked.sort("cell")
    assert chunked.to_dicts() == unchunked.to_dicts()


def test_run_chunked_empty_input_yields_empty_dataframe():
    from v2ecoli.workflow.analyses._helpers import run_chunked

    conn = duckdb.connect()
    conn.register("empty_t", pl.DataFrame({"cell": [], "value": []}))
    result = run_chunked(
        conn, "SELECT * FROM empty_t", lambda f: "SELECT 1", id_cols=["cell"]
    )
    assert result.is_empty()


# ---------------------------------------------------------------------------
# End-to-end parity tests: each exploding cd1_* module against hand-computed
# expected values, run through the real Analysis Step (registration + config
# + analyze()), exactly as analysis_runner.py invokes them in production.
# ---------------------------------------------------------------------------


def _history_conn(columns: dict) -> tuple[duckdb.DuckDBPyConnection, str]:
    conn = duckdb.connect()
    df = pl.DataFrame(columns)
    conn.register("history_tbl", df)
    return conn, "SELECT * FROM history_tbl"


_BASE_ID_COLS = {
    "experiment_id": ["e"] * 4,
    "variant": [0] * 4,
    "lineage_seed": [0, 0, 1, 1],
    "generation": [0, 0, 0, 0],
    "agent_id": ["0"] * 4,
    "global_time": [0.0, 1.0, 0.0, 1.0],
}


def _run_step(step_cls, sim_data, conn, history_sql, config=None):
    step = step_cls(config or {}, core=allocate_core())
    return step.update(
        {
            "conn": conn,
            "history_sql": history_sql,
            "config_sql": "",
            "success_sql": "",
            "sim_data": sim_data,
            "validation_data": None,
            "variant_metadata": {},
        }
    )


def test_cd1_proteomics_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_proteomics import Cd1Proteomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "listeners__monomer_counts": [
                [10.0, 20.0], [20.0, 30.0], [100.0, 200.0], [200.0, 300.0],
            ],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            translation=types.SimpleNamespace(monomer_data={"id": ["M1[c]", "M2[c]"]})
        )
    )
    out = _run_step(Cd1Proteomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    assert rows["M1"][3:5] == ["15.0", "150.0"]  # Cell: 0_0, Cell: 1_0
    assert rows["M2"][3:5] == ["25.0", "250.0"]


def test_cd1_fluxomics_chunked_matches_hand_computed_unit_converted_means():
    from wholecell.utils import units

    from v2ecoli.workflow.analyses.cd1_fluxomics import Cd1Fluxomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "listeners__fba_results__base_reaction_fluxes": [
                [1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0],
            ],
            "listeners__mass__cell_mass": [20.0] * 4,
            "listeners__mass__dry_mass": [10.0] * 4,
        }
    )
    cell_density = 1100.0
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            metabolism=types.SimpleNamespace(base_reaction_ids=["R1", "R2"])
        ),
        constants=types.SimpleNamespace(cell_density=cell_density * units.g / units.L),
    )
    out = _run_step(Cd1Fluxomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}

    def expected(fluxes, dry, cell_mass):
        coeff = dry / cell_mass * cell_density
        avg = sum(f / coeff for f in fluxes) / len(fluxes)
        return avg * 3600.0  # mmol/g/s -> mmol/g/h, matches the module's own unit conversion

    assert float(rows["R1"][3]) == pytest.approx(expected([1.0, 3.0], 10.0, 20.0))
    assert float(rows["R1"][4]) == pytest.approx(expected([10.0, 30.0], 10.0, 20.0))
    assert float(rows["R2"][3]) == pytest.approx(expected([2.0, 4.0], 10.0, 20.0))
    assert float(rows["R2"][4]) == pytest.approx(expected([20.0, 40.0], 10.0, 20.0))


def test_cd1_metabolomics_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_metabolomics import Cd1Metabolomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "bulk__id": [["A[c]", "B[c]", "C[c]"]] * 4,
            "bulk__count": [[10, 20, 30], [20, 30, 40], [100, 200, 300], [200, 300, 400]],
        }
    )
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            metabolism=types.SimpleNamespace(conc_dict={"A[c]": 1, "C[c]": 1})
        )
    )
    out = _run_step(Cd1Metabolomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    assert rows["A"][3:5] == ["15.0", "150.0"]
    assert rows["C"][3:5] == ["35.0", "350.0"]


def test_cd1_transcriptomics_chunked_matches_hand_computed_means():
    from v2ecoli.workflow.analyses.cd1_transcriptomics import Cd1Transcriptomics

    conn, history_sql = _history_conn(
        {
            **_BASE_ID_COLS,
            "listeners__rna_counts__mRNA_cistron_counts": [
                [5.0, 6.0], [7.0, 8.0], [50.0, 60.0], [70.0, 80.0],
            ],
        }
    )
    cistron_data = {
        "id": np.array(["G1_RNA", "G2_RNA"]),
        "is_mRNA": np.array([True, True]),
    }
    sim_data = types.SimpleNamespace(
        process=types.SimpleNamespace(
            transcription=types.SimpleNamespace(cistron_data=cistron_data)
        )
    )
    out = _run_step(Cd1Transcriptomics, sim_data, conn, history_sql)
    rows = {r[0]: r for r in [line.split("\t") for line in out["data"]["tsv"].splitlines()[1:]]}
    assert rows["G1"][3:5] == ["6.0", "60.0"]
    assert rows["G2"][3:5] == ["7.0", "70.0"]
