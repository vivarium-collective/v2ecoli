"""Acceptance gate: content checks that turn silent-success into loud failure.

Covers the failure classes from sms-ecoli#210: a required column missing, present
but all-null, no hive parquet at all, and a declared process absent from what ran.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2ecoli.workflow.acceptance_gate import (
    check_columns,
    check_composition,
    process_names_from_state,
    run_gate,
    _self_test,
)

pytestmark = pytest.mark.fast


def _write_hive(root: Path, table: pa.Table) -> None:
    leaf = (
        root / "history" / "experiment_id=t" / "variant=0"
        / "lineage_seed=0" / "generation=1" / "agent_id=1"
    )
    leaf.mkdir(parents=True)
    pq.write_table(table, str(leaf / "0.pq"))


@pytest.fixture
def sweep(tmp_path):
    _write_hive(tmp_path, pa.table({
        "global_time": [0.0, 1.0, 2.0],
        "listeners__mass__dry_mass": [430.0, 431.0, 432.0],
        "environment__exchange__VIOLACEIN": [0.0, 0.01, 0.02],
        "all_null_col": pa.array([None, None, None], type=pa.float64()),
    }))
    return str(tmp_path)


def test_present_columns_pass(sweep):
    r = check_columns(sweep, ["global_time", "listeners__mass__dry_mass"])
    assert r["passed"]
    assert r["columns"]["listeners__mass__dry_mass"]["nonnull_rows"] == 3


def test_missing_column_fails_and_is_named(sweep):
    r = check_columns(sweep, ["listeners__mass__dry_mass", "listeners__mass__cell_mass"])
    assert not r["passed"]
    assert r["columns"]["listeners__mass__cell_mass"] == {"present": False, "nonnull_rows": 0}


def test_all_null_column_fails(sweep):
    """Present but all-null is the silent case: the column exists, reads nothing."""
    r = check_columns(sweep, ["all_null_col"])
    assert not r["passed"]
    assert r["columns"]["all_null_col"]["present"] is True
    assert r["columns"]["all_null_col"]["nonnull_rows"] == 0


def test_no_hive_parquet_fails(tmp_path):
    r = check_columns(str(tmp_path), ["global_time"])
    assert not r["passed"]
    assert "no hive history parquet" in r["error"]


def test_a_stray_flat_parquet_is_not_counted(tmp_path):
    """history_files scopes to experiment_id=* hive trees; a flat default/history
    file must not make the gate think output landed."""
    flat = tmp_path / "default" / "history"
    flat.mkdir(parents=True)
    pq.write_table(pa.table({"global_time": [0.0]}), str(flat / "1.pq"))
    r = check_columns(str(tmp_path), ["global_time"])
    assert not r["passed"] and r["n_files"] == 0


def test_process_names_from_state_extracts_addresses():
    state = {
        "agents": {"0": {
            "ecoli-metabolism": {"_type": "process", "address": "local:MetabolismReduxClassic"},
            "division": {"_type": "step", "address": "local:Division"},
            "bulk": {"count": [1, 2, 3]},  # not a process
        }},
        "global_time": 0.0,
    }
    names = process_names_from_state(state)
    assert names == {"local:MetabolismReduxClassic", "local:Division"}


def test_composition_passes_when_declared_class_present():
    """Declared is the EXPECTED mounted class; matches the mounted address."""
    ran = ["local:MetabolismReduxClassic", "local:Division"]
    r = check_composition(ran, ["MetabolismReduxClassic"])
    assert r["passed"] and r["missing"] == []


def test_composition_fails_on_wild_type_run():
    """The declared swap class is entirely absent -> the run was wild-type."""
    ran = ["local:MetabolismFBA", "local:Division"]
    r = check_composition(ran, ["MetabolismReduxClassic"])
    assert not r["passed"] and r["missing"] == ["MetabolismReduxClassic"]


def test_run_gate_ands_both_checks(sweep):
    ran = ["local:MetabolismReduxClassic"]
    v = run_gate(sweep, ["listeners__mass__dry_mass"],
                 ran_processes=ran, declared_processes=["MetabolismReduxClassic"])
    assert v["passed"]
    # a wrong composition sinks an otherwise-good column check
    v2 = run_gate(sweep, ["listeners__mass__dry_mass"],
                  ran_processes=["local:MetabolismFBA"], declared_processes=["MetabolismReduxClassic"])
    assert not v2["passed"] and v2["columns_check"]["passed"]


def test_self_test_validates_the_verifier():
    assert _self_test() == 0
