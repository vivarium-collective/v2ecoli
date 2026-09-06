"""Acceptance gate: turn silent-success into loud failure for a run's output.

A run can exit 0, divide, and write a valid-looking store while producing NONE of
the observables the study exists to measure. A declared emit path that yields no
column is silent (sms-ecoli#210: "the check is the column list, not the exit
code"); a swapped process can be dropped between API and runner and the run still
completes green on the wild-type; a KPI column can be present and read exactly
zero. Presence/effect checks (a non-empty store, global_time advanced) pass all of
these.

This gate asserts, FROM THE ARTIFACT, two things a study's output must satisfy:

  1. Required columns exist in the hive-partitioned history parquet, each with at
     least one non-null row. The required list is the study's OUTPUT CONTRACT and
     belongs in the versioned run config, not in code.
  2. (optional) The composite that actually ran contained the processes the
     submission declared -- closes the wild-type / dropped-injected_processes
     class without first winning the root-cause hunt for where they drop.

It is route-independent: importable from run_pbg, a raw compute job, or a
post-run check, and it reads local paths or s3:// the same way (via sweep_io).
It writes a verdict.json so status can be derived from the verdict rather than a
container exit code (which has been wrong in both directions, sms-ecoli#210).

Exit codes: 0 = PASS, 1 = FAIL, 2 = the gate could not run (bad inputs). Kept
distinct from check_run_complete.py's exit-2-for-expected-partial convention so a
wrapper never confuses "gate could not run" with "run incomplete".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from v2ecoli.library.sweep_io import configure_duckdb_s3, history_files, is_s3_uri


def check_columns(sweep_dir: str, required_columns: Iterable[str]) -> dict[str, Any]:
    """Assert each required column exists in the sweep's history parquet with at
    least one non-null row.

    Column names are the double-underscore-flattened emit paths the parquet
    carries (e.g. ``listeners__mass__dry_mass``), NOT slash paths. Returns a dict
    with ``passed`` and a per-column ``{present, nonnull_rows}`` map. A missing
    column and a present-but-all-null column both fail -- both are the silent
    failure this gate exists to catch.
    """
    import duckdb

    required = list(required_columns)
    files = history_files(sweep_dir)
    if not files:
        return {
            "passed": False,
            "error": f"no hive history parquet under {sweep_dir!r}",
            "columns": {c: {"present": False, "nonnull_rows": 0} for c in required},
            "n_files": 0,
        }
    con = duckdb.connect()
    if is_s3_uri(sweep_dir):
        configure_duckdb_s3(con)
    flist = "[" + ",".join(_sql_str(f) for f in files) + "]"
    src = f"read_parquet({flist}, hive_partitioning=true)"
    available = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}

    columns: dict[str, Any] = {}
    for col in required:
        if col not in available:
            columns[col] = {"present": False, "nonnull_rows": 0}
            continue
        n = con.execute(f"SELECT count({_sql_ident(col)}) FROM {src}").fetchone()[0]
        columns[col] = {"present": True, "nonnull_rows": int(n)}
    passed = bool(required) and all(
        c["present"] and c["nonnull_rows"] > 0 for c in columns.values()
    )
    return {"passed": passed, "columns": columns, "n_files": len(files)}


def process_names_from_state(state: dict) -> set[str]:
    """The process ADDRESSES mounted in a built composite state (agents included).

    Walks the state tree for nodes that look like a process/step instance
    (``_type`` in {process, step} with an ``address``), and returns the set of
    addresses. Feed this to :func:`check_composition` as the "what actually ran"
    side. Address, not store key, because the store key is a wiring name
    (``ecoli-metabolism``) while the address is what was mounted
    (``local:MetabolismReduxClassic``) -- a swap changes the address.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            t = node.get("_type")
            addr = node.get("address")
            if isinstance(addr, str) and t in ("process", "step"):
                found.add(addr)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(state)
    return found


def check_composition(
    ran: Iterable[str], declared: Iterable[str]
) -> dict[str, Any]:
    """Assert every declared process is present in what actually ran.

    ``ran``: the process addresses mounted in the composite (e.g. from
    :func:`process_names_from_state`). ``declared``: the EXPECTED mounted
    class/address for each process the run must contain -- e.g. for a
    ``swap_processes={ecoli-metabolism: ecoli-metabolism-redux}`` submission the
    study declares ``MetabolismReduxClassic`` (the class the swap mounts), because
    the registry name ``ecoli-metabolism-redux`` does not textually appear in the
    mounted address ``local:MetabolismReduxClassic``. The study's config owns that
    mapping, the same way it owns the required-column list. Case-insensitive
    substring match, so ``MetabolismReduxClassic`` matches
    ``local:MetabolismReduxClassic``; the failure this catches is a declared
    class ENTIRELY ABSENT (the wild-type run that completed green).
    """
    ran_low = [r.lower() for r in ran]
    declared_list = list(declared)
    missing = [
        d for d in declared_list
        if not any(d.lower() in r or r in d.lower() for r in ran_low)
    ]
    return {"passed": not missing, "missing": missing, "declared": declared_list}


def run_gate(
    sweep_dir: str,
    required_columns: Iterable[str],
    *,
    ran_processes: Iterable[str] | None = None,
    declared_processes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run the column check and (if a composition is given) the composition check,
    and return a single verdict dict. ``passed`` is the AND of both."""
    verdict: dict[str, Any] = {
        "passed": False,
        "sweep_dir": sweep_dir,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "columns_check": check_columns(sweep_dir, required_columns),
    }
    passed = verdict["columns_check"]["passed"]
    if declared_processes is not None:
        verdict["composition_check"] = check_composition(
            ran_processes or [], declared_processes
        )
        passed = passed and verdict["composition_check"]["passed"]
    verdict["passed"] = passed
    return verdict


def write_verdict(verdict: dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(verdict, indent=2))


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _self_test() -> int:
    """Validate the verifier itself against a known-bad artifact (sms-ecoli#210:
    "a positive control only protects the checks you actually run against it").
    Builds a tiny hive parquet, then asserts the gate PASSES a present column and
    FAILS a missing one and an all-null one. Returns 0 iff the gate behaves."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    ok = True
    with tempfile.TemporaryDirectory() as d:
        # Minimal hive tree history_files() will find:
        # <d>/history/experiment_id=t/variant=0/lineage_seed=0/generation=1/agent_id=1/0.pq
        leaf = (
            Path(d) / "history" / "experiment_id=t" / "variant=0"
            / "lineage_seed=0" / "generation=1" / "agent_id=1"
        )
        leaf.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "global_time": [0.0, 1.0],
                "listeners__mass__dry_mass": [430.0, 431.0],
                "all_null_col": pa.array([None, None], type=pa.float64()),
            }),
            str(leaf / "0.pq"),
        )

        present = check_columns(d, ["global_time", "listeners__mass__dry_mass"])
        if not present["passed"]:
            print(f"SELF-TEST FAIL: present columns should pass: {present}")
            ok = False

        missing = check_columns(d, ["listeners__mass__dry_mass", "not_a_column"])
        if missing["passed"] or missing["columns"]["not_a_column"]["present"]:
            print(f"SELF-TEST FAIL: a missing column must fail: {missing}")
            ok = False

        allnull = check_columns(d, ["all_null_col"])
        if allnull["passed"]:
            print(f"SELF-TEST FAIL: an all-null column must fail: {allnull}")
            ok = False

    print("SELF-TEST PASS: gate flags missing and all-null columns" if ok
          else "SELF-TEST FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", help="Run output dir (local or s3://), hive history under it.")
    p.add_argument("--required-columns", help="JSON list of required column names, or @file.")
    p.add_argument("--declared-processes", default=None,
                   help="JSON list of process names/addresses the submission declared (optional).")
    p.add_argument("--process-table", default=None,
                   help="Path to a JSON composite state / final_state.json; the mounted "
                        "process addresses are extracted for the composition check.")
    p.add_argument("--out", default=None, help="Write verdict.json here.")
    p.add_argument("--self-test", action="store_true",
                   help="Validate the gate against a known-bad artifact and exit.")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.sweep_dir or not args.required_columns:
        p.error("--sweep-dir and --required-columns are required (unless --self-test)")

    spec = args.required_columns
    required = json.loads(Path(spec[1:]).read_text() if spec.startswith("@") else spec)

    ran = None
    declared = None
    if args.declared_processes is not None:
        declared = json.loads(args.declared_processes)
        ran = []
        if args.process_table:
            state = json.loads(Path(args.process_table).read_text())
            ran = sorted(process_names_from_state(state))

    verdict = run_gate(args.sweep_dir, required,
                       ran_processes=ran, declared_processes=declared)
    print(json.dumps(verdict, indent=2))
    if args.out:
        write_verdict(verdict, args.out)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
