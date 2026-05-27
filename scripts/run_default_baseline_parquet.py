"""Run a v2ecoli composite once with ParquetEmitter for the history backend.

Sibling to ``scripts/run_default_baseline.py`` (which uses SQLiteEmitter).
Wraps in ``pbg_runner`` (sqlite run-registry, for dashboard run tracking) +
``parquet_emitter`` (history target, hive-partitioned column store under
``.pbg/default-baseline/parquet/``).

Use this as the canonical wiring example. ``tests/test_parquet_run_smoke.py``
exercises the same machinery against a stub composite (54 emitter tests
green); this script lets you point it at any real composite recipe.

**Real-composite status (2026-05-26).** The emitter is unit-test-green
and smoke-test-green, but real cell composites that emit multi-dimensional
ndarrays into the Polars fallback path hit a polars "cannot parse
dtype('O')" wall. The fix path is per-field ``dtype_overrides`` (mirroring
how vEcoli locks listener types via USE_UINT16/USE_UINT32) — that's a
known iteration step the user must do per composite until the
listener-dtype-override list is exhaustive. Tracking as a follow-up.

**Dashboard interop note.** Until the vivarium-dashboard parquet-reader
PR lands, the dashboard's auto-discovery of default-baseline charts
still reads ``runs.db`` (sqlite) — so do NOT remove
``run_default_baseline.py`` yet. After the dashboard PR merges, the
parquet path becomes the source of truth and this script can be
promoted to the primary runner.

Usage:
    python scripts/run_default_baseline_parquet.py [--composite baseline] \\
        [--steps 5] [--interval-s 60] [--force]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli import build_composite
from v2ecoli.composites._helpers import parquet_emitter
from v2ecoli.library.parquet_emitter import ParquetEmitter
from pbg_superpowers.runner import pbg_runner


def _parquet_dir_has_runs(parquet_root: Path) -> bool:
    """Look for any *.pq under the experiment's history dir."""
    if not parquet_root.is_dir():
        return False
    history = parquet_root.glob("*/history/**/*.pq")
    return any(True for _ in history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--composite", default="baseline",
                   help="Composite recipe name to build (default: baseline).")
    p.add_argument("--steps", type=int, default=5,
                   help="Number of interval_s-sized chunks to run (default: 5).")
    p.add_argument("--interval-s", type=int, default=60,
                   help="Composite tick chunk size in sim-seconds (default: 60).")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if a complete parquet baseline already exists.")
    p.add_argument("--sim-name", default="workspace-default-baseline-parquet",
                   help="Run label.")
    args = p.parse_args()

    recipe_name = args.composite
    interval_s = int(args.interval_s)
    max_duration_s = int(args.steps) * interval_s
    params: dict = {}
    stop_on = "division"

    out_root = REPO_ROOT / ".pbg" / "default-baseline" / "parquet"
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.force and _parquet_dir_has_runs(out_root):
        print(f"Default parquet baseline already present at "
              f"{out_root.relative_to(REPO_ROOT)}. Use --force to re-run.")
        return

    print(f"Running v2ecoli composite (parquet):")
    print(f"  composite:  {recipe_name}")
    print(f"  params:     {params}")
    print(f"  stop on:    {stop_on}  (safety ceiling: {max_duration_s}s)")
    print(f"  interval:   {interval_s}s")
    print(f"  out_dir:    {out_root.relative_to(REPO_ROOT)}")
    print()

    # pbg_runner writes a tiny sqlite run-registry; the dashboard's run-list
    # tab uses that for run tracking independent of the history backend.
    registry_db = REPO_ROOT / ".pbg" / "default-baseline" / "runs.db"
    with pbg_runner(
        study="__workspace_default__",
        name=args.sim_name,
        params={**params, "max_duration_s": max_duration_s,
                "interval_s": interval_s, "stop_on": stop_on,
                "emitter": "parquet"},
        workspace_root=REPO_ROOT,
        db_file=registry_db,
    ) as run:
        with parquet_emitter(
            out_dir=str(out_root),
            experiment_id=run.run_id,
            study_slug="__workspace_default__",
            investigation_slug="__workspace_default__",
            agent_id="0",
            # batch_size matched to interval_s so on a 60-s default interval
            # we get one flush per ~400 minutes simulated. Threaded for I/O
            # parallelism — composite ticks don't block on the writer.
            batch_size=400,
            threaded=True,
        ):
            t0 = time.time()
            composite = build_composite(recipe_name, **params)
            load_time = time.time() - t0

            t_run = time.time()
            total = 0
            divided = False
            stop_reason = "unknown"
            n_snapshots = 0
            while total < max_duration_s:
                chunk = min(interval_s, max_duration_s - total)
                try:
                    composite.run(chunk)
                except Exception:
                    divided = True
                    stop_reason = "division (composite raised)"
                    break
                total += chunk
                n_snapshots += 1
                run.heartbeat(n_snapshots)
                agents = (composite.state or {}).get("agents") or {}
                if agents.get("0") is None:
                    divided = True
                    stop_reason = "division (agent removed)"
                    break
            else:
                stop_reason = f"max_duration_s reached ({max_duration_s}s without division)"
            wall_time = time.time() - t_run
            run.n_steps = n_snapshots

            # Composite's ParquetEmitter step holds a partial last batch
            # (rows since the most recent batch_size flush) in memory. The
            # parquet_emitter() context manager only manages the override
            # flag, not the step instance — without an explicit flush the
            # trailing batch never lands on disk.
            n_flushed = ParquetEmitter.flush_all_in_composite(
                composite, success=not divided or stop_reason.startswith("division"),
            )
            print(f"  flushed {n_flushed} ParquetEmitter instance(s) at end-of-run")

            summary = {
                "composite":       recipe_name,
                "params":          params,
                "stop_on":         stop_on,
                "max_duration_s":  max_duration_s,
                "interval_s":      interval_s,
                "sim_time":        total,
                "load_time":       load_time,
                "wall_time":       wall_time,
                "divided":         divided,
                "stop_reason":     stop_reason,
                "run_id":          run.run_id,
                "out_dir":         str(out_root.relative_to(REPO_ROOT)),
                "n_snapshots":     n_snapshots,
                "emitter":         "parquet",
            }
            (out_root / "result.json").write_text(json.dumps(summary, indent=2))

    # Report disk usage for the sanity check.
    total_bytes = sum(p.stat().st_size for p in out_root.rglob("*.pq"))
    print(f"\nWrote {out_root.relative_to(REPO_ROOT)}")
    print(f"  sim_time={total}s  wall={wall_time:.1f}s  divided={divided}")
    print(f"  stop_reason: {stop_reason}")
    print(f"  run_id={run.run_id}")
    print(f"  parquet bytes on disk: {total_bytes:,}")
    print(f"  summary: {out_root.relative_to(REPO_ROOT)}/result.json")


if __name__ == "__main__":
    main()
