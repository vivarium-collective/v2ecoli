"""Backfill per-study ``runs.db`` from the JSON outputs my dnaa-* runners
wrote to ``out/<study>/``.

The dashboard's Simulations DB indexes runs by walking
``.pbg/composite-runs.db`` and every ``studies/<name>/runs.db``. My
hand-rolled runners wrote raw JSON to ``out/`` and never touched those
SQLite files, so the dashboard correctly reports zero simulations even
though three studies have produced real output.

This script reads each ``out/<study>/<run_name>.json`` and inserts an
equivalent ``runs_meta`` row into ``studies/<study>/runs.db``. The
per-step ``history`` table (which SQLiteEmitter normally owns) is left
empty — the dashboard's auto-viz panel from runs.db will still be
missing chart data, but the Simulations DB will at least show that
runs HAPPENED, with their params, durations, and status.

This is a backfill bandaid. The proper fix is to have the runners
write via SQLiteEmitter at run time. Logged in the friction-log for
the meta-agent.

Usage:
    python scripts/backfill_runs_db.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from vivarium_dashboard.lib.composite_runs import connect, generate_run_id


# Mapping from each study slug to the runs we want to backfill.
# Each entry is (json_filename, label, spec_id_hint).
RUNS_TO_BACKFILL = {
    "dnaa-01-expression-dynamics": [
        ("baseline_seed0.json", "baseline seed 0", "baseline"),
        ("baseline_seed1.json", "baseline seed 1", "baseline"),
        ("baseline_seed2.json", "baseline seed 2", "baseline"),
    ],
    "dnaa-02-atp-hydrolysis": [
        ("seed0_default.json",          "seed 0, default rate (deterministic)",
         "dnaa_02_with_intrinsic_hydrolysis"),
        ("seed0_stochastic.json",       "seed 0, default rate (Poisson)",
         "dnaa_02_with_intrinsic_hydrolysis"),
        ("seed0_clamped.json",          "seed 0, clamp [0.2, 0.5] (pre EQ-04)",
         "dnaa_02_with_intrinsic_hydrolysis"),
        ("seed0_intrinsic_only.json",   "seed 0, clamp OFF (post EQ-04 pivot)",
         "dnaa_02_with_intrinsic_hydrolysis"),
        ("seed0_clamped_b.json",        "seed 0, clamp [0.2, 0.5] (post EQ-04)",
         "dnaa_02_with_intrinsic_hydrolysis"),
        ("seed0_no_hydrolysis.json",    "seed 0, rate=0 (no-hydrolysis diagnostic)",
         "dnaa_02_with_intrinsic_hydrolysis"),
        ("seed0_fast_hydrolysis.json",  "seed 0, rate=10x (fast diagnostic)",
         "dnaa_02_with_intrinsic_hydrolysis"),
    ],
    "dnaa-03-box-binding": [
        ("seed0_baseline.json",     "seed 0, 30 min, default DnaA = 115",
         "dnaa_03_with_box_binding"),
        ("seed0_lit_dnaa.json",     "seed 0, 60 min, initial DnaA = 500 (divided)",
         "dnaa_03_with_box_binding"),
        ("seed0_v1_500dnaa.json",   "seed 0, 30 min, initial DnaA = 500 (no division)",
         "dnaa_03_with_box_binding"),
    ],
}


def _read_run(json_path: Path) -> dict | None:
    """Read a runner JSON and extract metadata. Returns None if missing."""
    if not json_path.is_file():
        return None
    try:
        with json_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  SKIP {json_path}: {e}", file=sys.stderr)
        return None

    sim_time = data.get("sim_time") or data.get("duration_requested") or 0
    interval = data.get("interval") or 60
    n_steps = int(sim_time // max(interval, 1))
    snapshots = data.get("snapshots") or []
    if snapshots:
        n_steps = max(n_steps, len(snapshots) - 1)
    return {
        "params":     data.get("params") or {
            "seed": data.get("seed"),
            "rate_per_min": data.get("rate_per_min"),
            "deterministic": data.get("deterministic"),
            "clamp": data.get("clamp"),
        },
        "n_steps":    n_steps,
        "wall_time":  data.get("wall_time"),
        "sim_time":   sim_time,
        "divided":    data.get("divided", False),
    }


def backfill_study(study_slug: str, entries: list[tuple[str, str, str]]) -> int:
    # Output dir convention: "dnaa-01-..." -> "dnaa-01".
    short = "-".join(study_slug.split("-")[:2])
    out_dir = REPO_ROOT / "out" / short

    runs_db = REPO_ROOT / "studies" / study_slug / "runs.db"
    runs_db.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(runs_db)
    inserted = 0
    try:
        for filename, label, spec_id in entries:
            jp = out_dir / filename
            meta = _read_run(jp)
            if meta is None:
                print(f"  - SKIP  {short}/{filename} (not found)")
                continue
            started_at = jp.stat().st_mtime - (meta.get("wall_time") or 0)
            completed_at = jp.stat().st_mtime
            params = meta["params"]
            run_id = generate_run_id(spec_id, params=params, now=started_at)
            # status: "complete" if wall_time present, else "complete"
            status = "complete"
            sim_name = filename.rstrip(".json")
            try:
                conn.execute(
                    "INSERT INTO runs_meta "
                    "(run_id, spec_id, label, params_json, started_at, completed_at, "
                    " n_steps, status, sim_name, progress_step) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, spec_id, label, json.dumps(params, default=str),
                     started_at, completed_at, meta["n_steps"], status,
                     sim_name, meta["n_steps"]),
                )
                inserted += 1
                print(f"  + {short}/{filename}  ->  run_id={run_id[:48]}...  ({meta['n_steps']} steps)")
            except sqlite3.IntegrityError:
                print(f"  = {short}/{filename}  already registered")
        conn.commit()
    finally:
        conn.close()
    return inserted


def main() -> None:
    print(f"Backfilling runs.db for {len(RUNS_TO_BACKFILL)} studies under "
          f"{REPO_ROOT}/studies/")
    total = 0
    for study_slug, entries in RUNS_TO_BACKFILL.items():
        print(f"\n{study_slug}:")
        total += backfill_study(study_slug, entries)
    print(f"\nDone. Inserted {total} runs_meta rows total.")
    print("Tip: refresh the dashboard's Simulations tab to see them.")


if __name__ == "__main__":
    main()
