"""Run the workspace default-baseline composite once.

Reads ``workspace.yaml::default_baseline`` for the composite + params +
duration, wraps in ``pbg_runner`` + ``sqlite_emitter`` pointed at
``.pbg/default-baseline/runs.db``, and writes a JSON summary alongside.

The dashboard's viz pipeline auto-discovers this db when a study has no
runs of its own yet — so newly-created studies show populated charts
out of the box, against the workspace baseline.

Usage:
    python scripts/run_default_baseline.py [--force]

``--force`` re-runs even if .pbg/default-baseline/runs.db already exists
with a non-zero history. Default: skip if a complete run is present.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import yaml

from v2ecoli import build_composite
from v2ecoli.composites._helpers import sqlite_emitter
from pbg_superpowers.runner import pbg_runner


def _existing_run_count(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "runs_meta" not in tables:
            return 0
        return conn.execute(
            "SELECT COUNT(*) FROM runs_meta WHERE status='complete'"
        ).fetchone()[0]
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-run even if a complete default baseline already exists.")
    p.add_argument("--sim-name", default="workspace-default-baseline",
                   help="Run label (default: workspace-default-baseline).")
    args = p.parse_args()

    ws_yaml = yaml.safe_load((REPO_ROOT / "workspace.yaml").read_text())
    cfg = ws_yaml.get("default_baseline")
    if not cfg:
        sys.exit("workspace.yaml has no `default_baseline:` block — "
                 "set one before running this script.")

    composite_path = cfg["composite"]
    params         = dict(cfg.get("params") or {})
    duration_s     = int(cfg.get("duration_s", 1800))
    interval_s     = int(cfg.get("interval_s", 60))

    # Composite path is `<package>.<module>.<func>`; build_composite takes
    # the shortcut name; map back from the dotted path to the recipe name.
    recipe_name = composite_path.rsplit(".", 1)[-1]

    db_dir = REPO_ROOT / ".pbg" / "default-baseline"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "runs.db"

    if not args.force and _existing_run_count(db_path) > 0:
        print(f"Default baseline already present at {db_path.relative_to(REPO_ROOT)} "
              f"({_existing_run_count(db_path)} complete runs). Use --force to re-run.")
        return

    print(f"Running workspace default baseline:")
    print(f"  composite: {composite_path}  (recipe={recipe_name})")
    print(f"  params:    {params}")
    print(f"  duration:  {duration_s}s @ {interval_s}s intervals")
    print(f"  db_path:   {db_path.relative_to(REPO_ROOT)}")
    print()

    with pbg_runner(
        study="__workspace_default__",
        name=args.sim_name,
        params={**params, "duration_s": duration_s, "interval_s": interval_s},
        workspace_root=REPO_ROOT,
        db_file=db_path,
    ) as run:
        with sqlite_emitter(
            file_path=str(db_path.parent),
            db_file=db_path.name,
            simulation_id=run.run_id,
            name=args.sim_name,
            study_slug="__workspace_default__",
            investigation_slug="__workspace_default__",
        ):
            t0 = time.time()
            composite = build_composite(recipe_name, **params)
            load_time = time.time() - t0

            t_run = time.time()
            total = 0
            divided = False
            n_snapshots = 0
            while total < duration_s:
                chunk = min(interval_s, duration_s - total)
                try:
                    composite.run(chunk)
                except Exception:
                    divided = True
                    break
                total += chunk
                n_snapshots += 1
                run.heartbeat(n_snapshots)
            wall_time = time.time() - t_run
            run.n_steps = n_snapshots

            summary = {
                "composite":  composite_path,
                "params":     params,
                "duration_s": duration_s,
                "interval_s": interval_s,
                "sim_time":   total,
                "load_time":  load_time,
                "wall_time":  wall_time,
                "divided":    divided,
                "run_id":     run.run_id,
                "db_path":    str(db_path.relative_to(REPO_ROOT)),
                "n_snapshots": n_snapshots,
            }
            (db_dir / "result.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {db_path.relative_to(REPO_ROOT)}")
    print(f"  sim_time={total}s  wall={wall_time:.1f}s  divided={divided}")
    print(f"  run_id={run.run_id}")
    print(f"  summary: .pbg/default-baseline/result.json")


if __name__ == "__main__":
    main()
