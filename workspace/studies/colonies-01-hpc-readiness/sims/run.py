"""Perf harness for colonies-01-hpc-readiness.

Run a named simulation_set entry from the study's study.yaml, instrument
per-tick costs, and write rows to the study's runs.db.

Tables written:
  runs  — one row per completed run (totals + provenance)
  ticks — one row per tick (wall time + per-cell update + pymunk step + rss + cells)

Typical usage::

    python studies/colonies-01-hpc-readiness/sims/run.py --sim-name nsweep-n1
    python studies/colonies-01-hpc-readiness/sims/run.py --sim-name build-smoke-n2 \
        --duration-min 0.1 --force-divide

The harness reads ``n_cells``, ``duration_min``, and ``seeds`` from the
matching entry in ``study.yaml::simulation_set``. Use ``--duration-min``
to override (useful for smoke tests). ``--force-divide`` triggers the
divide flag on every initial cell after the warmup tick — handy for
exercising the structural-update path without waiting ~2500s sim time.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import psutil
import yaml

STUDY_DIR = Path(__file__).resolve().parent.parent
STUDY_YAML = STUDY_DIR / "study.yaml"
DEFAULT_DB = STUDY_DIR / "runs.db"

# Make sure the worktree's v2ecoli wins over any system-installed copy.
# When invoked as `python studies/…/sims/run.py`, sys.path[0] is this
# script's dir, not the worktree root.
_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def load_sim_entry(sim_name: str) -> dict:
    with open(STUDY_YAML) as f:
        study = yaml.safe_load(f)
    for sim in study.get("simulation_set", []):
        if sim["name"] == sim_name:
            return sim
    raise ValueError(
        f"sim {sim_name!r} not found in {STUDY_YAML}. "
        f"Available: {[s['name'] for s in study.get('simulation_set', [])]}"
    )


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_name TEXT NOT NULL,
            n_cells_initial INTEGER,
            n_cells_final INTEGER,
            duration_s REAL,
            seed INTEGER,
            wall_seconds REAL,
            peak_rss_mb REAL,
            n_division_events INTEGER,
            started_at TEXT,
            completed_at TEXT,
            status TEXT,
            note TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            run_id INTEGER NOT NULL,
            tick INTEGER NOT NULL,
            sim_time REAL,
            wall_ms REAL,
            per_cell_update_ms_sum REAL,
            pymunk_step_ms REAL,
            live_cell_count INTEGER,
            rss_mb REAL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_ticks_run ON ticks(run_id, tick)")
    conn.commit()
    return conn


def install_timers() -> dict:
    """Monkey-patch EcoliWCM.update + PymunkProcess.update to capture
    per-call wall time. Returns a mutable dict the caller can reset per
    tick to read the cumulative cost."""
    from v2ecoli.bridge import EcoliWCM
    from viva_munk.processes.multibody import PymunkProcess

    timings = {"ecoli_ms": 0.0, "pymunk_ms": 0.0}

    _orig_ecoli = EcoliWCM.update

    def timed_ecoli(self, state, interval):
        t0 = time.perf_counter()
        try:
            return _orig_ecoli(self, state, interval)
        finally:
            timings["ecoli_ms"] += (time.perf_counter() - t0) * 1000.0
    EcoliWCM.update = timed_ecoli

    _orig_pymunk = PymunkProcess.update

    def timed_pymunk(self, state, interval):
        t0 = time.perf_counter()
        try:
            return _orig_pymunk(self, state, interval)
        finally:
            timings["pymunk_ms"] += (time.perf_counter() - t0) * 1000.0
    PymunkProcess.update = timed_pymunk

    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-name", required=True,
                        help="simulation_set entry name from study.yaml")
    parser.add_argument("--duration-min", type=float, default=None,
                        help="override sim duration (minutes); default uses study.yaml")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help=f"path to runs.db (default: {DEFAULT_DB})")
    parser.add_argument("--force-divide", action="store_true",
                        help="force-divide every initial cell after warmup tick")
    parser.add_argument("--commit-every", type=int, default=10,
                        help="commit + flush every N ticks (default: 10)")
    parser.add_argument("--env-size", type=float, default=30.0,
                        help="2D environment edge length um (default: 30)")
    args = parser.parse_args(argv)

    sim = load_sim_entry(args.sim_name)
    perturbation = sim.get("perturbation") or {}
    n_cells = perturbation.get("n_cells", 2)
    duration_min = args.duration_min if args.duration_min is not None else sim["duration_min"]
    duration_s = float(duration_min) * 60.0
    seed = sim["seeds"][0]
    n_ticks = int(duration_s)

    print(f"[{args.sim_name}] n_cells={n_cells} duration_min={duration_min} "
          f"seed={seed} ticks={n_ticks}")
    print(f"  db={args.db}")

    conn = init_db(Path(args.db))
    cur = conn.cursor()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cur.execute(
        "INSERT INTO runs (sim_name, n_cells_initial, duration_s, seed, "
        "started_at, status) VALUES (?,?,?,?,?,?)",
        (args.sim_name, n_cells, duration_s, seed, started_at, "running"),
    )
    run_id = cur.lastrowid
    conn.commit()
    print(f"  run_id={run_id}")

    # Install timers BEFORE building colony (so the build-time imports of
    # EcoliWCM / PymunkProcess pick up the wrapped methods).
    timings = install_timers()
    proc = psutil.Process()

    # Lazy import to keep --help fast.
    from v2ecoli.colony import make_colony

    note = ""
    n_div_events = 0
    n_after = 0
    wall_main = 0.0
    peak_rss = 0.0
    status = "ok"

    try:
        print("Building colony…")
        t_build = time.perf_counter()
        comp = make_colony(
            n_cells=n_cells,
            env_size=args.env_size,
            cache_dir="out/cache",
            seed=seed,
        )
        print(f"  built in {time.perf_counter() - t_build:.1f}s")

        print("Warmup tick (builds inner EcoliWCM composites lazily)…")
        timings["ecoli_ms"] = 0.0
        timings["pymunk_ms"] = 0.0
        t0 = time.perf_counter()
        comp.run(1.0)
        warmup_s = time.perf_counter() - t0
        print(f"  warmup done in {warmup_s:.1f}s "
              f"(ecoli {timings['ecoli_ms']:.0f}ms, pymunk {timings['pymunk_ms']:.0f}ms)")

        if args.force_divide:
            print("Force-dividing all initial cells…")
            for cid in list(comp.state["cells"].keys()):
                inst = comp.state["cells"][cid]["ecoli"]["instance"]
                inst._composite.state["agents"]["0"]["divide"] = True
            note += "force-divide; "

        print(f"Main loop: {n_ticks} ticks at 1s each")
        t_main = time.perf_counter()

        for tick in range(n_ticks):
            timings["ecoli_ms"] = 0.0
            timings["pymunk_ms"] = 0.0
            n_before = len(comp.state["cells"])

            t_tick = time.perf_counter()
            comp.run(1.0)
            wall_ms = (time.perf_counter() - t_tick) * 1000.0

            n_after = len(comp.state["cells"])
            if n_after != n_before:
                n_div_events += 1

            rss_mb = proc.memory_info().rss / 1024 / 1024
            peak_rss = max(peak_rss, rss_mb)

            cur.execute(
                "INSERT INTO ticks (run_id, tick, sim_time, wall_ms, "
                "per_cell_update_ms_sum, pymunk_step_ms, live_cell_count, "
                "rss_mb) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, tick, float(comp.state.get("global_time", tick + 1)),
                 wall_ms, timings["ecoli_ms"], timings["pymunk_ms"],
                 n_after, rss_mb),
            )

            if tick % args.commit_every == 0:
                conn.commit()
                print(f"  tick {tick:4d}: wall={wall_ms:6.0f}ms  "
                      f"ecoli={timings['ecoli_ms']:6.0f}ms  "
                      f"pymunk={timings['pymunk_ms']:5.1f}ms  "
                      f"cells={n_after:3d}  rss={rss_mb:5.0f}MB")

        wall_main = time.perf_counter() - t_main

    except KeyboardInterrupt:
        status = "interrupted"
        note += "KeyboardInterrupt; "
        print("\n[interrupted]")
    except Exception as e:  # noqa: BLE001
        status = "error"
        note += f"{type(e).__name__}: {e}; "
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur.execute(
            "UPDATE runs SET n_cells_final=?, wall_seconds=?, peak_rss_mb=?, "
            "n_division_events=?, completed_at=?, status=?, note=? "
            "WHERE run_id=?",
            (n_after, wall_main, peak_rss, n_div_events, completed_at,
             status, note.strip(), run_id),
        )
        conn.commit()
        conn.close()

    print(f"\nDone. run_id={run_id} status={status} wall={wall_main:.1f}s "
          f"peak_rss={peak_rss:.0f}MB n_div={n_div_events} cells_final={n_after}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
