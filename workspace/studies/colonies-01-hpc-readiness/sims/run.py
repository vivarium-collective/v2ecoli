"""Perf harness for colonies-01-hpc-readiness.

Run a named simulation_set entry from the study's study.yaml, instrument
per-tick costs, and write rows to two parquet files under the study's
``runs/`` directory (replaces the legacy sqlite ``runs.db`` per
workspace-default-emitter parity — see workspace.yaml::runtime.default_emitter).

Files written under ``<study>/runs/``:
  runs.parquet   — one row per completed run (totals + provenance)
  ticks.parquet  — one row per tick (wall time + per-cell update + pymunk
                   step + rss + cells), hive-key column ``run_id``

The harness is append-aware: a new run appends one row to runs.parquet
and ~duration_s rows to ticks.parquet (read-rewrite per flush at
``--commit-every`` cadence, atomic via a ``.tmp`` swap).

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
import sys
import time
import uuid
from pathlib import Path

import polars as pl
import psutil
import yaml

STUDY_DIR = Path(__file__).resolve().parent.parent
STUDY_YAML = STUDY_DIR / "study.yaml"
DEFAULT_RUNS_DIR = STUDY_DIR / "runs"
RUNS_PARQUET = "runs.parquet"
TICKS_PARQUET = "ticks.parquet"

# Make sure the worktree's v2ecoli wins over any system-installed copy.
# When invoked as `python studies/…/sims/run.py`, sys.path[0] is this
# script's dir, not the worktree root.
_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


# ---- schemas ----
# Pre-declared polars schemas so empty buffers + appends preserve dtypes
# across runs (otherwise reading-then-rewriting a parquet file flips
# column types based on the in-memory buffer's content).
RUNS_SCHEMA = {
    "run_id":            pl.Utf8,
    "sim_name":          pl.Utf8,
    "n_cells_initial":   pl.Int64,
    "n_cells_final":     pl.Int64,
    "duration_s":        pl.Float64,
    "seed":              pl.Int64,
    "wall_seconds":      pl.Float64,
    "peak_rss_mb":       pl.Float64,
    "n_division_events": pl.Int64,
    "started_at":        pl.Utf8,
    "completed_at":      pl.Utf8,
    "status":            pl.Utf8,
    "note":              pl.Utf8,
}

TICKS_SCHEMA = {
    "run_id":                  pl.Utf8,
    "tick":                    pl.Int64,
    "sim_time":                pl.Float64,
    "wall_ms":                 pl.Float64,
    "per_cell_update_ms_sum":  pl.Float64,
    "pymunk_step_ms":          pl.Float64,
    "live_cell_count":         pl.Int64,
    "rss_mb":                  pl.Float64,
}


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


def _ensure_runs_dir(runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)


def _read_existing(path: Path, schema: dict) -> pl.DataFrame:
    """Load an existing parquet, or return an empty DF with the declared
    schema if absent. Schema lock means appends + atomic rewrite stay
    type-stable across runs."""
    if path.is_file():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=schema)


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    """Write parquet via .tmp swap so an interrupted run never leaves a
    half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


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
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR),
                        help=f"directory holding runs.parquet + ticks.parquet "
                             f"(default: {DEFAULT_RUNS_DIR})")
    parser.add_argument("--force-divide", action="store_true",
                        help="force-divide every initial cell after warmup tick")
    parser.add_argument("--commit-every", type=int, default=10,
                        help="flush ticks buffer every N ticks (default: 10)")
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

    runs_dir = Path(args.runs_dir)
    _ensure_runs_dir(runs_dir)
    runs_path = runs_dir / RUNS_PARQUET
    ticks_path = runs_dir / TICKS_PARQUET

    run_id = str(uuid.uuid4())

    print(f"[{args.sim_name}] n_cells={n_cells} duration_min={duration_min} "
          f"seed={seed} ticks={n_ticks}")
    print(f"  runs_dir={runs_dir}")
    print(f"  run_id={run_id}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Pre-seed a "running" row so an interrupted/crashed run still appears
    # in runs.parquet. We rewrite it post-loop with the final totals.
    runs_df = _read_existing(runs_path, RUNS_SCHEMA)
    new_row = pl.DataFrame([{
        "run_id":            run_id,
        "sim_name":          args.sim_name,
        "n_cells_initial":   int(n_cells),
        "n_cells_final":     None,
        "duration_s":        float(duration_s),
        "seed":              int(seed),
        "wall_seconds":      None,
        "peak_rss_mb":       None,
        "n_division_events": None,
        "started_at":        started_at,
        "completed_at":      None,
        "status":            "running",
        "note":              None,
    }], schema=RUNS_SCHEMA)
    runs_df = pl.concat([runs_df, new_row], how="vertical")
    _atomic_write(runs_df, runs_path)

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
    ticks_buffer: list[dict] = []

    def _flush_ticks_buffer() -> None:
        nonlocal ticks_buffer
        if not ticks_buffer:
            return
        new = pl.DataFrame(ticks_buffer, schema=TICKS_SCHEMA)
        existing = _read_existing(ticks_path, TICKS_SCHEMA)
        combined = pl.concat([existing, new], how="vertical")
        _atomic_write(combined, ticks_path)
        ticks_buffer = []

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

            ticks_buffer.append({
                "run_id":                 run_id,
                "tick":                   int(tick),
                "sim_time":               float(comp.state.get("global_time", tick + 1)),
                "wall_ms":                float(wall_ms),
                "per_cell_update_ms_sum": float(timings["ecoli_ms"]),
                "pymunk_step_ms":         float(timings["pymunk_ms"]),
                "live_cell_count":        int(n_after),
                "rss_mb":                 float(rss_mb),
            })

            if tick % args.commit_every == 0:
                _flush_ticks_buffer()
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
        # Final flush of any buffered ticks
        _flush_ticks_buffer()

        # Rewrite the run row with final totals.
        completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        runs_df = _read_existing(runs_path, RUNS_SCHEMA)
        runs_df = (
            runs_df
            .with_columns(
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(int(n_after)))
                  .otherwise(pl.col("n_cells_final")).alias("n_cells_final"),
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(float(wall_main)))
                  .otherwise(pl.col("wall_seconds")).alias("wall_seconds"),
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(float(peak_rss)))
                  .otherwise(pl.col("peak_rss_mb")).alias("peak_rss_mb"),
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(int(n_div_events)))
                  .otherwise(pl.col("n_division_events")).alias("n_division_events"),
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(completed_at))
                  .otherwise(pl.col("completed_at")).alias("completed_at"),
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(status))
                  .otherwise(pl.col("status")).alias("status"),
                pl.when(pl.col("run_id") == run_id)
                  .then(pl.lit(note.strip() or None))
                  .otherwise(pl.col("note")).alias("note"),
            )
        )
        _atomic_write(runs_df, runs_path)

    print(f"\nDone. run_id={run_id} status={status} wall={wall_main:.1f}s "
          f"peak_rss={peak_rss:.0f}MB n_div={n_div_events} cells_final={n_after}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
