"""Perf harness for colonies-02-parallel-multigen-perf.

Run a natural-division colony from ONE cell, let it double (1->2->4->8),
and record per-tick compute cost so we can read per-cell wall + RSS by
generation. Supports two transports:

  * local (default) — single-threaded, GIL-bound (the colonies-01 regime)
  * ray             — one OS process per cell via the process-bigraph Ray
                      protocol (transport='ray', parallel_processes=True)

Adapted from studies/colonies-01-hpc-readiness/sims/run.py. Differences:
  * NO --force-divide: division is natural (EcoliWCM signals divide at its
    biological mass threshold ~3000s in).
  * reads jitter_per_second / init_mass / parallel_processes from the
    simulation_set perturbation so the build-phase physics fix is applied.
  * adds `transport` to both tables and records live_cell_count every tick
    (generation is derived in analysis as ceil(log2(count))).

NOTE on Ray timing: the per-cell EcoliWCM.update runs inside the Ray actor
process, so the in-process monkeypatch CANNOT see it — per_cell_update_ms_sum
is ~0 under transport=ray. For Ray runs the headline metric is wall_ms
(end-to-end) and main+actor RSS; per-cell = wall_ms / live_cell_count.

Usage:
    python .../sims/run.py --sim-name seq-1cell-4div
    python .../sims/run.py --sim-name ray-1cell-4div
    python .../sims/run.py --sim-name seq-1cell-4div --duration-min 2   # smoke
"""
from __future__ import annotations

import argparse
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

_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

RUNS_SCHEMA = {
    "run_id":            pl.Utf8,
    "sim_name":          pl.Utf8,
    "transport":         pl.Utf8,
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
    "transport":               pl.Utf8,
    "tick":                    pl.Int64,
    "sim_time":                pl.Float64,
    "wall_ms":                 pl.Float64,
    "per_cell_update_ms_sum":  pl.Float64,   # ~0 under ray (runs in actor)
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


def _read_existing(path: Path, schema: dict) -> pl.DataFrame:
    if path.is_file():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=schema)


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def install_timers() -> dict:
    """Monkey-patch EcoliWCM.update + PymunkProcess.update for per-call wall.
    Under transport=ray the EcoliWCM patch is inert (update runs remotely)."""
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
    parser.add_argument("--sim-name", required=True)
    parser.add_argument("--duration-min", type=float, default=None)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--commit-every", type=int, default=10)
    parser.add_argument("--env-size", type=float, default=30.0)
    args = parser.parse_args(argv)

    sim = load_sim_entry(args.sim_name)
    pert = sim.get("perturbation") or {}
    n_cells = int(pert.get("n_cells", 1))
    parallel = bool(pert.get("parallel_processes", False))
    transport = "ray" if parallel else "local"
    jitter = float(pert.get("jitter_per_second", 1e-4))
    init_mass = pert.get("init_mass", 200.0)
    duration_min = args.duration_min if args.duration_min is not None else sim["duration_min"]
    duration_s = float(duration_min) * 60.0
    seed = sim["seeds"][0]
    n_ticks = int(duration_s)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_path = runs_dir / RUNS_PARQUET
    ticks_path = runs_dir / TICKS_PARQUET
    run_id = str(uuid.uuid4())

    print(f"[{args.sim_name}] transport={transport} n_cells={n_cells} "
          f"jitter={jitter:g} init_mass={init_mass} "
          f"duration_min={duration_min} seed={seed} ticks={n_ticks}")
    print(f"  run_id={run_id}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runs_df = _read_existing(runs_path, RUNS_SCHEMA)
    runs_df = pl.concat([runs_df, pl.DataFrame([{
        "run_id": run_id, "sim_name": args.sim_name, "transport": transport,
        "n_cells_initial": n_cells, "n_cells_final": None,
        "duration_s": float(duration_s), "seed": int(seed),
        "wall_seconds": None, "peak_rss_mb": None, "n_division_events": None,
        "started_at": started_at, "completed_at": None,
        "status": "running", "note": None,
    }], schema=RUNS_SCHEMA)], how="vertical")
    _atomic_write(runs_df, runs_path)

    timings = install_timers()
    proc = psutil.Process()

    from v2ecoli.colony import make_colony

    note = f"transport={transport}; "
    n_div_events = 0
    n_after = n_cells
    wall_main = 0.0
    peak_rss = 0.0
    status = "ok"
    ticks_buffer: list[dict] = []

    def _flush() -> None:
        nonlocal ticks_buffer
        if not ticks_buffer:
            return
        new = pl.DataFrame(ticks_buffer, schema=TICKS_SCHEMA)
        combined = pl.concat([_read_existing(ticks_path, TICKS_SCHEMA), new],
                             how="vertical")
        _atomic_write(combined, ticks_path)
        ticks_buffer = []

    try:
        print("Building colony…")
        t_build = time.perf_counter()
        comp = make_colony(
            n_cells=n_cells, env_size=args.env_size, cache_dir="out/cache",
            seed=seed, transport=transport, parallel_processes=parallel,
            jitter_per_second=jitter, init_mass=init_mass,
        )
        print(f"  built in {time.perf_counter() - t_build:.1f}s")

        print("Warmup tick…")
        comp.run(1.0)

        print(f"Main loop: {n_ticks} ticks at 1s each (natural division)")
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
                "run_id": run_id, "transport": transport, "tick": int(tick),
                "sim_time": float(comp.state.get("global_time", tick + 1)),
                "wall_ms": float(wall_ms),
                "per_cell_update_ms_sum": float(timings["ecoli_ms"]),
                "pymunk_step_ms": float(timings["pymunk_ms"]),
                "live_cell_count": int(n_after),
                "rss_mb": float(rss_mb),
            })
            if tick % args.commit_every == 0:
                _flush()
                print(f"  tick {tick:4d}: wall={wall_ms:7.0f}ms  "
                      f"ecoli={timings['ecoli_ms']:7.0f}ms  "
                      f"cells={n_after:3d}  rss={rss_mb:6.0f}MB")
        wall_main = time.perf_counter() - t_main

    except KeyboardInterrupt:
        status = "interrupted"
        note += "KeyboardInterrupt; "
    except Exception as e:  # noqa: BLE001
        status = "error"
        note += f"{type(e).__name__}: {e}; "
        import traceback
        traceback.print_exc()
    finally:
        _flush()
        if transport == "ray":
            try:
                from process_bigraph.protocols.ray import shutdown_all_runtimes
                shutdown_all_runtimes()
            except Exception:
                pass
        completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        runs_df = _read_existing(runs_path, RUNS_SCHEMA)
        m = pl.col("run_id") == run_id
        runs_df = runs_df.with_columns(
            pl.when(m).then(pl.lit(int(n_after))).otherwise(pl.col("n_cells_final")).alias("n_cells_final"),
            pl.when(m).then(pl.lit(float(wall_main))).otherwise(pl.col("wall_seconds")).alias("wall_seconds"),
            pl.when(m).then(pl.lit(float(peak_rss))).otherwise(pl.col("peak_rss_mb")).alias("peak_rss_mb"),
            pl.when(m).then(pl.lit(int(n_div_events))).otherwise(pl.col("n_division_events")).alias("n_division_events"),
            pl.when(m).then(pl.lit(completed_at)).otherwise(pl.col("completed_at")).alias("completed_at"),
            pl.when(m).then(pl.lit(status)).otherwise(pl.col("status")).alias("status"),
            pl.when(m).then(pl.lit(note.strip() or None)).otherwise(pl.col("note")).alias("note"),
        )
        _atomic_write(runs_df, runs_path)

    print(f"\nDone. run_id={run_id} status={status} wall={wall_main:.1f}s "
          f"peak_rss={peak_rss:.0f}MB n_div={n_div_events} cells_final={n_after}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
