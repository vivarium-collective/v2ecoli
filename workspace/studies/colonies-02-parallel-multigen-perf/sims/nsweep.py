"""Static N-sweep: wall-vs-N for local (GIL-bound) vs ray (per-process).

The confirmed unbounded per-cell EcoliWCM RSS leak (~7.7 MB/sim-s, see
seq-1cell-4div) means a 180-min growing-colony run OOM-kills before it
finishes, so per-tick wall at high cell counts is contaminated by swap
thrashing. This driver isolates the GIL-ceiling question cleanly: hold N
*static* and run a SHORT window (default 60 ticks) at low RSS, for both
transports, so total wall vs N is measured before the leak dominates.

Sequential (local) walks all N cells' 55-step WCM updates on one GIL-bound
thread, so total wall ~ N x single-cell wall and crosses realtime (~1000ms
for a 1s tick) near N~13 (the colonies-01 ceiling). Under ray each cell's
update runs in its own OS process, so independent cells solve concurrently
and total wall should stay far below N x single-cell.

Runs ONE (transport, n) config per invocation and APPENDS to nsweep.parquet
so an OOM at large N does not lose the smaller-N rows.

Usage:
    python .../sims/nsweep.py --transport local --n 4
    python .../sims/nsweep.py --transport ray   --n 8 --ticks 60
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import polars as pl
import psutil

STUDY_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = STUDY_DIR / "runs"
NSWEEP_PARQUET = "nsweep.parquet"

_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

SCHEMA = {
    "row_id":          pl.Utf8,
    "transport":       pl.Utf8,
    "n_cells":         pl.Int64,
    "ticks":           pl.Int64,
    "total_wall_s":    pl.Float64,
    "median_wall_ms":  pl.Float64,
    "per_cell_ms":     pl.Float64,   # median_wall_ms / n_cells
    "realtime_ratio":  pl.Float64,   # median_wall_ms / 1000 (>1 = slower than realtime)
    "main_rss_mb":     pl.Float64,   # main-process RSS at end
    "total_py_rss_mb": pl.Float64,   # sum RSS over all python procs (incl. ray workers)
    "seed":            pl.Int64,
}


def total_python_rss_mb() -> float:
    total = 0.0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            nm = (p.info.get("name") or "").lower()
            if "python" in nm and p.info.get("memory_info"):
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / 1024 / 1024


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", choices=["local", "ray"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--env-size", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / NSWEEP_PARQUET

    parallel = args.transport == "ray"
    print(f"[nsweep] transport={args.transport} n={args.n} ticks={args.ticks}")

    from v2ecoli.colony import make_colony
    proc = psutil.Process()

    t_build = time.perf_counter()
    comp = make_colony(
        n_cells=args.n, env_size=args.env_size, cache_dir="out/cache",
        seed=args.seed, transport=args.transport, parallel_processes=parallel,
        jitter_per_second=1e-4, init_mass=200.0,
    )
    print(f"  built in {time.perf_counter() - t_build:.1f}s")
    comp.run(1.0)  # warmup (hydrate actors / inner composites)

    walls = []
    t0 = time.perf_counter()
    for tick in range(args.ticks):
        t = time.perf_counter()
        comp.run(1.0)
        walls.append((time.perf_counter() - t) * 1000.0)
    total_wall = time.perf_counter() - t0

    walls_sorted = sorted(walls)
    median_wall = walls_sorted[len(walls_sorted) // 2]
    main_rss = proc.memory_info().rss / 1024 / 1024
    total_rss = total_python_rss_mb()
    n_live = len(comp.state["cells"])

    row = {
        "row_id": str(uuid.uuid4()),
        "transport": args.transport,
        "n_cells": int(args.n),
        "ticks": int(args.ticks),
        "total_wall_s": float(total_wall),
        "median_wall_ms": float(median_wall),
        "per_cell_ms": float(median_wall / max(1, args.n)),
        "realtime_ratio": float(median_wall / 1000.0),
        "main_rss_mb": float(main_rss),
        "total_py_rss_mb": float(total_rss),
        "seed": int(args.seed),
    }

    existing = pl.read_parquet(out) if out.is_file() else pl.DataFrame(schema=SCHEMA)
    combined = pl.concat([existing, pl.DataFrame([row], schema=SCHEMA)], how="vertical")
    tmp = out.with_suffix(out.suffix + ".tmp")
    combined.write_parquet(tmp)
    tmp.replace(out)

    print(f"  n_live={n_live} median_wall={median_wall:.0f}ms "
          f"per_cell={row['per_cell_ms']:.0f}ms realtime_ratio={row['realtime_ratio']:.2f} "
          f"main_rss={main_rss:.0f}MB total_py_rss={total_rss:.0f}MB")

    if args.transport == "ray":
        try:
            from process_bigraph.protocols.ray import shutdown_all_runtimes
            shutdown_all_runtimes()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
