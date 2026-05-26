"""
Colony composite CLI
====================

Build and run the v2ecoli colony composite (N whole-cell E. coli embedded
in a pymunk 2D environment via the EcoliWCM bridge) from the command line.

Usage::

    # 1 cell, 60 simulated seconds, defaults
    v2ecoli-colony

    # 1 cell, run past natural division (~2460s sim)
    v2ecoli-colony --n-cells 1 --duration-min 50

    # Smaller environment, deterministic seed
    v2ecoli-colony --n-cells 2 --env-size 20 --seed 42

    # Quiet (final summary only) + result JSON
    v2ecoli-colony --duration-s 120 --quiet --out out/run.json

Prints per-tick wall / RSS / cell-count to stdout and a JSON summary on
exit. With ``--out``, also writes the summary + per-tick samples to the
given file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2ecoli-colony",
        description="Run the v2ecoli colony composite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-cells", type=int, default=1,
                        help="initial cell count (default: 1)")
    dur = parser.add_mutually_exclusive_group()
    dur.add_argument("--duration-min", type=float, default=None,
                     help="sim duration in MINUTES (e.g. 50 for one cell cycle)")
    dur.add_argument("--duration-s", type=float, default=None,
                     help="sim duration in SECONDS (default: 60)")
    parser.add_argument("--env-size", type=float, default=30.0,
                        help="2D environment edge length in um (default: 30)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="out/cache",
                        help="ParCa cache dir (default: out/cache)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="tick interval seconds (default: 1.0)")
    parser.add_argument("--progress-every", type=int, default=30,
                        help="print a progress line every N ticks (default: 30)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-tick output (final summary only)")
    parser.add_argument("--out", default=None,
                        help="write run summary + tick samples to this JSON file")
    args = parser.parse_args(argv)

    if args.duration_min is not None:
        duration_s = float(args.duration_min) * 60.0
    elif args.duration_s is not None:
        duration_s = float(args.duration_s)
    else:
        duration_s = 60.0
    n_ticks = max(1, int(duration_s / args.interval))

    print(f"v2ecoli-colony: n_cells={args.n_cells} duration_s={duration_s:.0f} "
          f"seed={args.seed} env_size={args.env_size} ticks={n_ticks}",
          file=sys.stderr)

    import psutil
    from v2ecoli.colony import make_colony

    proc = psutil.Process()
    t_build = time.perf_counter()
    comp = make_colony(
        n_cells=args.n_cells,
        env_size=args.env_size,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    build_s = time.perf_counter() - t_build
    print(f"built in {build_s:.1f}s; warmup tick…", file=sys.stderr)

    t_warmup = time.perf_counter()
    comp.run(args.interval)
    warmup_s = time.perf_counter() - t_warmup

    samples: list[dict] = []
    peak_rss = proc.memory_info().rss / 1024 / 1024
    n_div = 0
    n_after = len(comp.state.get("cells", {})) or args.n_cells
    t_main = time.perf_counter()
    status = "ok"
    try:
        for tick in range(n_ticks):
            n_before = len(comp.state.get("cells", {}))
            t_tick = time.perf_counter()
            comp.run(args.interval)
            wall_ms = (time.perf_counter() - t_tick) * 1000.0
            n_after = len(comp.state.get("cells", {}))
            if n_after != n_before:
                n_div += 1
            rss_mb = proc.memory_info().rss / 1024 / 1024
            peak_rss = max(peak_rss, rss_mb)
            samples.append({
                "tick": tick, "wall_ms": round(wall_ms, 1),
                "cells": n_after, "rss_mb": round(rss_mb, 1),
            })
            if not args.quiet and (tick % args.progress_every == 0 or tick == n_ticks - 1):
                sim_t = (tick + 1) * args.interval
                print(f"  tick {tick:5d}  t_sim={sim_t:6.0f}s  "
                      f"wall={wall_ms:7.1f}ms  cells={n_after:3d}  rss={rss_mb:6.0f}MB",
                      file=sys.stderr)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[interrupted]", file=sys.stderr)

    wall_main = time.perf_counter() - t_main

    summary = {
        "status": status,
        "n_cells_initial": args.n_cells,
        "n_cells_final": n_after,
        "duration_s": duration_s,
        "n_ticks": len(samples),
        "n_division_events": n_div,
        "seed": args.seed,
        "build_seconds": round(build_s, 2),
        "warmup_seconds": round(warmup_s, 2),
        "wall_seconds": round(wall_main, 2),
        "peak_rss_mb": round(peak_rss, 1),
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "ticks": samples}, f, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
