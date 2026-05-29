"""cProfile the baseline (kFBA) composite — build + warm excluded, time
only a representative chunk of simulated wall.

Usage:
    .venv/bin/python scripts/profile_baseline.py [--warm 60] [--chunk 120] \
        [--composite baseline] [--out /tmp/baseline.prof]
"""

from __future__ import annotations
import argparse
import cProfile
import pstats
import time
import warnings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", type=int, default=60)
    p.add_argument("--chunk", type=int, default=120)
    p.add_argument("--composite", default="baseline")
    p.add_argument("--out", default="/tmp/baseline.prof")
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    from v2ecoli import build_composite

    t0 = time.perf_counter()
    c = build_composite(args.composite)
    print(f"build: {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    c.run(args.warm)
    print(f"warm {args.warm}s: {time.perf_counter() - t0:.2f}s", flush=True)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    c.run(args.chunk)
    pr.disable()
    wall = time.perf_counter() - t0
    print(f"profiled chunk: sim={args.chunk}s wall={wall:.2f}s", flush=True)

    pr.dump_stats(args.out)
    print(f"dumped: {args.out}", flush=True)

    print(f"\n=== top {args.top} by cumulative ===\n", flush=True)
    s = pstats.Stats(pr).sort_stats("cumulative")
    s.print_stats(args.top)

    print(f"\n=== top {args.top} by tottime (self) ===\n", flush=True)
    s2 = pstats.Stats(pr).sort_stats("tottime")
    s2.print_stats(args.top)


if __name__ == "__main__":
    main()
