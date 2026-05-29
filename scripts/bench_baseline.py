"""Quick wall-time benchmark for the baseline composite.

Builds, warms with one chunk, then times N chunks of `--chunk` simulated
seconds. Prints per-chunk times so jitter is visible. Used to validate
performance changes against the kFBA baseline without spinning up the
full workflow_report pipeline.

Usage:
    .venv/bin/python scripts/bench_baseline.py [--warm 60] [--chunk 600] [--repeats 1]
"""

from __future__ import annotations
import argparse
import time
import warnings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", type=int, default=60,
                   help="Warm-up simulated seconds before timed chunks.")
    p.add_argument("--chunk", type=int, default=600,
                   help="Simulated seconds per timed chunk.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Number of timed chunks to average over.")
    p.add_argument("--composite", default="baseline",
                   help="Composite recipe name (default: baseline).")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    from v2ecoli import build_composite

    t0 = time.perf_counter()
    c = build_composite(args.composite)
    print(f"build: {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    c.run(args.warm)
    print(f"warm {args.warm}s: {time.perf_counter() - t0:.2f}s", flush=True)

    times = []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        c.run(args.chunk)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"chunk {i + 1}/{args.repeats}  sim={args.chunk}s  wall={dt:.2f}s  "
              f"({args.chunk / dt:.1f}x real-time)", flush=True)

    if args.repeats > 1:
        avg = sum(times) / len(times)
        print(f"avg wall/chunk: {avg:.2f}s", flush=True)


if __name__ == "__main__":
    main()
