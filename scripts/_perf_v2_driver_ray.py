#!/usr/bin/env python3
"""Ray-parallel v2ecoli multiseed/multigen driver — the "catch up with vEcoli"
runner.

Same per-seed work as scripts/_perf_v2_driver.py, but each seed runs as an
independent @ray.remote task (its own worker process), so N seeds simulate
CONCURRENTLY — mirroring vEcoli's Nextflow fan-out. Total wall becomes the
critical path (slowest seed) instead of the sum.

Because each seed is a separate process, each worker reports its OWN peak RSS
via resource.getrusage(RUSAGE_SELF) — giving an accurate, bounded per-seed RAM
number (the measurement vEcoli couldn't provide on macOS-local). This only
holds up because the RAM fix is in place (null-emitter override + the
chromosome_history leak removed); without it each worker would still balloon.

Writes the same summary.json shape _perf_v2_driver / perf_compare consume,
plus per-seed peak_rss_mb.

  python scripts/_perf_v2_driver_ray.py --n-seeds 2 --max-generations 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ray

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(".pbg/runs/perf-v2-ray")
CACHE_DIR = "out/cache"
EMIT_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.volume",
    "listeners.mass.growth",
    "listeners.mass.instantaneous_growth_rate",
]


@ray.remote
def run_seed(seed: int, max_steps: int, max_generations: int, chunk: int,
             repo: str) -> dict:
    """One seed, in its own Ray worker process. Returns wall + peak RSS."""
    import resource
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, repo)
    # Same RAM fix the sequential driver applies — must run in EACH worker.
    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)
    from v2ecoli import build_composite
    from v2ecoli.library.sqlite_run import run_multigen_sqlite

    out_dir = _Path(repo) / ".pbg" / "runs" / "perf-v2-ray" / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_file = out_dir / "run.db"
    if db_file.exists():
        db_file.unlink()

    t0 = time.time()
    try:
        composite = build_composite("ecoli_baseline", cache_dir=str(_Path(repo) / CACHE_DIR),
                                    seed=seed)
        result = run_multigen_sqlite(
            composite, run_id=f"perf-v2-ray-seed{seed:02d}", db_file=str(db_file),
            emit_paths=EMIT_PATHS, max_steps=max_steps,
            max_generations=max_generations, chunk=chunk,
            initial_agent_id="0", single_daughters=True)
    except Exception as e:
        return {"seed": seed, "error": str(e), "type": type(e).__name__,
                "wall_seconds": round(time.time() - t0, 2)}
    wall = time.time() - t0
    # ru_maxrss is BYTES on macOS, KiB on Linux — normalise to MB.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = round(maxrss / (1024 * 1024 if _sys.platform == "darwin" else 1024), 1)
    s = {"seed": seed, "wall_seconds": round(wall, 2), "peak_rss_mb": peak_mb,
         "max_steps": max_steps, "max_generations": max_generations,
         "actual_steps": result.get("steps"),
         "actual_generations_seen": result.get("generations", [])}
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--max-generations", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--chunk", type=int, default=60)
    a = p.parse_args()
    if not (REPO / CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found under {REPO}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # CRITICAL for real parallelism: cap each worker's BLAS/OpenMP threads so N
    # concurrent workers don't oversubscribe the cores. Without this every
    # worker's numpy/FBA grabs ALL cores → 32 threads on 16 cores → they thrash
    # and total wall ≈ sum instead of critical path. Allocate threads_per_worker
    # = cores / n_seeds so the host is fully used but never oversubscribed (e.g.
    # 2 seeds on 16 cores → 8 threads each: fast per-seed AND truly parallel).
    # Set via runtime_env so it lands in the worker env BEFORE numpy import, and
    # reserve the matching num_cpus per task so Ray schedules them disjointly.
    import os as _os
    cpu_total = _os.cpu_count() or 1
    threads = max(1, cpu_total // max(1, a.n_seeds))
    env_threads = {k: str(threads) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    ray.init(ignore_reinit_error=True, log_to_driver=False,
             runtime_env={"env_vars": env_threads})
    n_cpu = int(ray.available_resources().get("CPU", cpu_total))
    print(f"v2ecoli RAY perf driver: {a.n_seeds} seeds × ≤{a.max_generations} gens "
          f"(single_daughters, max_steps={a.max_steps}); {n_cpu:.0f} CPUs, "
          f"{threads} thread(s)/seed", flush=True)

    t0 = time.time()
    futures = [run_seed.options(num_cpus=threads).remote(
                   s, a.max_steps, a.max_generations, a.chunk, str(REPO))
               for s in range(a.n_seeds)]
    per_seed = ray.get(futures)
    total = time.time() - t0                      # critical path (concurrent)
    ray.shutdown()

    ok = [r for r in per_seed if "error" not in r]
    for r in per_seed:
        if "error" in r:
            print(f"  seed={r['seed']:02d} FAILED: {r['type']}: {r['error'][:80]}", flush=True)
        else:
            print(f"  seed={r['seed']:02d}: wall={r['wall_seconds']}s "
                  f"rss={r['peak_rss_mb']}MB gens={r['actual_generations_seen']}", flush=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "n_seeds_requested": a.n_seeds, "n_seeds_successful": len(ok),
        "max_generations": a.max_generations, "max_steps": a.max_steps,
        "single_daughters": True, "parallel": True, "n_cpu": n_cpu,
        "total_wall_seconds": round(total, 2), "per_seed": per_seed,
    }, indent=2))
    sum_wall = sum(r.get("wall_seconds", 0) for r in ok)
    print(f"\nDone: {len(ok)}/{a.n_seeds} seeds | total(critical-path) {total/60:.1f} min "
          f"| Σ per-seed {sum_wall/60:.1f} min | speedup {sum_wall/max(total,1e-9):.2f}×",
          flush=True)


if __name__ == "__main__":
    main()
