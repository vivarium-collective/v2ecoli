"""Multi-generation Phase 0 pilot — N seeds × max_generations × v2ecoli baseline.

Uses v2ecoli.library.sqlite_run.run_multigen_sqlite to capture per-generation
listener trajectories. Output: per-seed sqlite db at
.pbg/runs/phase0-multigen/seed_NN/run.db (queryable via standard sqlite
tools or vivarium_dashboard.lib.composite_runs).

Wall projection from 600-step single-gen baseline (48s wall): for 3 generations
(~2100 ticks), expect ~3 min wall per replicate. N=4 → ~12 min total serial.

Goal: unblock pdmp-02's multi-gen-inheritance-binomial primary test —
variance of daughter1−daughter2 should be consistent with binomial(mother, 0.5)
partition across many divisions.

Run from worktree root:
    python scripts/run_phase0_multigen.py [--n-seeds 4] [--max-steps 2100] [--max-generations 3]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli import build_composite
from v2ecoli.library.sqlite_run import run_multigen_sqlite


CACHE_DIR = "out/cache"
OUT_ROOT = Path(".pbg/runs/phase0-multigen")

EMIT_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.volume",
    "listeners.mass.growth",
    "listeners.mass.instantaneous_growth_rate",
]


def run_one(seed: int, max_steps: int, max_generations: int, chunk: int) -> dict:
    out_dir = OUT_ROOT / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_file = out_dir / "run.db"
    if db_file.exists():
        db_file.unlink()

    run_id = f"phase0-multigen-seed{seed:02d}"
    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    try:
        result = run_multigen_sqlite(
            composite,
            run_id=run_id,
            db_file=str(db_file),
            emit_paths=EMIT_PATHS,
            max_steps=max_steps,
            max_generations=max_generations,
            chunk=chunk,
            initial_agent_id="0",
        )
    except Exception as e:
        print(f"  seed={seed:02d} FAILED: {type(e).__name__}: {str(e)[:80]}")
        return {"seed": seed, "error": str(e), "type": type(e).__name__,
                "wall_seconds": round(time.time() - t0, 2)}

    wall = time.time() - t0
    summary = {
        "seed": seed,
        "max_steps": max_steps,
        "max_generations": max_generations,
        "actual_steps": result.get("steps"),
        "actual_generations_seen": result.get("generations", []),
        "wall_seconds": round(wall, 2),
        "db_file": str(db_file),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  seed={seed:02d}: wall={wall:6.1f}s  steps={result.get('steps')}  "
          f"generations={result.get('generations')}  db={db_file.stat().st_size//1024} KB")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2100)
    p.add_argument("--max-generations", type=int, default=3)
    p.add_argument("--chunk", type=int, default=60)
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0 multi-gen pilot: N={args.n_seeds} seeds × {args.max_steps} steps × "
          f"≤{args.max_generations} generations (chunk={args.chunk})")
    t0 = time.time()
    results = []
    for seed in range(args.n_seeds):
        results.append(run_one(seed, args.max_steps, args.max_generations, args.chunk))
    total = time.time() - t0
    successful = [r for r in results if "error" not in r]

    ensemble = {
        "n_seeds_requested": args.n_seeds,
        "n_seeds_successful": len(successful),
        "max_steps": args.max_steps,
        "max_generations": args.max_generations,
        "total_wall_seconds": round(total, 2),
        "per_seed": results,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(ensemble, indent=2))
    print(f"\nDone: {len(successful)}/{args.n_seeds} runs, total wall {total/60:.1f} min")
    for r in successful:
        print(f"  seed_{r['seed']:02d}: {len(r.get('actual_generations_seen', []))} generations, "
              f"{r.get('actual_steps')} steps")


if __name__ == "__main__":
    main()
