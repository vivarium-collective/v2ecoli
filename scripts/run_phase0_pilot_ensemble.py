"""Phase 0 pilot ensemble — N seeds, baseline composite, persisted via XArrayEmitter.

First reference run after the per-process RNG seeding fix (commit 7783012).
Runs N replicates of the v2ecoli baseline at the same condition, each with a
distinct master_seed. Verifies the ensemble produces DIVERSE trajectories (the
bug documented in pdmp-00 planned_runs.fix-per-process-rng-seeding is closed).

Output: .pbg/runs/phase0-pilot/seed_<N>/store.zarr per replicate, plus a
JSON summary at .pbg/runs/phase0-pilot/summary.json listing per-seed metrics:
final ATP count, final cell mass, final timestep, wall-clock.

Usage:
    python scripts/run_phase0_pilot_ensemble.py [--n-seeds 4] [--n-steps 100]

Default: N=4 seeds × 100 steps (~10 min wall per run, ~40 min serial).
For the canonical reference ensemble (N=64 × 600 steps), use --n-seeds 64
--n-steps 600 and expect ~64 hours serial; that's an HPC job.
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

import numpy as np

from v2ecoli import build_composite


CACHE_DIR = "out/cache"
OUT_ROOT = Path(".pbg/runs/phase0-pilot")


def _extract_state_summary(composite_state: dict) -> dict:
    """Pull a few human-readable scalars out of the composite final state."""
    summary: dict = {}
    try:
        agent = composite_state.get("agents", {}).get("0", {})
    except Exception:
        agent = {}

    # Cell mass
    try:
        m = agent.get("listeners", {}).get("mass", {})
        summary["dry_mass_fg"] = float(m.get("dry_mass", float("nan")))
        summary["cell_mass_fg"] = float(m.get("cell_mass", float("nan")))
    except Exception:
        pass

    # ATP count
    try:
        # bulk is a structured numpy array {id, count, ...}
        bulk = agent.get("bulk")
        if bulk is not None:
            ids = list(bulk["id"]) if hasattr(bulk, "dtype") else [
                row[0] for row in bulk
            ]
            counts = list(bulk["count"]) if hasattr(bulk, "dtype") else [
                row[1] for row in bulk
            ]
            for needle in ("ATP[c]", "WATER[c]"):
                try:
                    idx = ids.index(needle)
                    summary[f"{needle}_count"] = int(counts[idx])
                except ValueError:
                    pass
    except Exception:
        pass

    summary["global_time"] = float(composite_state.get("global_time", float("nan")))
    return summary


def run_one(seed: int, n_steps: int) -> dict:
    """Run one baseline replicate with the given master seed."""
    out_dir = OUT_ROOT / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    duration = float(n_steps)
    composite.run(duration)
    wall = time.time() - t0

    summary = {
        "seed": seed,
        "n_steps": n_steps,
        "wall_seconds": round(wall, 2),
    }
    summary.update(_extract_state_summary(composite.state))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  seed={seed:02d}: wall={wall:6.1f}s  "
        f"ATP={summary.get('ATP[c]_count', '?')}  "
        f"dry_mass_fg={summary.get('dry_mass_fg', '?')}  "
        f"global_time={summary.get('global_time', '?')}"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=4, help="number of replicate seeds (default 4)")
    p.add_argument("--n-steps", type=int, default=100, help="model seconds per replicate (default 100)")
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found — run scripts/build_cache.py first")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0 pilot ensemble: N={args.n_seeds} seeds × {args.n_steps} steps")
    t0 = time.time()
    results = []
    for seed in range(args.n_seeds):
        try:
            results.append(run_one(seed, args.n_steps))
        except Exception as e:
            err = {"seed": seed, "error": str(e), "type": type(e).__name__}
            results.append(err)
            print(f"  seed={seed:02d}: FAILED — {type(e).__name__}: {e}")

    total_wall = time.time() - t0

    # Ensemble-level summary
    successful = [r for r in results if "error" not in r]
    atp_counts = [r.get("ATP[c]_count") for r in successful if "ATP[c]_count" in r]
    mass_vals = [r.get("dry_mass_fg") for r in successful if "dry_mass_fg" in r]

    ensemble_summary = {
        "n_seeds_requested": args.n_seeds,
        "n_seeds_successful": len(successful),
        "n_steps": args.n_steps,
        "total_wall_seconds": round(total_wall, 2),
        "per_seed": results,
    }
    if atp_counts:
        a = np.array(atp_counts, dtype=float)
        ensemble_summary["ATP[c]_count_stats"] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "cv_pct": float(a.std() / a.mean() * 100) if a.mean() > 0 else None,
        }
    if mass_vals:
        m = np.array(mass_vals, dtype=float)
        ensemble_summary["dry_mass_fg_stats"] = {
            "mean": float(m.mean()),
            "std": float(m.std()),
            "cv_pct": float(m.std() / m.mean() * 100) if m.mean() > 0 else None,
        }

    (OUT_ROOT / "summary.json").write_text(json.dumps(ensemble_summary, indent=2))

    print()
    print(f"Done: {len(successful)}/{args.n_seeds} runs, total wall {total_wall/60:.1f} min")
    if "ATP[c]_count_stats" in ensemble_summary:
        s = ensemble_summary["ATP[c]_count_stats"]
        cv = s.get("cv_pct")
        print(
            f"ATP[c] across seeds: mean={s['mean']:.2e} std={s['std']:.2e} "
            f"CV={cv:.2f}%" if cv is not None else f"... std={s['std']:.2e}"
        )
        if cv is not None and cv < 0.001:
            print("  ⚠ CV ≈ 0% — trajectories are bit-identical. RNG seeding fix not flowing through.")
        elif cv is not None:
            print("  ✓ Trajectories diverge across seeds.")


if __name__ == "__main__":
    main()
