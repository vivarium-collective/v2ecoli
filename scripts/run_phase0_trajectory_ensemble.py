"""Phase 0 ensemble with FULL trajectory capture via direct JSON snapshot.

Pragmatic alternative to XArrayEmitter wiring: snapshot listener state every
SNAPSHOT_STRIDE ticks into a Python list, persist as JSON per replicate.

Why not XArrayEmitter:
  Validated reference (docs/superpowers/notes/2026-05-22-xarray-multigen-emit-validated.py)
  uses MAX_STEPS=9000 to amortize the trailing-buffer-loss issue. For shorter
  runs (<2000 steps) the trailing buffer at close() loses data silently — the
  async writer's executor may not complete the final flush before close()
  returns. For the Phase 0 N=64 × 600-step pilot, that means ~20% of timepoints
  are lost per replicate. Endpoint-state-plus-trajectory-snapshots via JSON is
  simpler and bit-exact for our purposes (matplotlib plotting + numerical
  analysis don't need the full zarr partitioning machinery).

Per-seed output:
  .pbg/runs/phase0-traj/seed_<NN>/summary.json    endpoint state
  .pbg/runs/phase0-traj/seed_<NN>/trajectory.json {time: [], <listener>: [], ...}

Plus ensemble-level summary.json with per-seed wall + endpoint stats.

Usage:
    python scripts/run_phase0_trajectory_ensemble.py [--n-seeds 64] [--n-steps 600] [--stride 5]
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
OUT_ROOT = Path(".pbg/runs/phase0-traj")

# Listener observables captured per snapshot.
LISTENER_PATHS = [
    ("listeners", "mass", "cell_mass"),
    ("listeners", "mass", "dry_mass"),
    ("listeners", "mass", "water_mass"),
    ("listeners", "mass", "protein_mass"),
    ("listeners", "mass", "rRna_mass"),
    ("listeners", "mass", "mRna_mass"),
    ("listeners", "mass", "tRna_mass"),
    ("listeners", "mass", "volume"),
    ("listeners", "mass", "growth"),
    ("listeners", "mass", "instantaneous_growth_rate"),
    ("listeners", "mass", "dry_mass_fold_change"),
]


def _get_path(d: dict, path: tuple) -> float | None:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _snapshot(state: dict, t: float) -> dict:
    """Pull current listener values out of composite state."""
    snap = {"time": t}
    agents = state.get("agents") or {}
    agent_id = next(iter(agents), None)
    if agent_id is None:
        return snap
    agent = agents[agent_id]
    for path in LISTENER_PATHS:
        key = ".".join(path)
        snap[key] = _get_path(agent, path)
    return snap


def _endpoint_summary(state: dict, seed: int, wall: float, n_steps: int) -> dict:
    summary = {"seed": seed, "n_steps": n_steps, "wall_seconds": round(wall, 2)}
    try:
        agents = state.get("agents") or {}
        agent_id = next(iter(agents), None)
        agent = agents.get(agent_id, {}) if agent_id else {}
    except Exception:
        agent = {}
    try:
        m = agent.get("listeners", {}).get("mass", {})
        summary["dry_mass_fg"] = float(m.get("dry_mass", float("nan")))
        summary["cell_mass_fg"] = float(m.get("cell_mass", float("nan")))
    except Exception:
        pass
    try:
        bulk = agent.get("bulk")
        if bulk is not None:
            ids = list(bulk["id"]) if hasattr(bulk, "dtype") else [row[0] for row in bulk]
            counts = list(bulk["count"]) if hasattr(bulk, "dtype") else [row[1] for row in bulk]
            for needle in ("ATP[c]", "WATER[c]"):
                try:
                    idx = ids.index(needle)
                    summary[f"{needle}_count"] = int(counts[idx])
                except ValueError:
                    pass
    except Exception:
        pass
    summary["global_time"] = float(state.get("global_time", float("nan")))
    return summary


def run_one(seed: int, n_steps: int, stride: int) -> dict:
    out_dir = OUT_ROOT / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)

    trajectory: list[dict] = []
    # Warm-up tick so listeners materialize.
    composite.run(1)
    trajectory.append(_snapshot(composite.state, 1.0))

    elapsed = 1
    while elapsed < n_steps:
        step = min(stride, n_steps - elapsed)
        try:
            composite.run(step)
        except Exception as e:
            print(f"  seed={seed:02d}: composite stopped at {elapsed}: {str(e)[:80]}")
            break
        elapsed += step
        trajectory.append(_snapshot(composite.state, float(elapsed)))

    wall = time.time() - t0
    summary = _endpoint_summary(composite.state, seed, wall, n_steps)

    # Persist trajectory as JSON (compact key:list_of_values form for easy
    # pandas/matplotlib loading).
    cols: dict[str, list] = {"time": []}
    for path in LISTENER_PATHS:
        cols[".".join(path)] = []
    for snap in trajectory:
        cols["time"].append(snap.get("time"))
        for path in LISTENER_PATHS:
            cols[".".join(path)].append(snap.get(".".join(path)))

    (out_dir / "trajectory.json").write_text(json.dumps(cols))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  seed={seed:02d}: wall={wall:6.1f}s  ATP={summary.get('ATP[c]_count', '?')}  "
        f"dry_mass_fg={summary.get('dry_mass_fg', '?')}  "
        f"snapshots={len(trajectory)}"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=600)
    p.add_argument("--stride", type=int, default=5,
                   help="snapshot interval in model seconds (default: 5)")
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0 trajectory ensemble: N={args.n_seeds} × {args.n_steps} steps "
          f"(snapshot every {args.stride}s)")
    t0 = time.time()
    results = []
    for seed in range(args.n_seeds):
        try:
            results.append(run_one(seed, args.n_steps, args.stride))
        except Exception as e:
            print(f"  seed={seed:02d}: FAILED -- {type(e).__name__}: {e}")
            results.append({"seed": seed, "error": str(e), "type": type(e).__name__})
    total_wall = time.time() - t0

    successful = [r for r in results if "error" not in r]
    atp_counts = [r.get("ATP[c]_count") for r in successful if "ATP[c]_count" in r]
    mass_vals = [r.get("dry_mass_fg") for r in successful if "dry_mass_fg" in r]

    ensemble = {
        "n_seeds_requested": args.n_seeds,
        "n_seeds_successful": len(successful),
        "n_steps": args.n_steps,
        "stride": args.stride,
        "total_wall_seconds": round(total_wall, 2),
        "per_seed": results,
    }
    if atp_counts:
        a = np.array(atp_counts, dtype=float)
        ensemble["ATP[c]_count_stats"] = {
            "mean": float(a.mean()), "std": float(a.std()),
            "cv_pct": float(a.std() / a.mean() * 100) if a.mean() > 0 else None,
        }
    if mass_vals:
        m = np.array(mass_vals, dtype=float)
        ensemble["dry_mass_fg_stats"] = {
            "mean": float(m.mean()), "std": float(m.std()),
            "cv_pct": float(m.std() / m.mean() * 100) if m.mean() > 0 else None,
        }

    (OUT_ROOT / "summary.json").write_text(json.dumps(ensemble, indent=2))
    print()
    print(f"Done: {len(successful)}/{args.n_seeds} runs, total wall {total_wall/60:.1f} min")
    if "ATP[c]_count_stats" in ensemble:
        s = ensemble["ATP[c]_count_stats"]
        print(f"ATP[c] across seeds: mean={s['mean']:.2e} std={s['std']:.2e} CV={s['cv_pct']:.2f}%")


if __name__ == "__main__":
    main()
