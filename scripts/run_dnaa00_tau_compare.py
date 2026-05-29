"""Minimal standalone dnaa-00 baseline runner for the tau-gap A/B/C comparison.

Builds the dnaa_00_baseline_with_dnaa_readout composite against the
specified cache, runs N sim-seconds with per-minute snapshots, and
emits:

  - cell_mass / dry_mass / volume trajectory
  - DnaA monomer count (total)
  - DnaA-ATP fraction
  - number_of_oric (replication-init event)
  - chromosomes / forks
  - division time (first tick at which agents/0 disappears)

The point is to compare:
  A) out/cache-stage1-glycerol            (existing Round-3 baseline)
  B) out/cache-stage1-glycerol-tau150     (hand-tuned dry_mass_inc[minimal_glycerol] = 144 fg)

both against Haochen's declared target tau = 150 min cell cycle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np

from v2ecoli import build_composite


def _get(node, *path, default=None):
    cur = node
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
    return cur if cur is not None else default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                    help="ParCa cache to build the composite against")
    ap.add_argument("--duration", type=int, default=12000,
                    help="Sim seconds (default 12000 = 200 min)")
    ap.add_argument("--interval", type=int, default=60,
                    help="Snapshot interval in sim seconds (default 60)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True,
                    help="JSON file to write the run summary")
    args = ap.parse_args()

    t0 = time.time()
    composite = build_composite(
        "dnaa_00_baseline_with_dnaa_readout",
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    load_time = time.time() - t0

    snapshots: list[dict] = []
    division_time = None
    t_run = time.time()
    total = 0.0

    # Initial snapshot at t=0.
    def _snap(t: float) -> dict:
        cell = _get(composite.state, "agents", "0")
        if cell is None:
            return {"t": t, "agent_present": False}
        listeners = cell.get("listeners") or {}
        cycle = listeners.get("dnaA_cycle") or {}
        mass = listeners.get("mass") or {}
        replication = listeners.get("replication_data") or {}
        monomers = listeners.get("monomer_counts")
        # dnaA monomer index 3861 per the workspace convention.
        try:
            dnaa_count = int(monomers[3861]) if monomers is not None else None
        except (IndexError, TypeError):
            dnaa_count = None
        return {
            "t": t,
            "agent_present": True,
            "dnaa_count": dnaa_count,
            "atp_fraction": float(cycle.get("atp_fraction")) if cycle.get("atp_fraction") is not None else None,
            "cell_mass": float(mass.get("cell_mass")) if mass.get("cell_mass") is not None else None,
            "dry_mass": float(mass.get("dry_mass")) if mass.get("dry_mass") is not None else None,
            "expected_mass_fold_change": float(mass.get("expected_mass_fold_change")) if mass.get("expected_mass_fold_change") is not None else None,
            "n_oric": int(replication.get("number_of_oric")) if replication.get("number_of_oric") is not None else None,
        }

    snapshots.append({"t": 0, "phase": "initial", **_snap(0.0)})

    while total < args.duration:
        chunk = min(args.interval, args.duration - total)
        try:
            composite.run(chunk)
        except Exception as e:
            snapshots.append({"t": total, "phase": "error", "error": str(e)})
            break
        total += chunk
        s = _snap(total)
        if not s.get("agent_present"):
            division_time = total
            snapshots.append({"t": total, "phase": "divided", **s})
            break
        snapshots.append({"t": total, **s})

    wall_time = time.time() - t_run

    # Summaries from the pre-division window.
    pre_div = [s for s in snapshots if s.get("agent_present") and s.get("dnaa_count") is not None]
    dnaa_vals = [s["dnaa_count"] for s in pre_div]
    cm_vals = [s["cell_mass"] for s in pre_div if s.get("cell_mass") is not None]
    dm_vals = [s["dry_mass"] for s in pre_div if s.get("dry_mass") is not None]
    atp_vals = [s["atp_fraction"] for s in pre_div if s.get("atp_fraction") is not None]
    summary = {
        "n_snapshots": len(snapshots),
        "n_pre_division": len(pre_div),
        "dnaa_count": {
            "first": dnaa_vals[0] if dnaa_vals else None,
            "median": float(np.median(dnaa_vals)) if dnaa_vals else None,
            "max": int(max(dnaa_vals)) if dnaa_vals else None,
            "min": int(min(dnaa_vals)) if dnaa_vals else None,
        },
        "cell_mass": {
            "first": cm_vals[0] if cm_vals else None,
            "last": cm_vals[-1] if cm_vals else None,
            "delta": (cm_vals[-1] - cm_vals[0]) if len(cm_vals) >= 2 else None,
        },
        "dry_mass": {
            "first": dm_vals[0] if dm_vals else None,
            "last": dm_vals[-1] if dm_vals else None,
        },
        "atp_fraction_median": float(np.median(atp_vals)) if atp_vals else None,
    }

    result = {
        "cache_dir": args.cache_dir,
        "duration": args.duration,
        "interval": args.interval,
        "seed": args.seed,
        "load_time": load_time,
        "wall_time": wall_time,
        "sim_time_completed": total,
        "division_time_s": division_time,
        "division_time_min": (division_time / 60.0) if division_time else None,
        "summary": summary,
        "snapshots": snapshots,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {args.out}")
    print(f"  cache: {args.cache_dir}")
    print(f"  sim_time: {total}s  wall_time: {wall_time:.1f}s")
    if division_time is not None:
        print(f"  DIVIDED at t={division_time}s = {division_time/60:.1f} min")
    else:
        print(f"  no division within {args.duration}s = {args.duration/60:.1f} min")
    if dnaa_vals:
        print(f"  DnaA pre-division: median {summary['dnaa_count']['median']:.0f}, "
              f"range {summary['dnaa_count']['min']}-{summary['dnaa_count']['max']}")
    if cm_vals:
        print(f"  cell_mass: {cm_vals[0]:.0f} -> {cm_vals[-1]:.0f} fg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
