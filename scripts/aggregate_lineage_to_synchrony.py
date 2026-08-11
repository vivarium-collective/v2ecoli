"""Aggregate all lineage_seed*.json in a directory into synchrony_summary.json
with the FULL pair schema (matching the older 0%/25% v2 files that include
e2_after, first_vol/second_vol, e1_before/after, etc.).

Fixes two issues in the raw extract pipeline:
  1. Each call to extract_initiation_lineage.py overwrites synchrony_summary.json
     with only the roots passed on that invocation. This aggregator re-scans
     all per-seed JSONs and rebuilds the aggregate.
  2. The current mother_cycle_deltas drops cell_volume + oriC-after fields.
     This aggregator recomputes pairs from init events and emits the full
     schema so downstream filters (strict, volume overlays) work.

Usage:
  aggregate_lineage_to_synchrony.py <dir_with_lineage_seed*.json>
"""
import argparse
import glob
import json
import os
import re
import numpy as np


def pairs_from_lineage(lineage):
    seed = lineage["seed"]
    out = []
    for g in lineage["generations"]:
        inits = g.get("initiations", [])
        # Mother-cycle events: firing while already at ≥ 2 oriC.
        mother_events = [e for e in inits if e.get("oric_before", 0) >= 2]
        if len(mother_events) < 2:
            continue
        e1, e2 = mother_events[0], mother_events[1]
        out.append({
            "seed": seed,
            "gen": g["gen"],
            "delta_t_s": e2["time_s_from_birth"] - e1["time_s_from_birth"],
            "first_time_s": e1["time_s_from_birth"],
            "second_time_s": e2["time_s_from_birth"],
            "first_mass": e1.get("cell_mass"),
            "second_mass": e2.get("cell_mass"),
            "first_vol": e1.get("cell_volume"),
            "second_vol": e2.get("cell_volume"),
            "tau_min": g["tau_min"],
            "birth_oric": g["birth_oric"],
            "n_mother_initiations": len(mother_events),
            "n_all_initiations": len(inits),
            "e1_before": e1.get("oric_before"),
            "e1_after":  e1.get("oric_after"),
            "e2_before": e2.get("oric_before"),
            "e2_after":  e2.get("oric_after"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="dir containing lineage_seed*.json files")
    args = ap.parse_args()

    lineage_files = sorted(glob.glob(os.path.join(args.dir, "lineage_seed*.json")),
                           key=lambda p: int(re.search(r"seed(\d+)", p).group(1)))
    if not lineage_files:
        raise SystemExit(f"no lineage_seed*.json in {args.dir}")

    all_pairs = []
    all_taus = []
    n_lineages = 0
    for f in lineage_files:
        lineage = json.load(open(f))
        n_lineages += 1
        all_pairs.extend(pairs_from_lineage(lineage))
        all_taus.extend([g["tau_min"] for g in lineage["generations"]])

    dts = np.array([p["delta_t_s"] for p in all_pairs], dtype=float)
    tau_mean_min = float(np.mean(all_taus)) if all_taus else float("nan")
    tau_mean_s = tau_mean_min * 60.0
    cv_intrinsic = (float(np.std(dts, ddof=1) / tau_mean_s)
                    if len(dts) > 1 and tau_mean_s else float("nan"))

    summary = {
        "n_seeds": n_lineages,
        "n_pairs": len(all_pairs),
        "n_gens_total": len(all_taus),
        "tau_mean_min": tau_mean_min,
        "delta_t_stats_s": ({
            "mean": float(np.mean(dts)),
            "std": float(np.std(dts, ddof=1)) if len(dts) > 1 else float("nan"),
            "min": float(np.min(dts)),
            "max": float(np.max(dts)),
            "median": float(np.median(dts)),
        } if len(dts) else {}),
        "intrinsic_CV": cv_intrinsic,
        "pairs": all_pairs,
    }
    out_json = os.path.join(args.dir, "synchrony_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    strict = [p for p in all_pairs
              if p.get("birth_oric") == 2 and p.get("e2_after", 0) >= 4]
    print(f"aggregated {n_lineages} seeds  |  total gens {len(all_taus)}"
          f"  mother pairs {len(all_pairs)}  strict {len(strict)}"
          f"  mean τ {tau_mean_min:.1f} min")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
