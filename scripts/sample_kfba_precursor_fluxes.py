"""Measure kFBA's per-second turnover of biomass precursors.

The PDMP composite drops kFBA's biomass-production objective, so the WCM's
downstream consumers (transcription, translation) deplete substrate pools
without replenishment — the W₂ comparison vs Phase-0 surfaces this as the
remaining biology gap (task #21).

This script samples the kFBA baseline composite once per simulated second
for `--duration` s and computes, for each precursor (21 AAs + 4 NTPs + 4
dNTPs), the average per-second NET INCREASE driven by kFBA's biomass +
homeostatic-target machinery. Those rates are the constants the
ref_growth_driver Step should add per tick to keep the PDMP composite's
precursor pools at kFBA-equivalent levels.

Output: JSON mapping {bulk_id: rate_per_s}. Saved by default to
.pbg/runs/kfba-precursor-fluxes.json.

Usage:
    .venv/bin/python scripts/sample_kfba_precursor_fluxes.py
        [--duration 600]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")


# Canonical precursor IDs (matches v2ecoli/steps/ref_growth_driver.py).
AA_BULK_IDS = (
    "L-ALPHA-ALANINE[c]", "ARG[c]", "ASN[c]", "L-ASPARTATE[c]", "CYS[c]",
    "GLT[c]", "GLN[c]", "GLY[c]", "HIS[c]", "ILE[c]", "LEU[c]", "LYS[c]",
    "MET[c]", "PHE[c]", "PRO[c]", "SER[c]", "THR[c]", "TRP[c]", "TYR[c]",
    "L-SELENOCYSTEINE[c]", "VAL[c]",
)
NTP_BULK_IDS = ("ATP[c]", "GTP[c]", "CTP[c]", "UTP[c]")
DNTP_BULK_IDS = ("DATP[c]", "DGTP[c]", "DCTP[c]", "TTP[c]")
# Water is sampled here too so the ref_growth_driver's consumption_matched
# mode can read its rate from data instead of a hardcoded WATER_RATE_PER_S
# in source — the value drifts with the live baseline trajectory, so it
# wants to live with the AAs / NTPs in this JSON.
WATER_BULK_IDS = ("WATER[c]",)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--sample-every", type=int, default=10,
                   help="Sample interval in sim-seconds (smaller = finer "
                        "delta resolution; runtime cost ~linear).")
    p.add_argument("--out",
                   default=".pbg/runs/kfba-precursor-fluxes.json")
    args = p.parse_args()

    from v2ecoli import build_composite

    precursor_ids = AA_BULK_IDS + NTP_BULK_IDS + DNTP_BULK_IDS + WATER_BULK_IDS

    print(f"Building kFBA baseline composite...")
    t0 = time.perf_counter()
    c = build_composite("baseline")
    print(f"  build: {time.perf_counter() - t0:.1f}s", flush=True)

    cell = c.state["agents"]["0"]
    bulk_ids = cell["bulk"]["id"]
    # Resolve precursor ids → bulk indices once.
    idx_lookup: dict[str, int] = {}
    for vid in precursor_ids:
        match = np.where(bulk_ids == vid)[0]
        if match.size:
            idx_lookup[vid] = int(match[0])
    print(f"  resolved {len(idx_lookup)} / {len(precursor_ids)} precursor ids",
          flush=True)

    # Initial counts.
    indices = np.asarray(list(idx_lookup.values()), dtype=np.int64)
    ids_in_order = list(idx_lookup.keys())
    initial_counts = cell["bulk"]["count"][indices].astype(np.int64)

    # Per-tick sampling.
    print(f"\nRunning kFBA baseline for {args.duration}s, "
          f"sampling every {args.sample_every}s...")
    samples_t: list[int] = [0]
    samples_counts: list[np.ndarray] = [initial_counts.copy()]
    t_run = time.perf_counter()
    sim_t = 0
    while sim_t < args.duration:
        chunk = min(args.sample_every, args.duration - sim_t)
        c.run(chunk)
        sim_t += chunk
        samples_t.append(sim_t)
        samples_counts.append(cell["bulk"]["count"][indices].copy().astype(np.int64))
    print(f"  run wall: {time.perf_counter() - t_run:.1f}s", flush=True)

    samples_arr = np.asarray(samples_counts, dtype=np.int64)
    times = np.asarray(samples_t, dtype=np.int64)
    # Net flux per sim-second: linear fit (count vs t), slope = rate/s.
    rates_per_s = {}
    for j, vid in enumerate(ids_in_order):
        ts = times.astype(np.float64)
        cs = samples_arr[:, j].astype(np.float64)
        # numpy.polyfit deg=1 → [slope, intercept]
        slope, intercept = np.polyfit(ts, cs, deg=1)
        rates_per_s[vid] = float(slope)

    # Summary stats.
    rates_summary = {
        "duration_s": int(args.duration),
        "sample_every_s": int(args.sample_every),
        "n_precursors": len(ids_in_order),
        "initial_counts": {
            vid: int(samples_arr[0, j]) for j, vid in enumerate(ids_in_order)
        },
        "final_counts": {
            vid: int(samples_arr[-1, j]) for j, vid in enumerate(ids_in_order)
        },
        "net_rate_per_s": rates_per_s,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rates_summary, indent=2))
    print(f"\nWrote {out_path}\n")

    print("Top 10 production rates (count/s):")
    sorted_rates = sorted(rates_per_s.items(), key=lambda kv: -kv[1])
    for vid, r in sorted_rates[:10]:
        print(f"  {vid:>25s}: {r:>12,.0f}")
    print("Bottom 5 (largest negative / consumption-net):")
    for vid, r in sorted_rates[-5:]:
        print(f"  {vid:>25s}: {r:>12,.0f}")


if __name__ == "__main__":
    main()
