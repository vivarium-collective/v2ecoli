"""Analyze a dnaa-1 Mechanism-A re-run for in-band DnaA oscillation.

Rashmi 2026-05-31: at V=2e-3 the DnaA pool's cycle-MEAN is in [300,800] but the
pre-division PEAK overshoots (~1000-1022). She asked to lower V "a little" so the
whole oscillation stays within the band. This computes, per generation, the DnaA
monomer trough / cycle-mean / peak and flags whether the FULL oscillation (not
just the mean) lies in [300, 800], plus the dnaA mRNA initiation rate.

Usage:
    .venv/bin/python scripts/analyze_dnaa1_band.py --exp-id dnaa1-mechA-1.5e-3-7gen-2026-05-31
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import polars as pl

DNAA_MONOMER_IDX = 3861
DNAA_CISTRON_IDX = 227   # rnap_data cistron_ids[227] = EG10235_RNA (dnaA)
BAND_LOW, BAND_HIGH = 300.0, 800.0


def _gen_df(run_dir: str, exp_id: str, gen: int) -> pl.DataFrame | None:
    aid = "0" * gen
    pat = (f"{run_dir}/history/experiment_id={exp_id}/variant=0/lineage_seed=0/"
           f"generation={gen}/agent_id={aid}/*.pq")
    files = sorted(glob.glob(pat),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not files:
        return None
    cols = ["global_time", "listeners__mass__cell_mass",
            "listeners__monomer_counts"]
    # init-event column name varies; include if present.
    sample = pl.read_parquet(files[0])
    init_col = next((c for c in sample.columns
                     if "rna_init_event_per_cistron" in c
                     or c == "listeners__rnap_data__rna_init_event"), None)
    if init_col:
        cols.append(init_col)
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files]).sort(
        "global_time")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--run-dir", default=None,
                    help="default: studies/dnaa-1-expression-dynamics/"
                         "parquet-runs/<exp-id>")
    ap.add_argument("--max-gens", type=int, default=7)
    ap.add_argument("--steady-from", type=int, default=3)
    args = ap.parse_args()
    run_dir = args.run_dir or (
        f"studies/dnaa-1-expression-dynamics/parquet-runs/{args.exp_id}")

    print(f"run: {run_dir}")
    print(f"band: [{BAND_LOW:.0f}, {BAND_HIGH:.0f}]  (whole oscillation must fit)")
    rows = []
    for g in range(1, args.max_gens + 1):
        df = _gen_df(run_dir, args.exp_id, g)
        if df is None:
            continue
        dnaa = (df["listeners__monomer_counts"].list.get(DNAA_MONOMER_IDX)
                .to_numpy())
        t = df["global_time"].to_numpy()
        dur = (t[-1] - t[0]) / 60.0 if len(t) > 1 else 0.0
        init_col = next((c for c in df.columns if "rna_init_event" in c), None)
        rate = None
        if init_col and dur > 0:
            inits = (df[init_col].list.get(DNAA_CISTRON_IDX).fill_null(0)
                     .to_numpy())
            rate = float(inits.sum() / dur)
        rows.append({
            "gen": g, "trough": float(dnaa.min()), "mean": float(dnaa.mean()),
            "peak": float(dnaa.max()), "dur": dur, "rate": rate,
        })
        rstr = f"{rate:.2f}/min" if rate is not None else "n/a"
        print(f"  gen {g}: trough={dnaa.min():4.0f}  mean={dnaa.mean():4.0f}  "
              f"peak={dnaa.max():4.0f}  τ={dur:4.1f}min  initrate={rstr}")

    steady = [r for r in rows if r["gen"] >= args.steady_from]
    if not steady:
        print("no steady-state generations found"); return 1
    troughs = np.array([r["trough"] for r in steady])
    peaks = np.array([r["peak"] for r in steady])
    means = np.array([r["mean"] for r in steady])
    rates = [r["rate"] for r in steady if r["rate"] is not None]
    whole_in_band = bool((troughs >= BAND_LOW).all() and (peaks <= BAND_HIGH).all())
    mean_in_band = bool((means >= BAND_LOW).all() and (means <= BAND_HIGH).all())
    cv_mean = float(means.std() / means.mean()) if means.mean() else 1.0
    print(f"\nSteady state (gens {args.steady_from}+):")
    print(f"  trough range: {troughs.min():.0f}–{troughs.max():.0f}  "
          f"(floor {BAND_LOW:.0f})")
    print(f"  peak range:   {peaks.min():.0f}–{peaks.max():.0f}  "
          f"(ceiling {BAND_HIGH:.0f})")
    print(f"  cycle-mean:   {means.min():.0f}–{means.max():.0f}  "
          f"(cross-gen CV {cv_mean*100:.1f}%)")
    if rates:
        print(f"  init rate:    median {np.median(rates):.2f}/min  "
              f"mean {np.mean(rates):.2f}/min  (target ≈1/min)")
    print(f"\n  WHOLE-OSCILLATION in band [{BAND_LOW:.0f},{BAND_HIGH:.0f}]: "
          f"{'YES ✓' if whole_in_band else 'NO'}")
    print(f"  cycle-MEAN in band: {'YES' if mean_in_band else 'NO'}")
    return 0 if whole_in_band else 2


if __name__ == "__main__":
    raise SystemExit(main())
