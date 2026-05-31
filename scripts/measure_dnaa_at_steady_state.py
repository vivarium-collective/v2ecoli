"""Measure mean DnaA count over a converged generation.

Reads parquet output from a multigen run and reports mean DnaA monomer
count + concentration + RNA count for a chosen generation (default: gen 3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import polars as pl

DNAA_MONOMER_IDX = 3861
DNAA_MRNA_IDX = 2700
AVOGADRO = 6.022e23


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True, help="e.g. out/dnaaFC30_run/dnaaFC30_run")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--gen", type=int, default=3, help="which gen to measure (default 3)")
    args = ap.parse_args()

    agent_id = "0" * args.gen
    pat = (f"{args.exp_root}/history/experiment_id={args.experiment_id}/"
           f"variant=0/lineage_seed=0/generation={args.gen}/agent_id={agent_id}/*.pq")
    files = sorted(glob.glob(pat), key=lambda p: int(p.split("/")[-1].split(".")[0]))
    if not files:
        print(f"ERROR: no parquet files found at {pat}")
        sys.exit(1)

    df = pl.concat([pl.read_parquet(f, columns=[
        "global_time",
        "listeners__mass__volume",
        "listeners__monomer_counts",
        "listeners__rna_counts__mRNA_counts",
        "listeners__rnap_data__rna_init_event",
    ]) for f in files]).sort("global_time")

    t = df["global_time"].to_numpy()
    dnaA_mon = df["listeners__monomer_counts"].list.get(DNAA_MONOMER_IDX).to_numpy()
    dnaA_mrna = df["listeners__rna_counts__mRNA_counts"].list.get(DNAA_MRNA_IDX).to_numpy().astype(float)
    init_events = df["listeners__rnap_data__rna_init_event"].list.get(DNAA_MRNA_IDX).fill_null(0).to_numpy()
    vol_fL = df["listeners__mass__volume"].to_numpy()

    dur_min = (t[-1] - t[0]) / 60
    conc = dnaA_mon / (vol_fL * AVOGADRO * 1e-15) * 1e9   # nM

    print(f"=== DnaA measurement, gen {args.gen} ===")
    print(f"  duration:       {dur_min:.1f} min, {len(t)} ticks")
    print(f"  DnaA monomer:   mean={dnaA_mon.mean():.1f}  min={dnaA_mon.min():.0f}  "
          f"max={dnaA_mon.max():.0f}  end={dnaA_mon[-1]:.0f}")
    print(f"  DnaA conc (nM): mean={conc.mean():.1f}  min={conc.min():.1f}  max={conc.max():.1f}")
    print(f"  dnaA mRNA:      mean={dnaA_mrna.mean():.3f}  max={dnaA_mrna.max():.0f}")
    print(f"  init events:    total={int(init_events.sum())}  rate={init_events.sum()/dur_min:.3f}/min")

    band_lo, band_hi = 300, 800
    if band_lo <= dnaA_mon.mean() <= band_hi:
        print(f"  ✓ mean DnaA in PDF target band [{band_lo}, {band_hi}]")
    elif dnaA_mon.mean() < band_lo:
        ratio = band_lo / dnaA_mon.mean()
        print(f"  ✗ below band by {band_lo - dnaA_mon.mean():.0f}; "
              f"need ~{ratio:.1f}× more")
    else:
        ratio = dnaA_mon.mean() / band_hi
        print(f"  ✗ above band by {dnaA_mon.mean() - band_hi:.0f}; "
              f"need ~{ratio:.1f}× less")


if __name__ == "__main__":
    main()
