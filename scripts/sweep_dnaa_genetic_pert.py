"""Sweep dnaA via sim_data.genetic_perturbations (Mechanism A — runtime overwrite).

This pins dnaA's per-TU promoter_init_probs to a fixed value via
``transcript_initiation.py:_rescale_initiation_probs``, which runs AFTER
the ppGpp basal_prob branch — so the perturbation actually lands at
runtime.

For each value V in --values:
  1. Hydrate sim_data from the shipped fixture (no ParCa rerun)
  2. Set ``sim_data.genetic_perturbations = {"TU00259[c]": V}``
  3. save_sim_input → cache
  4. Run N gens from the burned-in dill
  5. Measure mean DnaA monomer at chosen gen

V is interpreted by ``_rescale_initiation_probs`` as the TARGET sum of
promoter_init_probs across all dnaA promoter copies before normalization.
After normalization, dnaA's share of total init events ≈ V (since other
TUs sum to ~1.0).

Usage:
    .venv/bin/python scripts/sweep_dnaa_genetic_pert.py \\
        --values 1e-4,5e-4,1e-3,5e-3,1e-2 \\
        --generations 5 --measure-gen 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import polars as pl

from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)
from v2ecoli.core import save_sim_input

DNAA_MONOMER_IDX = 3861
DNAA_TU_ID = "TU00259[c]"
AVOGADRO = 6.022e23


def build_cache_with_pert(fixture: str, cache_dir: str, condition: str,
                          pert_value: float) -> None:
    state = load_parca_state(fixture)
    sim_data = hydrate_sim_data_from_state(state)
    sim_data.genetic_perturbations = {DNAA_TU_ID: pert_value}
    print(f"  → set sim_data.genetic_perturbations = "
          f"{{{DNAA_TU_ID!r}: {pert_value:.3e}}}")
    save_sim_input(sim_data, cache_dir, condition=condition)


def measure(exp_root: str, exp_id: str, gen: int) -> dict:
    agent_id = "0" * gen
    pat = (f"{exp_root}/history/experiment_id={exp_id}/"
           f"variant=0/lineage_seed=0/generation={gen}/agent_id={agent_id}/*.pq")
    files = sorted(glob.glob(pat), key=lambda p: int(p.split("/")[-1].split(".")[0]))
    if not files:
        return {"status": "missing", "gen": gen}
    df = pl.concat([pl.read_parquet(f, columns=[
        "global_time",
        "listeners__mass__volume",
        "listeners__monomer_counts",
        "listeners__rnap_data__rna_init_event",
    ]) for f in files]).sort("global_time")
    t = df["global_time"].to_numpy()
    dnaA = df["listeners__monomer_counts"].list.get(DNAA_MONOMER_IDX).to_numpy()
    inits = df["listeners__rnap_data__rna_init_event"].list.get(2778).fill_null(0).to_numpy()
    vol = df["listeners__mass__volume"].to_numpy()
    conc = dnaA / (vol * AVOGADRO * 1e-15) * 1e9
    dur = (t[-1] - t[0]) / 60.0
    return {
        "status":             "ok",
        "gen":                gen,
        "duration_min":       float(dur),
        "n_ticks":            int(len(t)),
        "dnaA_mean":          float(dnaA.mean()),
        "dnaA_max":           float(dnaA.max()),
        "dnaA_end":           float(dnaA[-1]),
        "dnaA_conc_mean_nM":  float(conc.mean()),
        "init_events_total":  int(inits.sum()),
        "init_rate_per_min":  float(inits.sum() / dur) if dur > 0 else 0.0,
    }


def run_one(value: float, generations: int, resume_dill: str,
            measure_gen: int) -> dict:
    label = f"pert={value:.3e}"
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
    cache_dir = f"out/cache_dnaaPert{value:.0e}".replace("+", "")
    out_dir   = f"out/dnaaPert{value:.0e}_parquet".replace("+", "")
    exp_id    = f"dnaaPert{value:.0e}".replace("+", "")
    exp_root  = f"{out_dir}/{exp_id}"
    t_start = time.time()

    # 1) build cache with perturbation
    for d in (cache_dir, out_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
    print(f"[{label}] build_cache with genetic_perturbations injection...")
    build_cache_with_pert("models/parca/parca_state.pkl.gz", cache_dir,
                          "succinate", value)

    # 2) run sim from burned-in dill
    cmd = [
        sys.executable, "scripts/extend_multigen_from_dill.py",
        "--cache-dir", cache_dir,
        "--out-dir", out_dir,
        "--experiment-id", exp_id,
        "--resume-dill", resume_dill,
        "--start-gen", "1",
        "--generations", str(generations),
        "--seed", "0",
    ]
    print(f"[{label}] running {generations} gens ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! sim failed:\n{r.stderr[-2000:]}")
        return {"value": value, "status": "sim_failed",
                "stderr": r.stderr[-500:]}

    # 3) measure across all gens
    measurements = []
    for g in range(1, generations + 1):
        m = measure(exp_root, exp_id, g)
        measurements.append(m)
        if m.get("status") == "ok":
            print(f"  gen {g}: τ={m['duration_min']:.1f} min  "
                  f"DnaA mean={m['dnaA_mean']:.0f} max={m['dnaA_max']:.0f}  "
                  f"({m['dnaA_conc_mean_nM']:.0f} nM)  "
                  f"init_rate={m['init_rate_per_min']:.3f}/min")

    target = measurements[measure_gen - 1] if measure_gen <= len(measurements) else None
    return {
        "value":         value,
        "status":        "ok",
        "wall_seconds":  time.time() - t_start,
        "measurements":  measurements,
        "target_gen":    measure_gen,
        "target_dnaA":   target["dnaA_mean"] if (target and target.get("status") == "ok") else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default="1e-4,5e-4,1e-3,5e-3,1e-2",
                    help="comma-separated genetic_perturbation values to sweep")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--measure-gen", type=int, default=3)
    ap.add_argument("--resume-dill",
                    default="out/steady_state_inputs/succinate_default_gen3_start.dill")
    ap.add_argument("--output", default="out/dnaa_pert_sweep_results.json")
    args = ap.parse_args()

    values = [float(v) for v in args.values.split(",") if v.strip()]
    print(f"Sweeping sim_data.genetic_perturbations[TU00259[c]] = {values}")
    print(f"Measuring at gen {args.measure_gen}, target band [300, 800]")

    results = []
    for v in values:
        r = run_one(v, args.generations, args.resume_dill, args.measure_gen)
        results.append(r)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 70}\n  SUMMARY — DnaA at gen {args.measure_gen}\n{'=' * 70}")
    print(f"{'pert value':>12}  {'mean DnaA':>10}  {'band [300-800]':>20}")
    for r in results:
        if r.get("status") != "ok" or r.get("target_dnaA") is None:
            print(f"{r['value']:>12.2e}  {'(failed)':>10}  {r.get('status')}")
            continue
        d = r["target_dnaA"]
        tag = ("✓ IN BAND" if 300 <= d <= 800
               else f"below by {300-d:.0f}" if d < 300
               else f"above by {d-800:.0f}")
        print(f"{r['value']:>12.2e}  {d:>10.0f}  {tag:>20}")


if __name__ == "__main__":
    main()
