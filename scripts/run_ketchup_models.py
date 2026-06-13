#!/usr/bin/env python3
"""Produce the live model artifacts for the ketchup-exchange-comparison study.

This regenerates the JSON inputs that ``scripts/ketchup_baseline_report_cards.py``
grades + renders. Two of the five models are run live here; the other two are
pre-cached real fits (see provenance below).

Subcommands (each needs a DIFFERENT python env):

  fdh      Run the KETCHUP dynamic (time-series) FDH fit and write
           ``fdh_dynamic.json`` (NADH(t) data + fit, SSE, status).
           ENV: the pbg-ketchup micromamba env (has IPOPT + cyipopt/pyomo) with
           its bin on PATH so pyomo finds the ipopt executable, e.g.
             KENV=/opt/homebrew/Cellar/micromamba/2.5.0_4/envs/pbg-ketchup
             PATH="$KENV/bin:$PATH" "$KENV/bin/python" \
                 scripts/run_ketchup_models.py fdh

  bridge   Run the ``millard_fba_bridge_harness`` composite (real whole-cell
           tFBA + Millard 2017 ODE flux source) for N ticks off the ParCa cache
           and write ``fba_bridge_exchange.json`` (time-averaged exchange vector
           from listeners.fba_results.external_exchange_fluxes).
           ENV: the v2ecoli workspace venv:  .venv/bin/python

Pre-cached real fits (NOT regenerated here — committed in OUT):
  * ``ketchup_exchange.json`` — KETCHUP k-ecoli74 / k-ecoli307 steady-state
    fitted exchange fluxes (pbg-ketchup KetchupEstimator, IPOPT; boundary
    reactions mapped to EcoCyc IDs, glucose-normalized to -100).
  * ``millard_exchange.json`` — Millard 2017 central-carbon ODE steady-state
    glucose + acetate exchange.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "report_cards" / "ketchup_vs_baseline"

SHARED = ["GLC[p]", "OXYGEN-MOLECULE[p]", "CARBON-DIOXIDE[p]", "ACET[p]",
          "AMMONIUM[c]", "SULFATE[p]"]


def run_fdh(out_dir: str = "") -> None:
    """KETCHUP dynamic FDH fit -> fdh_dynamic.json. Run in the pbg-ketchup env."""
    from process_bigraph import allocate_core
    from pbg_ketchup import KetchupDynamicEstimator

    work = Path(out_dir or "/tmp/fdh_run")
    work.mkdir(parents=True, exist_ok=True)
    opt = work / "ipopt.opt"
    opt.write_text("tol 0.001\nmax_iter 300\nmax_cpu_time 120\n"
                   "print_user_options no\n")
    step = KetchupDynamicEstimator(
        config={"model_name": "FDH", "solver_options": str(opt),
                "output_dir": str(work)},
        core=allocate_core())
    cwd = os.getcwd()
    os.chdir(work)
    try:
        r = step.update({"seed": 0})
    finally:
        os.chdir(cwd)
    keep = ("nadh_time", "nadh_fit", "data_time", "data_nadh", "sse", "status",
            "n_experiments", "n_parameters", "solve_time")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({k: r[k] for k in keep}, open(OUT / "fdh_dynamic.json", "w"))
    print(f"fdh: status={r['status']} sse={r['sse']:.3g} "
          f"n_exp={r['n_experiments']} n_par={r['n_parameters']} "
          f"-> {OUT / 'fdh_dynamic.json'}")


def run_bridge(ticks: int = 60, burn: int = 10,
               cache_dir: str = "out/cache", seed: int = 0) -> None:
    """millard_fba_bridge_harness exchange vector -> fba_bridge_exchange.json.

    Run in the v2ecoli venv. Builds the real whole-cell composite (~55 procs +
    Millard ODE flux source + fba-flux-coupler), runs `ticks` 1-second steps,
    and time-averages listeners.fba_results.external_exchange_fluxes over the
    post-burn-in window. The 87-vector is in the same order as the baseline
    reference fixture's flux_ids, so we index it directly.
    """
    warnings.filterwarnings("ignore")
    import numpy as np
    from process_bigraph import Composite
    from pbg_superpowers.composite_generator import _REGISTRY, build_generator
    from v2ecoli.core import build_core

    fx = json.load(open(
        REPO / "tests/fixtures/population_phenotype_basal_reference.json"))
    flux_ids = fx["axes"]["fluxes.exchange"]["criterion"]["flux_ids"]

    core = build_core()
    entry = [e for e in _REGISTRY.values()
             if e.name == "millard_fba_bridge_harness"][0]
    doc = build_generator(
        entry, overrides={"cache_dir": cache_dir, "seed": seed}, core=core)
    comp = Composite(doc, core=core)

    rows, dm = [], []
    t0 = time.time()
    for _ in range(ticks):
        comp.run(1)
        ag = comp.state["agents"]["0"]
        ex = ((ag.get("listeners") or {}).get("fba_results") or {}).get(
            "external_exchange_fluxes")
        if ex is not None and len(ex) == len(flux_ids):
            rows.append(np.asarray(ex, float))
        m = ((ag.get("listeners") or {}).get("mass") or {}).get("dry_mass")
        dm.append(float(getattr(m, "magnitude", m)) if m is not None else float("nan"))
    wall = time.time() - t0

    arr = np.array(rows)
    avg = arr[burn:].mean(axis=0)
    exch = {fid: float(avg[i]) for i, fid in enumerate(flux_ids)}
    shared = {s: exch[s] for s in SHARED}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"exchange_full": exch, "shared": shared,
               "n_ticks_avg": len(rows) - burn, "ticks": len(rows),
               "dry_mass_first": dm[0], "dry_mass_last": dm[-1]},
              open(OUT / "fba_bridge_exchange.json", "w"), indent=2)
    print(f"bridge: {len(rows)} ticks in {wall:.1f}s, dry_mass "
          f"{dm[0]:.1f}->{dm[-1]:.1f} fg -> {OUT / 'fba_bridge_exchange.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fdh", help="run the KETCHUP dynamic FDH fit (pbg-ketchup env)")
    pb = sub.add_parser("bridge", help="run the FBA-bridge harness (v2ecoli venv)")
    pb.add_argument("--ticks", type=int, default=60)
    pb.add_argument("--burn", type=int, default=10)
    pb.add_argument("--cache-dir", default="out/cache")
    pb.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "fdh":
        run_fdh()
    elif args.cmd == "bridge":
        run_bridge(ticks=args.ticks, burn=args.burn,
                   cache_dir=args.cache_dir, seed=args.seed)
