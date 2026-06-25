#!/usr/bin/env python3
"""GUR-titration sweep runner for the overflow-acetate-vs-growth study.

Sweeps the aerobic glucose-uptake CAP (the FBA import bound, normally 20.0
mmol/gDCW/h; knob = env var ``V2ECOLI_GLC_UPTAKE_CAP_AEROBIC``, see
external_state.py) and, at each cap, runs the *basal-card ensemble*
(``v2ecoli-workflow`` on the population_phenotype_basal config, scaled down,
run-to-division, Ray across seeds) — then reads out the EMERGENT phenotype with
the SAME reviewed extractors the basal card uses:

  * growth rate μ ← analysis_runner.build_cell_records (per-cell division_time,
    aggregated across cells; μ = ln2 / mean doubling time)
  * exchange phenotype ← card_vectors.extract_vectors (ensemble-mean
    external_exchange_fluxes; acetate / glucose / CO2 / any secreted byproduct)

and derives two extra diagnostics (cheap, from those numbers):

  * biomass yield  Yxs = μ / (|q_glc| · M_glc)   [gDW / g glucose]
  * carbon balance closure — C_in (glucose·6) vs C_out (CO2 + secreted
    C-byproducts) + implied biomass-C. The ASSUMPTION-FREE part: if
    C_out_exchanges > C_in, carbon is created → hard violation (no biomass
    C-content needed). A softer consistency check compares the implied
    biomass-C to μ·(C_content) using a flagged literature C-content.

This is the model analogue of Basan 2015's titratable-PtsG experiment (set the
uptake knob → growth emerges → measure what's secreted). NOTE: the cap only
titrates GUR below the FBA's unconstrained glucose optimum (~5.4 mmol/gDCW/h);
above that it goes slack (replicates at the natural upper bound).

Two modes:
  run (default)  loop caps → run ensemble per cap → extract → JSON.
  --from-sweep   extract from an existing sweep dir (validation / baseline), no run.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
M_GLC = 0.180156  # g/mmol glucose
M_C = 0.012011    # g/mmol carbon
GLC, O2, CO2, ACET = "GLC[p]", "OXYGEN-MOLECULE[p]", "CARBON-DIOXIDE[p]", "ACET[p]"

# Carbon atoms per exchange molecule (the C-containing ones that can be active on
# glucose minimal media). Anything secreted/taken up above floor that is NOT here
# is reported as `carbon_uncounted` so we never silently miss carbon.
CARBON_COUNT = {
    "GLC[p]": 6, "CARBON-DIOXIDE[p]": 1, "ACET[p]": 2, "D-LACTATE[p]": 3,
    "ETOH[p]": 2, "FORMATE[p]": 1, "SUC[p]": 4, "PYRUVATE[p]": 3, "PYR[p]": 3,
    "GLYCOLLATE[c]": 2, "GLYCOLALDEHYDE[c]": 2, "GLC-D-LACTONE[c]": 6,
    "METOH[p]": 1, "IMIDAZOLE-PYRUVATE[c]": 6,
}


def _flux_ids() -> list:
    fx = json.load(open(REPO / "tests/fixtures/population_phenotype_basal_reference.json"))
    return fx["axes"]["fluxes.exchange"]["criterion"]["flux_ids"]


def extract_cap(sweep_dir: str, gen_lb: int, floor: float, c_content: float) -> dict:
    """Pull μ + exchange phenotype + yield + carbon balance from one sweep dir."""
    import numpy as np
    from v2ecoli.workflow.analysis_runner import build_cell_records
    from v2ecoli.library.card_vectors import extract_vectors

    # --- growth rate from per-cell division_time (the basal-card physiology) ---
    recs = list(build_cell_records(sweep_dir).values())
    div_times = [r["division_time"] for r in recs
                 if r.get("divided") and r.get("generation", 0) >= gen_lb
                 and r.get("division_time", 0) > 0]
    n_div = len(div_times)
    doubling_s = float(np.mean(div_times)) if div_times else None
    mu = (math.log(2) / (doubling_s / 3600.0)) if doubling_s else None  # 1/h

    # --- exchange phenotype (ensemble-mean external_exchange_fluxes) ---
    flux_ids = _flux_ids()
    vec = (extract_vectors(sweep_dir, gen_lb).get("fluxes") or {}).get("exchange")
    exch = {}
    if vec and vec.get("vector") and len(vec["vector"]) == len(flux_ids):
        exch = {fid: float(v) for fid, v in zip(flux_ids, vec["vector"])}
    glucose, co2, acet = exch.get(GLC), exch.get(CO2), exch.get(ACET)
    secreted = {k: v for k, v in sorted(exch.items(), key=lambda kv: -kv[1]) if v > floor}

    # --- biomass yield (gDW / g glucose) ---
    q_glc = abs(glucose) if glucose else None
    yxs = (mu / (q_glc * M_GLC)) if (mu and q_glc) else None

    # --- carbon balance (mmol C / gDCW / h) ---
    c_in = c_out = 0.0
    uncounted = []
    for fid, v in exch.items():
        if abs(v) <= floor:
            continue
        nc = CARBON_COUNT.get(fid)
        if nc is None:
            # only flag C-relevant misses: positive (secreted) or a non-trivial uptake
            if v > floor or (v < -floor and fid not in (O2, "AMMONIUM[c]", "SULFATE[p]",
                             "Pi[p]", "PROTON[p]", "WATER[p]", "OXYGEN-MOLECULE[p]")):
                uncounted.append(fid)
            continue
        if v < 0:
            c_in += -v * nc       # uptake carbon in
        else:
            c_out += v * nc       # secreted carbon out
    implied_biomass_c = c_in - c_out          # carbon that must go to biomass
    # softer consistency (uses the flagged literature C_content)
    expected_biomass_c = (mu * c_content / M_C) if mu else None  # mmol C/gDCW/h
    closure = None
    if expected_biomass_c is not None and c_in > 0:
        # fraction of C accounted: (C_out + expected biomass C) / C_in ; ≈1 if balanced
        closure = (c_out + expected_biomass_c) / c_in
    return {
        "growth_rate_per_h": mu, "doubling_min": (doubling_s / 60.0) if doubling_s else None,
        "n_cells": len(recs), "n_divided": n_div,
        "glucose": glucose, "oxygen": exch.get(O2), "co2": co2, "acetate": acet,
        "yield_gDW_per_g_glc": yxs,
        "carbon": {
            "c_in_mmolC": c_in, "c_out_exchange_mmolC": c_out,
            "implied_biomass_C_mmolC": implied_biomass_c,
            "violation_C_created": implied_biomass_c < -1e-9,  # ASSUMPTION-FREE flag
            "expected_biomass_C_mmolC_litC": expected_biomass_c,
            "closure_fraction_litC": closure, "c_content_gC_per_gDW": c_content,
            "uncounted_C_exchanges": uncounted,
        },
        "secreted": secreted, "exchange_full": exch,
    }


def run_cap(cap: float, n_seeds: int, generations: int, base_cfg: dict,
            work: Path, cache_dir: str) -> str:
    """Run the scaled basal ensemble at one cap; return the sweep dir.

    Config-driven cap (replaces the old V2ECOLI_GLC_UPTAKE_CAP_AEROBIC env var,
    which only reached generation 0): a single-value `variants` block targets
    the ExchangeData step's `glc_uptake_cap_aerobic` config field. meta_composite
    threads it into each LineageProcess as a `config_overrides` entry, and
    baseline() re-applies that override on EVERY generation's rebuild — so the
    cap survives the division->daughter rebuild. `skip_baseline` keeps only the
    capped variant (no extra uncapped baseline branch).
    """
    cfg = dict(base_cfg)
    experiment_id = f"gur_cap_{str(cap).replace('.', 'p')}"
    # Pin out_dir per cap: run.py uses setdefault, so the inherited base_cfg
    # out_dir would otherwise win and land parquet at out/population_phenotype_basal
    # (the bug the old extractor tripped on). Own dir per cap == known read path.
    sweep_dir = work / experiment_id
    cfg.update({
        "experiment_id": experiment_id,
        "out_dir": str(sweep_dir),
        "n_init_sims": n_seeds, "generations": generations,
        "emitter": "parquet",  # build_cell_records / extract_vectors read parquet
        "cache_dir": cache_dir,
        "skip_baseline": True,
        "variants": {
            "glc_cap": {
                "target": "exchange_data.glc_uptake_cap_aerobic",
                "value": [cap],
            }
        },
    })
    cfg_path = work / f"{experiment_id}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    wf = str(REPO / ".venv/bin/v2ecoli-workflow")
    print(f"  [cap={cap}] running ensemble ({n_seeds}×{generations}) …", flush=True)
    subprocess.run([wf, "--config", str(cfg_path), "--out", str(sweep_dir)],
                   check=True)
    return str(sweep_dir)


def main():
    ap = argparse.ArgumentParser(description="GUR-titration sweep (baseline arm).")
    ap.add_argument("--caps", default="1,2,3,4,5,6",
                    help="comma-separated aerobic glucose-uptake caps (mmol/gDCW/h)")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--gen-lb", type=int, default=1, help="generation_lower_bound (burn-in)")
    ap.add_argument("--floor", type=float, default=1e-3, help="flux floor (mmol/gDCW/h)")
    ap.add_argument("--c-content", type=float, default=0.45,
                    help="biomass carbon content (gC/gDW) — FLAGGED literature fallback")
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--out", default=str(HERE / "gur_titration.json"))
    ap.add_argument("--from-sweep", default=None,
                    help="extract from an existing sweep dir (validation); no run")
    ap.add_argument("--keep-sweeps", action="store_true",
                    help="keep per-cap parquet sweeps (default: delete after extract)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.from_sweep:
        r = extract_cap(args.from_sweep, args.gen_lb, args.floor, args.c_content)
        print(json.dumps(r, indent=2, default=str))
        return

    base_cfg = json.load(open(REPO / "v2ecoli/configs/population_phenotype_basal.json"))
    caps = [float(c) for c in args.caps.split(",") if c.strip()]
    work = Path(tempfile.mkdtemp(prefix="gur_titration_", dir=str(REPO / "out")))
    rows = []
    for cap in caps:
        sweep = run_cap(cap, args.n_seeds, args.generations, base_cfg, work, args.cache_dir)
        r = extract_cap(sweep, args.gen_lb, args.floor, args.c_content)
        r["cap"] = cap
        rows.append(r)
        c = r["carbon"]
        print(f"cap={cap:>4} | μ={r['growth_rate_per_h']} (Td={r['doubling_min']} min, "
              f"{r['n_divided']} div) | GLC={r['glucose']} ACET={r['acetate']} | "
              f"Yxs={r['yield_gDW_per_g_glc']} | C_in={c['c_in_mmolC']:.1f} "
              f"C_out={c['c_out_exchange_mmolC']:.1f} implied_bioC={c['implied_biomass_C_mmolC']:.1f} "
              f"closure={c['closure_fraction_litC']} violation={c['violation_C_created']}", flush=True)
        if not args.keep_sweeps:
            shutil.rmtree(sweep, ignore_errors=True)

    out = {"arm": "baseline", "knob": "exchange_data.glc_uptake_cap_aerobic",
           "n_seeds": args.n_seeds, "generations": args.generations,
           "gen_lb": args.gen_lb, "c_content_gC_per_gDW": args.c_content,
           "units": {"growth_rate": "1/h", "fluxes": "mmol/gDCW/h (neg=uptake, pos=secretion)",
                     "yield": "gDW/g glucose", "carbon": "mmol C/gDCW/h"},
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    if not args.keep_sweeps:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {len(rows)} points -> {args.out}")


if __name__ == "__main__":
    main()
