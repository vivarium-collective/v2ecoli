#!/usr/bin/env python
"""Score a dnaa multigen run for ONCE-PER-CELL-CYCLE oriC-low saturation.

Reads a run out-dir (parquet history) and computes, per generation:
  - oriC-low occupancy = oriC_low_bound_atp / (oriC_low_bound_atp + oriC_low_free)
  - n_saturation_events: rising-edge crossings into occ >= SAT (a "fill" event)
  - frac_saturated: fraction of the gen at occ >= SAT
  - max_oric / clean (oriC stays in {1,2}; reaching >=4 = overlapping re-init)
  - divided (did the gen end in a division)

Target ("fills once, fires once, resets"): every STEADY-state generation (>= STEADY_GEN)
has exactly ONE saturation event, oriC max == 2, and divides. Score = sum of squared
deviations from 1 event/gen over steady gens, + penalties for re-init (oriC>2) and
missed divisions. Lower is better; 0.0 is ideal.

Usage:
  score_dnaa_saturation.py <out-dir> [<out-dir> ...] [--sat 0.9] [--steady-gen 4] [--json]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import polars as pl

RD = "listeners__replication_data__"
SAT_DEFAULT = 6      # oriC_low_bound_atp fire threshold (the mechanistic-low trigger)
STEADY_GEN_DEFAULT = 4


def _gen_shards(out_dir):
    """Map generation -> sorted list of (step, path) for the followed lineage (agent_id=0*)."""
    fs = glob.glob(os.path.join(out_dir, "**", "history", "**", "*.pq"), recursive=True)
    gens = {}
    for f in fs:
        if "/agent_id=0" not in f:  # follow the tracked daughter lineage
            continue
        gm = [p for p in f.split("/") if p.startswith("generation=")]
        sm = os.path.basename(f)[:-3]
        if not gm or not sm.isdigit():
            continue
        g = int(gm[0].split("=")[1])
        gens.setdefault(g, []).append((int(sm), f))
    for g in gens:
        gens[g].sort()
    return gens


def _read_cols(paths, cols):
    """Read `cols` from many parquet shards whose schemas may differ (some carry
    extra listener columns). Polars errors on a column-subset read across a
    schema union, so read per-file and concatenate the needed columns."""
    frames = []
    for p in paths:
        avail = pl.read_parquet_schema(p)
        sel = [c for c in cols if c in avail]
        if sel:
            frames.append(pl.read_parquet(p, columns=sel))
    if not frames:
        return None
    return pl.concat(frames, how="diagonal")


def _occ_oric(series_bound, series_free):
    b = np.asarray(series_bound, float)
    f = np.asarray(series_free, float)
    return b / np.maximum(b + f, 1e-9)


def score_run(out_dir, sat=SAT_DEFAULT, steady_gen=STEADY_GEN_DEFAULT):
    gens = _gen_shards(out_dir)
    if not gens:
        return {"out_dir": out_dir, "error": "no parquet shards"}
    per_gen = []
    for g in sorted(gens):
        paths = [p for _, p in gens[g]]
        cols = pl.read_parquet(paths[0]).columns
        need = [RD + "oriC_low_bound_atp", RD + "oriC_low_free", RD + "number_of_oric"]
        if any(c not in cols for c in need):
            continue
        # ATP-fraction (open band [0.2,0.5]) from the binding listener, if present
        B = "listeners__dnaA_binding__"
        atpfr = None
        if B + "free_DnaA_ATP_nM" in cols and B + "free_DnaA_ADP_nM" in cols:
            frdf = _read_cols(paths, [B + "free_DnaA_ATP_nM", B + "free_DnaA_ADP_nM"])
            fa = frdf[B + "free_DnaA_ATP_nM"].to_numpy().astype(float)
            fd = frdf[B + "free_DnaA_ADP_nM"].to_numpy().astype(float)
            atpfr = float(np.mean(fa / np.maximum(fa + fd, 1e-9)))
        df = _read_cols(paths, need)
        bound = df[RD + "oriC_low_bound_atp"].to_numpy().astype(float)
        occ = _occ_oric(df[RD + "oriC_low_bound_atp"], df[RD + "oriC_low_free"])
        oric = df[RD + "number_of_oric"].to_numpy().astype(float)
        # FIRE events = rising-edge crossings of the trigger condition
        # (oriC_low_bound_atp >= the mechanistic-low threshold). This directly
        # measures "the cluster fills+fires" per generation.
        hi = bound >= sat
        # Debounced FIRE episodes: a new episode only when bound crosses the
        # threshold after being below it for >= MIN_GAP steps. This collapses the
        # rapid up/down jitter WITHIN one brief fill-spike into a single event
        # (the bound count flickers across the threshold during one fill), so the
        # metric measures distinct fillings, not signal noise.
        MIN_GAP = 60
        events = 0
        below_run = MIN_GAP + 1
        for v in hi:
            if v:
                if below_run >= MIN_GAP:
                    events += 1
                below_run = 0
            else:
                below_run += 1
        per_gen.append({
            "gen": g,
            "n_sat_events": events,
            "frac_saturated": round(float(hi.mean()), 3),
            "max_bound": int(bound.max()) if len(bound) else 0,
            "max_occ": round(float(occ.max()), 3),
            "max_oric": int(np.nanmax(oric)) if len(oric) else 0,
            "atpfr": round(atpfr, 3) if atpfr is not None else None,
            "steps": len(occ),
        })
    n_gens = len(per_gen)
    # division: a gen divided if a later gen exists (the lineage continued)
    divided_gens = n_gens - 1 if n_gens else 0
    steady = [r for r in per_gen if r["gen"] >= steady_gen]
    # score over steady gens (fall back to last 2 if run too short)
    scored = steady if steady else per_gen[-2:]
    sat_dev = sum((r["n_sat_events"] - 1) ** 2 for r in scored)
    reinit_pen = sum(3.0 for r in scored if r["max_oric"] >= 4)
    underfill_pen = sum(2.0 for r in scored if r["max_bound"] < sat)  # never fires = bad
    # missed-division penalty: ideally divides every gen
    requested = None
    score = sat_dev + reinit_pen + underfill_pen
    once_per_cycle = bool(scored) and all(r["n_sat_events"] == 1 and r["max_oric"] == 2 for r in scored)
    # ATP-fraction band [0.2,0.5] over steady gens (exclude terminal partial gen)
    fr_vals = [r["atpfr"] for r in per_gen[:-1] if r["gen"] >= steady_gen and r["atpfr"] is not None]
    atpfr_steady = round(sum(fr_vals) / len(fr_vals), 3) if fr_vals else None
    atpfr_in_band = atpfr_steady is not None and 0.2 <= atpfr_steady <= 0.5
    return {
        "out_dir": os.path.basename(out_dir.rstrip("/")),
        "n_gens_emitted": n_gens,
        "divided_gens": divided_gens,
        "steady_from_gen": steady_gen,
        "score": round(score, 3),
        "once_per_cycle": once_per_cycle,
        "atpfr_steady": atpfr_steady,
        "atpfr_in_band": atpfr_in_band,
        "per_gen": per_gen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--sat", type=float, default=SAT_DEFAULT)
    ap.add_argument("--steady-gen", type=int, default=STEADY_GEN_DEFAULT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = [score_run(d, args.sat, args.steady_gen) for d in args.out_dirs]
    results.sort(key=lambda r: r.get("score", 1e9))
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        if "error" in r:
            print(f"{r['out_dir']:45s}  ERROR: {r['error']}")
            continue
        tag = "✓ once/cycle" if r["once_per_cycle"] else ""
        fr = r.get("atpfr_steady")
        frtag = (f"ATPfr={fr}{'✓band' if r.get('atpfr_in_band') else '✗band'}" if fr is not None else "")
        pg = " ".join(f"g{x['gen']}:{x['n_sat_events']}ev/oc{x['max_oric']}" for x in r["per_gen"])
        print(f"{r['out_dir']:45s} score={r['score']:6.2f} gens={r['n_gens_emitted']} {tag} {frtag}")
        print(f"    {pg}")


if __name__ == "__main__":
    main()
