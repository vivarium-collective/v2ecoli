#!/usr/bin/env python
"""Lineage-trajectory extraction + cyclic/steady-band metric for the dnaa n×K sweep.

This is the *simulation-results* companion to the analytic dnaa-5 switch/K curves:
for each multigen run under the mechanistic oriC-low trigger it pulls the four
lineage trajectories Rashmi asked to see —

  1. number of oriC          listeners__replication_data__number_of_oric
  2. total DnaA (counts)     PD03831[c] + MONOMER0-160[c] + MONOMER0-4565[c]  (apo+ATP+ADP)
  3. DnaA-ATP fraction       MONOMER0-160[c] / total DnaA
  4. oriC-low occupancy      oriC_low_bound_atp / (oriC_low_bound_atp + oriC_low_free)

and scores the run on the two things the new sweep metric cares about:

  * CYCLIC replication  — exactly one oriC-low fill→fire→reset per generation,
    oriC strictly 1↔2 (never ≥3), a division every steady gen.
  * STEADY DnaA BAND    — per-gen mean total DnaA sits inside [300, 800] and does
    not drift (low across-generation CV; small end-to-end slope).

The DnaA forms are read straight from the BULK store (the dnaA_cycle listener was
removed post-bf8b82e; the equilibrium machinery interconverts the three bulk
species, so counts are the faithful source — same recipe as
render_dnaa2_sixpanel.py).

Usage:
  dnaa_sweep_analysis.py <out-dir> [<out-dir> ...] [--steady-gen 4] [--json]
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import polars as pl

RD = "listeners__replication_data__"
# bulk species for the three DnaA nucleotide forms
DNAA_APO = "PD03831[c]"          # apo-DnaA
DNAA_ATP = "MONOMER0-160[c]"     # DnaA-ATP
DNAA_ADP = "MONOMER0-4565[c]"    # DnaA-ADP

# metric bands / thresholds
DNAA_BAND = (300.0, 800.0)       # total DnaA homeostatic band (counts)
ATP_BAND = (0.2, 0.5)            # DnaA-ATP fraction band (Boesen)
SAT = 6                          # oriC_low_bound_atp fire threshold (mechanistic-low)
MIN_GAP = 60                     # debounce: steps below threshold before a new fire episode
STEADY_GEN_DEFAULT = 4
DRIFT_TOL = 0.20                 # |end/start - 1| of per-gen mean DnaA over steady gens


def _lineage_files(out_dir):
    """All history shards on the followed (all-zeros) daughter lineage."""
    fs = glob.glob(os.path.join(out_dir, "**", "history", "**", "*.pq"), recursive=True)
    keep = []
    for f in fs:
        m = re.search(r"agent_id=([^/]+)", f)
        if m and re.fullmatch(r"0+", m.group(1)):   # 0, 00, 000, … (not the …01 sibling)
            keep.append(f)
    return keep


def load_trajectory(out_dir, downsample=6000):
    """Return (df, gen_bounds). df has one row per step on the followed lineage:
    generation, t_min (cumulative minutes), number_of_oric, total_dnaa,
    atp_fraction, oric_low_occ. gen_bounds = cumulative t_min at each division."""
    files = _lineage_files(out_dir)
    if not files:
        return None, None
    ids = (pl.scan_parquet(files[0]).select("bulk__id").head(1)
           .collect()["bulk__id"][0].to_list())

    def _idx(mol):
        try:
            return ids.index(mol)
        except ValueError:
            raise SystemExit(f"{out_dir}: bulk species {mol!r} absent from output")

    i_apo, i_atp, i_adp = _idx(DNAA_APO), _idx(DNAA_ATP), _idx(DNAA_ADP)
    bc = pl.col("bulk__count")
    BOUND = "listeners__dnaA_binding__total_DnaA_bound"   # DnaA bound to chromosomal boxes
    MASS = "listeners__mass__cell_mass"
    need = [RD + "number_of_oric", RD + "oriC_low_bound_atp", RD + "oriC_low_free", BOUND, MASS]
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains(r"^0+$"))
          .select(["generation", "global_time"] + need + [
              bc.list.get(i_apo).alias("_apo"),
              bc.list.get(i_atp).alias("_atp"),
              bc.list.get(i_adp).alias("_adp")])
          .sort(["generation", "global_time"]).collect())
    if df.height == 0:
        return None, None
    # drop <5-min daughter-stub partitions (the final division spawns a sliver gen)
    dur = (df.group_by("generation")
             .agg(((pl.col("global_time").max() - pl.col("global_time").min()) / 60.0)
                  .alias("_d")))
    real = dur.filter(pl.col("_d") >= 5.0)["generation"].to_list()
    df = df.filter(pl.col("generation").is_in(real))
    # TOTAL cellular DnaA = free pool (apo+ATP+ADP) + DnaA bound to chromosomal
    # boxes; this is the ~[300,800] homeostatic quantity. The free-pool alone is
    # only ~150 — the box-bound ~420 makes up the physiological total.
    free = pl.col("_apo") + pl.col("_atp") + pl.col("_adp")
    tot = free + pl.col(BOUND).fill_null(0)
    df = df.with_columns([
        tot.alias("total_dnaa"),
        # DnaA-ATP fraction = free-pool ATP fraction (matches the Boesen [0.2,0.5]
        # band + the prior dnaa figures; the box-bound pool is not counted here).
        (pl.col("_atp") / pl.max_horizontal(free, pl.lit(1))).alias("atp_fraction"),
        (pl.col(RD + "oriC_low_bound_atp")
         / pl.max_horizontal(pl.col(RD + "oriC_low_bound_atp") + pl.col(RD + "oriC_low_free"),
                             pl.lit(1e-9))).alias("oric_low_occ"),
        pl.col(RD + "number_of_oric").alias("number_of_oric"),
        pl.col(MASS).alias("cell_mass"),
    ])
    # cumulative-minutes x-axis with generation offsets
    offset, cum, bounds = 0.0, [], []
    for g in sorted(df["generation"].unique().to_list()):
        t = df.filter(pl.col("generation") == g)["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0)
        offset += float(t.max())
        bounds.append(offset / 60.0)
    df = df.with_columns(pl.Series("t_min", cum))
    df = df.select(["generation", "t_min", "number_of_oric", "total_dnaa",
                    "atp_fraction", "oric_low_occ", "cell_mass",
                    RD + "oriC_low_bound_atp"])
    n = df.height
    if n > downsample:
        df = df.gather_every(max(1, n // downsample))
    return df, bounds[:-1]


def total_atp_fraction(out_dir, steady_gen=STEADY_GEN_DEFAULT):
    """Per-generation TOTAL DnaA-ATP fraction = (free DnaA-ATP + all box-bound
    DnaA-ATP) / (all DnaA forms, free + bound). This is the biologically-meaningful
    Boesen [0.2,0.5] quantity — the free-pool-only fraction is misleadingly low
    because active DnaA-ATP is sequestered on the ~300 chromosomal DnaA boxes."""
    import glob, re
    files = _lineage_files(out_dir)
    if not files:
        return None
    RDp = RD
    b_atp = [RDp+"chromosomal_high_bound_atp", RDp+"oric_high_bound_atp",
             RDp+"oriC_low_bound_atp", RDp+"promoter_high_bound_atp"]
    b_adp = [RDp+"chromosomal_high_bound_adp", RDp+"oric_high_bound_adp",
             RDp+"promoter_high_bound_adp"]
    avail = pl.read_parquet_schema(files[0])
    have = [c for c in b_atp + b_adp if c in avail]
    ids = pl.scan_parquet(files[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {m: ids.index(m) for m in (DNAA_APO, DNAA_ATP, DNAA_ADP)}
    bc = pl.col("bulk__count")
    d = (pl.scan_parquet(files, hive_partitioning=True)
         .filter(pl.col("agent_id").cast(pl.Utf8).str.contains(r"^0+$"))
         .select(["generation"] + have + [bc.list.get(v).alias(k) for k, v in idx.items()])
         .collect())
    per_gen = {}
    for g in sorted(d["generation"].unique().to_list()):
        gd = d.filter(pl.col("generation") == g)
        fa, fd, apo = gd[DNAA_ATP].mean(), gd[DNAA_ADP].mean(), gd[DNAA_APO].mean()
        batp = sum(gd[c].mean() for c in b_atp if c in have)
        badp = sum(gd[c].mean() for c in b_adp if c in have)
        tot = apo + fa + fd + batp + badp
        per_gen[int(g)] = (fa + batp) / tot if tot > 0 else 0.0
    steady = [v for g, v in per_gen.items() if g >= steady_gen] or list(per_gen.values())[-2:]
    mean = sum(steady) / len(steady) if steady else 0.0
    return {"mean": round(mean, 3), "in_band": ATP_BAND[0] <= mean <= ATP_BAND[1],
            "per_gen": {g: round(v, 3) for g, v in per_gen.items()}}


def _fire_events(bound):
    """Debounced count of oriC-low fire episodes (bound >= SAT after MIN_GAP below)."""
    hi = np.asarray(bound, float) >= SAT
    events, below = 0, MIN_GAP + 1
    for v in hi:
        if v:
            if below >= MIN_GAP:
                events += 1
            below = 0
        else:
            below += 1
    return events


def compute_metric(out_dir, steady_gen=STEADY_GEN_DEFAULT):
    df, bounds = load_trajectory(out_dir)
    if df is None:
        return {"tag": os.path.basename(out_dir.rstrip("/")), "error": "no data"}
    tag = os.path.basename(out_dir.rstrip("/"))
    gens = sorted(df["generation"].unique().to_list())
    per_gen = []
    for g in gens:
        gd = df.filter(pl.col("generation") == g)
        bound = gd[RD + "oriC_low_bound_atp"].to_numpy()
        oric = gd["number_of_oric"].to_numpy().astype(float)
        per_gen.append({
            "gen": int(g),
            "events": _fire_events(bound),
            "max_oric": int(np.nanmax(oric)) if len(oric) else 0,
            "mean_dnaa": float(gd["total_dnaa"].mean()),
            "mean_atpfr": float(gd["atp_fraction"].mean()),
        })
    n_gens = len(per_gen)
    steady = [r for r in per_gen if r["gen"] >= steady_gen] or per_gen[-2:]
    # CYCLIC: every steady gen fires exactly once, oriC max == 2; lineage keeps dividing
    cyclic_gens = sum(1 for r in steady if r["events"] == 1 and r["max_oric"] == 2)
    cyclic_frac = cyclic_gens / len(steady)
    cyclic_ok = cyclic_frac == 1.0 and n_gens >= steady_gen + 1
    # STEADY DnaA BAND: per-gen mean total DnaA in band, low across-gen drift
    band_gens = sum(1 for r in steady if DNAA_BAND[0] <= r["mean_dnaa"] <= DNAA_BAND[1])
    band_frac = band_gens / len(steady)
    dm = np.array([r["mean_dnaa"] for r in steady], float)
    drift = abs(dm[-1] / dm[0] - 1.0) if dm[0] > 0 else 1.0
    cv = float(dm.std() / dm.mean()) if dm.mean() > 0 else 1.0
    band_ok = band_frac == 1.0 and drift <= DRIFT_TOL
    # ATP fraction band (reported; known low — a k_r/code lever, not n/K)
    afr = float(np.mean([r["mean_atpfr"] for r in steady]))
    atp_ok = ATP_BAND[0] <= afr <= ATP_BAND[1]
    # composite 0..1 for the heatmap
    steadiness = 1.0 / (1.0 + 5.0 * cv)      # cv 0 -> 1.0, cv 0.1 -> 0.67
    composite = round((cyclic_frac + band_frac + steadiness) / 3.0, 3)
    return {
        "tag": tag,
        "n_gens": n_gens,
        "cyclic_ok": cyclic_ok, "cyclic_frac": round(cyclic_frac, 3),
        "band_ok": band_ok, "band_frac": round(band_frac, 3),
        "dnaa_mean": round(float(dm.mean()), 1), "dnaa_drift": round(drift, 3),
        "dnaa_cv": round(cv, 3),
        "atp_frac": round(afr, 3), "atp_ok": atp_ok,
        "pass": bool(cyclic_ok and band_ok),
        "composite": composite,
        "per_gen": per_gen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--steady-gen", type=int, default=STEADY_GEN_DEFAULT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = [compute_metric(d, args.steady_gen) for d in args.out_dirs]
    res.sort(key=lambda r: (-r.get("composite", -1)))
    if args.json:
        print(json.dumps(res, indent=2))
        return
    for r in res:
        if "error" in r:
            print(f"{r['tag']:16s}  ERROR: {r['error']}")
            continue
        pg = " ".join(f"g{x['gen']}:{x['events']}ev/oc{x['max_oric']}/{x['mean_dnaa']:.0f}"
                      for x in r["per_gen"])
        flags = f"{'CYC' if r['cyclic_ok'] else '   '} {'BAND' if r['band_ok'] else '    '}"
        print(f"{r['tag']:16s} comp={r['composite']:.2f} {flags} "
              f"dnaa={r['dnaa_mean']:.0f}(drift {r['dnaa_drift']:.2f},cv {r['dnaa_cv']:.2f}) "
              f"atpfr={r['atp_frac']:.2f}{'✓' if r['atp_ok'] else '✗'}")
        print(f"    {pg}")


if __name__ == "__main__":
    main()
