"""dnaa-3 Rashmi item 1 — SINK capacity sweep.

Sweep the chromosomal high-affinity DnaA-box capacity (DNAA_N_CHROM_HIGH_CAP)
with the dnaa_sink feature ENABLED, to find the setting where free DnaA-ATP
lands in a physiological window (not ~0, not >K_d over-binding) while the
gen-1 cell cycle is preserved.

Literature (study.yaml references): ChIP-seq (Smith 2024) finds ~13 strongly
DnaA-bound regions / 32 strict-consensus sites genome-wide — the ~302
loose-motif chromosomal pool is NOT the in-vivo titration sink. So we sweep the
chromosomal capacity toward the literature {302, 32, 13} (302 = the current
un-capped behaviour that over-titrates).

Each value runs in its OWN subprocess (per the v2ecoli subprocess-isolation
memory: one crash must not kill the whole sweep). Each subprocess enables
dnaa_sink (via _run_dnaa_sink.py) and sets DNAA_N_CHROM_HIGH_CAP, runs a single
generation from the burned-in resume dill, and writes parquet. We then parse
free DnaA-ATP, oriC occupancy, division time, and the oriC 1<->2 pattern.

Usage:
  .venv/bin/python scripts/sweep_dnaa_sink_capacity.py \
      --cache-dir out/cache_dnaa3 \
      --resume-dill out/steady_state_inputs/succinate_default_gen3_start.dill \
      --caps 302,32,13 --max-min 90 --seed 1
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_run(run_out_dir):
    """Pull sink metrics from the parquet history of a single-gen run."""
    import polars as pl
    fs = sorted(glob.glob(os.path.join(run_out_dir, "**/history/**/*.pq"),
                          recursive=True))
    fs = [f for f in fs if "/agent_id=0" in f or "agent_id=0" in f]
    if not fs:
        return {"error": "no parquet"}
    lf = pl.scan_parquet(fs)
    cols = lf.collect_schema().names()

    def col(name):
        full = "listeners__dnaA_binding__" + name
        return full if full in cols else None

    want = {
        "free_atp": col("free_DnaA_ATP_nM"),
        "free_adp": col("free_DnaA_ADP_nM"),
        "oric_low": col("oric__low_affinity_occupied"),
        "oric_high": col("oric__high_affinity_occupied"),
        "total_bound": col("total_DnaA_bound"),
        "chrom_count": col("chromosome__occupied_count"),
        "chrom_total": col("chromosome__n_total"),
    }
    oric_col = "listeners__replication_data__number_of_oric"
    mass_col = "listeners__mass__cell_mass"
    time_col = "time" if "time" in cols else (
        "global_time" if "global_time" in cols else None)

    sel = [pl.col(c).alias(k) for k, c in want.items() if c]
    if oric_col in cols:
        sel.append(pl.col(oric_col).alias("n_oric"))
    if mass_col in cols:
        sel.append(pl.col(mass_col).alias("cell_mass"))
    if time_col:
        sel.append(pl.col(time_col).alias("t"))
    df = lf.select(sel).collect()
    n = df.height
    if n == 0:
        return {"error": "empty parquet"}

    def stat(k, fn):
        if k not in df.columns:
            return None
        s = df[k].drop_nulls()
        return round(float(fn(s)), 3) if s.len() else None

    res = {
        "ticks": n,
        "free_atp_mean": stat("free_atp", lambda s: s.mean()),
        "free_atp_median": stat("free_atp", lambda s: s.median()),
        "free_atp_min": stat("free_atp", lambda s: s.min()),
        "free_atp_max": stat("free_atp", lambda s: s.max()),
        "free_adp_mean": stat("free_adp", lambda s: s.mean()),
        "oric_low_mean": stat("oric_low", lambda s: s.mean()),
        "oric_low_t0": stat("oric_low", lambda s: s.head(1).item() if s.len() else None)
            if "oric_low" in df.columns else None,
        "oric_high_mean": stat("oric_high", lambda s: s.mean()),
        "total_bound_mean": stat("total_bound", lambda s: s.mean()),
        "chrom_total": stat("chrom_total", lambda s: s.max()),
    }
    if "n_oric" in df.columns:
        oric_vals = df["n_oric"].drop_nulls()
        res["oric_min"] = int(oric_vals.min())
        res["oric_max"] = int(oric_vals.max())
        res["oric_reached_2"] = bool(oric_vals.max() >= 2)
        res["oric_reinit"] = bool(oric_vals.max() >= 4)
    # division: a single-gen run "divides" if the run terminated by division
    # (the runner stops the gen at division). We infer it from whether the run
    # ended before max_min (division) vs ran to the cap (observer artifact).
    if "t" in df.columns:
        res["t_end_s"] = round(float(df["t"].max()), 1)
        res["t_end_min"] = round(float(df["t"].max()) / 60.0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="out/cache_dnaa3")
    ap.add_argument("--resume-dill",
                    default="out/steady_state_inputs/succinate_default_gen3_start.dill")
    ap.add_argument("--caps", default="302,32,13",
                    help="comma-separated chromosomal high-aff capacities")
    ap.add_argument("--max-min", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--out-root", default="out/dnaa3_sink_sweep")
    ap.add_argument("--results-json",
                    default="out/dnaa3_sink_sweep/sweep_results.json")
    args = ap.parse_args()

    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    os.makedirs(args.out_root, exist_ok=True)
    results = {}

    for cap in caps:
        label = f"cap{cap}"
        out_dir = os.path.join(args.out_root, label)
        exp_id = f"dnaa3_sink_{label}"
        print("=" * 70)
        print(f"[{time.strftime('%H:%M:%S')}] SINK sweep cap={cap} "
              f"(chromosomal high-aff capacity)")
        print("=" * 70)
        env = dict(os.environ)
        env["DNAA_N_CHROM_HIGH_CAP"] = str(cap)
        env["V2ECOLI_EMIT_UNIQUE"] = "1"
        cmd = [
            sys.executable, os.path.join(HERE, "_run_dnaa_sink.py"),
            "--cache-dir", args.cache_dir,
            "--resume-dill", args.resume_dill,
            "--start-gen", "1",
            "--generations", str(args.generations),
            "--max-min", str(args.max_min),
            "--seed", str(args.seed),
            "--out-dir", out_dir,
            "--experiment-id", exp_id,
            "--dill-dir", os.path.join(out_dir, "gen_dills"),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        wall = round(time.time() - t0, 1)
        if proc.returncode != 0:
            print(f"  cap={cap} subprocess FAILED (rc={proc.returncode})")
            results[label] = {"cap": cap, "error": f"rc={proc.returncode}",
                              "wall_s": wall}
        else:
            metrics = parse_run(out_dir)
            metrics["cap"] = cap
            metrics["wall_s"] = wall
            results[label] = metrics
            print(f"  cap={cap} done in {wall}s: "
                  f"free_atp_mean={metrics.get('free_atp_mean')} nM, "
                  f"oric_low_mean={metrics.get('oric_low_mean')}, "
                  f"oric_max={metrics.get('oric_max')}, "
                  f"t_end_min={metrics.get('t_end_min')}")
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    print(f"\nwritten: {args.results_json}")


if __name__ == "__main__":
    main()
