#!/usr/bin/env python
"""Trajectory plot for the DnaA-oriC mechanism — the reference's signature view.

Four stacked panels over a multi-generation lineage:
  1. bulk free DnaA-ATP (nM)         — the sawtooth (fill -> fire -> reset)
  2. oriC-low bound DnaA-ATP (0..8/origin) + number_of_oric
  3. DnaA-ATP fraction  = ATP / (ATP + ADP)
  4. cell mass (fg) with the oriC 1->2 initiation instants marked

Usage:
  python scripts/plot_dnaa_trajectory.py \
     --exp-root out/dnaa5_succ_mech_seed1_parquet/dnaa5_succ_mech_seed1 \
     --out out/analysis/trajectory_succinate.svg --title "succinate (seed 1)"
"""
import argparse, glob, json, os
import duckdb
import numpy as np


def load(exp_root):
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT generation, agent_id, global_time,
               listeners__dnaA_binding__free_DnaA_ATP_nM AS atp,
               listeners__dnaA_binding__free_DnaA_ADP_nM AS adp,
               listeners__replication_data__oriC_low_bound_atp AS low8,
               listeners__replication_data__number_of_oric AS noric,
               listeners__mass__cell_mass AS mass
        FROM read_parquet('{exp_root}/history/**/*.pq', hive_partitioning=true)
        WHERE listeners__dnaA_binding__free_DnaA_ATP_nM IS NOT NULL
        ORDER BY generation, agent_id, global_time
    """).fetchdf()
    # build a continuous lineage clock: cumulative time across generations
    # (global_time resets to 0 each generation) following ONE daughter lineage.
    t, offset = [], 0.0
    keys = list(df.groupby(["generation", "agent_id"]).groups.keys())
    # pick the primary lineage: the agent_id that is the all-zeros prefix chain
    prim = sorted(keys, key=lambda k: (k[0], len(str(k[1])), str(k[1])))
    seen_gen = {}
    lineage = []
    for g, a in prim:
        if g in seen_gen:
            continue  # one agent per generation (primary daughter)
        seen_gen[g] = a
        lineage.append((g, a))
    frames = []
    for g, a in lineage:
        sub = df[(df.generation == g) & (df.agent_id == a)].copy()
        sub["tabs"] = sub.global_time + offset
        offset = sub.tabs.max()
        frames.append(sub)
    import pandas as pd
    return pd.concat(frames), lineage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--out", default="out/analysis/trajectory.svg")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    df, lineage = load(args.exp_root)
    tm = df.tabs.values / 60.0
    frac = df.atp.values / np.clip(df.atp.values + df.adp.values, 1e-9, None)
    # initiation instants (oriC 1->2, per generation)
    init_t = []
    off = 0.0
    con = duckdb.connect()
    for g, a in lineage:
        sub = df[(df.generation == g) & (df.agent_id == a)]
        n = sub.noric.values
        inc = np.where(np.diff(n) > 0.5)[0]
        for i in inc:
            init_t.append(sub.tabs.values[i + 1] / 60.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    ax[0].plot(tm, df.atp.values, color="#2563eb", lw=1.1)
    ax[0].axhspan(10, 30, color="#16a34a", alpha=0.12, label="reference sawtooth 10-30 nM")
    ax[0].set_ylabel("free DnaA-ATP\n(nM)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].set_title("Bulk DnaA-ATP sawtooth (fill → fire → reset)", fontsize=10)

    ax[1].plot(tm, df.low8.values, color="#7c3aed", lw=1.1, label="oriC-low bound DnaA-ATP")
    ax1b = ax[1].twinx()
    ax1b.plot(tm, df.noric.values, color="#dc2626", lw=1.3, drawstyle="steps-post",
              label="number of oriC")
    ax[1].set_ylabel("oriC-low\nbound ATP")
    ax1b.set_ylabel("# oriC", color="#dc2626")
    ax[1].set_title("oriC-low saturation + origin count", fontsize=10)

    ax[2].plot(tm, frac, color="#0891b2", lw=1.1)
    ax[2].axhspan(0.2, 0.5, color="#f59e0b", alpha=0.14, label="Boesen band [0.2,0.5]")
    ax[2].set_ylabel("DnaA-ATP\nfraction")
    ax[2].legend(frameon=False, fontsize=8, loc="upper right")
    ax[2].set_title("DnaA-ATP fraction (Haochen H3 target band)", fontsize=10)

    ax[3].plot(tm, df.mass.values, color="#334155", lw=1.1)
    for it in init_t:
        ax[3].axvline(it, color="#dc2626", ls=":", lw=0.9, alpha=0.7)
    ax[3].set_ylabel("cell mass\n(fg)")
    ax[3].set_xlabel("lineage time (min)")
    ax[3].set_title("Cell mass — red dotted = initiation (oriC 1→2)", fontsize=10)

    fig.suptitle("DnaA-oriC mechanistic trajectory"
                 + (f" — {args.title}" if args.title else ""),
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    meta = {"title": f"DnaA-oriC mechanistic trajectory ({args.title})",
            "caption": "Bulk DnaA-ATP sawtooth, oriC-low saturation + origin count, "
                       "DnaA-ATP fraction vs the Boesen band, and cell mass with "
                       "initiation instants marked, across a multi-generation lineage.",
            "atp_peak_nM": float(np.nanpercentile(df.atp.values, 98)),
            "n_initiations": len(init_t)}
    json.dump(meta, open(args.out.rsplit(".", 1)[0] + ".meta.json", "w"))
    print(f"{args.title}: peak DnaA-ATP ~{meta['atp_peak_nM']:.0f} nM, {len(init_t)} initiations")
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
