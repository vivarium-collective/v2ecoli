#!/usr/bin/env python
"""Quantify replication-initiation ASYNCHRONY under the mechanistic (sat-init)
trigger and plot the inter-initiation-time distribution as a bar graph.

Biology: under the inherited cell-mass clock every origin in a cell fires at
the same critical mass, so sister chromosomes initiate synchronously (delay
~0). The DnaA-oriC sat-init gate makes initiation depend on stochastic
DnaA-ATP filament assembly at oriC, and the daughter POST_INIT_UNLOCK gate
holds fresh origins until they accumulate 60 s of rising bulk DnaA-ATP — so
sisters fire at different cell ages, i.e. asynchronously.

We measure, per generation, the CELL AGE AT INITIATION = the within-generation
time (global_time resets to 0 at each cell birth) at which
``listeners__replication_data__number_of_oric`` first steps 1 -> 2. Pooling
these across seeds and generations gives the initiation-age distribution for a
condition. Under the mass clock it is a sharp spike; under sat-init it spreads.

The INTER-INITIATION DELAY between two sister chromosomes is then the absolute
difference in initiation age between two cells of the condition; we form its
distribution from all within-seed pairwise differences and plot it as a 3-min
bar graph, mechanism vs. mass-clock control.

Usage:
    python scripts/analyze_initiation_asynchrony.py \
        --mech 'out/dnaa5_succ_mech_seed*_parquet/*/history' \
        --control 'out/dnaa5_succ_ctrl_seed*_parquet/*/history' \
        --out out/analysis/asynchrony_succinate.svg --title "succinate"
"""
import argparse, glob, json, os
import duckdb
import numpy as np


def initiation_ages(history_globs):
    """Return {seed_key: [cell-age-at-init (min), ...]} over all matching runs.

    seed_key groups cells that belong to the same lineage/run so pairwise
    differences stay within a biologically comparable population.
    """
    con = duckdb.connect()
    out = {}
    roots = sorted(set(p.rstrip("/") for g in history_globs
                       for p in glob.glob(g)))
    for root in roots:
        files = glob.glob(f"{root}/**/*.pq", recursive=True)
        if not files:
            continue
        rows = con.execute(f"""
            SELECT generation, agent_id, global_time,
                   listeners__replication_data__number_of_oric AS noric
            FROM read_parquet('{root}/**/*.pq', hive_partitioning=true)
            WHERE listeners__replication_data__number_of_oric IS NOT NULL
            ORDER BY generation, agent_id, global_time
        """).fetchall()
        ages = []
        cur, series = None, []
        def flush(series):
            if len(series) < 2:
                return
            t = np.array([s[0] for s in series], float) / 60.0
            n = np.array([s[1] for s in series], float)
            inc = np.where(np.diff(n) > 0.5)[0]
            if len(inc):
                ages.append(float(t[inc[0] + 1]))  # first 1->2 crossing age
        for r in rows:
            key = (r[0], r[1])
            if key != cur:
                flush(series); series, cur = [], key
            series.append((r[2], r[3]))
        flush(series)
        out[root] = ages
    return out


def pairwise_delays(ages_by_run):
    """All within-run pairwise |Δ initiation age| (min) — the sister-delay proxy."""
    d = []
    for ages in ages_by_run.values():
        a = np.array(ages, float)
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                d.append(abs(a[i] - a[j]))
    return np.array(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", action="append", required=True,
                    help="history-root glob(s) for the sat-init runs")
    ap.add_argument("--control", action="append", default=None,
                    help="history-root glob(s) for the mass-clock control runs")
    ap.add_argument("--out", default="out/analysis/asynchrony.svg")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    mech_ages = initiation_ages(args.mech)
    mech_all = np.array([x for v in mech_ages.values() for x in v])
    mech_delay = pairwise_delays(mech_ages)
    ctrl_ages = ctrl_delay = None
    if args.control:
        ctrl_ages = initiation_ages(args.control)
        ctrl_delay = pairwise_delays(ctrl_ages)

    print(f"{args.title}: mech {len(mech_all)} initiations across {len(mech_ages)} runs; "
          + (f"age median {np.median(mech_all):.1f} min, std {np.std(mech_all):.1f}; "
             if len(mech_all) else "none; ")
          + (f"{len(mech_delay)} sister-delay pairs, median {np.median(mech_delay):.1f} min"
             if len(mech_delay) else "no pairs"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: initiation-age distribution (spread == asynchrony)
    if len(mech_all):
        lo, hi = 0, max(mech_all.max(), (np.max([x for v in (ctrl_ages or {}).values()
                        for x in v]) if ctrl_ages else 0)) + 6
        bins = np.arange(lo, hi, 4.0)
        axL.hist(mech_all, bins=bins, color="#2563eb", alpha=0.75,
                 edgecolor="#1e3a8a", label=f"sat-init (std {np.std(mech_all):.1f} min)")
        if ctrl_ages:
            ca = np.array([x for v in ctrl_ages.values() for x in v])
            if len(ca):
                axL.hist(ca, bins=bins, color="#9ca3af", alpha=0.65,
                         edgecolor="#4b5563",
                         label=f"mass-clock (std {np.std(ca):.1f} min)")
        axL.legend(frameon=False, fontsize=9)
    axL.set_xlabel("cell age at initiation (min)")
    axL.set_ylabel("count")
    axL.set_title("Initiation-age distribution\n(broader = more asynchronous)", fontsize=11)

    # Right: inter-initiation (sister) delay distribution — the bar graph
    if len(mech_delay):
        hi = max(mech_delay.max(), (ctrl_delay.max() if ctrl_delay is not None
                 and len(ctrl_delay) else 0)) + 4
        bins = np.arange(0, max(hi, 20), 3.0)
        axR.hist(mech_delay, bins=bins, color="#2563eb", alpha=0.8,
                 edgecolor="#1e3a8a",
                 label=f"sat-init  median {np.median(mech_delay):.1f} min (n={len(mech_delay)})")
        if ctrl_delay is not None and len(ctrl_delay):
            axR.hist(ctrl_delay, bins=bins, color="#9ca3af", alpha=0.6,
                     edgecolor="#4b5563",
                     label=f"mass-clock  median {np.median(ctrl_delay):.1f} min")
        axR.axvline(0, color="#94a3b8", ls=":", lw=1)
        axR.legend(frameon=False, fontsize=9)
    axR.set_xlabel("inter-initiation delay between sister oriCs (min)")
    axR.set_ylabel("count")
    axR.set_title("Sister inter-initiation-time distribution", fontsize=11)

    fig.suptitle("Replication-initiation asynchrony"
                 + (f" — {args.title}" if args.title else ""),
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    meta = {"title": f"Replication-initiation asynchrony ({args.title})" if args.title
                     else "Replication-initiation asynchrony",
            "caption": "Left: cell-age-at-initiation distribution — the mass clock "
                       "fires synchronously (sharp), sat-init spreads it. Right: "
                       "the resulting inter-initiation delay between sister oriCs; "
                       "the mass-clock control sits near 0, sat-init shifts positive.",
            "n_initiations": int(len(mech_all)),
            "init_age_std_min": (float(np.std(mech_all)) if len(mech_all) else None),
            "sister_delay_median_min": (float(np.median(mech_delay)) if len(mech_delay) else None),
            "n_delay_pairs": int(len(mech_delay))}
    json.dump(meta, open(args.out.rsplit(".", 1)[0] + ".meta.json", "w"))
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
