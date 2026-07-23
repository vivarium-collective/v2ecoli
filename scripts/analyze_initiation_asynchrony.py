#!/usr/bin/env python
"""Quantify replication-initiation ASYNCHRONY between sister/daughter chromosomes
under the mechanistic (sat-init) replication, and plot the inter-initiation-time
distribution as a bar graph.

Under the mass-clock, a cell's oriCs fire synchronously (delay ~0). The
sat-init gate + daughter POST_INIT_UNLOCK gate make daughter oriCs fire late,
so the delay distribution shifts to positive values. We detect each oriC firing
as a unit increase in ``listeners__replication_data__number_of_oric`` and, within
an initiation ROUND (the increments that carry the count from 2^k up to the next
power of two), measure the time between the first and each subsequent firing.

Usage:
    python scripts/analyze_initiation_asynchrony.py \
        --exp-root out/dnaa5_succ_valid_seed4_parquet/dnaa5_succ_valid_seed4 \
        --out out/analysis/asynchrony.svg --title "succinate (seed 4)"
Multiple --exp-root may be given to pool seeds.
"""
import argparse, glob, json, os
import duckdb
import numpy as np


def initiation_delays(exp_root):
    """Return a list of inter-initiation delays (min) across all generations of
    one experiment root, plus per-round diagnostics."""
    files = glob.glob(f"{exp_root}/history/**/*.pq", recursive=True)
    if not files:
        return [], []
    con = duckdb.connect()
    # Per generation (hive: generation=<n>, agent_id=<id>), read the oriC-count
    # time series ordered by time.
    rows = con.execute(f"""
        SELECT generation, agent_id, global_time,
               listeners__replication_data__number_of_oric AS noric
        FROM read_parquet('{exp_root}/history/**/*.pq', hive_partitioning=true)
        WHERE listeners__replication_data__number_of_oric IS NOT NULL
        ORDER BY generation, agent_id, global_time
    """).fetchall()
    delays, rounds = [], []
    # group by (generation, agent_id)
    cur_key, series = None, []
    def flush(series):
        if len(series) < 2:
            return
        t = np.array([s[0] for s in series], float) / 60.0  # → minutes
        n = np.array([s[1] for s in series], float)
        # firing times = times where n increases by >=1
        inc_idx = np.where(np.diff(n) > 0.5)[0] + 1
        if len(inc_idx) < 2:
            return
        fire_t = t[inc_idx]
        fire_dn = np.diff(n)[inc_idx - 1]
        # Group consecutive firings into rounds: a round is a maximal run of
        # firings separated by < ROUND_GAP min (co-initiation of sister oriCs).
        ROUND_GAP = 25.0
        i = 0
        while i < len(fire_t):
            j = i
            while j + 1 < len(fire_t) and (fire_t[j + 1] - fire_t[j]) < ROUND_GAP:
                j += 1
            if j > i:  # a multi-oriC round
                t0 = fire_t[i]
                for k in range(i + 1, j + 1):
                    delays.append(float(fire_t[k] - t0))
                rounds.append({"n_fire": int(j - i + 1),
                               "span_min": float(fire_t[j] - fire_t[i])})
            i = j + 1
    for r in rows:
        key = (r[0], r[1])
        if key != cur_key:
            flush(series)
            series, cur_key = [], key
        series.append((r[2], r[3]))
    flush(series)
    return delays, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", action="append", required=True)
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--out", default="out/analysis/asynchrony.svg")
    ap.add_argument("--title", default="Replication-initiation asynchrony")
    args = ap.parse_args()

    all_delays = []
    per_label = {}
    for i, root in enumerate(args.exp_root):
        d, rounds = initiation_delays(root)
        lbl = (args.label[i] if args.label and i < len(args.label)
               else os.path.basename(root))
        per_label[lbl] = d
        all_delays += d
        print(f"{lbl}: {len(d)} inter-initiation delays from {len(rounds)} multi-oriC rounds"
              + (f"  (median {np.median(d):.1f} min, max {max(d):.1f})" if d else "  (none)"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if all_delays:
        bins = np.arange(0, max(max(all_delays) + 4, 20), 3.0)  # 3-min bars
        ax.hist(all_delays, bins=bins, color="#2563eb", edgecolor="#1e3a8a", alpha=0.85)
        med = np.median(all_delays)
        ax.axvline(med, color="#dc2626", ls="--", lw=1.6,
                   label=f"median = {med:.1f} min  (n={len(all_delays)})")
        ax.axvline(0, color="#94a3b8", ls=":", lw=1, label="synchronous (mass-clock)")
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "no multi-oriC initiation rounds detected",
                ha="center", va="center", transform=ax.transAxes, color="#6b7280")
    ax.set_xlabel("inter-initiation time between sister oriCs (min)")
    ax.set_ylabel("count")
    ax.set_title(args.title + "\n" + r"$\Delta t$ between the two daughter chromosomes' initiations",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    # meta for the workbench Results tab
    meta = {"title": "Replication-initiation asynchrony (inter-initiation times)",
            "caption": "Distribution of the time delay between the two daughter "
                       "chromosomes' replication initiations under the mechanistic "
                       "sat-init gate. Mass-clock initiation would be synchronous "
                       "(delay ~0); the daughter POST_INIT_UNLOCK gate shifts it "
                       "positive.",
            "n_delays": len(all_delays),
            "median_min": (float(np.median(all_delays)) if all_delays else None)}
    json.dump(meta, open(args.out.rsplit(".", 1)[0] + ".meta.json", "w"))
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
