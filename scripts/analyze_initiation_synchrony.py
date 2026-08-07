#!/usr/bin/env python
"""Change in replication-initiation SYNCHRONY: mass clock vs the DnaA-oriC
sat-init mechanism, measured the CORRECT way — within a single cell.

When a cell's origins double (N -> 2N), we ask whether they fire together
(synchronous) or one at a time with a delay (asynchronous). From the origin
count time series we group increments of
``listeners__replication_data__number_of_oric`` that belong to the same
doubling event (consecutive increments < GROUP_GAP min apart) and record:
  - jump pattern: a single +N step (synchronous) vs several +1 steps (staggered)
  - event SPAN: minutes from the first to the last origin firing in the event
    (0 = perfectly synchronous; > 0 = asynchronous, the inter-origin delay)

Biology: the inherited cell-mass clock fires every origin at one critical mass,
so sisters co-initiate synchronously (span ~0). The sat-init gate makes each
origin fire on its own DnaA-ATP filament assembly, so they stagger — the
initiation asynchrony the study set out to quantify.

Usage:
  python scripts/analyze_initiation_synchrony.py \
     --mech 'out/dnaa5_succ_mech_seed*_parquet/*/history' \
     --control 'out/dnaa5_succ_ctrl_seed*_parquet/*/history' \
     --out out/analysis/synchrony_succinate.svg --title "succinate"
"""
import argparse, glob, json, os
from itertools import groupby
import duckdb
import numpy as np

GROUP_GAP = 10.0  # min; increments closer than this belong to one doubling event


def events(history_globs):
    """Return (spans, jump_sizes) for all doubling events across the runs.

    spans: minutes between first and last origin firing in each event.
    jump_sizes: the individual +N increments (1 = one origin at a time)."""
    con = duckdb.connect()
    spans, jumps = [], []
    roots = sorted(set(p.rstrip("/") for g in history_globs for p in glob.glob(g)))
    for root in roots:
        if not glob.glob(f"{root}/**/*.pq", recursive=True):
            continue
        rows = con.execute(f"""
            SELECT generation, agent_id, global_time,
                   listeners__replication_data__number_of_oric AS noric
            FROM read_parquet('{root}/**/*.pq', hive_partitioning=true)
            WHERE listeners__replication_data__number_of_oric IS NOT NULL
            ORDER BY generation, agent_id, global_time
        """).fetchall()
        for _key, grp in groupby(rows, key=lambda r: (r[0], r[1])):
            g = list(grp)
            t = np.array([x[2] for x in g], float) / 60.0
            n = np.array([x[3] for x in g], float)
            d = np.diff(n)
            idx = np.where(d > 0.5)[0]
            if len(idx) == 0:
                continue
            ft = t[idx + 1]          # firing times
            fd = d[idx]              # jump size at each firing
            jumps.extend(int(round(x)) for x in fd)
            # group firings into doubling events
            i = 0
            while i < len(ft):
                j = i
                while j + 1 < len(ft) and (ft[j + 1] - ft[j]) < GROUP_GAP:
                    j += 1
                spans.append(float(ft[j] - ft[i]))  # 0 if single firing
                i = j + 1
    return np.array(spans), jumps


def summarize(jumps):
    from collections import Counter
    c = Counter(jumps)
    total = sum(c.values())
    plus1 = c.get(1, 0)
    plusN = total - plus1
    return c, plus1, plusN, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", action="append", required=True)
    ap.add_argument("--control", action="append", default=None)
    ap.add_argument("--out", default="out/analysis/synchrony.svg")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    m_span, m_jump = events(args.mech)
    c_span, c_jump = (events(args.control) if args.control else (np.array([]), []))
    mc, m1, mN, mt = summarize(m_jump)
    cc, c1, cN, ct = summarize(c_jump)

    # asynchrony index = fraction of doubling events that are staggered (span > 0.05 min)
    m_async = float(np.mean(m_span > 0.05)) if len(m_span) else 0.0
    c_async = float(np.mean(c_span > 0.05)) if len(c_span) else 0.0
    m_span_pos = m_span[m_span > 0.05]
    c_span_pos = c_span[c_span > 0.05]
    print(f"{args.title}: MECH events={len(m_span)} async_frac={m_async:.2f} "
          f"median_span={np.median(m_span_pos) if len(m_span_pos) else 0:.2f} min "
          f"| jumps +1={m1} +N={mN}")
    print(f"{args.title}: CTRL events={len(c_span)} async_frac={c_async:.2f} "
          f"| jumps +1={c1} +N={cN}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Panel A: synchronous (single +N) vs asynchronous (staggered) fraction
    labels = ["mass-clock", "sat-init"]
    sync_frac = [1 - c_async, 1 - m_async]
    async_frac = [c_async, m_async]
    x = np.arange(2)
    axA.bar(x, sync_frac, 0.6, label="synchronous (origins fire together)",
            color="#9ca3af", edgecolor="#4b5563")
    axA.bar(x, async_frac, 0.6, bottom=sync_frac,
            label="asynchronous (staggered)", color="#2563eb", edgecolor="#1e3a8a")
    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_ylabel("fraction of doubling events")
    axA.set_ylim(0, 1.0)
    for xi, af in zip(x, async_frac):
        axA.text(xi, 0.5, f"{af*100:.0f}%\nasync", ha="center", va="center",
                 fontsize=10, color="white", fontweight="bold")
    axA.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    axA.set_title("Initiation synchrony\n(mass-clock fires together; sat-init staggers)",
                  fontsize=11)

    # Panel B: inter-origin firing delay (span) distribution — the bar graph
    hi = max(m_span_pos.max() if len(m_span_pos) else 1,
             c_span_pos.max() if len(c_span_pos) else 1) + 0.1
    bins = np.linspace(0, max(hi, 0.5), 16)
    if len(m_span_pos):
        axB.hist(m_span_pos, bins=bins, color="#2563eb", alpha=0.8, edgecolor="#1e3a8a",
                 label=f"sat-init (median {np.median(m_span_pos):.2f} min, n={len(m_span_pos)})")
    if len(c_span_pos):
        axB.hist(c_span_pos, bins=bins, color="#9ca3af", alpha=0.6, edgecolor="#4b5563",
                 label=f"mass-clock (n={len(c_span_pos)})")
    axB.axvline(0, color="#94a3b8", ls=":", lw=1, label="synchronous (Δt=0)")
    axB.set_xlabel("inter-origin initiation delay within a cell (min)")
    axB.set_ylabel("count")
    axB.legend(frameon=False, fontsize=9)
    axB.set_title("Sister inter-initiation-time distribution", fontsize=11)

    fig.suptitle("Change in initiation synchrony"
                 + (f" — {args.title}" if args.title else ""),
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    png = args.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    meta = {"title": f"Change in initiation synchrony ({args.title})" if args.title
                     else "Change in initiation synchrony",
            "caption": "The inherited cell-mass clock co-initiates sister origins "
                       "SYNCHRONOUSLY (single +N origin-count jump, delay ~0); the "
                       "DnaA-oriC sat-init mechanism STAGGERS them (origins fire one "
                       "at a time). Left: fraction of doubling events that are "
                       "asynchronous. Right: the inter-origin initiation delay.",
            "mech_async_fraction": m_async,
            "ctrl_async_fraction": c_async,
            "mech_median_span_min": (float(np.median(m_span_pos)) if len(m_span_pos) else 0.0),
            "mech_jump_plus1": m1, "mech_jump_plusN": mN,
            "ctrl_jump_plus1": c1, "ctrl_jump_plusN": cN}
    json.dump(meta, open(args.out.rsplit(".", 1)[0] + ".meta.json", "w"))
    print("wrote", args.out, "and", png)


if __name__ == "__main__":
    main()
