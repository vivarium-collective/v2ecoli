"""Render the colony memory-scaling figures from the staged measurement run.

Reads runs/mem_measure.csv (one row per sample: tick, n_cells, rss_mb, numpy_mb,
native_mb) produced by mem_measure.py — a controlled run that stages the colony
through N=1 -> 2 -> 4 (force-divide for timing) while sampling memory.

Produces three committed SVGs under charts/ that DEMONSTRATE the two distinct
memory factors + where the memory lives:
  01_rss_vs_tick_by_ncells.svg  — RSS vs tick, plateaus shaded by cell count:
        the STEP-UPS = per-cell footprint; the within-plateau SLOPE = the leak.
  02_footprint_and_leak_by_ncells.svg — per-cell footprint (GB/cell) and
        per-tick leak rate (MB/tick) at each cell count.
  03_memory_decomposition.svg — RSS split into numpy-object bytes (flat) vs
        native (growing): the leak is native, not a Python/numpy object.

    python .../colonies-02-parallel-multigen-perf/sims/make_memory_figs.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STUDY_DIR = Path(__file__).resolve().parent.parent
CSV = STUDY_DIR / "runs" / "mem_measure.csv"
CHARTS = STUDY_DIR / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(CSV)))
tick = np.array([int(r["tick"]) for r in rows])
rss = np.array([float(r["rss_mb"]) for r in rows])
nc = np.array([int(r["n_cells"]) for r in rows])
npm = np.array([float(r["numpy_mb"]) for r in rows])
nat = np.array([float(r["native_mb"]) for r in rows])

# Plateau segments = maximal runs of constant n_cells (drop 1-2 transient ticks).
segs = []  # (n_cells, idx_lo, idx_hi)
i = 0
while i < len(nc):
    j = i
    while j + 1 < len(nc) and nc[j + 1] == nc[i]:
        j += 1
    if j - i >= 2:
        segs.append((int(nc[i]), i, j))
    i = j + 1
plateaus = {}
for n, lo, hi in segs:
    plateaus.setdefault(n, (lo, hi))  # first stable plateau per count
PAL = {1: "#cfe8ff", 2: "#cdeccd", 4: "#ffe0b3", 8: "#f3cccc"}

# ---- Fig 1: RSS vs tick, plateaus shaded by cell count ----
fig, ax = plt.subplots(figsize=(8.4, 5.0))
for n, lo, hi in segs:
    ax.axvspan(tick[lo], tick[hi], color=PAL.get(n, "#eee"), alpha=0.6, lw=0)
ax.plot(tick, rss, color="#c0392b", lw=1.8, label="process RSS")
for n, (lo, hi) in plateaus.items():
    sl = np.polyfit(tick[lo:hi + 1], rss[lo:hi + 1], 1)[0]  # MB/tick
    midx = tick[(lo + hi) // 2]
    ax.annotate(f"N={n}\n+{sl:.2f} MB/tick", (midx, rss[(lo + hi) // 2]),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=9, color="#333")
ax.set_xlabel("tick (sim seconds)")
ax.set_ylabel("RSS (MB)")
ax.set_title("Colony memory vs time — step-ups = per-cell footprint, "
             "within-plateau slope = the leak")
ax.legend(handles=[mpatches.Patch(color=PAL[n], label=f"{n} cell(s)")
                   for n in sorted(plateaus)], loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(CHARTS / "01_rss_vs_tick_by_ncells.svg"); plt.close(fig)

# ---- Fig 2: per-cell footprint + per-tick leak rate by cell count ----
ns = sorted(plateaus)
foot = [rss[plateaus[n][0]] / n / 1024.0 for n in ns]          # GB per cell
leak = [np.polyfit(tick[plateaus[n][0]:plateaus[n][1] + 1],
                   rss[plateaus[n][0]:plateaus[n][1] + 1], 1)[0] for n in ns]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.4))
a1.bar([str(n) for n in ns], foot, color="#2c7fb8")
a1.set_xlabel("cells"); a1.set_ylabel("GB / cell"); a1.set_title("Per-cell footprint\n(each cell = a full baseline WCM + own sim_data)")
a2.bar([str(n) for n in ns], leak, color="#c0392b")
a2.set_xlabel("cells"); a2.set_ylabel("MB / tick"); a2.set_title("Per-tick leak rate\n(within-plateau RSS slope; scales with cells)")
fig.tight_layout(); fig.savefig(CHARTS / "02_footprint_and_leak_by_ncells.svg"); plt.close(fig)

# ---- Fig 3: RSS decomposition — numpy objects (flat) vs native (growing) ----
fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.fill_between(tick, 0, npm, color="#2c7fb8", alpha=0.8, label="numpy-object bytes (gc-visible)")
ax.fill_between(tick, npm, npm + nat, color="#c0392b", alpha=0.55, label="native / C (not a Python object)")
ax.plot(tick, rss, color="black", lw=1.0, label="total RSS")
ax.set_xlabel("tick (sim seconds)"); ax.set_ylabel("MB")
ax.set_title("Where the memory lives — numpy stays flat; the growth is native\n"
             "(tracemalloc saw <2 MB; macOS leaks: ~13 MB true LLVM/Numba leak)")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(CHARTS / "03_memory_decomposition.svg"); plt.close(fig)

print("wrote:", *(p.name for p in sorted(CHARTS.glob("0*.svg"))))
