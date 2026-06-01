#!/usr/bin/env python3
"""
The Money Figure — One model, four stresses, four correct responses.
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = "out/money_figure"; os.makedirs(OUT, exist_ok=True)

def load(p):
    with open(p) as f: d=json.load(f)
    s=d['snapshots']
    return {k: np.array([x.get(k,0) for x in s]) for k in s[0]}

def ml(a, f=0.5):
    n=len(a); return float(np.nanmean(a[int(n*(1-f)):]))

# Load all 4 conditions
print("Loading ...")
EXP = load("out/scenarios/mol_baseline.json")      # exponential
H2O = load("out/scenarios/mol_h2o2.json")           # H₂O₂ challenge
STV = load("out/scenarios/timed_starvation.json")   # starvation
OSM = load("out/scenarios/osm_severe.json")         # osmotic (1.0 Osm)

conditions = ["Exponential", "H₂O₂\nChallenge", "Nutrient\nStarvation", "Osmotic\nStress"]
colors = ["#4DAC26", "#D6604D", "#B2182B", "#FF7F00"]
datasets = [EXP, H2O, STV, OSM]

# ══════════════════════════════════════════════════════════════════════════════
# THE MONEY FIGURE — 2 rows × 4 columns
# Top row: time series (ppGpp, σ70, σ38, growth)
# Bottom row: bar charts comparing final values across conditions
# ══════════════════════════════════════════════════════════════════════════════
print("Generating money figure ...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
              top=0.90, bottom=0.08, left=0.06, right=0.97)

fig.suptitle("One Sigma Factor Competition Model — Four Stress Conditions — Four Correct Responses\n"
             "Mauri & Klumpp (2014) framework with ppGpp modulation (Jishage 2002)",
             fontsize=14, fontweight="bold", y=0.97)

# ── Top row: Time series ─────────────────────────────────────────────────────

# A — ppGpp
ax = fig.add_subplot(gs[0, 0])
for D, c, lb in zip(datasets, colors, conditions):
    t = D["time"]/60.0
    ax.plot(t, D["ppgpp_uM"], color=c, lw=1.4, label=lb.replace("\n"," "))
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp — The Master Regulator")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# B — σ70
ax = fig.add_subplot(gs[0, 1])
for D, c, lb in zip(datasets, colors, conditions):
    t = D["time"]/60.0
    ax.plot(t, D["f_sigma70"], color=c, lw=1.4, label=lb.replace("\n"," "))
ax.set_xlabel("Time (min)"); ax.set_ylabel("σ70 holoenzyme fraction")
ax.set_title("σ70 (RpoD) — Housekeeping")
ax.legend(frameon=False, fontsize=7)
ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# C — σ38
ax = fig.add_subplot(gs[0, 2])
for D, c, lb in zip(datasets, colors, conditions):
    t = D["time"]/60.0
    ax.plot(t, D["f_sigma38"], color=c, lw=1.4, label=lb.replace("\n"," "))
ax.set_xlabel("Time (min)"); ax.set_ylabel("σ38 holoenzyme fraction")
ax.set_title("σ38 (RpoS) — Stress Response")
ax.legend(frameon=False, fontsize=7)
ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# D — Growth rate
ax = fig.add_subplot(gs[0, 3])
for D, c, lb in zip(datasets, colors, conditions):
    t = D["time"]/60.0
    ax.plot(t, np.maximum(D["growth_rate"]*3600, 0), color=c, lw=1.4, label=lb.replace("\n"," "))
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth Rate — The Trade-off")
ax.legend(frameon=False, fontsize=7)
ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# ── Bottom row: Bar charts (final values) ────────────────────────────────────

x = np.arange(4)
w = 0.6

# E — ppGpp bar
ax = fig.add_subplot(gs[1, 0])
vals = [ml(D["ppgpp_uM"]) for D in datasets]
bars = ax.bar(x, vals, w, color=colors, edgecolor="black", lw=0.5)
for xi, v in enumerate(vals):
    ax.text(xi, v + max(vals)*0.02, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp Level")
# Literature reference lines
ax.axhline(50, color="#888", ls=":", lw=0.8)
ax.text(3.5, 53, "Cashel 1996\n(exponential)", fontsize=6, color="#888", ha="right")
ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# F — σ70 bar
ax = fig.add_subplot(gs[1, 1])
vals = [ml(D["f_sigma70"]) for D in datasets]
bars = ax.bar(x, vals, w, color=colors, edgecolor="black", lw=0.5)
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("σ70 holoenzyme fraction")
ax.set_title("σ70 Fraction")
ax.axhline(0.85, color="#888", ls=":", lw=0.8)
ax.text(3.5, 0.86, "Grigorova 2006", fontsize=6, color="#888", ha="right")
ax.set_ylim(0, 1.0)
ax.text(-0.12, 1.05, "F", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# G — σ38 bar
ax = fig.add_subplot(gs[1, 2])
vals = [ml(D["f_sigma38"]) for D in datasets]
bars = ax.bar(x, vals, w, color=colors, edgecolor="black", lw=0.5)
for xi, v in enumerate(vals):
    ax.text(xi, v + max(vals)*0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("σ38 holoenzyme fraction")
ax.set_title("σ38 Fraction")
ax.axhline(0.05, color="#888", ls=":", lw=0.8)
ax.text(3.5, 0.055, "Grigorova 2006", fontsize=6, color="#888", ha="right")
ax.text(-0.12, 1.05, "G", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

# H — Growth rate bar
ax = fig.add_subplot(gs[1, 3])
vals = [ml(D["growth_rate"])*3600 for D in datasets]
bars = ax.bar(x, vals, w, color=colors, edgecolor="black", lw=0.5)
for xi, v in enumerate(vals):
    ax.text(xi, v + max(vals)*0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth Rate")
ax.axhline(1.04, color="#888", ls=":", lw=0.8)
ax.text(3.5, 1.07, "Bremer &\nDennis 2008", fontsize=6, color="#888", ha="right")
ax.text(-0.12, 1.05, "H", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

for e in ("png", "pdf"):
    fig.savefig(f"{OUT}/money_figure.{e}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  money_figure")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  THE MONEY FIGURE — SUMMARY")
print("="*70)
print(f"\n  {'Condition':<20} {'ppGpp (µM)':>12} {'σ70':>8} {'σ38':>8} {'Growth (h⁻¹)':>14}")
print("  " + "-"*65)
for lb, D in zip(["Exponential","H₂O₂ challenge","Starvation","Osmotic (1.0 Osm)"], datasets):
    print(f"  {lb:<20} {ml(D['ppgpp_uM']):>12.1f} {ml(D['f_sigma70']):>8.3f} {ml(D['f_sigma38']):>8.4f} {ml(D['growth_rate'])*3600:>14.3f}")
print(f"\n  Figure saved to: {OUT}/money_figure.{{png,pdf}}")
print("="*70)
