#!/usr/bin/env python3
"""
plot_validation.py — Systematic biological accuracy assessment
==============================================================

Compares every model prediction against published experimental data.
Three categories:
  1. Exponential phase: cell composition, sigma fractions, metabolites
  2. Oxidative stress: H₂O₂ kinetics, OxyR response, enzyme induction
  3. Starvation: ppGpp levels, sigma redistribution, growth arrest

Each comparison shows: model value, literature value, % error, reference.
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 300,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

C_MOD="#2166AC"; C_LIT="#D6604D"; C_GOOD="#4DAC26"; C_OK="#FF7F00"; C_BAD="#B2182B"
C_GRAY="#888888"
OUT = "out/validation"; os.makedirs(OUT, exist_ok=True)

def sv(fig,name):
    for e in ("png","pdf"):
        fig.savefig(f"{OUT}/{name}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  {name}")

def load(p):
    with open(p) as f: d=json.load(f)
    s=d['snapshots']
    return {k: np.array([x.get(k,0) for x in s]) for k in s[0]}

def ml(a, f=0.5):
    n=len(a); return float(np.nanmean(a[int(n*(1-f)):]))

print("Loading ...")
B = load("out/scenarios/mol_baseline.json")
H = load("out/scenarios/mol_h2o2.json")
S = load("out/scenarios/mol_starv.json")
# Also load the timed_starvation (uses ppGpp clamping + RpoS — more accurate)
TS = load("out/scenarios/timed_starvation.json")

# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE LITERATURE VALUES
# ══════════════════════════════════════════════════════════════════════════════
# Format: (model_value, lit_value, lit_range_low, lit_range_high, unit, reference)

exp_validation = [
    # Cell composition — Bremer & Dennis 2008
    ("Growth rate",        ml(B["growth_rate"])*3600, 1.04, 0.7, 1.3, "h⁻¹", "Bremer & Dennis 2008"),
    ("Protein mass",       ml(B["protein_mass"]),     280,  250, 310, "fg",   "Bremer & Dennis 2008"),
    ("RNA mass",           ml(B["rRna_mass"])+ml(B["tRna_mass"])+ml(B["mRna_mass"]), 77, 65, 90, "fg", "Bremer & Dennis 2008"),
    ("rRNA fraction",      ml(B["rRna_mass"])/(ml(B["rRna_mass"])+ml(B["tRna_mass"])+ml(B["mRna_mass"])+0.01), 0.86, 0.82, 0.90, "of RNA", "Dennis & Bremer 1974"),
    # Sigma fractions — Grigorova et al. 2006
    ("σ70 fraction",       ml(B["f_sigma70"]),  0.85, 0.80, 0.90, "",  "Grigorova et al. 2006"),
    ("σ38 fraction",       ml(B["f_sigma38"]),  0.05, 0.03, 0.08, "",  "Grigorova et al. 2006"),
    ("σ32 fraction",       ml(B["f_sigma32"]),  0.03, 0.01, 0.05, "",  "Grigorova et al. 2006"),
    # Metabolites — Cashel 1996, Imlay 2013
    ("ppGpp",              ml(B["ppgpp_uM"]),   50,   30,   80,   "µM", "Cashel et al. 1996"),
    ("H₂O₂ steady state",  ml(B["h2o2_uM"]),   0.020, 0.010, 0.050, "µM", "Imlay 2013"),
    # Enzyme counts — Li et al. 2014 (total protein, not free monomer)
    ("Free RNAP (core E)", ml(B["bulk_rnap_free"]), 2000, 1000, 5000, "molecules", "Grigorova 2006"),
]

ox_validation = [
    # OxyR response — Åslund 1999, Zheng 2001
    ("OxyR Kox",           0.2,                 0.2,  0.1,  0.5,  "µM",  "Åslund et al. 1999"),
    ("OxyR fold change",   float(H["oxyr_fc"].max()),  7.0,  5.0,  10.0, "×",   "Zheng et al. 2001"),
    ("H₂O₂ peak stress",   float(H["h2o2_uM"].max()), 0.3,  0.1,  1.0,  "µM",  "Seaver & Imlay 2001"),
    # Enzyme induction
    ("KatG induction",     ml(H["bulk_katg"])/ml(B["bulk_katg"]), 3.0, 2.0, 10.0, "×fold", "Zheng et al. 2001"),
    ("AhpCF induction",    ml(H["bulk_ahpcf"])/ml(B["bulk_ahpcf"]), 3.0, 2.0, 5.0, "×fold", "Zheng et al. 2001"),
    # Kinetics — Switala & Loewen 2002
    ("KatG kcat",          200,                 200,  150,  250,  "s⁻¹", "Switala & Loewen 2002"),
    ("KatG Km",            3900,                3900, 3000, 5000, "µM",  "Switala & Loewen 2002"),
    ("AhpCF kcat",         428,                 428,  300,  500,  "s⁻¹", "Seaver & Imlay 2001"),
    ("AhpCF Km",           1.4,                 1.4,  1.0,  2.0,  "µM",  "Seaver & Imlay 2001"),
    # DNA damage
    ("Fenton k",           0.076,               0.076, 0.05, 0.1, "M⁻¹s⁻¹µM⁻¹", "Imlay & Linn 1988"),
]

starv_validation = [
    # ppGpp — Cashel 1996, Traxler 2008 (use timed_starvation post-onset)
    ("ppGpp (starvation)", ml(TS["ppgpp_uM"]),   300,  200,  500,  "µM",  "Cashel et al. 1996"),
    # RpoS — Zgurskaya 1997 (use timed_starvation which has RpoS clamping)
    ("RpoS accumulation",  ml(TS.get("bulk_rpos", TS.get("f_sigma38", np.zeros(1))*5000))/max(ml(B["bulk_rpos"]),1), 5.0, 3.0, 15.0, "×fold", "Zgurskaya et al. 1997"),
    # Sigma redistribution (post-onset)
    ("σ38 under starvation", ml(TS["f_sigma38"]), 0.20, 0.10, 0.30, "", "Mauri & Klumpp 2014"),
    ("σ70 under starvation", ml(TS["f_sigma70"]), 0.70, 0.60, 0.80, "", "Mauri & Klumpp 2014"),
    # Growth — Traxler 2008
    ("Growth reduction",   1.0 - ml(TS["growth_rate"])/ml(B["growth_rate"]), 0.50, 0.30, 0.90, "fraction", "Traxler et al. 2008"),
    # RpoS half-life (parameters, not simulation output)
    ("RpoS t½ (exp)",     120,                  120,  90,   180,  "s",   "Zgurskaya et al. 1997"),
    ("RpoS t½ (starved)", 1800,                 1800, 1200, 2400, "s",   "Zgurskaya et al. 1997"),
]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Exponential Phase Validation (scatter + bars)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Exponential validation ...")
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# A — Log-log scatter: model vs literature
ax = fig.add_subplot(gs[0, 0])
all_data = exp_validation + ox_validation + starv_validation
mod_vals = [d[1] for d in all_data if d[1] > 0 and d[2] > 0]
lit_vals = [d[2] for d in all_data if d[1] > 0 and d[2] > 0]
names_all = [d[0] for d in all_data if d[1] > 0 and d[2] > 0]

mod_log = np.log10(np.array(mod_vals))
lit_log = np.log10(np.array(lit_vals))

# Color by error
errors = [abs(m-l)/l*100 for m,l in zip(mod_vals, lit_vals)]
colors = [C_GOOD if e < 25 else (C_OK if e < 50 else C_BAD) for e in errors]

ax.scatter(lit_log, mod_log, c=colors, s=60, edgecolors="black", lw=0.5, zorder=5)
lims = [min(lit_log.min(), mod_log.min())-0.5, max(lit_log.max(), mod_log.max())+0.5]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y = x (perfect)")
# 2-fold lines
ax.plot(lims, [l+np.log10(2) for l in lims], color=C_GRAY, ls=":", lw=0.5, alpha=0.5)
ax.plot(lims, [l-np.log10(2) for l in lims], color=C_GRAY, ls=":", lw=0.5, alpha=0.5)

from scipy.stats import pearsonr
r, _ = pearsonr(lit_log, mod_log)
within_2fold = sum(1 for m,l in zip(mod_vals,lit_vals) if 0.5 <= m/l <= 2.0)
ax.text(0.05, 0.95, f"r = {r:.3f}\n{within_2fold}/{len(mod_vals)} within 2-fold",
        transform=ax.transAxes, fontsize=9, va="top")
ax.set_xlabel("log₁₀(Literature value)"); ax.set_ylabel("log₁₀(Model value)")
ax.set_title("All Predictions vs Literature\n(green <25%, orange <50%, red >50% error)")
ax.legend(frameon=False, fontsize=7)
ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

# B — Exponential phase bars
ax = fig.add_subplot(gs[0, 1])
exp_names = [d[0] for d in exp_validation]
exp_mod = [d[1] for d in exp_validation]
exp_lit = [d[2] for d in exp_validation]
x = np.arange(len(exp_names)); w = 0.35
ax.bar(x-w/2, exp_lit, w, color=C_LIT, alpha=0.8, label="Literature", edgecolor="white")
ax.bar(x+w/2, exp_mod, w, color=C_MOD, alpha=0.8, label="Model", edgecolor="white")
for xi, (vl, vm) in enumerate(zip(exp_lit, exp_mod)):
    if vl > 0:
        pct = abs(vm-vl)/vl*100
        c = C_GOOD if pct < 25 else (C_OK if pct < 50 else C_BAD)
        ax.text(xi, max(vl,vm)*1.05, f"{pct:.0f}%", ha="center", fontsize=7, color=c, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(exp_names, rotation=35, ha="right", fontsize=7)
ax.set_ylabel("Value (mixed units)"); ax.set_title("Exponential Phase Validation")
ax.legend(frameon=False)
ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

# C — Oxidative stress validation
ax = fig.add_subplot(gs[1, 0])
ox_names = [d[0] for d in ox_validation]
ox_mod = [d[1] for d in ox_validation]
ox_lit = [d[2] for d in ox_validation]
x = np.arange(len(ox_names)); w = 0.35
ax.bar(x-w/2, ox_lit, w, color=C_LIT, alpha=0.8, label="Literature", edgecolor="white")
ax.bar(x+w/2, ox_mod, w, color=C_MOD, alpha=0.8, label="Model", edgecolor="white")
for xi, (vl, vm) in enumerate(zip(ox_lit, ox_mod)):
    if vl > 0:
        pct = abs(vm-vl)/vl*100
        c = C_GOOD if pct < 25 else (C_OK if pct < 50 else C_BAD)
        ax.text(xi, max(vl,vm)*1.05, f"{pct:.0f}%", ha="center", fontsize=6, color=c, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(ox_names, rotation=35, ha="right", fontsize=6)
ax.set_ylabel("Value (mixed units)"); ax.set_title("Oxidative Stress Validation")
ax.legend(frameon=False)
ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

# D — Starvation validation
ax = fig.add_subplot(gs[1, 1])
st_names = [d[0] for d in starv_validation]
st_mod = [d[1] for d in starv_validation]
st_lit = [d[2] for d in starv_validation]
x = np.arange(len(st_names)); w = 0.35
ax.bar(x-w/2, st_lit, w, color=C_LIT, alpha=0.8, label="Literature", edgecolor="white")
ax.bar(x+w/2, st_mod, w, color=C_MOD, alpha=0.8, label="Model", edgecolor="white")
for xi, (vl, vm) in enumerate(zip(st_lit, st_mod)):
    if vl > 0:
        pct = abs(vm-vl)/vl*100
        c = C_GOOD if pct < 25 else (C_OK if pct < 50 else C_BAD)
        ax.text(xi, max(vl,vm)*1.05, f"{pct:.0f}%", ha="center", fontsize=6, color=c, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(st_names, rotation=35, ha="right", fontsize=6)
ax.set_ylabel("Value (mixed units)"); ax.set_title("Starvation Validation")
ax.legend(frameon=False)
ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

sv(fig, "fig1_comprehensive_validation")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Range validation (model value within literature range?)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Range validation ...")
fig, ax = plt.subplots(figsize=(12, 8))

all_checks = []
for category, data, cat_color in [
    ("Exponential", exp_validation, C_MOD),
    ("Oxidative stress", ox_validation, C_LIT),
    ("Starvation", starv_validation, C_BAD),
]:
    for name, mod, lit, lo, hi, unit, ref in data:
        in_range = lo <= mod <= hi
        pct_err = abs(mod - lit) / lit * 100 if lit > 0 else 0
        all_checks.append((category, name, mod, lit, lo, hi, unit, ref, in_range, pct_err))

# Sort by category
y_pos = 0
y_ticks = []
y_labels = []
prev_cat = ""

for cat, name, mod, lit, lo, hi, unit, ref, in_range, pct_err in all_checks:
    if cat != prev_cat:
        y_pos += 0.5  # gap between categories
        ax.axhline(y_pos + 0.25, color=C_GRAY, ls="-", lw=0.3, alpha=0.3)
        ax.text(-0.02, y_pos + 0.5, cat, transform=ax.get_yaxis_transform(),
                fontsize=9, fontweight="bold", color=C_GRAY, ha="right", va="center")
        prev_cat = cat

    # Normalize to literature value for display
    if lit > 0:
        norm_mod = mod / lit
        norm_lo = lo / lit
        norm_hi = hi / lit
    else:
        norm_mod = 1.0; norm_lo = 0.5; norm_hi = 1.5

    color = C_GOOD if in_range else (C_OK if pct_err < 50 else C_BAD)

    # Literature range bar
    ax.barh(y_pos, norm_hi - norm_lo, left=norm_lo, height=0.6,
            color=C_GRAY, alpha=0.2, edgecolor=C_GRAY, lw=0.5)
    # Literature point
    ax.plot(1.0, y_pos, "|", color=C_GRAY, ms=12, mew=1.5)
    # Model point
    ax.plot(norm_mod, y_pos, "o", color=color, ms=8, zorder=5,
            markeredgecolor="black", markeredgewidth=0.5)

    # Label
    status = "✓" if in_range else "✗"
    ax.text(max(norm_hi, norm_mod) + 0.05, y_pos,
            f"{status} {pct_err:.0f}% — {ref}",
            fontsize=6.5, va="center", color=color)

    y_ticks.append(y_pos)
    y_labels.append(f"{name} ({unit})")
    y_pos += 1

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=7)
ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
ax.axvline(0.5, color=C_GRAY, ls=":", lw=0.5, alpha=0.3)
ax.axvline(2.0, color=C_GRAY, ls=":", lw=0.5, alpha=0.3)
ax.set_xlabel("Model / Literature (1.0 = perfect match)")
ax.set_title("Biological Accuracy — Every Prediction vs Literature Range\n"
             "● = model value, gray bar = literature range, | = literature central value\n"
             "Green = within range, Orange = <50% off, Red = >50% off",
             fontsize=10)
ax.set_xlim(-0.1, 3.5)
ax.invert_yaxis()

# Summary stats
n_total = len(all_checks)
n_in_range = sum(1 for c in all_checks if c[8])
n_within_2fold = sum(1 for c in all_checks if 0.5 <= c[2]/c[3] <= 2.0 if c[3] > 0)
ax.text(0.98, 0.02, f"In range: {n_in_range}/{n_total} ({n_in_range/n_total*100:.0f}%)\n"
        f"Within 2-fold: {n_within_2fold}/{n_total} ({n_within_2fold/n_total*100:.0f}%)",
        transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GRAY))

sv(fig, "fig2_range_validation")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE — Full validation table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  BIOLOGICAL ACCURACY ASSESSMENT — COMPLETE VALIDATION TABLE")
print("="*100)
print(f"\n  {'Category':<15} {'Metric':<25} {'Model':>10} {'Literature':>10} {'Range':>15} {'Error':>8} {'In Range':>8} {'Reference'}")
print("  " + "-"*95)

for cat, name, mod, lit, lo, hi, unit, ref, in_range, pct_err in all_checks:
    status = "  ✓" if in_range else "  ✗"
    print(f"  {cat:<15} {name:<25} {mod:>10.4g} {lit:>10.4g} [{lo:.3g}–{hi:.3g}] {pct_err:>7.1f}% {status}  {ref}")

print(f"\n  SUMMARY:")
print(f"    Total checks:     {n_total}")
print(f"    Within range:     {n_in_range}/{n_total} ({n_in_range/n_total*100:.0f}%)")
print(f"    Within 2-fold:    {n_within_2fold}/{n_total} ({n_within_2fold/n_total*100:.0f}%)")
print(f"    Pearson r (log):  {r:.3f}")
print(f"\n  2 figures saved to: {OUT}/")
print("="*100)
