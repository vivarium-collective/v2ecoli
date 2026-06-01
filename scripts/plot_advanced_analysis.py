#!/usr/bin/env python3
"""
plot_advanced_analysis.py — Figures for timed stress, OxyR feedback,
multi-generation, and experimental validation.

Reads: out/scenarios/{sigma_exp, h2o2_stress, starvation,
       timed_h2o2, timed_starvation, multigen_starv}.json
Outputs: out/advanced_analysis/*.{png,pdf}
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

C_BASE="#2166AC"; C_SIG="#4DAC26"; C_H2O2="#D6604D"; C_STARV="#B2182B"
C_S70="#2166AC"; C_S38="#D6604D"; C_GRAY="#AAAAAA"; C_FEED="#FF7F00"
OUT = "out/advanced_analysis"; os.makedirs(OUT, exist_ok=True)

def safe(v, fl=1e-9): return np.maximum(np.asarray(v,float), fl)
def ml(a, f=0.5):
    n=len(a); return float(np.nanmean(a[int(n*(1-f)):]))
def plbl(ax,s,x=-0.12,y=1.05):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
def sv(fig,name):
    for e in ("png","pdf"):
        fig.savefig(f"{OUT}/{name}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  {name}")

def load(path):
    with open(path) as f: d = json.load(f)
    snaps = d["snapshots"]
    out = {k: np.array([s[k] for s in snaps]) for k in snaps[0]}
    out["t_min"] = out["time"]/60.0
    out["_meta"] = {k:v for k,v in d.items() if k!="snapshots"}
    return out

print("Loading data ...")
D2 = load("out/scenarios/sigma_exp.json")
TH = load("out/scenarios/timed_h2o2.json")
TS = load("out/scenarios/timed_starvation.json")
MG = load("out/scenarios/multigen_starv.json")
print("  All loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Timed H₂O₂ Challenge with OxyR Feedback (2×3)
# Shows: pre-stress → stress onset → adaptive homeostasis
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Timed H₂O₂ + OxyR feedback ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)
t = TH["t_min"]; onset = 10.0  # minutes

ax=axes[0,0]
ax.semilogy(t, safe(TH["h2o2_uM"],1e-4), color=C_H2O2, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.text(onset+0.5, ax.get_ylim()[1]*0.5, "stress\nonset", fontsize=7, color=C_GRAY)
ax.set_xlabel("Time (min)"); ax.set_ylabel("[H₂O₂] (µM)")
ax.set_title("H₂O₂ concentration\n(100 µM/s challenge at t=10 min)")
plbl(ax,"A")

ax=axes[0,1]
ax.plot(t, TH["oxyr_ox"], color=C_H2O2, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OxyR oxidised fraction")
ax.set_title("OxyR activation\n(Kox=0.2 µM, Åslund 1999)")
ax.set_ylim(-0.05, 1.0); plbl(ax,"B")

ax=axes[0,2]
ax.plot(t, TH["oxyr_fc"], color=C_H2O2, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.axhline(1.0, color=C_GRAY, ls=":", lw=0.6)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Fold change")
ax.set_title("OxyR gene upregulation\n(katG, ahpCF, dps)")
plbl(ax,"C")

ax=axes[1,0]
ax.plot(t, TH.get("extra_katg", np.zeros_like(t)), color=C_FEED, lw=1.2, label="Extra KatG")
ax.plot(t, TH.get("extra_ahpcf", np.zeros_like(t)), color="#4DAC26", lw=1.2, label="Extra AhpCF")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Extra enzymes (molecules)")
ax.set_title("OxyR feedback: enzyme induction\n(adaptive scavenging)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
ax.semilogy(t, safe(TH["dna_dmg"],1e-5), color=C_H2O2, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("DNA damage rate (a.u./s)")
ax.set_title("DNA damage\n(Fenton chemistry)")
plbl(ax,"E")

ax=axes[1,2]
ax.plot(t, np.maximum(TH["growth_rate"]*3600, 0), color=C_H2O2, lw=1.2, label="H₂O₂ challenge")
ax.plot(D2["t_min"], np.maximum(D2["growth_rate"]*3600, 0), color=C_SIG, lw=1.0, alpha=0.5, label="Exponential")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig1_timed_h2o2_feedback")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Timed Starvation with Sigma Redistribution (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Timed starvation ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)
t = TS["t_min"]; onset = 10.0

ax=axes[0,0]
ax.plot(t, TS["ppgpp_uM"], color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.axhline(200, color=C_GRAY, ls=":", lw=0.6)
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp concentration\n(starvation at t=10 min)")
plbl(ax,"A")

ax=axes[0,1]
ax.plot(t, TS["f_sigma70"], color=C_S70, lw=1.2, label="σ70")
ax.plot(t, TS["f_sigma38"], color=C_S38, lw=1.2, label="σ38")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("Sigma factor redistribution")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
ax.plot(t, np.maximum(TS["growth_rate"]*3600, 0), color=C_STARV, lw=1.2, label="Starvation")
ax.plot(D2["t_min"], np.maximum(D2["growth_rate"]*3600, 0), color=C_SIG, lw=1.0, alpha=0.5, label="Exponential")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
ax.plot(t, TS["K_E70_eff_nM"], color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("$K_{d,σ70}^{eff}$ (nM)")
ax.set_title("σ70–RNAP affinity\n(ppGpp weakens binding)")
plbl(ax,"D")

ax=axes[1,1]
ax.plot(t, TS["phase"], color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Phase index (0=exp, 1=stat)")
ax.set_title("Growth phase\n(0=exp, 1=stationary)")
ax.set_ylim(-0.05, 1.1); plbl(ax,"E")

ax=axes[1,2]
ax.plot(t, TS["dry_mass"], color=C_STARV, lw=1.2, label="Starvation")
ax.plot(D2["t_min"], D2["dry_mass"], color=C_SIG, lw=1.0, alpha=0.5, label="Exponential")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
ax.set_title("Cell dry mass")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig2_timed_starvation")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Multi-Generation Starvation (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Multi-generation starvation ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)
t = MG["t_min"]; onset = 5.0

ax=axes[0,0]
ax.plot(t, MG["ppgpp_uM"], color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.axvline(42, color="#999", ls=":", lw=0.6, alpha=0.5)
ax.text(42.5, ax.get_ylim()[0]*1.1 if ax.get_ylim()[0]>0 else 10, "gen 1→2", fontsize=6, color="#999")
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp across generations\n(starvation at t=5 min)")
plbl(ax,"A")

ax=axes[0,1]
ax.plot(t, MG["f_sigma70"], color=C_S70, lw=1.2, label="σ70")
ax.plot(t, MG["f_sigma38"], color=C_S38, lw=1.2, label="σ38")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("Sigma factors across generations")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
ax.plot(t, np.maximum(MG["growth_rate"]*3600, 0), color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate across generations")
plbl(ax,"C")

ax=axes[1,0]
ax.plot(t, MG["dry_mass"], color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
ax.set_title("Cell dry mass\n(2 generations)")
plbl(ax,"D")

ax=axes[1,1]
ax.plot(t, MG["phase"], color=C_STARV, lw=1.2)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Phase index (0=exp, 1=stat)")
ax.set_title("Growth phase persistence")
ax.set_ylim(-0.05, 1.1); plbl(ax,"E")

ax=axes[1,2]
ax.plot(t, MG["protein_mass"], color=C_STARV, lw=1.2, label="Protein")
ax.plot(t, MG["rRna_mass"], color=C_S70, lw=1.2, ls="--", label="rRNA")
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.8, alpha=0.7)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Mass (fg)")
ax.set_title("Protein & rRNA mass")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig3_multigen_starvation")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Experimental Validation (2×2)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Experimental validation ...")
fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
fig.subplots_adjust(hspace=0.50, wspace=0.42)

# Literature values (Bremer & Dennis 2008, Grigorova 2006, Cashel 1996, Imlay 2013)
lit = {
    "Growth rate (h⁻¹)":  1.04,   # Bremer & Dennis 2008
    "Dry mass (fg)":       380,    # Bremer & Dennis 2008 (at µ=1.04)
    "Protein mass (fg)":   280,    # Bremer & Dennis 2008
    "RNA mass (fg)":       77,     # Bremer & Dennis 2008
    "ppGpp (µM)":          50,     # Cashel et al. 1996
    "H₂O₂ SS (µM)":       0.020,  # Imlay 2013 (~20 nM)
    "f_σ70":               0.85,   # Grigorova et al. 2006
    "f_σ38":               0.05,   # Grigorova et al. 2006
}

# Model values (from sigma_exp, last 50%)
mod = {
    "Growth rate (h⁻¹)":  ml(D2["growth_rate"]) * 3600,
    "Dry mass (fg)":       ml(D2["dry_mass"]),
    "Protein mass (fg)":   ml(D2["protein_mass"]),
    "RNA mass (fg)":       ml(D2["rRna_mass"]) + ml(D2["tRna_mass"]) + ml(D2["mRna_mass"]),
    "ppGpp (µM)":          ml(D2["ppgpp_uM"]),
    "H₂O₂ SS (µM)":       ml(D2["h2o2_uM"]),
    "f_σ70":               ml(D2["f_sigma70"]),
    "f_σ38":               ml(D2["f_sigma38"]),
}

# A — Log-log scatter: model vs experiment
ax = axes[0,0]
names = list(lit.keys())
lit_vals = np.array([lit[n] for n in names])
mod_vals = np.array([mod[n] for n in names])
valid = (lit_vals > 0) & (mod_vals > 0)
ax.scatter(np.log10(lit_vals[valid]), np.log10(mod_vals[valid]),
           c=C_BASE, s=50, zorder=5, edgecolors="black", lw=0.5)
for i, n in enumerate(names):
    if valid[i]:
        ax.annotate(n, (np.log10(lit_vals[i]), np.log10(mod_vals[i])),
                    fontsize=6, xytext=(5,5), textcoords="offset points")
lims = [min(np.log10(lit_vals[valid]).min(), np.log10(mod_vals[valid]).min()) - 0.5,
        max(np.log10(lit_vals[valid]).max(), np.log10(mod_vals[valid]).max()) + 0.5]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
from scipy.stats import pearsonr
r, _ = pearsonr(np.log10(lit_vals[valid]), np.log10(mod_vals[valid]))
ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9, va="top", style="italic")
ax.set_xlabel("log₁₀(Experimental)"); ax.set_ylabel("log₁₀(Model)")
ax.set_title("Model vs Experiment\n(log-log correlation)")
plbl(ax,"A")

# B — Bar chart: key metrics
ax = axes[0,1]
bar_names = ["Growth\n(h⁻¹)", "Protein\n(fg)", "RNA\n(fg)", "ppGpp\n(µM)"]
bar_lit = [lit["Growth rate (h⁻¹)"], lit["Protein mass (fg)"], lit["RNA mass (fg)"], lit["ppGpp (µM)"]]
bar_mod = [mod["Growth rate (h⁻¹)"], mod["Protein mass (fg)"], mod["RNA mass (fg)"], mod["ppGpp (µM)"]]
x = np.arange(len(bar_names)); w = 0.35
ax.bar(x-w/2, bar_lit, w, color=C_GRAY, alpha=0.8, label="Experiment", edgecolor="white")
ax.bar(x+w/2, bar_mod, w, color=C_BASE, alpha=0.8, label="Model", edgecolor="white")
for xi, (vl, vm) in enumerate(zip(bar_lit, bar_mod)):
    if vl > 0:
        pct = (vm-vl)/vl*100
        ax.text(xi, max(vl,vm)*1.05, f"{pct:+.0f}%", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(bar_names, fontsize=8)
ax.set_ylabel("Value (mixed units)"); ax.set_title("Cell Composition\n(Bremer & Dennis 2008)")
ax.legend(frameon=False); plbl(ax,"B")

# C — Sigma fractions: model vs Grigorova 2006
ax = axes[1,0]
sig_names = ["σ70", "σ38", "σ32", "σ24", "σ54"]
sig_lit = [0.85, 0.05, 0.03, 0.03, 0.04]  # Grigorova 2006
sig_mod = [ml(D2[f"f_sigma{k}"]) for k in [70,38,32,24,54]]
x = np.arange(5); w = 0.35
ax.bar(x-w/2, sig_lit, w, color=C_GRAY, alpha=0.8, label="Grigorova 2006", edgecolor="white")
ax.bar(x+w/2, sig_mod, w, color=C_BASE, alpha=0.8, label="Model", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(sig_names, fontsize=9)
ax.set_ylabel("Holoenzyme fraction")
ax.set_title("Sigma Factor Fractions\n(Grigorova et al. 2006)")
ax.legend(frameon=False); plbl(ax,"C")

# D — Starvation response: model vs Traxler 2008
ax = axes[1,1]
# Traxler 2008: ppGpp ~300 µM, growth drops ~60%, rRNA drops ~50%
trax_names = ["ppGpp\n(µM)", "Growth\n(h⁻¹)", "rRNA\n(fg)"]
trax_exp = [ml(D2["ppgpp_uM"]), ml(D2["growth_rate"])*3600, ml(D2["rRna_mass"])]
trax_starv = [ml(TS["ppgpp_uM"]), ml(TS["growth_rate"])*3600, ml(TS["rRna_mass"])]
trax_lit_starv = [300, 0.4, 35]  # approximate from Traxler 2008
x = np.arange(3); w = 0.25
ax.bar(x-w, trax_exp, w, color=C_SIG, alpha=0.8, label="Model (exp)", edgecolor="white")
ax.bar(x, trax_starv, w, color=C_STARV, alpha=0.8, label="Model (starv)", edgecolor="white")
ax.bar(x+w, trax_lit_starv, w, color=C_GRAY, alpha=0.8, label="Traxler 2008", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(trax_names, fontsize=8)
ax.set_ylabel("Value (mixed units)"); ax.set_title("Starvation Response\n(Traxler et al. 2008)")
ax.legend(frameon=False, fontsize=7); plbl(ax,"D")

sv(fig, "fig4_experimental_validation")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  ADVANCED ANALYSIS SUMMARY")
print("="*70)

print("\n  Timed H₂O₂ (onset=10 min, 100 µM/s):")
print(f"    Pre-stress H₂O₂:  {TH['h2o2_uM'][50]:.4f} µM")
print(f"    Peak H₂O₂:        {TH['h2o2_uM'].max():.4f} µM")
print(f"    Final H₂O₂:       {TH['h2o2_uM'][-1]:.4f} µM (adaptive homeostasis)")
print(f"    Peak OxyR FC:      {TH['oxyr_fc'].max():.2f}×")
print(f"    Final extra KatG:  {TH.get('extra_katg',np.zeros(1))[-1]:.0f} molecules")

print("\n  Timed Starvation (onset=10 min, ppGpp=250k):")
print(f"    Pre-stress ppGpp:  {TS['ppgpp_uM'][50]:.1f} µM")
print(f"    Post-stress ppGpp: {TS['ppgpp_uM'][70]:.1f} µM")
print(f"    Pre-stress growth: {TS['growth_rate'][50]*3600:.3f} h⁻¹")
print(f"    Post-stress growth:{TS['growth_rate'][-1]*3600:.3f} h⁻¹")

print("\n  Multi-Generation Starvation:")
print(f"    Total sim time:    {MG['_meta']['sim_time']:.0f}s ({MG['_meta']['sim_time']/60:.0f} min)")
print(f"    Final growth:      {MG['growth_rate'][-1]*3600:.3f} h⁻¹")
print(f"    Final ppGpp:       {MG['ppgpp_uM'][-1]:.1f} µM")

print("\n  Experimental Validation:")
for n in names:
    if lit[n] > 0 and mod[n] > 0:
        err = abs(mod[n]-lit[n])/lit[n]*100
        print(f"    {n:<25} Lit={lit[n]:.4g}  Model={mod[n]:.4g}  Error={err:.1f}%")

print(f"\n  4 figures (PNG + PDF) saved to: {OUT}/")
print("="*70)
