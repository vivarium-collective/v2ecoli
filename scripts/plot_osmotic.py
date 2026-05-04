#!/usr/bin/env python3
"""
plot_osmotic.py — Osmotic stress analysis and experimental validation.
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

C_BASE="#4DAC26"; C_MOD="#FF7F00"; C_SEV="#B2182B"; C_GRAY="#AAAAAA"
C_S70="#2166AC"; C_S38="#D6604D"; C_LIT="#888888"
OUT = "out/osmotic_analysis"; os.makedirs(OUT, exist_ok=True)

def safe(v, fl=1e-9): return np.maximum(np.asarray(v,float), fl)
def ml(a, f=0.5):
    n=len(a); return float(np.nanmean(a[int(n*(1-f)):]))
def plbl(ax,s,x=-0.12,y=1.05):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
def sv(fig,name):
    for e in ("png","pdf"):
        fig.savefig(f"{OUT}/{name}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  {name}")

def load(p):
    with open(p) as f: d=json.load(f)
    s=d['snapshots']
    return {k: np.array([x.get(k,0) for x in s]) for k in s[0]}

print("Loading ...")
B = load("out/scenarios/osm_baseline.json")
M = load("out/scenarios/osm_moderate.json")
S = load("out/scenarios/osm_severe.json")
for D in [B,M,S]: D["t_min"] = D["time"]/60.0
print(f"  Baseline: {len(B['time'])} pts, Moderate: {len(M['time'])} pts, Severe: {len(S['time'])} pts\n")

ALL = [(B,"t_min",C_BASE,"Exponential"),(M,"t_min",C_MOD,"Moderate (0.6 Osm)"),(S,"t_min",C_SEV,"Severe (1.0 Osm)")]
onset = 10.0

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Osmotic Stress Response: Physical & Signaling (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Physical & signaling ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["turgor_atm"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Turgor pressure (atm)")
ax.set_title("Turgor pressure\n(Pilizota & Shaevitz 2012)")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["volume_fraction"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Cell volume (% of normal)")
ax.set_title("Cell volume\n(water efflux)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["ompr_p_fraction"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OmpR-P (% of total)")
ax.set_title("EnvZ/OmpR signaling\n(Batchelor & Goulian 2003)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["k_plus_mM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.axhline(600, color=C_GRAY, ls=":", lw=0.5)
ax.text(1, 610, "max K⁺ (Dinnbier 1988)", fontsize=6, color=C_GRAY)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Cytoplasmic K⁺ (mM)")
ax.set_title("K⁺ accumulation\n(TrkA/Kdp uptake)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["glutamate_mM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Cytoplasmic glutamate (mM)")
ax.set_title("Glutamate synthesis\n(charge balance with K⁺)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["trehalose_mM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Trehalose (mM)")
ax.set_title("Trehalose synthesis\n(otsBA, σ38-dependent)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig1_osmotic_physical_signaling")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sigma Factors & Growth Under Osmotic Stress (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Sigma & growth ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["f_sigma70"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("σ70 (RpoD) fraction"); ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["f_sigma38"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("σ38 (RpoS) fraction\n(stabilised under osmotic stress)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["ppgpp_uM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp concentration"); ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], np.maximum(D["growth_rate"]*3600,0), color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate\n(Cayley et al. 1991)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["dry_mass"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
ax.set_title("Cell dry mass"); ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["growth_inhibition"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth inhibition (%)")
ax.set_title("Growth inhibition\n(linear above 0.28 Osm)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig2_osmotic_sigma_growth")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Experimental Validation (2×2)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Validation ...")
fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
fig.subplots_adjust(hspace=0.50, wspace=0.42)

# A — Growth rate vs osmolarity (model vs Cayley 1991)
ax = axes[0,0]
osm_range = np.linspace(0.2, 1.5, 50)
# Cayley 1991: linear decrease above 0.28 Osm, zero at 1.8 Osm
growth_lit = np.maximum(0, 1.0 - (osm_range - 0.28) / 1.52)
ax.plot(osm_range, growth_lit, color=C_LIT, lw=2, ls="--", label="Cayley et al. 1991")
# Model points
osm_pts = [0.28, 0.6, 1.0]
growth_pts = [ml(B["growth_rate"])*3600, ml(M["growth_rate"])*3600, ml(S["growth_rate"])*3600]
growth_norm = [g / growth_pts[0] for g in growth_pts]
ax.scatter(osm_pts, growth_norm, c=[C_BASE, C_MOD, C_SEV], s=80, zorder=5,
           edgecolors="black", lw=0.5, label="Model")
ax.set_xlabel("Osmolarity (Osm)"); ax.set_ylabel("Relative growth rate")
ax.set_title("Growth vs Osmolarity\n(Cayley et al. 1991)")
ax.legend(frameon=False); plbl(ax,"A")

# B — K⁺ accumulation (model vs Dinnbier 1988)
ax = axes[0,1]
# Dinnbier 1988: K⁺ rises from ~200 to ~600 mM at 0.5 Osm upshift
lit_osm = [0.28, 0.5, 0.8, 1.0]
lit_k = [200, 400, 550, 600]
ax.plot(lit_osm, lit_k, "s--", color=C_LIT, ms=8, lw=1.5, label="Dinnbier et al. 1988")
mod_k = [ml(B["k_plus_mM"]), ml(M["k_plus_mM"]), ml(S["k_plus_mM"])]
ax.scatter([0.28, 0.6, 1.0], mod_k, c=[C_BASE, C_MOD, C_SEV], s=80, zorder=5,
           edgecolors="black", lw=0.5, label="Model")
ax.set_xlabel("Osmolarity (Osm)"); ax.set_ylabel("Cytoplasmic K⁺ (mM)")
ax.set_title("K⁺ Accumulation\n(Dinnbier et al. 1988)")
ax.legend(frameon=False); plbl(ax,"B")

# C — σ38 increase under osmotic stress (Jishage et al. 1996)
ax = axes[1,0]
# Jishage 1996: σ38 increases 2-3× under high osmolarity
bar_names = ["Exponential\n(0.28 Osm)", "Moderate\n(0.6 Osm)", "Severe\n(1.0 Osm)"]
bar_s38 = [ml(B["f_sigma38"]), ml(M["f_sigma38"]), ml(S["f_sigma38"])]
bar_colors = [C_BASE, C_MOD, C_SEV]
x = np.arange(3)
ax.bar(x, bar_s38, color=bar_colors, edgecolor="white", width=0.6)
ax.axhline(0.05, color=C_LIT, ls="--", lw=1, label="Grigorova 2006 (exp)")
for xi, v in enumerate(bar_s38):
    ax.text(xi, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(bar_names, fontsize=8)
ax.set_ylabel("σ38 holoenzyme fraction")
ax.set_title("σ38 Under Osmotic Stress\n(Jishage et al. 1996: 2-3× increase)")
ax.legend(frameon=False); plbl(ax,"C")

# D — Porin switch (OmpR-P dependent)
ax = axes[1,1]
bar_ompr = [ml(B["ompr_p_fraction"])*100, ml(M["ompr_p_fraction"])*100, ml(S["ompr_p_fraction"])*100]
bar_ompC = [ml(B["porin_ompC_up"])*100, ml(M["porin_ompC_up"])*100, ml(S["porin_ompC_up"])*100]
x = np.arange(3); w = 0.35
ax.bar(x-w/2, bar_ompr, w, color=[C_BASE,C_MOD,C_SEV], alpha=0.7, label="OmpR-P (%)")
ax.bar(x+w/2, bar_ompC, w, color=[C_BASE,C_MOD,C_SEV], alpha=0.4, hatch="//", label="OmpC activation (%)")
ax.set_xticks(x); ax.set_xticklabels(bar_names, fontsize=8)
ax.set_ylabel("Percentage (%)")
ax.set_title("Porin Switch\n(OmpR-P → OmpC ↑, OmpF ↓)")
ax.legend(frameon=False); plbl(ax,"D")

sv(fig, "fig3_osmotic_validation")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  OSMOTIC STRESS ANALYSIS — SUMMARY")
print("="*80)
print(f"\n  {'Metric':<30} {'Exponential':>12} {'Moderate':>12} {'Severe':>12}")
print("  " + "-"*70)
rows = [
    ("Osmolarity (Osm)",       ml(B["osmolarity"]),    ml(M["osmolarity"]),    ml(S["osmolarity"])),
    ("Turgor (atm)",           ml(B["turgor_atm"]),    ml(M["turgor_atm"]),    ml(S["turgor_atm"])),
    ("Volume (%)",             ml(B["volume_fraction"])*100, ml(M["volume_fraction"])*100, ml(S["volume_fraction"])*100),
    ("OmpR-P (%)",             ml(B["ompr_p_fraction"])*100, ml(M["ompr_p_fraction"])*100, ml(S["ompr_p_fraction"])*100),
    ("K⁺ (mM)",               ml(B["k_plus_mM"]),     ml(M["k_plus_mM"]),     ml(S["k_plus_mM"])),
    ("Glutamate (mM)",        ml(B["glutamate_mM"]),  ml(M["glutamate_mM"]),  ml(S["glutamate_mM"])),
    ("Trehalose (mM)",        ml(B["trehalose_mM"]),  ml(M["trehalose_mM"]),  ml(S["trehalose_mM"])),
    ("Growth inhibition (%)", ml(B["growth_inhibition"])*100, ml(M["growth_inhibition"])*100, ml(S["growth_inhibition"])*100),
    ("Growth rate (h⁻¹)",    ml(B["growth_rate"])*3600, ml(M["growth_rate"])*3600, ml(S["growth_rate"])*3600),
    ("f_σ70",                  ml(B["f_sigma70"]),     ml(M["f_sigma70"]),     ml(S["f_sigma70"])),
    ("f_σ38",                  ml(B["f_sigma38"]),     ml(M["f_sigma38"]),     ml(S["f_sigma38"])),
    ("ppGpp (µM)",             ml(B["ppgpp_uM"]),      ml(M["ppgpp_uM"]),      ml(S["ppgpp_uM"])),
    ("Dry mass (fg)",          ml(B["dry_mass"]),       ml(M["dry_mass"]),       ml(S["dry_mass"])),
]
for name, *vals in rows:
    print(f"  {name:<30}" + "".join(f"{v:>12.4g}" for v in vals))

print(f"\n  3 figures (PNG + PDF) saved to: {OUT}/")
print("="*80)
