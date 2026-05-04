#!/usr/bin/env python3
"""
plot_osmotic_molecules.py — Every molecule in the osmotic stress pathway.
Compares exponential vs moderate (0.6 Osm) vs severe (1.0 Osm).
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
C_S70="#2166AC"; C_S38="#D6604D"
OUT = "out/osmotic_molecules"; os.makedirs(OUT, exist_ok=True)

def safe(v, fl=1e-9): return np.maximum(np.asarray(v,float), fl)
def plbl(ax,s,x=-0.12,y=1.05):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
def sv(fig,name):
    for e in ("png","pdf"):
        fig.savefig(f"{OUT}/{name}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  {name}")

def load(p):
    with open(p) as f: d=json.load(f)
    s=d['snapshots']
    out = {k: np.array([x.get(k,0) for x in s]) for k in s[0]}
    out["t_min"] = out["time"]/60.0
    return out

print("Loading ...")
B = load("out/scenarios/osm_baseline.json")
M = load("out/scenarios/osm_moderate.json")
S = load("out/scenarios/osm_severe.json")
print(f"  Loaded: {len(B['time'])}, {len(M['time'])}, {len(S['time'])} pts\n")

ALL = [(B,C_BASE,"Exponential"),(M,C_MOD,"Moderate (0.6 Osm)"),(S,C_SEV,"Severe (1.0 Osm)")]
onset = 10.0  # minutes

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Osmolytes: K⁺, Glutamate, Trehalose, Turgor, Volume (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Osmolytes & physical ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["k_plus_mM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.axhline(600, color=C_GRAY, ls=":", lw=0.5); ax.text(1, 610, "max (Dinnbier 1988)", fontsize=6, color=C_GRAY)
ax.set_xlabel("Time (min)"); ax.set_ylabel("K⁺ (mM)")
ax.set_title("Cytoplasmic K⁺\n(TrkA/Kdp uptake)")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["glutamate_mM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Glutamate (mM)")
ax.set_title("Cytoplasmic glutamate\n(charge balance with K⁺)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["trehalose_mM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Trehalose (mM)")
ax.set_title("Trehalose (compatible solute)\n(otsBA, σ38-dependent)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["turgor_atm"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Turgor pressure (atm)")
ax.set_title("Turgor pressure\n(Pilizota & Shaevitz 2012)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["volume_fraction"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Cell volume (% of normal)")
ax.set_title("Cell volume\n(water efflux on upshift)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["osmolarity"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Osmolarity (Osm)")
ax.set_title("External osmolarity\n(stress applied at t=10 min)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig1_osmolytes_physical")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — EnvZ/OmpR Signaling & Porin Switch (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: EnvZ/OmpR & porins ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["ompr_p_fraction"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OmpR-P (% of total OmpR)")
ax.set_title("OmpR phosphorylation\n(EnvZ → OmpR-P)")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["porin_ompC_up"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OmpC activation (%)")
ax.set_title("OmpC induction\n(small pore, high osmolarity)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["porin_ompF_down"]*100, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OmpF repression (%)")
ax.set_title("OmpF repression\n(large pore, low osmolarity)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_rpos"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free RpoS (molecules)")
ax.set_title("σ38 (RpoS) protein\n(stabilised under osmotic stress)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_rpod"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free RpoD (molecules)")
ax.set_title("σ70 (RpoD) protein\n(free monomer)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_rnap_free"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free core RNAP (molecules)")
ax.set_title("Core RNAP (E)\n(sigma factors compete for this)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig2_envz_ompr_porins")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Sigma Factors & Metabolites (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Sigma & metabolites ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["f_sigma70"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("σ70 holoenzyme fraction")
ax.set_title("σ70 (RpoD) fraction\n(Eσ70 / total holoenzyme)")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["f_sigma38"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("σ38 holoenzyme fraction")
ax.set_title("σ38 (RpoS) fraction\n(rises under osmotic stress)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["ppgpp_uM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp concentration\n(increases via growth slowdown)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_ppgpp"]/1000, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("ppGpp (×10³ molecules)")
ax.set_title("ppGpp molecules\n(GTP + ATP → ppGpp)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_gtp"]/1e6, color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("GTP (×10⁶ molecules)")
ax.set_title("GTP pool\n(consumed by ppGpp synthesis)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["K_E70_eff_nM"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("$K_{d,σ70}^{eff}$ (nM)")
ax.set_title("σ70–RNAP effective Kd\n(ppGpp weakens binding)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig3_sigma_metabolites")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Growth, Mass & Cross-Protection (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Growth & cross-protection ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,c,lb in ALL: ax.plot(D["t_min"], np.maximum(D["growth_rate"]*3600,0), color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["dry_mass"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
ax.set_title("Cell dry mass")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["protein_mass"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Protein mass (fg)")
ax.set_title("Protein mass")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,c,lb in ALL: ax.plot(D["t_min"], D["rRna_mass"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("rRNA mass (fg)")
ax.set_title("rRNA mass\n(ribosome biogenesis)")
ax.legend(frameon=False); plbl(ax,"D")

# Cross-protection: KatG and AhpCF (OxyR regulon, also σ38-dependent)
ax=axes[1,1]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_katg"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("KatG (molecules)")
ax.set_title("KatG (catalase HPI)\n(cross-protection via σ38)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,c,lb in ALL: ax.plot(D["t_min"], D["bulk_ahpcf"], color=c, lw=1.2, label=lb)
ax.axvline(onset, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("AhpCF (molecules)")
ax.set_title("AhpCF (peroxidase)\n(cross-protection via σ38)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig4_growth_crossprotection")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def ml(a, f=0.5):
    n=len(a); return float(np.nanmean(a[int(n*(1-f)):]))

print("\n" + "="*80)
print("  OSMOTIC STRESS — ALL MOLECULES")
print("="*80)
hdr = f"  {'Molecule':<30} {'Exponential':>12} {'Moderate':>12} {'Severe':>12}"
print(hdr); print("  "+"-"*70)
rows = [
    ("K⁺ (mM)",               ml(B["k_plus_mM"]),     ml(M["k_plus_mM"]),     ml(S["k_plus_mM"])),
    ("Glutamate (mM)",        ml(B["glutamate_mM"]),  ml(M["glutamate_mM"]),  ml(S["glutamate_mM"])),
    ("Trehalose (mM)",        ml(B["trehalose_mM"]),  ml(M["trehalose_mM"]),  ml(S["trehalose_mM"])),
    ("Turgor (atm)",          ml(B["turgor_atm"]),    ml(M["turgor_atm"]),    ml(S["turgor_atm"])),
    ("Volume (%)",            ml(B["volume_fraction"])*100, ml(M["volume_fraction"])*100, ml(S["volume_fraction"])*100),
    ("OmpR-P (%)",            ml(B["ompr_p_fraction"])*100, ml(M["ompr_p_fraction"])*100, ml(S["ompr_p_fraction"])*100),
    ("OmpC activation (%)",   ml(B["porin_ompC_up"])*100, ml(M["porin_ompC_up"])*100, ml(S["porin_ompC_up"])*100),
    ("OmpF repression (%)",   ml(B["porin_ompF_down"])*100, ml(M["porin_ompF_down"])*100, ml(S["porin_ompF_down"])*100),
    ("Free RpoS (molecules)", ml(B["bulk_rpos"]),     ml(M["bulk_rpos"]),     ml(S["bulk_rpos"])),
    ("Free RpoD (molecules)", ml(B["bulk_rpod"]),     ml(M["bulk_rpod"]),     ml(S["bulk_rpod"])),
    ("Free RNAP (molecules)", ml(B["bulk_rnap_free"]),ml(M["bulk_rnap_free"]),ml(S["bulk_rnap_free"])),
    ("f_σ70",                 ml(B["f_sigma70"]),     ml(M["f_sigma70"]),     ml(S["f_sigma70"])),
    ("f_σ38",                 ml(B["f_sigma38"]),     ml(M["f_sigma38"]),     ml(S["f_sigma38"])),
    ("ppGpp (µM)",            ml(B["ppgpp_uM"]),      ml(M["ppgpp_uM"]),      ml(S["ppgpp_uM"])),
    ("ppGpp (×10³ mol)",      ml(B["bulk_ppgpp"])/1e3,ml(M["bulk_ppgpp"])/1e3,ml(S["bulk_ppgpp"])/1e3),
    ("GTP (×10⁶ mol)",       ml(B["bulk_gtp"])/1e6,  ml(M["bulk_gtp"])/1e6,  ml(S["bulk_gtp"])/1e6),
    ("K_E70_eff (nM)",        ml(B["K_E70_eff_nM"]),  ml(M["K_E70_eff_nM"]),  ml(S["K_E70_eff_nM"])),
    ("KatG (molecules)",      ml(B["bulk_katg"]),     ml(M["bulk_katg"]),     ml(S["bulk_katg"])),
    ("AhpCF (molecules)",     ml(B["bulk_ahpcf"]),    ml(M["bulk_ahpcf"]),    ml(S["bulk_ahpcf"])),
    ("Growth rate (h⁻¹)",    ml(B["growth_rate"])*3600, ml(M["growth_rate"])*3600, ml(S["growth_rate"])*3600),
    ("Dry mass (fg)",         ml(B["dry_mass"]),       ml(M["dry_mass"]),       ml(S["dry_mass"])),
    ("rRNA mass (fg)",        ml(B["rRna_mass"]),      ml(M["rRna_mass"]),      ml(S["rRna_mass"])),
]
for name, *vals in rows:
    print(f"  {name:<30}" + "".join(f"{v:>12.4g}" for v in vals))

print(f"\n  4 figures (PNG + PDF) saved to: {OUT}/")
print("="*80)
