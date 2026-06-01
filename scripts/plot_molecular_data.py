#!/usr/bin/env python3
"""
plot_molecular_data.py — Plot every molecule from the biological flow diagrams.
Compares baseline vs H₂O₂ stress vs starvation for each molecule.
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

C_BASE="#4DAC26"; C_H2O2="#D6604D"; C_STARV="#B2182B"; C_GRAY="#AAAAAA"
OUT = "out/molecular_data"; os.makedirs(OUT, exist_ok=True)

def safe(v, fl=1e-9): return np.maximum(np.asarray(v,float), fl)
def plbl(ax,s,x=-0.12,y=1.05):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top")
def sv(fig,name):
    for e in ("png","pdf"):
        fig.savefig(f"{OUT}/{name}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"  {name}")

def load(path):
    with open(path) as f: d = json.load(f)
    snaps = d["snapshots"]
    out = {k: np.array([s.get(k,0) for s in snaps]) for k in snaps[0]}
    out["t_min"] = out["time"]/60.0
    return out

print("Loading ...")
B = load("out/scenarios/mol_baseline.json")
H = load("out/scenarios/mol_h2o2.json")
S = load("out/scenarios/mol_starv.json")
print(f"  Baseline: {len(B['time'])} pts, H2O2: {len(H['time'])} pts, Starv: {len(S['time'])} pts\n")

onset_h = 10.0  # H2O2 onset in minutes
onset_s = 2.0   # starvation onset in minutes


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Scavenging Enzymes: KatG, AhpCF, KatE (2×3)
# The OxyR regulon — what binds, what gets induced
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Scavenging enzymes ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ALL = [(B,"t_min",C_BASE,"Exponential"),(H,"t_min",C_H2O2,"H₂O₂ challenge"),(S,"t_min",C_STARV,"Starvation")]

ax=axes[0,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_katg"], color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("KatG molecules")
ax.set_title("KatG (catalase HPI)\nOxyR-induced, kcat=200 s⁻¹")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_ahpcf"], color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("AhpCF molecules")
ax.set_title("AhpCF (peroxidase)\nOxyR-induced, kcat=428 s⁻¹")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_kate"], color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("KatE molecules")
ax.set_title("KatE (catalase HPII)\nσ38-dependent (stationary)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D.get("extra_katg",np.zeros_like(D[tk])), color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Extra KatG (OxyR feedback)")
ax.set_title("OxyR-induced extra KatG\n(feedback loop)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,tk,c,lb in ALL:
    ax.semilogy(D[tk], safe(D["h2o2_uM"],1e-4), color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.axhline(0.2, color="red", ls=":", lw=0.6, alpha=0.5)
ax.text(1, 0.25, "OxyR Kox", fontsize=6, color="red")
ax.set_xlabel("Time (min)"); ax.set_ylabel("[H₂O₂] (µM)")
ax.set_title("H₂O₂ concentration\n(scavenging keeps it low)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["oxyr_fc"], color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.axhline(1.0, color=C_GRAY, ls=":", lw=0.6)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OxyR fold change")
ax.set_title("OxyR transcriptional activation\n(drives KatG/AhpCF induction)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig1_scavenging_enzymes")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sigma Factors & RNAP: RpoD, RpoS, RpoH, free RNAP (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Sigma factors & RNAP ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_rpod"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free RpoD molecules")
ax.set_title("σ70 (RpoD) free monomer\n(most is in holoenzyme)")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_rpos"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free RpoS molecules")
ax.set_title("σ38 (RpoS) free monomer\nppGpp stabilises (↑ half-life)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_rpoh"], color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free RpoH molecules")
ax.set_title("σ32 (RpoH) free monomer\n(heat shock sigma)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_rnap_free"], color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Free core RNAP")
ax.set_title("Inactive RNAP (core E)\n(sigma factors compete for this)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["f_sigma70"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("σ70 holoenzyme fraction")
ax.set_title("σ70 fraction\n(Eσ70 / total holoenzyme)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["f_sigma38"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("σ38 holoenzyme fraction")
ax.set_title("σ38 fraction\n(Eσ38 / total holoenzyme)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig2_sigma_factors_rnap")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Metabolites: ppGpp, GTP, ATP, NADPH (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Metabolites ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_ppgpp"]/1000, color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("ppGpp (×10³ molecules)")
ax.set_title("ppGpp (alarmone)\nRelA synthesises from GTP+ATP")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["ppgpp_uM"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.axhline(200, color=C_GRAY, ls=":", lw=0.6)
ax.text(1, 210, "stationary", fontsize=6, color=C_GRAY)
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp concentration\n(75 µM exp → 330 µM starved)")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_gtp"]/1e6, color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("GTP (×10⁶ molecules)")
ax.set_title("GTP pool\n(ppGpp synthesis consumes GTP)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_atp"]/1e6, color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("ATP (×10⁶ molecules)")
ax.set_title("ATP pool\n(ppGpp synthesis consumes ATP)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["bulk_nadph"]/1e3, color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("NADPH (×10³ molecules)")
ax.set_title("NADPH pool\n(AhpCF consumes 1 per H₂O₂)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["K_E70_eff_nM"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("$K_{d,σ70}^{eff}$ (nM)")
ax.set_title("σ70–RNAP effective Kd\n(ppGpp weakens: 1→5 nM)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig3_metabolites")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Growth & Mass: the downstream consequence (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Growth & mass ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], np.maximum(D["growth_rate"]*3600,0), color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate\n(stress → growth arrest)")
ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["dry_mass"], color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
ax.set_title("Cell dry mass")
ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["protein_mass"], color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Protein mass (fg)")
ax.set_title("Protein mass\n(translation capacity)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["rRna_mass"], color=c, lw=1.2, label=lb)
ax.axvline(onset_s, color=C_STARV, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("rRNA mass (fg)")
ax.set_title("rRNA mass\n(ppGpp shuts down rRNA promoters)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["mRna_mass"], color=c, lw=1.2, label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("mRNA mass (fg)")
ax.set_title("mRNA mass")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
for D,tk,c,lb in ALL: ax.plot(D[tk], D["dna_dmg"], color=c, lw=1.2, label=lb)
ax.axvline(onset_h, color=C_H2O2, ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Time (min)"); ax.set_ylabel("DNA damage rate (a.u./s)")
ax.set_title("DNA damage\n(Fe²⁺ + H₂O₂ → •OH)")
ax.legend(frameon=False); plbl(ax,"F")

sv(fig, "fig4_growth_mass")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def ml(a, f=0.5):
    n=len(a); return float(np.nanmean(a[int(n*(1-f)):]))

print("\n" + "="*80)
print("  MOLECULAR TRACKING — KEY MOLECULES COMPARISON")
print("="*80)
hdr = f"  {'Molecule':<25} {'Baseline':>12} {'H₂O₂ stress':>12} {'Starvation':>12}"
print(hdr); print("  "+"-"*65)
rows = [
    ("KatG (molecules)",      ml(B["bulk_katg"]),  ml(H["bulk_katg"]),  ml(S["bulk_katg"])),
    ("AhpCF (molecules)",     ml(B["bulk_ahpcf"]), ml(H["bulk_ahpcf"]), ml(S["bulk_ahpcf"])),
    ("KatE (molecules)",      ml(B["bulk_kate"]),  ml(H["bulk_kate"]),  ml(S["bulk_kate"])),
    ("Extra KatG (feedback)", ml(B["extra_katg"]), ml(H["extra_katg"]), ml(S["extra_katg"])),
    ("RpoD free (molecules)", ml(B["bulk_rpod"]),  ml(H["bulk_rpod"]),  ml(S["bulk_rpod"])),
    ("RpoS free (molecules)", ml(B["bulk_rpos"]),  ml(H["bulk_rpos"]),  ml(S["bulk_rpos"])),
    ("Free RNAP (molecules)", ml(B["bulk_rnap_free"]),ml(H["bulk_rnap_free"]),ml(S["bulk_rnap_free"])),
    ("ppGpp (molecules)",     ml(B["bulk_ppgpp"]), ml(H["bulk_ppgpp"]), ml(S["bulk_ppgpp"])),
    ("ppGpp (µM)",            ml(B["ppgpp_uM"]),   ml(H["ppgpp_uM"]),   ml(S["ppgpp_uM"])),
    ("GTP (×10⁶)",           ml(B["bulk_gtp"])/1e6,ml(H["bulk_gtp"])/1e6,ml(S["bulk_gtp"])/1e6),
    ("NADPH (×10³)",         ml(B["bulk_nadph"])/1e3,ml(H["bulk_nadph"])/1e3,ml(S["bulk_nadph"])/1e3),
    ("H₂O₂ (µM)",           ml(B["h2o2_uM"]),    ml(H["h2o2_uM"]),    ml(S["h2o2_uM"])),
    ("OxyR FC",               ml(B["oxyr_fc"]),    ml(H["oxyr_fc"]),    ml(S["oxyr_fc"])),
    ("f_σ70",                 ml(B["f_sigma70"]),  ml(H["f_sigma70"]),  ml(S["f_sigma70"])),
    ("f_σ38",                 ml(B["f_sigma38"]),  ml(H["f_sigma38"]),  ml(S["f_sigma38"])),
    ("Growth (h⁻¹)",        ml(B["growth_rate"])*3600,ml(H["growth_rate"])*3600,ml(S["growth_rate"])*3600),
]
for name, vb, vh, vs in rows:
    print(f"  {name:<25} {vb:>12.4g} {vh:>12.4g} {vs:>12.4g}")

print(f"\n  4 figures (PNG + PDF) saved to: {OUT}/")
print("="*80)
