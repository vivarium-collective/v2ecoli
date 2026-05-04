#!/usr/bin/env python3
"""
plot_full_analysis.py — Generate all figures from saved scenario data.
Reads out/scenarios/*.json, produces out/full_analysis/*.{png,pdf}
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
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

C_BASE="#2166AC"; C_SIG="#4DAC26"; C_H2O2="#D6604D"; C_STARV="#B2182B"
C_S70="#2166AC"; C_S38="#D6604D"; C_S32="#F4A582"; C_S24="#92C5DE"; C_S54="#762A83"
C_GRAY="#AAAAAA"
SC = [C_S70,C_S38,C_S32,C_S24,C_S54]
SL = ["σ70 (RpoD)","σ38 (RpoS)","σ32 (RpoH)","σ24 (RpoE)","σ54 (RpoN)"]
OXYR_KOX=0.2; OXYR_HILL=1.5; OXYR_FC_MAX=10.0

OUT = "out/full_analysis"; os.makedirs(OUT, exist_ok=True)

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
D1 = load("out/scenarios/baseline.json")
D2 = load("out/scenarios/sigma_exp.json")
D3 = load("out/scenarios/h2o2_stress.json")
D4 = load("out/scenarios/starvation.json")
print(f"  Baseline: {D1['_meta']['sim_time']}s, Sigma: {D2['_meta']['sim_time']}s, "
      f"H2O2: {D3['_meta']['sim_time']}s, Starv: {D4['_meta']['sim_time']}s")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Oxidative Stress Response (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Oxidative stress ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)
t2,t3 = D2["t_min"],D3["t_min"]

ax=axes[0,0]
ax.semilogy(t2,safe(D2["h2o2_uM"],1e-4),color=C_SIG,lw=1.2,label="Exponential")
ax.semilogy(t3,safe(D3["h2o2_uM"],1e-4),color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.axhline(0.02,color=C_GRAY,lw=0.8,ls="--",alpha=0.7,label="~20 nM SS (Imlay 2013)")
ax.set_xlabel("Time (min)"); ax.set_ylabel("[H₂O₂] (µM)")
ax.set_title("Intracellular H₂O₂"); ax.legend(frameon=False,fontsize=7); plbl(ax,"A")

ax=axes[0,1]
ax.plot(t2,D2["oxyr_ox"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,D3["oxyr_ox"],color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.axhline(0.5,color=C_GRAY,lw=0.6,ls="--",alpha=0.6)
ax.set_xlabel("Time (min)"); ax.set_ylabel("OxyR oxidised fraction")
ax.set_title("OxyR activation\n(Kox=0.2 µM, Åslund 1999)")
ax.set_ylim(-0.05,1.1); ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]
ax.plot(t2,D2["oxyr_fc"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,D3["oxyr_fc"],color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.axhline(1.0,color=C_GRAY,ls=":",lw=0.6)
fc_r = ml(D3["oxyr_fc"])/ml(D2["oxyr_fc"]) if ml(D2["oxyr_fc"])>0 else 0
ax.text(0.97,0.03,f"FC = {fc_r:.1f}×",transform=ax.transAxes,fontsize=9,
        ha="right",va="bottom",fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C_H2O2,lw=1.5))
ax.set_xlabel("Time (min)"); ax.set_ylabel("Fold change")
ax.set_title("OxyR gene upregulation\n(katG, ahpCF, dps — max 10×)")
ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]
ax.semilogy(t2,safe(D2["total_scav"],1e-3),color=C_SIG,lw=1.2,label="Exponential")
ax.semilogy(t3,safe(D3["total_scav"],1e-3),color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Scavenging (µM/s)")
ax.set_title("Total H₂O₂ scavenging"); ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
ax.semilogy(t2,safe(D2["dna_dmg"],1e-5),color=C_SIG,lw=1.2,label="Exponential")
ax.semilogy(t3,safe(D3["dna_dmg"],1e-5),color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("DNA damage rate (a.u./s)")
ax.set_title("DNA damage (Fenton)\n(k=76 M⁻¹s⁻¹, [Fe²⁺]=10 µM)")
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
h_r=np.logspace(-3,2,300)
oxyr_r=1.0+(OXYR_FC_MAX-1.0)*h_r**OXYR_HILL/(OXYR_KOX**OXYR_HILL+h_r**OXYR_HILL)
ax.semilogx(h_r,oxyr_r,color=C_H2O2,lw=2)
for hv,c_,lb in [(ml(D2["h2o2_uM"]),C_SIG,"Exponential"),(ml(D3["h2o2_uM"]),C_H2O2,"H₂O₂ challenge")]:
    fc_pt=1.0+9.0*hv**OXYR_HILL/(OXYR_KOX**OXYR_HILL+hv**OXYR_HILL)
    ax.plot(hv,fc_pt,"o",color=c_,ms=8,zorder=5,label=f"{lb} ({hv:.3f} µM)")
ax.axhline(1,color=C_GRAY,ls=":",lw=0.6)
ax.set_xlabel("[H₂O₂] (µM)"); ax.set_ylabel("OxyR fold change")
ax.set_title("OxyR dose-response\n(Kox=0.2 µM, Åslund 1999)")
ax.legend(frameon=False,fontsize=7); ax.set_ylim(0.5,11); plbl(ax,"F")
sv(fig,"fig1_oxidative_stress")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sigma Factor Dynamics Under H₂O₂ Stress (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Sigma dynamics ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)

ax=axes[0,0]; ax.plot(t2,D2["f_sigma70"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,D3["f_sigma70"],color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("σ70 (RpoD) fraction"); ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]; ax.plot(t2,D2["f_sigma38"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,D3["f_sigma38"],color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("σ38 (RpoS) fraction"); ax.legend(frameon=False); plbl(ax,"B")

ax=axes[0,2]; ax.plot(t2,D2["ppgpp_uM"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,D3["ppgpp_uM"],color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("ppGpp (µM)")
ax.set_title("ppGpp concentration"); ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]; ax.plot(t2,D2["K_E70_eff_nM"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,D3["K_E70_eff_nM"],color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("$K_{d,σ70}^{eff}$ (nM)")
ax.set_title("Effective σ70–RNAP affinity"); ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]
ax.semilogy(t2,safe(D2["Es70_count"]),color=C_S70,lw=1.2,ls="-",label="Eσ70 exp")
ax.semilogy(t3,safe(D3["Es70_count"]),color=C_S70,lw=1.2,ls="--",label="Eσ70 H₂O₂")
ax.semilogy(t2,safe(D2["EsS_count"]),color=C_S38,lw=1.2,ls="-",label="Eσ38 exp")
ax.semilogy(t3,safe(D3["EsS_count"]),color=C_S38,lw=1.2,ls="--",label="Eσ38 H₂O₂")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme count")
ax.set_title("Holoenzyme counts"); ax.legend(frameon=False,fontsize=7); plbl(ax,"E")

gr2=np.maximum(D2["growth_rate"]*3600,0); gr3=np.maximum(D3["growth_rate"]*3600,0)
ax=axes[1,2]; ax.plot(t2,gr2,color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t3,gr3,color=C_H2O2,lw=1.2,label="H₂O₂ challenge")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate"); ax.legend(frameon=False); plbl(ax,"F")
sv(fig,"fig2_sigma_dynamics_stress")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Cell Physiology 4-Way (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Cell physiology ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)
ALL=[(D1,C_BASE,"Baseline (no sigma)"),(D2,C_SIG,"Exponential"),(D3,C_H2O2,"H₂O₂ challenge"),(D4,C_STARV,"Starvation")]

ax=axes[0,0]
for D,c,lb in ALL: ax.plot(D["t_min"],np.maximum(D["growth_rate"]*3600,0),color=c,lw=1.2,label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate"); ax.legend(frameon=False,fontsize=7); plbl(ax,"A")

ax=axes[0,1]
for D,c,lb in ALL: ax.plot(D["t_min"],D["dry_mass"],color=c,lw=1.2,label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Dry mass (fg)")
ax.set_title("Cell dry mass"); ax.legend(frameon=False,fontsize=7); plbl(ax,"B")

ax=axes[0,2]
for D,c,lb in ALL:
    m0=D["dry_mass"][0]
    if m0>0: ax.plot(D["t_min"],D["dry_mass"]/m0,color=c,lw=1.2,label=lb)
ax.axhline(2.0,color=C_GRAY,ls=":",lw=0.6)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Mass fold change")
ax.set_title("Mass accumulation"); ax.legend(frameon=False,fontsize=7); plbl(ax,"C")

ax=axes[1,0]
for D,c,lb in ALL: ax.plot(D["t_min"],D["protein_mass"],color=c,lw=1.2,label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Protein mass (fg)")
ax.set_title("Protein mass"); ax.legend(frameon=False,fontsize=7); plbl(ax,"D")

ax=axes[1,1]
for rna,ls_,lbl_ in [("rRna_mass","-","rRNA"),("mRna_mass","--","mRNA"),("tRna_mass",":","tRNA")]:
    ax.plot(D2["t_min"],D2[rna],color=C_SIG,lw=1.2,ls=ls_,label=f"{lbl_} exp")
    ax.plot(D4["t_min"],D4[rna],color=C_STARV,lw=0.8,ls=ls_,alpha=0.6,label=f"{lbl_} starv")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Mass (fg)")
ax.set_title("RNA mass by type\n(exp vs starvation)")
ax.legend(frameon=False,fontsize=6,ncol=2); plbl(ax,"E")

ax=axes[1,2]
for D,c,lb in ALL: ax.plot(D["t_min"],D["dna_mass"],color=c,lw=1.2,label=lb)
ax.set_xlabel("Time (min)"); ax.set_ylabel("DNA mass (fg)")
ax.set_title("DNA mass"); ax.legend(frameon=False,fontsize=7); plbl(ax,"F")
sv(fig,"fig3_cell_physiology")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Starvation Mechanistic Chain (2×3)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Starvation chain ...")
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.55, wspace=0.42)
t2_,t4_ = D2["t_min"],D4["t_min"]

ax=axes[0,0]; ax.plot(t2_,D2["ppgpp_uM"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t4_,D4["ppgpp_uM"],color=C_STARV,lw=1.2,label="Starvation")
ax.axhline(200,color=C_GRAY,ls=":",lw=0.6)
ax.text(1,210,"stationary threshold",fontsize=6,color=C_GRAY)
ax.set_xlabel("Time (min)"); ax.set_ylabel("[ppGpp] (µM)")
ax.set_title("ppGpp concentration\n(stringent response)"); ax.legend(frameon=False); plbl(ax,"A")

ax=axes[0,1]
ax.plot(t2_,D2["f_sigma70"],color=C_S70,lw=1.2,label="σ70 exp")
ax.plot(t4_,D4["f_sigma70"],color=C_S70,lw=1.2,ls="--",label="σ70 starv")
ax.plot(t2_,D2["f_sigma38"],color=C_S38,lw=1.2,label="σ38 exp")
ax.plot(t4_,D4["f_sigma38"],color=C_S38,lw=1.2,ls="--",label="σ38 starv")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("Sigma factor redistribution\n(ppGpp weakens σ70)")
ax.legend(frameon=False,fontsize=7); plbl(ax,"B")

gr2_=np.maximum(D2["growth_rate"]*3600,0); gr4_=np.maximum(D4["growth_rate"]*3600,0)
ax=axes[0,2]; ax.plot(t2_,gr2_,color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t4_,gr4_,color=C_STARV,lw=1.2,label="Starvation")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Growth rate (h⁻¹)")
ax.set_title("Growth rate"); ax.legend(frameon=False); plbl(ax,"C")

ax=axes[1,0]; ax.plot(t2_,D2["K_E70_eff_nM"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t4_,D4["K_E70_eff_nM"],color=C_STARV,lw=1.2,label="Starvation")
ax.set_xlabel("Time (min)"); ax.set_ylabel("$K_{d,σ70}^{eff}$ (nM)")
ax.set_title("Effective σ70–RNAP affinity\n(ppGpp weakens binding)")
ax.legend(frameon=False); plbl(ax,"D")

ax=axes[1,1]; ax.plot(t2_,D2["phase"],color=C_SIG,lw=1.2,label="Exponential")
ax.plot(t4_,D4["phase"],color=C_STARV,lw=1.2,label="Starvation")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Phase index (0=exp, 1=stat)")
ax.set_title("Growth phase\n(0=exp, 1=stationary)"); ax.set_ylim(-0.05,1.1)
ax.legend(frameon=False); plbl(ax,"E")

ax=axes[1,2]
fracs=np.column_stack([D4[f"f_sigma{k}"] for k in [70,38,32,24,54]])
ax.stackplot(t4_,fracs.T,labels=SL,colors=SC,alpha=0.85)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Holoenzyme fraction")
ax.set_title("Sigma partitioning\n(starvation)"); ax.set_ylim(0,1)
ax.legend(loc="upper right",fontsize=6,frameon=False); plbl(ax,"F")
sv(fig,"fig4_starvation_chain")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Coupling Chain Summary (1×2)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Coupling chains ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.subplots_adjust(wspace=0.55)

# A — H₂O₂ stress vs exponential
ax=axes[0]
m1={
    "H₂O₂ (µM)":       (ml(D2["h2o2_uM"]),   ml(D3["h2o2_uM"])),
    "OxyR fold change": (ml(D2["oxyr_fc"]),    ml(D3["oxyr_fc"])),
    "SoxRS fold change":(ml(D2["soxrs_fc"]),   ml(D3["soxrs_fc"])),
    "ppGpp (µM)":       (ml(D2["ppgpp_uM"]),   ml(D3["ppgpp_uM"])),
    "K_E70 (nM)":       (ml(D2["K_E70_eff_nM"]),ml(D3["K_E70_eff_nM"])),
    "f_σ38":            (ml(D2["f_sigma38"]),   ml(D3["f_sigma38"])),
    "DNA damage":       (ml(D2["dna_dmg"]),     ml(D3["dna_dmg"])),
    "Growth (h⁻¹)":    (ml(gr2),               ml(gr3)),
}
n1=list(m1.keys()); p1=[(vs-vb)/vb*100 if vb>0 else 0 for vb,vs in m1.values()]
y1=np.arange(len(n1)); c1=[C_H2O2 if p>0 else C_SIG for p in p1]
ax.barh(y1,p1,color=c1,height=0.6,edgecolor="white",lw=0.3)
ax.set_yticks(y1); ax.set_yticklabels(n1,fontsize=8)
ax.set_xlabel("% change (H₂O₂ stress vs exponential)")
ax.set_title("Oxidative Stress Chain\nH₂O₂ → OxyR → ppGpp → σ redistribution → growth")
ax.axvline(0,color="black",lw=0.8)
for yi,p_ in zip(y1,p1):
    ax.text(p_+(3 if p_>=0 else -3),yi,f"{p_:+.1f}%",va="center",
            ha="left" if p_>=0 else "right",fontsize=7)
ax.invert_yaxis(); plbl(ax,"A",x=-0.20)

# B — Starvation vs exponential
ax=axes[1]
m2={
    "ppGpp (µM)":       (ml(D2["ppgpp_uM"]),   ml(D4["ppgpp_uM"])),
    "K_E70 (nM)":       (ml(D2["K_E70_eff_nM"]),ml(D4["K_E70_eff_nM"])),
    "f_σ70":            (ml(D2["f_sigma70"]),   ml(D4["f_sigma70"])),
    "f_σ38":            (ml(D2["f_sigma38"]),   ml(D4["f_sigma38"])),
    "Phase index":      (ml(D2["phase"]),       ml(D4["phase"])),
    "Growth (h⁻¹)":    (ml(gr2_),              ml(gr4_)),
    "Dry mass (fg)":    (ml(D2["dry_mass"]),    ml(D4["dry_mass"])),
    "Protein (fg)":     (ml(D2["protein_mass"]),ml(D4["protein_mass"])),
    "rRNA (fg)":        (ml(D2["rRna_mass"]),   ml(D4["rRna_mass"])),
}
n2=list(m2.keys()); p2=[(vs-vb)/vb*100 if vb>0 else 0 for vb,vs in m2.values()]
y2=np.arange(len(n2)); c2=[C_STARV if p>0 else C_SIG for p in p2]
ax.barh(y2,p2,color=c2,height=0.6,edgecolor="white",lw=0.3)
ax.set_yticks(y2); ax.set_yticklabels(n2,fontsize=8)
ax.set_xlabel("% change (starvation vs exponential)")
ax.set_title("Starvation Chain\nppGpp ↑ → σ70 Kd ↑ → σ38 ↑ → rRNA ↓ → growth ↓")
ax.axvline(0,color="black",lw=0.8)
for yi,p_ in zip(y2,p2):
    ax.text(p_+(3 if p_>=0 else -3),yi,f"{p_:+.1f}%",va="center",
            ha="left" if p_>=0 else "right",fontsize=7)
ax.invert_yaxis(); plbl(ax,"B",x=-0.20)
sv(fig,"fig5_coupling_chains")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  FULL SIGMA FACTOR ANALYSIS — SUMMARY")
print("="*80)
hdr=f"  {'Metric':<25} {'Baseline':>10} {'Sigma Exp':>10} {'H₂O₂ Str':>10} {'Starvation':>10}"
print(hdr); print("  "+"-"*70)
rows=[
    ("Growth (h⁻¹)",    ml(D1["growth_rate"])*3600,ml(D2["growth_rate"])*3600,
                          ml(D3["growth_rate"])*3600,ml(D4["growth_rate"])*3600),
    ("Dry mass (fg)",     ml(D1["dry_mass"]),ml(D2["dry_mass"]),ml(D3["dry_mass"]),ml(D4["dry_mass"])),
    ("f_σ70",             0,ml(D2["f_sigma70"]),ml(D3["f_sigma70"]),ml(D4["f_sigma70"])),
    ("f_σ38",             0,ml(D2["f_sigma38"]),ml(D3["f_sigma38"]),ml(D4["f_sigma38"])),
    ("ppGpp (µM)",        0,ml(D2["ppgpp_uM"]),ml(D3["ppgpp_uM"]),ml(D4["ppgpp_uM"])),
    ("H₂O₂ (µM)",        0,ml(D2["h2o2_uM"]),ml(D3["h2o2_uM"]),ml(D4["h2o2_uM"])),
    ("OxyR FC",           0,ml(D2["oxyr_fc"]),ml(D3["oxyr_fc"]),ml(D4["oxyr_fc"])),
    ("DNA damage",        0,ml(D2["dna_dmg"]),ml(D3["dna_dmg"]),ml(D4["dna_dmg"])),
    ("Phase",             0,ml(D2["phase"]),ml(D3["phase"]),ml(D4["phase"])),
    ("Protein (fg)",      ml(D1["protein_mass"]),ml(D2["protein_mass"]),
                          ml(D3["protein_mass"]),ml(D4["protein_mass"])),
    ("rRNA (fg)",         ml(D1["rRna_mass"]),ml(D2["rRna_mass"]),
                          ml(D3["rRna_mass"]),ml(D4["rRna_mass"])),
]
for name,*vals in rows:
    print(f"  {name:<25}"+"".join(f"{v:>10.4g}" for v in vals))

print(f"\n  Mass doubling:")
for lb,D in [("Baseline",D1),("Sigma exp",D2),("H₂O₂ stress",D3),("Starvation",D4)]:
    m0,mf=D["dry_mass"][0],D["dry_mass"][-1]
    print(f"    {lb:<15} {m0:.1f} → {mf:.1f} fg  ({mf/m0:.2f}×)")

print(f"\n  5 figures (PNG + PDF) saved to: {OUT}/")
print("="*80)
