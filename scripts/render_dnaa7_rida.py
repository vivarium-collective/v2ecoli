import glob, numpy as np, polars as pl, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dnaa_observables as dnaa   # canonical DnaA pool columns (single source of truth)
L=dnaa.L
BATP=dnaa.BOUND_ATP_COLS
BADP=dnaa.BOUND_ADP_COLS
BULK={"apo":dnaa.APO_ID,"atp":dnaa.ATP_ID,"adp":dnaa.ADP_ID}
def load(run):
    fs=sorted(glob.glob(f"{run}/history/**/*.pq",recursive=True))
    ids=pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx={k:ids.index(v) for k,v in BULK.items()}
    return pl.scan_parquet(fs,hive_partitioning=True).select(
        [pl.col("generation"),pl.col("global_time"),pl.col(L+"number_of_oric")]
        +[pl.col(c) for c in BATP+BADP]+[pl.col("bulk__count").list.get(i).alias(k) for k,i in idx.items()]
    ).collect().sort(["generation","global_time"])
mass=load("out/dnaa4_s06_F05_seed1_12gen/dnaa4_s06_F05_seed1_12gen")
nor=load("out/dnaa6_mech_low_coop_n4_seed1_8gen/dnaa6_mech_low_coop_n4_seed1_8gen")
rid=load("out/dnaa7_rida_full_seed1_8gen/dnaa7_rida_full_seed1_8gen")
a=lambda d,c: np.asarray(d[c].to_list(),float)
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(9,7),sharex=True)
ax1.set_title("dnaa-7 — RIDA restores one-initiation-per-cycle for the mechanistic trigger (seed 1)",fontsize=10.5)
tmax=a(rid,"global_time").max()/60+5
for d,c,lab in [(mass,"#1f77b4","MASS heuristic: oriC 1<->2"),
                (nor,"#d62728","mech (no RIDA): oriC 1-4 (over-init)"),
                (rid,"#16a34a","mech + RIDA/DDAH/DARS: oriC 1<->2 (fixed)")]:
    t=a(d,"global_time")/60; m=t<=tmax
    ax1.plot(t[m],a(d,L+"number_of_oric")[m],color=c,lw=1.2,label=lab)
ax1.axhline(2,color="0.6",ls=":",lw=1); ax1.set_ylim(0,4.5); ax1.set_ylabel("number of oriC")
ax1.legend(fontsize=8.5,loc="upper left"); ax1.grid(alpha=0.2)
# ATP fraction per gen
ax2.axhspan(0.2,0.5,color="#f1f5f9",zorder=0,label="DnaA-ATP fraction band [0.2,0.5]")
for d,c,lab in [(mass,"#1f77b4","MASS: 0.25 (in band)"),(rid,"#16a34a","mech+RIDA: 0.03 (CRASHES — rate needs tuning)")]:
    gens=sorted(set(d["generation"].to_list())); steady=[g for g in gens if 3<=g<=max(gens)-1]
    fr=[]
    for g in steady:
        s=d.filter(pl.col("generation")==g); b=lambda x: np.asarray(s[x].to_list(),float)
        batp=sum(b(x) for x in BATP); tot=b("apo")+b("atp")+b("adp")+batp+sum(b(x) for x in BADP)
        fr.append(((b("atp")+batp)/np.maximum(tot,1)).mean())
    ax2.plot(steady,fr,"-o",color=c,ms=4,label=lab)
ax2.set_ylabel("DnaA-ATP fraction"); ax2.set_xlabel("generation"); ax2.set_ylim(-0.02,0.6)
ax2.legend(fontsize=8.5,loc="center right"); ax2.grid(alpha=0.2)
for ax in (ax1,ax2): ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for ext in ("png","svg"): fig.savefig(f"out/dnaa7_rida_fixes_overinit.{ext}",dpi=140)
print("wrote dnaa7_rida_fixes_overinit")
