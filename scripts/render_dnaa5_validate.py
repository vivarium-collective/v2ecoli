import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dnaa_observables as dnaa   # canonical DnaA pool columns (single source of truth)
L=dnaa.L
BATP=dnaa.BOUND_ATP_COLS
BADP=dnaa.BOUND_ADP_COLS
BULK={"apo":dnaa.APO_ID,"atp":dnaa.ATP_ID,"adp":dnaa.ADP_ID}
RUNS={"Langmuir n=1":("out/dnaa4_s06_F05_seed1_12gen/dnaa4_s06_F05_seed1_12gen","#94a3b8"),
      "cooperative n=4":("out/dnaa5_coop_n4_k30_seed1_8gen/dnaa5_coop_n4_k30_seed1_8gen","#16a34a"),
      "cooperative n=6":("out/dnaa5_coop_n6_k30_seed1_8gen/dnaa5_coop_n6_k30_seed1_8gen","#d62728")}
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(9,7))
ax1.set_title("dnaa-5 — cooperativity makes oriC-low SWITCH while preserving DnaA homeostasis (mass trigger, seed 1)",fontsize=10.5)
for label,(run,col) in RUNS.items():
    fs=sorted(glob.glob(f"{run}/history/**/*.pq",recursive=True))
    ids=pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx={k:ids.index(v) for k,v in BULK.items()}
    df=pl.scan_parquet(fs,hive_partitioning=True).select(
        [pl.col("generation"),pl.col("global_time"),pl.col(L+"oriC_low_free")]
        +[pl.col(c) for c in BATP+BADP]+[pl.col("bulk__count").list.get(i).alias(k) for k,i in idx.items()]
    ).collect().sort(["generation","global_time"])
    a=lambda c: np.asarray(df[c].to_list(),float)
    t=a("global_time")/60
    occ=a(L+"oriC_low_bound_atp")/np.maximum(a(L+"oriC_low_bound_atp")+a(L+"oriC_low_free"),1)
    if label!="cooperative n=6":
        ax1.plot(t,occ,color=col,lw=1.0,alpha=0.8,label=f"{label} (max {occ.max():.2f})")
    gens=sorted(set(df["generation"].to_list())); steady=[g for g in gens if 3<=g<=max(gens)-1]
    pm=[]
    for g in steady:
        d=df.filter(pl.col("generation")==g)
        tot=a2=None
        b=lambda c: np.asarray(d[c].to_list(),float)
        tot=b("apo")+b("atp")+b("adp")+sum(b(c) for c in BATP)+sum(b(c) for c in BADP)
        pm.append(tot.mean())
    ax2.plot(steady,pm,"-o",color=col,ms=4,label=f"{label}")
ax1.set_ylabel("oriC-low occupancy"); ax1.set_ylim(-0.03,1.05); ax1.legend(fontsize=8,loc="upper right")
ax1.set_xlabel("time (min)"); ax1.grid(alpha=0.2)
ax1.annotate("cooperative: snaps to FULL (switch)\nLangmuir: gradual, partial (<=0.5)",xy=(0.02,0.7),xycoords="axes fraction",fontsize=8.5)
ax2.axhspan(300,800,color="#f1f5f9",zorder=0,label="DnaA band [300,800]")
ax2.set_ylabel("DnaA pool /gen (fg-eq count)"); ax2.set_xlabel("generation"); ax2.legend(fontsize=8,loc="upper right"); ax2.grid(alpha=0.2)
ax2.annotate("n=4 stays in band (preserved);\nn=6 too strong -> dips below",xy=(0.02,0.05),xycoords="axes fraction",fontsize=8.5)
for ax in (ax1,ax2): ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for ext in ("png","svg"): fig.savefig(f"out/dnaa5_cooperative_validation.{ext}",dpi=140)
print("wrote dnaa5_cooperative_validation")
