import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
L="listeners__replication_data__"
def load(run):
    fs=sorted(glob.glob(f"{run}/history/**/*.pq",recursive=True))
    return pl.scan_parquet(fs,hive_partitioning=True).select(
        ["generation","global_time",pl.col(L+"number_of_oric"),pl.col("listeners__mass__cell_mass")]
    ).collect().sort(["generation","global_time"])
mass=load("out/dnaa4_s06_F05_seed1_12gen/dnaa4_s06_F05_seed1_12gen")
fp=load("out/dnaa5_mech_diag_seed1_8gen/dnaa5_mech_diag_seed1_8gen")
po=load("out/dnaa6_mech_low_coop_n4_seed1_8gen/dnaa6_mech_low_coop_n4_seed1_8gen")
a=lambda d,c: np.asarray(d[c].to_list(),float)
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(9,7),sharex=True)
ax1.set_title("dnaa-6 payoff — mechanistic DnaA-ATP initiation SUSTAINS the cycle with cooperativity (seed 1)",fontsize=10.5)
for d,c,lab in [(mass,"#1f77b4","MASS control (heuristic)"),
                (fp,"#d62728","MECH first-pass (oriC-high, no coop) — collapses gen 2"),
                (po,"#16a34a","MECH oriC-low switch + cooperativity — sustains 8/8")]:
    t=a(d,"global_time")/60; tm=t<=a(po,"global_time").max()/60+10
    ax1.plot(t[tm],a(d,"listeners__mass__cell_mass")[tm],color=c,lw=1.4,label=lab)
ax1.set_ylabel("cell mass (fg)"); ax1.legend(fontsize=8,loc="upper left"); ax1.grid(alpha=0.2)
# panel B: oriC for payoff vs mass
tp=a(po,"global_time")/60; tmm=a(mass,"global_time")/60
ax2.plot(tmm[tmm<=tp.max()+10],a(mass,L+"number_of_oric")[tmm<=tp.max()+10],color="#1f77b4",lw=1.2,label="MASS: oriC clean 1<->2")
ax2.plot(tp,a(po,L+"number_of_oric"),color="#16a34a",lw=1.2,label="MECH+coop: oriC 1..4 (OVER-initiates — needs SeqA eclipse)")
ax2.axhline(2,color="0.6",ls=":",lw=1)
ax2.set_xlabel("time (min)"); ax2.set_ylabel("number of oriC"); ax2.set_ylim(0,4.5)
ax2.legend(fontsize=8,loc="upper left"); ax2.grid(alpha=0.2)
for ax in (ax1,ax2): ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for ext in ("png","svg"): fig.savefig(f"out/dnaa6_payoff_sustains.{ext}",dpi=140)
print("wrote dnaa6_payoff_sustains")
