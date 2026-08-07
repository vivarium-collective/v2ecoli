import glob, numpy as np, polars as pl, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
L="listeners__replication_data__"
def oric_t(run):
    fs=sorted(glob.glob(f"{run}/history/**/*.pq",recursive=True))
    d=pl.scan_parquet(fs,hive_partitioning=True).select(["global_time",pl.col(L+"number_of_oric")]).collect().sort("global_time")
    return np.asarray(d["global_time"].to_list(),float)/60, np.asarray(d[L+"number_of_oric"].to_list(),float)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={"width_ratios":[2,1]})
tm,om=oric_t("out/dnaa4_s06_F05_seed1_12gen/dnaa4_s06_F05_seed1_12gen")
te,oe=oric_t("out/dnaa7_seqa_eclipse40_seed1_8gen/dnaa7_seqa_eclipse40_seed1_8gen")
ax1.plot(tm[tm<=te.max()+5],om[tm<=te.max()+5],color="#1f77b4",lw=1.3,label="MASS heuristic: clean 1<->2")
ax1.plot(te,oe,color="#d62728",lw=1.0,label="mechanistic + SeqA eclipse 40: still 1-4 (over-init)")
ax1.axhline(2,color="0.6",ls=":",lw=1); ax1.set_ylim(0,4.5)
ax1.set_xlabel("time (min)"); ax1.set_ylabel("number of oriC")
ax1.set_title("dnaa-7 — global eclipse does NOT restore one-init-per-cycle",fontsize=10.5)
ax1.legend(fontsize=8.5,loc="upper left"); ax1.grid(alpha=0.2)
# re-init bar
labels=["MASS","payoff","thr=8","K=45","eclipse40","eclipse20"]
reinit=[0,1400,3511,6530,2405,2578]
cols=["#1f77b4","#d62728","#d62728","#d62728","#f59e0b","#f59e0b"]
ax2.bar(range(len(labels)),reinit,color=cols)
ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels,rotation=40,ha="right",fontsize=8)
ax2.set_ylabel("re-initiation ticks (oriC>2)"); ax2.set_title("none of the simple fixes -> 0",fontsize=10)
ax2.grid(alpha=0.2,axis="y")
for ax in (ax1,ax2): ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for ext in ("png","svg"): fig.savefig(f"out/dnaa7_eclipse_insufficient.{ext}",dpi=140)
print("wrote dnaa7_eclipse_insufficient")
