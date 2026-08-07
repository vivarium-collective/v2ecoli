"""3-seed comparison of the dnaa-4 linear-s=0.6 + F-05 reference.
Top: total-DnaA per-gen mean (band [300,800]); bottom: DnaA-ATP fraction per-gen mean (band [0.2,0.5])."""
import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = "listeners__replication_data__"
BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]"}
BOUND_ATP = [L + c for c in ("chromosomal_high_bound_atp", "oriC_high_bound_atp",
                             "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP = [L + c for c in ("chromosomal_high_bound_adp", "oriC_high_bound_adp",
                             "promoter_high_bound_adp")]
SEEDS = [0, 1, 2]
COLORS = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}

def per_gen(seed):
    N = f"dnaa4_s06_F05_seed{seed}_12gen"
    fs = sorted(glob.glob(f"out/{N}/{N}/history/**/*.pq", recursive=True))
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {k: ids.index(v) for k, v in BULK.items()}
    df = pl.scan_parquet(fs, hive_partitioning=True).select(
        [pl.col(c) for c in (["generation"] + BOUND_ATP + BOUND_ADP)]
        + [pl.col("bulk__count").list.get(i).alias(k) for k, i in idx.items()]
    ).collect().sort("generation")
    a = lambda d, c: np.asarray(d[c].to_list(), float)
    gens, tot, frac = [], [], []
    for g in sorted(set(df["generation"].to_list())):
        d = df.filter(pl.col("generation") == g)
        free = a(d, "apo") + a(d, "atp") + a(d, "adp")
        batp = sum(a(d, c) for c in BOUND_ATP); badp = sum(a(d, c) for c in BOUND_ADP)
        t = free + batp + badp
        gens.append(g); tot.append(t.mean())
        frac.append(((a(d, "atp") + batp) / np.maximum(t, 1.0)).mean())
    return np.array(gens), np.array(tot), np.array(frac)

data = {s: per_gen(s) for s in SEEDS}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax1.axhspan(300, 800, color="0.85", zorder=0, label="acceptance band [300,800]")
ax2.axhspan(0.2, 0.5, color="0.85", zorder=0, label="acceptance band [0.2,0.5]")
fmean = {}
for s in SEEDS:
    g, tot, frac = data[s]
    steady = (g >= 4) & (g <= 11)
    fmean[s] = frac[steady].mean()
    lbl = f"seed {s}" + (" (reference)" if s == 1 else "")
    ax1.plot(g, tot, "-o", color=COLORS[s], ms=4, lw=1.6, label=lbl)
    ax2.plot(g, frac, "-o", color=COLORS[s], ms=4, lw=1.6, label=lbl)

ax1.set_ylabel("total DnaA / cell\n(per-gen mean)")
ax1.set_title("dnaa-4 3-seed confirmation — linear s=0.6 + F-05, V=1.5e-3, k_h=0.025/min\n"
              "homeostasis robust 3/3; DnaA-ATP fraction band seed-sensitive (1/3)", fontsize=10)
ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax1.set_ylim(0, 900)
ax2.set_ylabel("DnaA-ATP fraction\n(per-gen mean)")
ax2.set_xlabel("generation")
ax2.set_ylim(0, 0.6)
ax2.legend([f"seed {s}: g4-11 mean {fmean[s]:.3f}" + (" ✓" if 0.2 <= fmean[s] <= 0.5 else " ✗")
            for s in SEEDS], fontsize=8, loc="upper left", framealpha=0.9)
for ax in (ax1, ax2):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "out/dnaa4_s06_F05_3seed_comparison.png"
fig.savefig(out, dpi=140)
print("wrote", out)
print("steady g4-11 ATP-fraction means:", {s: round(fmean[s], 3) for s in SEEDS})
