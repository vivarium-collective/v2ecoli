"""dnaa-4 EFFECT figure: make the autoregulation feedback visible (v2, cleaned).

  A. Homeostasis plateau  - steady DnaA setpoint vs expression rate V.
  B. Phase portrait       - (promoter occupancy -> DnaA pool): feedback settles on
                            a tight attractor; control drifts up & out.
  C. The brake in action  - pool curves (control drifts out / dnaa-4 held) with the
                            smoothed repression factor R modulating underneath.
High-frequency box-binding noise is smoothed for the sensor/brake traces only.
"""
import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

L = "listeners__replication_data__"
M = "listeners__mass__"
BOUND_ATP = [L + c for c in ("chromosomal_high_bound_atp", "oriC_high_bound_atp",
                             "oriC_low_bound_atp", "promoter_high_bound_atp")]
BOUND_ADP = [L + c for c in ("chromosomal_high_bound_adp", "oriC_high_bound_adp",
                             "promoter_high_bound_adp")]
PROM = [L + "promoter_high_free", L + "promoter_high_bound_atp", L + "promoter_high_bound_adp"]
BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]"}
S, K, N = 0.7, 0.5, 4
BAND = (300, 800)


def smooth(x, w=81):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="same")
    h = w // 2  # fix edge roll-off
    y[:h] = x[:h].mean()
    y[-h:] = x[-h:].mean()
    return y


def frame(run_dir):
    fs = sorted(glob.glob(f"{run_dir}/**/history/**/*.pq", recursive=True))
    if not fs:
        raise FileNotFoundError(run_dir)
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    idx = {k: ids.index(v) for k, v in BULK.items()}
    cols = list(dict.fromkeys(["generation", "global_time", M + "cell_mass"]
                              + BOUND_ATP + BOUND_ADP + PROM))
    df = pl.scan_parquet(fs, hive_partitioning=True).select(
        [pl.col(c) for c in cols]
        + [pl.col("bulk__count").list.get(i).alias(k) for k, i in idx.items()]
    ).collect().sort(["generation", "global_time"])
    a = lambda c: np.asarray(df[c].to_list(), float)
    bound_atp = sum(a(c) for c in BOUND_ATP)
    bound_adp = sum(a(c) for c in BOUND_ADP)
    total = a("apo") + a("atp") + a("adp") + bound_atp + bound_adp
    gen = a("generation").astype(int)
    gtime = a("global_time")
    abs_min = np.zeros_like(gtime)
    off = 0.0
    for gg in sorted(set(gen)):
        m = gen == gg
        abs_min[m] = (off + gtime[m]) / 60.0
        off += gtime[m].max()
    prom_bound = a(PROM[1]) + a(PROM[2])
    occ = prom_bound / np.maximum(a(PROM[0]) + prom_bound, 1.0)
    return dict(gen=gen, t=abs_min, total=total, occ=occ)


def ss_mean(run_dir, ss_gen=3):
    f = frame(run_dir)
    return float(np.mean(f["total"][f["gen"] >= ss_gen]))


def brake(occ):
    occ = np.asarray(occ, float)
    return 1.0 - S * (occ ** N) / (K ** N + occ ** N)


CONTROL = {1.0: "out/dnaa4_control_v1e3_16gen", 1.5: "out/dnaa4_control_v15e3_16gen",
           2.0: "out/dnaa4_control_v2e3_16gen"}
AUTOREG = {1.0: "out/dnaa4_autoreg_v1e3_16gen", 1.5: "out/dnaa4_autoreg_hill_v15e3_16gen",
           2.0: "out/dnaa4_autoreg_hill_v2e3_16gen"}
ctrl_pts = {v: ss_mean(d) for v, d in CONTROL.items()}
auto_pts = {v: ss_mean(d) for v, d in AUTOREG.items()}
print("control:", {v: round(x) for v, x in ctrl_pts.items()})
print("autoreg:", {v: round(x) for v, x in auto_pts.items()})

fa = frame("out/dnaa4_s07_seed1_16gen_fixed")    # feedback ON, V=1.5e-3
fc = frame("out/dnaa4_control_v15e3_16gen")        # feedback OFF, V=1.5e-3

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11.5})
fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.92], hspace=0.34, wspace=0.22)
C_CTRL, C_AUTO, C_BRAKE = "#dc2626", "#2563eb", "#7c3aed"

# ---- A: homeostasis plateau ----
axA = fig.add_subplot(gs[0, 0])
axA.axhspan(*BAND, color="#16a34a", alpha=0.10, zorder=0)
for y in BAND:
    axA.axhline(y, color="#16a34a", lw=0.8, ls="--", alpha=0.6)
vs = sorted(ctrl_pts)
axA.plot(vs, [ctrl_pts[v] for v in vs], "-o", color=C_CTRL, lw=2.4, ms=9, label="feedback OFF (control)")
axA.plot(vs, [auto_pts[v] for v in vs], "-o", color=C_AUTO, lw=2.4, ms=9, label="feedback ON (dnaa-4)")
axA.annotate("runaway —\nDnaA tracks V", (vs[-1], ctrl_pts[vs[-1]]), xytext=(-12, -34),
             textcoords="offset points", color=C_CTRL, fontsize=9, ha="right",
             arrowprops=dict(arrowstyle="->", color=C_CTRL))
axA.annotate("homeostasis —\nsetpoint flat (~300)", (vs[1], auto_pts[vs[1]]), xytext=(6, 44),
             textcoords="offset points", color=C_AUTO, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=C_AUTO))
axA.text(0.97, 0.06, "homeostatic band [300, 800]", transform=axA.transAxes, ha="right",
         color="#15803d", fontsize=8)
axA.set_xlabel("constitutive dnaA expression  V  (×10$^{-3}$)")
axA.set_ylabel("steady-state DnaA pool (molecules)")
axA.set_title("A · The effect: feedback cancels the expression rate")
axA.set_xticks(vs)
axA.legend(loc="upper left", fontsize=9, framealpha=0.9)
axA.grid(alpha=0.25)

# ---- B: phase portrait — autoreg attractor density vs control drift ----
axB = fig.add_subplot(gs[0, 1])
axB.axhspan(*BAND, color="#16a34a", alpha=0.08, zorder=0)
mc = (fc["gen"] >= 4) & (fc["gen"] <= 12)
ma = (fa["gen"] >= 4) & (fa["gen"] <= 12)
# autoreg: 2D density -> the attractor shows up as a bright blob
xa, ya = smooth(fa["occ"][ma], 41), smooth(fa["total"][ma], 41)
hb = axB.hexbin(xa, ya, gridsize=34, cmap="Blues", mincnt=1, extent=(0, 1, 200, 1500), zorder=1)
xc, yc = smooth(fc["occ"][mc], 121), smooth(fc["total"][mc], 41)
axB.plot(xc, yc, color=C_CTRL, lw=2.6, alpha=0.95, solid_capstyle="round",
         label="control (drifts up & out →)", zorder=3)
axB.annotate("", (xc[-1], yc[-1]), (xc[-400], yc[-400]),
             arrowprops=dict(arrowstyle="-|>", color=C_CTRL, lw=2), zorder=4)
axB.plot(np.mean(xa), np.mean(ya), "*", color="#1e3a8a", ms=24, mec="white", mew=1.4, zorder=5)
axB.annotate("attractor\n(setpoint ≈ 330)", (np.mean(xa), np.mean(ya)), xytext=(16, -4),
             textcoords="offset points", color="#1e3a8a", fontsize=9, fontweight="bold")
axB.plot([], [], "s", color="#2563eb", ms=9, label="dnaa-4 (settles on attractor)")
axB.set_xlim(0, 1)
axB.set_ylim(200, 1500)
axB.set_xlabel("dnaA-promoter occupancy  f  (the sensor)")
axB.set_ylabel("DnaA pool (molecules)")
axB.set_title("B · Homeostasis is a stable attractor")
axB.legend(loc="upper left", fontsize=9, framealpha=0.92)
axB.grid(alpha=0.2)

# ---- C: the brake in action (pool + smoothed repression factor) ----
axC = fig.add_subplot(gs[1, :])
ma = (fa["gen"] >= 5) & (fa["gen"] <= 9)
mc = (fc["gen"] >= 5) & (fc["gen"] <= 9)
ta = fa["t"][ma] - fa["t"][ma].min()
tc = fc["t"][mc] - fc["t"][mc].min()
axC.axhspan(*BAND, color="#16a34a", alpha=0.08, zorder=0)
axC.plot(tc, smooth(fc["total"][mc], 41), color=C_CTRL, lw=1.8, alpha=0.9,
         label="DnaA pool — control (no brake, drifts out)")
axC.plot(ta, smooth(fa["total"][ma], 41), color=C_AUTO, lw=2.0,
         label="DnaA pool — dnaa-4 (held in band)")
axC.set_ylabel("DnaA pool (molecules)")
axC.set_xlabel("time (min) · mid-run generations 5–9")
axC.set_title("C · The brake engaging in real time:  pool ↑ → occupancy ↑ → repression factor R ↓ → synthesis cut → pool pulled back")
axC.set_xlim(0, max(ta.max(), tc.max()))
axC.grid(alpha=0.2)
axC.legend(loc="upper left", fontsize=9, framealpha=0.9)
axC2 = axC.twinx()
R = smooth(brake(fa["occ"][ma]), 81)
axC2.plot(ta, R, color=C_BRAKE, lw=1.8, label="synthesis factor R = 1 − s·fⁿ/(Kⁿ+fⁿ)  (the brake)")
axC2.fill_between(ta, 1.0, R, color=C_BRAKE, alpha=0.10)
axC2.set_ylabel("dnaA synthesis factor R  (1 = full, ↓ = repressed)", color=C_BRAKE)
axC2.set_ylim(0, 1.05)
axC2.tick_params(axis="y", labelcolor=C_BRAKE)
axC2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

fig.suptitle("The effect of dnaA self-autoregulation (dnaa-4): a molecular thermostat for the DnaA pool",
             fontsize=13.5, y=0.985)
out = "out/dnaa4_effect_thermostat.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
