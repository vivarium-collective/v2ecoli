"""Render the real deep-parameter Sobol figure for param-uq-04 from results JSON."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
STUDY = os.path.abspath(os.path.join(HERE, "..", "workspace", "studies", "param-uq-04-deep-params"))
ART = os.path.join(STUDY, "artifacts")
CHARTS = os.path.join(STUDY, "charts")
os.makedirs(CHARTS, exist_ok=True)

with open(os.path.join(ART, "sobol_results.json")) as f:
    R = json.load(f)

pnames = R["param_names"]
OBS = R["observables"]
# growth + mass observables highlighted
show_obs = ["instantaneous_growth_rate", "cell_mass"]
pretty = {"instantaneous_growth_rate": "growth rate", "cell_mass": "cell mass",
          "dry_mass": "dry mass"}
plabels = {"rnap_elongation_rate": "RNAP elong.\nrate (post-ParCa)",
           "cell_dry_mass_fraction": "dry-mass\nfraction (rebuild)",
           "kinetic_objective_weight": "FBA kinetic\nweight (post-ParCa)"}

fig, axes = plt.subplots(1, len(show_obs), figsize=(11, 4.4), sharey=True)
colors = {"rnap_elongation_rate": "#2c7fb8", "cell_dry_mass_fraction": "#d95f0e",
          "kinetic_objective_weight": "#999999"}
for ax, o in zip(axes, show_obs):
    a = R["aggregate"][o]
    mean = np.array(a["total_order_mean"])
    lo = np.array(a["total_order_min"]); hi = np.array(a["total_order_max"])
    x = np.arange(len(pnames))
    err = np.vstack([mean - lo, hi - mean])
    ax.bar(x, mean, yerr=err, capsize=5,
           color=[colors[p] for p in pnames], edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls="--", lw=1, color="crimson", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([plabels.get(p, p) for p in pnames], fontsize=8.5)
    ax.set_title(f"{pretty.get(o,o)}\n(PCE test err {a['relerr_test_mean']*100:.1f}%)", fontsize=10)
    ax.set_ylim(0, 1.05)
    for xi, m in zip(x, mean):
        ax.text(xi, min(m + 0.04, 1.0), f"{m:.2f}", ha="center", fontsize=8.5)

axes[0].set_ylabel("total-order Sobol index $S_T$")
axes[-1].text(0.98, 0.52, "dominance 0.5", color="crimson", fontsize=8,
              ha="right", va="bottom", transform=axes[-1].get_yaxis_transform())
fig.suptitle("param-uq-04 — total-order Sobol of DEEP sim_data parameters "
             f"(order-2 PCE, {R['n_train']}+{R['n_test']} samples, seeds {R['seeds']})",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(CHARTS, "deep_param_sobol.png")
fig.savefig(out, dpi=140)
print("wrote", out)
