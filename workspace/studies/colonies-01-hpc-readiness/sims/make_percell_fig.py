"""Render the per-cell RSS figure from percell_rss.csv (colonies hardening).

Two panels:
  (left)  steady-state RSS vs N (plateaus N=1,2,4), with the fitted per-cell
          slope and fixed baseline annotated -- the additive footprint model.
  (right) the reconciled cells-per-node budget across per-cell RSS assumptions:
          450 MB (stale, commit 2f950d9, emit_cells) vs 290 MB (current main,
          bounded recorder) vs the unsupported "1000" claim, using the
          64-actor x K-cells/actor packing model on a 64-core/256 GB node.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "runs", "percell_rss.csv")

rows = list(csv.DictReader(open(CSV)))
by = {r["label"]: r for r in rows}
plateaus = [("plateau_n1", 1), ("plateau_n2", 2), ("plateau_n4", 4)]
ns = [n for _, n in plateaus]
rss = [float(by[lab]["rss_mb"]) for lab, _ in plateaus]

# fit RSS = baseline + slope * N over the plateaus
slope = (rss[-1] - rss[0]) / (ns[-1] - ns[0])
baseline = rss[0] - slope * ns[0]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- left: RSS vs N ----------------------------------------------------------
ax0.plot(ns, rss, "o-", color="#1f77b4", lw=2, ms=8, label="measured RSS")
xs = [0, ns[-1]]
ax0.plot(xs, [baseline + slope * x for x in xs], "--", color="#888",
         label=f"fit: {baseline:.0f} + {slope:.0f}·N")
for n, r in zip(ns, rss):
    ax0.annotate(f"{r:.0f}", (n, r), textcoords="offset points",
                 xytext=(6, 8), fontsize=9)
ax0.set_xlabel("cells in one process (N)")
ax0.set_ylabel("steady-state RSS (MB)")
ax0.set_title(f"Within-process per-cell RSS ≈ {slope:.0f} MB/cell\n"
              "(sim_data lru-shared; numpy flat across N)")
ax0.set_xticks(ns)
ax0.grid(alpha=0.3)
ax0.legend(fontsize=8)

# --- right: cells-per-node budget vs per-cell assumption ---------------------
NODE_GB, CORES = 256, 64
per_actor_mb = NODE_GB * 1024 / CORES  # RAM budget per Ray actor
fixed_mb = baseline  # own sim_data + imports, paid once per actor (process)


def cells_per_node(per_cell_mb):
    k = int((per_actor_mb - fixed_mb) // per_cell_mb)  # cells per actor
    return max(k, 0) * CORES, k


scenarios = [
    (f"stale 450 MB/cell\n(2f950d9, emit_cells)", 450, "#d62728"),
    (f"current {slope:.0f} MB/cell\n(main, bounded recorder)", slope, "#2ca02c"),
]
labels, vals, colors = [], [], []
for name, pc, col in scenarios:
    tot, k = cells_per_node(pc)
    labels.append(name + f"\n→ {k} cells/actor")
    vals.append(tot)
    colors.append(col)
# the unsupported executive claim, for contrast
labels.append("exec claim\n(unsupported)")
vals.append(1000)
colors.append("#999999")

bars = ax1.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax1.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v),
                 ha="center", va="bottom", fontsize=10)
ax1.set_ylabel("cells / node (64-core, 256 GB)")
ax1.set_title("Reconciled RAM budget\n(64 Ray actors × K cells, own sim_data/actor)")
ax1.axhline(384, ls=":", color="#d62728", alpha=0.6)
ax1.annotate("prior study figure (384)", (2.3, 384), fontsize=8,
             color="#d62728", ha="right", va="bottom")
ax1.grid(axis="y", alpha=0.3)

fig.tight_layout()
out_dir = os.path.join(HERE, "..", "charts")
os.makedirs(out_dir, exist_ok=True)
out = os.path.abspath(os.path.join(out_dir, "percell_rss_budget.svg"))
fig.savefig(out, bbox_inches="tight")
print(f"per-cell slope = {slope:.1f} MB/cell; baseline = {baseline:.0f} MB")
print(f"per-actor RAM budget = {per_actor_mb:.0f} MB; "
      f"current-main cells/node = {cells_per_node(slope)[0]} "
      f"({cells_per_node(slope)[1]}/actor × {CORES})")
print(f"WROTE {out}")
