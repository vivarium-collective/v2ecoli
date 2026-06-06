"""5-gen acetate lineage cycle-time plot for the v2ecoli issue."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/meeting_20260526/acetate_multigen_cycle.png")
DATA = Path("out/acetate_multigen/acetate_multigen_timeseries.json")


def main():
    d = json.loads(DATA.read_text())
    gens = d["generations"]
    indices = [g["index"] for g in gens]
    cycle_min = [g["duration"] / 60.0 for g in gens]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(indices, cycle_min, width=0.6, color="#2470d6", alpha=0.9,
                  label="observed cell cycle (v2ecoli)")
    for x, v in zip(indices, cycle_min):
        ax.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=9, color="#1f2328")
    ax.axhline(136, color="#cf222e", linestyle="--", lw=1.5,
               label="declared τ = 136 min")
    ax.set_xticks(indices)
    ax.set_xlabel("generation", fontsize=10)
    ax.set_ylabel("cell cycle time (min)", fontsize=10)
    ax.set_title("5-gen v2ecoli acetate lineage — no equilibration toward declared τ",
                 fontsize=11)
    ax.set_ylim(0, 175)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
