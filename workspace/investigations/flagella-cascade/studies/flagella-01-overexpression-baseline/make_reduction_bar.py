"""Grouped-bar headline: regulation reduces flagella at every generation.

Plots the per-generation complete-flagella counts recorded by the 2-generation
run (run_studies_multigen.py): the unregulated lineage vs the flagella_regulation
lineage. Makes the central claim of the investigation undeniable at a glance.

Numbers are the recorded gen-end CPLX0-7452 counts from the committed 2-gen run
(study findings: OFF g1=53,g2=44 ; ON g1=44,g2=34). Re-run run_studies_multigen.py
to refresh them.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

GENS = ["generation 1", "generation 2"]
OFF = [53, 44]   # flagella_regulation OFF (unregulated WCM)
ON = [44, 34]    # flagella_regulation ON (Kalir & Alon SUM-gate)


def main():
    x = np.arange(len(GENS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b1 = ax.bar(x - w / 2, OFF, w, label="regulation OFF", color="#9467bd")
    b2 = ax.bar(x + w / 2, ON, w, label="regulation ON", color="#2ca02c")
    for bars in (b1, b2):
        ax.bar_label(bars, padding=2, fontsize=9)
    for i, (o, n) in enumerate(zip(OFF, ON)):
        ax.annotate(f"-{o - n}", (i, max(o, n) + 2.5), ha="center",
                    fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(GENS)
    ax.set_ylabel("complete flagella (CPLX0-7452)")
    ax.set_title("Transcriptional regulation trims flagellar overexpression\n"
                 "at every generation (2-gen lineage, seed 0)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(OFF) + 8)
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/03_reduction_OFF_vs_ON.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
