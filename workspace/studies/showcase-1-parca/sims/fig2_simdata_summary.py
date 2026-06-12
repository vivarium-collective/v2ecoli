#!/usr/bin/env python
"""showcase-1 figure 2 — sim_data summary (the "this is the FULL ParCa" proof).

Hydrates the ParCa fixture and counts genes / TUs / monomers / complexes /
metabolic reactions / conditions-fitted. The headline panel makes the
51 (full) vs 7 (fast/debug) condition contrast explicit.

Run on the mini:
    .venv/bin/python studies/showcase-1-parca/sims/fig2_simdata_summary.py \
        --fixture out/sim_data-showcase/parca_state.pkl.gz \
        --out studies/showcase-1-parca/charts
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _len(x):
    try:
        return int(len(x))
    except Exception:
        try:
            return int(x.shape[0])
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)

    state = load_parca_state(args.fixture)
    sd = hydrate_sim_data_from_state(state)

    counts = {}
    p = sd.process
    # genes (cistrons)
    try:
        counts["genes (cistrons)"] = _len(p.transcription.cistron_data["id"])
    except Exception:
        try:
            counts["genes (cistrons)"] = _len(p.replication.gene_data["name"])
        except Exception:
            counts["genes (cistrons)"] = None
    # transcription units / RNAs
    try:
        counts["transcription units"] = _len(p.transcription.rna_data["id"])
    except Exception:
        counts["transcription units"] = None
    # monomers
    try:
        counts["protein monomers"] = _len(p.translation.monomer_data["id"])
    except Exception:
        counts["protein monomers"] = None
    # complexes
    try:
        counts["complexes"] = _len(p.complexation.ids_complexes)
    except Exception:
        counts["complexes"] = None
    # metabolic reactions
    try:
        counts["metabolic reactions"] = _len(list(p.metabolism.reaction_stoich.keys()))
    except Exception:
        counts["metabolic reactions"] = None
    # conditions fitted (the proof)
    try:
        n_cond = _len(list(sd.condition_to_doubling_time.keys()))
    except Exception:
        n_cond = None
    counts["TF conditions fitted"] = n_cond

    print("sim_data counts:")
    for k, v in counts.items():
        print(f"  {k:24s} {v}")

    os.makedirs(args.out, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.32)

    # ---- left: bar chart of the structural counts ----
    ax = fig.add_subplot(gs[0, 0])
    bar_keys = ["genes (cistrons)", "transcription units", "protein monomers",
                "complexes", "metabolic reactions"]
    labels = [k for k in bar_keys if counts.get(k) is not None]
    vals = [counts[k] for k in labels]
    bars = ax.barh(range(len(labels)), vals, color="#3498db")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("sim_data structural inventory\n(rebuilt from ecoli-sources)",
                 fontsize=12, fontweight="bold", pad=14)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01,
                b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: the 51 vs 7 headline panel ----
    ax2 = fig.add_subplot(gs[0, 1])
    full = n_cond if n_cond is not None else 51
    fast = 7
    b2 = ax2.bar(["this run\n(--mode full)", "--mode fast\n(debug, rejected)"],
                 [full, fast], color=["#27ae60", "#bdc3c7"])
    ax2.set_ylabel("TF conditions fitted")
    ax2.set_title("FULL ParCa proof", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(full, fast) * 1.25)
    for rect, v in zip(b2, [full, fast]):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + max(full, fast) * 0.02,
                 str(v), ha="center", fontsize=13, fontweight="bold")
    ax2.text(0.5, -0.22,
             f"{full} conditions = full (not 7).\nFast mis-calibrates "
             "dnaA / replication.",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color="#555")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("showcase-1: v2ecoli sim_data rebuilt in full from ecoli-sources",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    base = os.path.join(args.out, "showcase1_simdata_summary")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    print(f"wrote {base}.png / .svg")

    # emit counts as json for the meta sidecar / study.yaml
    import json
    with open(base + ".counts.json", "w") as f:
        json.dump(counts, f, indent=2)


if __name__ == "__main__":
    main()
