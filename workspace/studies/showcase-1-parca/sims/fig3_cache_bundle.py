#!/usr/bin/env python
"""showcase-1 figure 3 — cache-bundle contents.

Table-figure of the 4 cache-bundle artifacts produced by build_cache.py
(initial_state.json, sim_data_cache.dill, metadata.json, .cache_version) with
their sizes, plus a short initial_state molecule-count summary.

Run on the mini:
    .venv/bin/python studies/showcase-1-parca/sims/fig3_cache_bundle.py \
        --cache out/cache-showcase \
        --out studies/showcase-1-parca/charts
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED = ["initial_state.json", "sim_data_cache.dill", "metadata.json",
            "cache_version.json"]


def fmt_size(b):
    if b >= 1e6:
        return f"{b/1e6:.2f} MB"
    if b >= 1e3:
        return f"{b/1e3:.1f} KB"
    return f"{b} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    present = set(os.listdir(args.cache))
    rows = []
    for name in REQUIRED:
        if name in present:
            size = os.path.getsize(os.path.join(args.cache, name))
            rows.append([name, "present", fmt_size(size)])
        else:
            rows.append([name, "MISSING", "--"])

    # initial_state molecule-count summary
    init_summary = []
    init_path = os.path.join(args.cache, "initial_state.json")
    if os.path.exists(init_path):
        try:
            with open(init_path) as f:
                init = json.load(f)
            bulk = init.get("bulk")
            n_bulk = None
            if isinstance(bulk, dict) and "shape" in bulk:
                # numpy-structured-array dump: shape == [n_species]
                try:
                    n_bulk = int(bulk["shape"][0])
                except Exception:
                    n_bulk = None
            elif hasattr(bulk, "__len__"):
                n_bulk = len(bulk)
            if n_bulk is not None:
                init_summary.append(("bulk molecule species", f"{n_bulk:,}"))
            uniq = init.get("unique")
            if isinstance(uniq, dict):
                init_summary.append(("unique molecule types", f"{len(uniq):,}"))
            # top-level store count
            init_summary.append(("top-level state stores",
                                 ", ".join(init.keys())))
        except Exception as e:
            init_summary.append(("initial_state parse", f"err: {e}"))

    os.makedirs(args.out, exist_ok=True)
    nrows = len(rows) + len(init_summary)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.5, 0.55 * nrows + 2.2),
        gridspec_kw={"height_ratios": [len(rows) + 1, max(len(init_summary), 1) + 1]})
    fig.suptitle("showcase-1: ParCa cache bundle — out/cache-showcase",
                 fontsize=13, fontweight="bold")

    # ---- top: the 4 artifacts ----
    ax = axes[0]
    ax.axis("off")
    ax.set_title("bundle artifacts (all four required for showcase-2 resume)",
                 fontsize=11, loc="left", pad=8)
    t = ax.table(cellText=rows, colLabels=["artifact", "status", "size"],
                 colWidths=[0.5, 0.25, 0.25], cellLoc="left", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.7)
    for c in range(3):
        t[0, c].set_facecolor("#2c3e50")
        t[0, c].set_text_props(color="white", fontweight="bold")
    for r in range(1, len(rows) + 1):
        col = "#27ae60" if rows[r - 1][1] == "present" else "#c0392b"
        t[r, 1].set_text_props(color=col, fontweight="bold")

    # ---- bottom: initial_state summary ----
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("initial_state.json molecule summary",
                  fontsize=11, loc="left", pad=8)
    if init_summary:
        t2 = ax2.table(cellText=[[k, v] for k, v in init_summary],
                       colLabels=["quantity", "value"],
                       colWidths=[0.6, 0.4], cellLoc="left", loc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(11)
        t2.scale(1, 1.7)
        for c in range(2):
            t2[0, c].set_facecolor("#2c3e50")
            t2[0, c].set_text_props(color="white", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "(initial_state.json not parseable)",
                 ha="center", va="center", color="#999")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base = os.path.join(args.out, "showcase1_cache_bundle")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    print(f"wrote {base}.png / .svg")
    print("bundle:", rows)
    print("init_summary:", init_summary)

    with open(base + ".bundle.json", "w") as f:
        json.dump({"artifacts": rows, "initial_state": init_summary}, f, indent=2)


if __name__ == "__main__":
    main()
