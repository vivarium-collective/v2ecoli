#!/usr/bin/env python
"""showcase-1 figure 1 — ecoli-sources flat-file manifest.

Groups the ~133 ecoli-sources flat files (the raw ParCa inputs) by domain and
renders a table-figure with per-group file counts + total sizes.

Run on the mini:
    .venv/bin/python studies/showcase-1-parca/sims/fig1_source_manifest.py \
        --flat v2ecoli/processes/parca/reconstruction/ecoli/flat \
        --out studies/showcase-1-parca/charts
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Domain classification by filename substring (first match wins).
GROUPS = [
    ("genes / RNAs / proteins", ["genes", "rnas", "rna_", "_rna", "proteins",
                                  "protein_", "monomer", "cistron", "trna", "rrna",
                                  "gene_", "sequence", "fasta", "base_codes",
                                  "transcription", "translation", "codon"]),
    ("complexation / equilibrium", ["complex", "equilibrium", "two_component",
                                    "endoribonuclease"]),
    ("metabolic reactions", ["reaction", "metabolic", "metabolite", "fba",
                             "kcat", "enzyme", "flux", "transport"]),
    ("biomass / mass", ["biomass", "mass_fraction", "mass", "dry_mass"]),
    ("transcription factors (tf_*)", ["tf_", "_tf", "fold_change", "regulation",
                                      "promoter", "ppgpp"]),
    ("growth / condition params", ["growth", "condition", "media", "doubling",
                                   "nutrient", "environment", "secretion"]),
    ("validation / reference", ["validation", "schmidt", "wisniewski", "toya",
                                "houser", "li_", "taniguchi"]),
]


def classify(name):
    low = name.lower()
    for label, keys in GROUPS:
        if any(k in low for k in keys):
            return label
    return "other / misc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = []
    for root, _dirs, fnames in os.walk(args.flat):
        for f in fnames:
            p = os.path.join(root, f)
            files.append((f, os.path.getsize(p)))

    agg = {}
    for label, _ in GROUPS:
        agg[label] = [0, 0]
    agg["other / misc"] = [0, 0]
    for name, size in files:
        g = classify(name)
        agg[g][0] += 1
        agg[g][1] += size

    # keep only non-empty groups, ordered
    order = [g for g, _ in GROUPS] + ["other / misc"]
    rows = [(g, agg[g][0], agg[g][1]) for g in order if agg[g][0] > 0]
    total_n = sum(r[1] for r in rows)
    total_b = sum(r[2] for r in rows)

    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 0.55 * (len(rows) + 3) + 1.2))
    ax.axis("off")
    ax.set_title(f"ecoli-sources flat-file manifest — {total_n} raw ParCa input "
                 f"files ({total_b/1e6:.1f} MB)",
                 fontsize=13, fontweight="bold", pad=16)

    table_rows = [[g, str(n), f"{b/1024:.0f} KB"] for g, n, b in rows]
    table_rows.append(["TOTAL", str(total_n), f"{total_b/1e6:.2f} MB"])
    tbl = ax.table(
        cellText=table_rows,
        colLabels=["domain group", "files", "size"],
        colWidths=[0.58, 0.16, 0.20],
        cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # header + total styling
    ncol = 3
    for c in range(ncol):
        tbl[0, c].set_facecolor("#2c3e50")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
        tbl[len(table_rows), c].set_facecolor("#ecf0f1")
        tbl[len(table_rows), c].set_text_props(fontweight="bold")

    fig.text(0.5, 0.02,
             "These flat files (TSV/CSV/FASTA) are the from-sources inputs the "
             "ParCa reads to build sim_data.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    base = os.path.join(args.out, "showcase1_source_manifest")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    print(f"wrote {base}.png / .svg  ({total_n} files, {total_b/1e6:.2f} MB)")
    for g, n, b in rows:
        print(f"  {g:38s} {n:3d}  {b/1024:8.0f} KB")


if __name__ == "__main__":
    main()
