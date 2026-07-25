"""ColE1 / pBR322 copy-number control-loop schematic for Figure 3.

A preliminary-results schematic showing the BP1993 molecular state machine
currently implemented in v2ecoli: RNA II preprimer maturation at oriV,
RNA I antisense inhibition, Rom-stabilized kissing complex, and RNase H
cleavage producing the replication-competent primer.

Callable as a standalone figure, and also as a helper that draws the
schematic into a provided matplotlib Axes (so it can be embedded as a
panel inside ``plot_plasmid_multiseed_multigen_figure.py``).

    uv run python scripts/plot_colE1_control_schematic.py
"""
from __future__ import annotations

import argparse
import os


def draw_schematic(ax, *, show_title: bool = True,
                   show_footnote: bool = True):
    """Draw the ColE1 control-loop schematic onto ``ax``.

    The axis is configured with a fixed coordinate range (0–14 × -1.2–7.5)
    so the same layout renders identically whether used standalone or as
    an embedded panel.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    ax.set_xlim(0, 14)
    ax.set_ylim(-1.2, 7.5)
    ax.axis("off")

    def node(xy, w, h, text, fc, ec="black", fontsize=10,
             fontweight="normal"):
        x, y = xy
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.2",
            fc=fc, ec=ec, lw=1.4, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, zorder=3)

    def arrow(xy1, xy2, color="black", rad=0.0, lw=1.4,
              style="-|>"):
        a = FancyArrowPatch(
            xy1, xy2,
            arrowstyle=style, mutation_scale=16, lw=lw,
            color=color, connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
        ax.add_patch(a)

    def label(xy, text, color="black", fs=9, fontweight="normal",
              rotation=0):
        ax.text(xy[0], xy[1], text, ha="center", va="center",
                fontsize=fs, color=color, fontweight=fontweight,
                rotation=rotation,
                bbox=dict(boxstyle="round,pad=0.22",
                          fc="white", ec="none", alpha=0.98),
                zorder=4)

    # -----------------------------------------------------------------
    # Left cluster (productive path, bottom): RNA II transcription →
    #   RNA II–DNA hybrid at oriV → RNase H cleavage → primer →
    #   replication initiation.
    # Top-left: RNA I antisense + Rom inhibitor feed into the kissing
    #   complex that aborts the productive path.
    # -----------------------------------------------------------------

    # ---- Productive path nodes (left → middle → right, bottom row) --
    rna_ii    = (2.0, 1.6)
    hybrid    = (6.0, 1.6)
    primer    = (9.7, 1.6)
    repl      = (12.7, 1.6)

    # ---- Inhibitory path nodes (top row) ----------------------------
    rna_i     = (2.0, 5.7)
    kiss      = (6.0, 5.7)
    rom       = (6.0, 3.6)  # Rom sits between RNA I and the kissing complex

    # ---- Nodes ------------------------------------------------------
    node(rna_ii, 2.6, 1.0, "RNA II\n(preprimer)",
         fc="#74b9ff", fontweight="bold")
    node(hybrid, 2.8, 1.0, "RNA II–DNA\nhybrid at oriV",
         fc="#a3d9ff")
    node(primer, 2.6, 1.0, "Mature primer\n(RNase H cleaved)",
         fc="#55efc4", fontweight="bold")
    node(repl, 2.2, 1.0, "Replication\ninitiation",
         fc="#00b894", fontweight="bold")

    node(rna_i, 2.6, 1.0, "RNA I\n(antisense)",
         fc="#ffeaa7", fontweight="bold")
    node(kiss, 3.0, 1.0,
         "RNA I–RNA II\nkissing complex\n(abortive)",
         fc="#fab1a0", fontsize=9.2)
    node(rom, 2.0, 0.75, "Rom protein",
         fc="#d1b3ff", fontsize=9.5)

    # ---- Productive path arrows (bottom row, left → right) ----------
    arrow((rna_ii[0] + 1.35, rna_ii[1]),
          (hybrid[0] - 1.45, hybrid[1]), color="black")
    label(((rna_ii[0] + hybrid[0]) / 2, rna_ii[1] + 0.55),
          "transcription\nfrom oriV", fs=9)

    arrow((hybrid[0] + 1.45, hybrid[1]),
          (primer[0] - 1.35, primer[1]), color="black")
    label(((hybrid[0] + primer[0]) / 2, hybrid[1] + 0.55),
          "RNase H\ncleavage", fs=9)

    arrow((primer[0] + 1.35, primer[1]),
          (repl[0] - 1.15, repl[1]),
          color="#00b894", lw=1.8)
    label(((primer[0] + repl[0]) / 2, primer[1] + 0.50),
          "DNA Pol I\nelongation", fs=9, color="#00b894")

    # ---- Inhibitory path: RNA I binds RNA II → kissing complex -----
    arrow((rna_i[0] + 1.35, rna_i[1]),
          (kiss[0] - 1.55, kiss[1]), color="#d63031", lw=1.6)
    label(((rna_i[0] + kiss[0]) / 2, rna_i[1] + 0.55),
          "antisense\nbinding", fs=9, color="#d63031")

    # Rom stabilizes the kissing complex (upward arrow)
    arrow((rom[0], rom[1] + 0.40),
          (kiss[0], kiss[1] - 0.55),
          color="#8e44ad", lw=1.5)
    label((rom[0] + 1.25, (rom[1] + kiss[1]) / 2),
          "stabilizes\nkissing", fs=9, color="#8e44ad")

    # Kissing complex aborts RNA II → hybrid (T-bar blocking arrow,
    # drawn as a bent red line ending just above the productive path).
    arrow((kiss[0] - 0.6, kiss[1] - 0.55),
          (hybrid[0] - 0.4, hybrid[1] + 0.55),
          color="#d63031", rad=0.25, lw=1.6, style="-[")
    label(((kiss[0] + hybrid[0]) / 2 - 1.0, (kiss[1] + hybrid[1]) / 2),
          "blocks primer\nmaturation", fs=9, color="#d63031")

    # ---- Feedback: copy number up → RNA I up (closing the loop) ----
    # Drawn as a long arc from the replication node back up to RNA I,
    # annotated as "higher copy # → more RNA I transcription".
    arrow((repl[0], repl[1] + 0.55),
          (rna_i[0] + 1.3, rna_i[1] - 0.1),
          color="0.35", rad=-0.35, lw=1.2, style="-|>")
    label((9.0, 5.1),
          "↑ copy number → ↑ RNA I pool\n(negative feedback)",
          fs=9, color="0.2")

    # ---- Title ------------------------------------------------------
    if show_title:
        ax.set_title(
            "ColE1 / pBR322 copy-number control loop "
            "(BP1993 module)",
            fontsize=11.5, loc="left", pad=10,
        )

    # ---- Abbreviation footnote --------------------------------------
    if show_footnote:
        ax.text(
            0.01, 0.02,
            "oriV = plasmid replication origin;   "
            "RNase H = ribonuclease H;   "
            "Rom = RNA-one modulator (encoded by rop);   "
            "BP1993 = Brendel & Perelson 1993.",
            transform=ax.transAxes, fontsize=8.5, color="0.3",
            ha="left", va="bottom",
        )


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
        default="out/plasmid/figure3_colE1_control_schematic.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(13, 7.5))
    draw_schematic(ax)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
