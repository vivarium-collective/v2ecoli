"""dnaa-3 Phase 2 / dnaa-4 mechanism schematic.

Renders a single-page diagram of what the box-binding stack wires into the
simulation. The ``--variant`` flag toggles between:

  dnaa3 (default)
    Phase 2 only: 4 affinity pools, K_d_high = 1 nM, intrinsic hydrolysis at
    k_h = 0.046/min (Sekimizu 1987), mass-only initiation gate, no
    self-autoregulation.

  dnaa4
    Phase 2 + dynamic dnaA self-autoregulation. Three changes:
      • K_d_high raised to 3 nM (chromosomal_high, oriC_high, promoter_high)
        so the promoter genuinely de-occupies at low DnaA and the
        autoregulation loop has a titration handle.
      • k_h lowered to 0.025/min so ATPfr settles inside the 0.2-0.5 band.
      • Hill-form repression on TU00259[c] (dnaA operon):
            scale = 1 − s · f^n / (K^n + f^n)
        with s = 0.8 (5x at saturation), n = 4, K = 0.5, where f is the
        promoter_high bound-fraction read from the replication_data listener.

Usage:
    python scripts/plot_dnaa3_phase2_schematic.py \\
        --variant dnaa4 \\
        --out out/figures/dnaa4_schematic.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _box(ax, xy, w, h, label, fc="#f1f5f9", ec="#334155", lw=1.4,
         fontsize=9, color="#0f172a"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center", fontsize=fontsize, color=color, zorder=3)


def _arrow(ax, xy_from, xy_to, color="#0f172a", lw=1.3, ls="-",
           connectionstyle="arc3,rad=0.0"):
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw, linestyle=ls,
        connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(arrow)


def _label(ax, xy, text, fontsize=8, color="#475569",
           ha="left", va="center", bbox=None):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color,
            ha=ha, va=va, bbox=bbox, zorder=4)


def _bar_arrow(ax, xy_from, xy_to, color="#b91c1c", lw=1.6,
               connectionstyle="arc3,rad=0.0", zorder=1):
    """Repression arrow ending in a T-bar instead of a normal arrowhead."""
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-[, widthB=0.5, lengthB=0.18",
        mutation_scale=12, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(arrow)


def draw(out_path: Path, variant: str = "dnaa3",
         linear_s: float | None = None,
         te_mult: float = 1.0) -> None:
    if variant not in ("dnaa3", "dnaa4"):
        raise ValueError(f"unknown variant {variant!r}; expected dnaa3 or dnaa4")
    kd_high_nM = 3 if variant == "dnaa4" else 1
    k_h = 0.025 if variant == "dnaa4" else 0.046
    kd_note = "   [was 1]" if variant == "dnaa4" else ""
    kh_note = "   [was 0.046]" if variant == "dnaa4" else ""
    # When linear_s is set on the dnaa-4 variant, switch the autoregulation
    # form from Hill to linear. Hill default is kept for backward compatibility.
    is_linear = variant == "dnaa4" and linear_s is not None
    has_te_callout = variant == "dnaa4" and te_mult > 1.0
    # Extend canvas to make room for the CHANGES sidebar on dnaa-4.
    fig_w = 17 if variant == "dnaa4" else 14
    x_max = 17 if variant == "dnaa4" else 14

    fig, ax = plt.subplots(figsize=(fig_w, 9))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    if variant == "dnaa4":
        autoreg_label = "linear" if is_linear else "Hill"
        title = (f"dnaa-4 — DnaA-box binding + dynamic dnaA self-autoregulation "
                 f"({autoreg_label} on promoter occupancy)")
    else:
        title = "dnaa-3 Phase 2 — DnaA-box binding mechanism (what we wired into v2ecoli)"
    fig.suptitle(title, fontsize=12, y=0.96)

    # Translation source -------------------------------------------------
    if variant == "dnaa4":
        # Add an explicit transcript_initiation box upstream so the
        # autoregulation T-bar has somewhere clear to land.
        _box(ax, (-0.05, 7.5), 1.4, 0.7,
             "transcript_initiation\nTU00259[c]",
             fc="#fde68a", ec="#92400e", fontsize=7)
        _box(ax, (1.55, 7.5), 1.20, 0.7,
             "Translation\n(mRNA → apo)",
             fc="#fef3c7", ec="#b45309", fontsize=7)
        _arrow(ax, (1.35, 7.85), (1.55, 7.85))
    else:
        _box(ax, (0.3, 7.5), 1.8, 0.7,
             "Translation\n(dnaA mRNA → apo DnaA)",
             fc="#fef3c7", ec="#b45309", fontsize=8)

    # Bulk apo + charging ------------------------------------------------
    apo_x = 2.85 if variant == "dnaa4" else 2.8
    apo_w = 1.55 if variant == "dnaa4" else 1.6
    _box(ax, (apo_x, 7.5), apo_w, 0.7,
         "apo DnaA\n[PD03831]",
         fc="#fef9c3", ec="#a16207", fontsize=8)

    _box(ax, (4.9, 7.55), 0.9, 0.6,
         "DnaA\ncharging",
         fc="#dbeafe", ec="#1d4ed8", fontsize=8)

    _box(ax, (6.3, 7.5), 2.6, 0.7,
         "bulk DnaA-ATP    bulk DnaA-ADP\n[MONOMER0-160] [MONOMER0-4565]",
         fc="#dcfce7", ec="#166534", fontsize=8)

    # Translation → apo arrow: dnaa-4 has narrower Translation, so the
    # arrow starts further right.
    _arrow(ax, (2.75 if variant == "dnaa4" else 2.10, 7.85),
           (apo_x, 7.85))
    _arrow(ax, (4.4, 7.85), (4.9, 7.85))
    _arrow(ax, (5.8, 7.85), (6.3, 7.85))
    _label(ax, (5.35, 8.30), "apo + ATP ⇌ DnaA-ATP", fontsize=7,
           color="#1e3a8a", ha="center")
    _label(ax, (5.35, 8.55), "apo + ADP ⇌ DnaA-ADP", fontsize=7,
           color="#1e3a8a", ha="center")

    # Intrinsic hydrolysis (with the Phase 2 extension to bound pool) -----
    _box(ax, (9.3, 7.4), 4.4, 0.95,
         "DnaA-ATP intrinsic hydrolysis (Sekimizu 1987)\n"
         f"k_h = {k_h}/min on (free + bound) DnaA-ATP{kh_note}\n"
         "→ DnaA-ADP + Pi + PROTON − WATER (stoich-tracked)\n"
         "Bound-pool: in-place ATP→ADP form swap; molecule\n"
         "stays attached, then re-equilibrates next tick",
         fc="#fce7f3", ec="#9d174d", fontsize=7)
    _arrow(ax, (8.9, 7.85), (9.3, 7.85))

    # The 4 pools (binding-step targets) ---------------------------------
    pool_y = 4.3
    pool_h = 1.0
    pool_w = 3.0

    _box(ax, (0.4, pool_y), pool_w, pool_h,
         f"chromosomal_high\n302 sites\nK_d = {kd_high_nM} nM{kd_note}\nbinds ATP or ADP",
         fc="#e0e7ff", ec="#3730a3", fontsize=8)

    _box(ax, (3.7, pool_y), pool_w, pool_h,
         f"oriC_high\n3 sites (R1, R2, R4)\nK_d = {kd_high_nM} nM{kd_note}\nbinds ATP or ADP",
         fc="#fae8ff", ec="#6b21a8", fontsize=8)

    _box(ax, (7.0, pool_y), pool_w, pool_h,
         "oriC_low\n8 sites (R5M, τ2, I1, I2,\nC3, C2, I3, C1)\n"
         "K_d = 100 nM   ATP only",
         fc="#cffafe", ec="#0e7490", fontsize=8)

    _box(ax, (10.3, pool_y), pool_w, pool_h,
         f"promoter_high\n2 sites (box1, box2)\nK_d = {kd_high_nM} nM{kd_note}\nbinds ATP or ADP",
         fc="#fff7ed", ec="#9a3412", fontsize=8)

    # Binding step label between bulk and pools --------------------------
    _box(ax, (5.4, 5.85), 4.4, 0.85,
         "dnaa_box_binding step — fast-equilibrium competitive Langmuir\n"
         "Mass-balance system in (A_free, D_free) solved by scipy.optimize.root\n"
         "(MINPACK hybr, Newton-like; converges ~10 iter, mass-balance exact)",
         fc="#ecfeff", ec="#155e75", fontsize=7)

    # Arrows from bulk → binding step → pools ----------------------------
    _arrow(ax, (7.6, 7.5), (7.6, 6.6), color="#155e75")
    # binding step → each pool
    for px in (1.9, 5.2, 8.5, 11.8):
        _arrow(ax, (7.6, 6.05), (px, pool_y + pool_h),
               color="#155e75", lw=1.0,
               connectionstyle="arc3,rad=0.0")

    # Autoregulation loop (dnaa-4 only) ----------------------------------
    if variant == "dnaa4":
        # Autoreg formula box sits directly above promoter_high, visually
        # tying the math to the pool that drives the feedback signal.
        if is_linear:
            autoreg_text = (
                "Dynamic dnaA self-autoregulation\n"
                "prob_TU00259[c]  ×=  1 − s · f\n"
                f"f = promoter_high bound-fraction\n"
                f"s = {linear_s}   (≈{1.0/(1.0 - linear_s):.1f}× repression at saturation)"
            )
        else:
            autoreg_text = (
                "Dynamic dnaA self-autoregulation\n"
                "prob_TU00259[c]  ×=  1 − s · f^n / (K^n + f^n)\n"
                "f = promoter_high bound-fraction\n"
                "s = 0.8, n = 4, K = 0.5   (≈5× repression at saturation)"
            )
        autoreg_box_xy = (9.95, 5.45)
        autoreg_box_w  = 3.55
        autoreg_box_h  = 1.05
        _box(
            ax, autoreg_box_xy, autoreg_box_w, autoreg_box_h,
            autoreg_text,
            fc="#fee2e2", ec="#b91c1c", fontsize=7, color="#7f1d1d",
        )
        # T-bar arrow: route as L-shape through the empty corridor between
        # binding-step (top y=6.7) and the bulk row (bottom y=7.5) so the
        # path doesn't cross any box or equilibrium label. Lands on the
        # BOTTOM edge of transcript_initiation (centered at x=0.65, y=7.5).
        corridor_y = 7.10
        x_right = autoreg_box_xy[0] + 0.30           # left edge of autoreg box
        y_top_autoreg = autoreg_box_xy[1] + autoreg_box_h
        x_left = 0.65                                  # center of transcript_initiation
        # Plain polyline for the up + across segments (no arrowhead/T-bar).
        ax.plot(
            [x_right, x_right, x_left],
            [y_top_autoreg, corridor_y, corridor_y],
            color="#b91c1c", lw=1.6, zorder=5,
        )
        # Final short vertical with the T-bar pressing UP against the
        # transcript_initiation box's bottom edge.
        _bar_arrow(
            ax,
            (x_left, corridor_y),
            (x_left, 7.48),
            color="#b91c1c", lw=1.6,
            zorder=5,
        )
        # Short connector arrow promoter_high (top edge) → autoreg box
        # (bottom edge) so the signal source is unambiguous.
        _arrow(
            ax,
            (10.4, pool_y + pool_h),
            (autoreg_box_xy[0] + 0.6, autoreg_box_xy[1]),
            color="#b91c1c", lw=1.2,
        )

    # Fork-passage release -----------------------------------------------
    # Place it directly under the bulk DnaA pool so the arrow can be a short
    # vertical line that doesn't cross any other box.
    _box(ax, (5.7, 1.8), 5.6, 1.4,
         "chromosome_structure.py — fork-passage DnaA release (Phase 2 fix)\n\n"
         "When a replication fork crosses a DnaA box:\n"
         " • parent box is deleted, 2 fresh child boxes added (bound_form=0)\n"
         " • bound DnaA-ATP / DnaA-ADP on the parent → released back to bulk\n"
         " • mass conserved; matches Katayama 2017 fork-passage dissociation",
         fc="#fee2e2", ec="#991b1b", fontsize=7)
    # Per-pool occupancy listener moved to make room (see below).

    # Per-pool occupancy listener (moved to left side) -------------------
    _box(ax, (0.3, 1.8), 5.2, 1.4,
         "replication_data listener — 11 per-pool occupancy counts\n\n"
         " chromosomal_high_{free,bound_atp,bound_adp}\n"
         " oriC_high_{free,bound_atp,bound_adp}\n"
         " oriC_low_{free,bound_atp}\n"
         " promoter_high_{free,bound_atp,bound_adp}",
         fc="#f3f4f6", ec="#374151", fontsize=7)
    for px in (1.9, 5.2, 8.5, 11.8):
        _arrow(ax, (px, pool_y), (2.9, 3.2),
               color="#6b7280", lw=0.8,
               connectionstyle="arc3,rad=-0.05")

    # Fork-release → bulk DnaA: straight vertical arrow that threads the
    # 0.3-unit gap between oriC_high (ends at x=6.7) and oriC_low (x=7.0).
    _arrow(ax, (6.85, 3.2), (6.85, 7.5),
           color="#991b1b", lw=1.4,
           connectionstyle="arc3,rad=0.0")
    _label(ax, (6.95, 5.5), "fork-released DnaA-ATP/ADP\n→ back to bulk",
           fontsize=7, color="#991b1b", ha="left")

    # TE callout (dnaa-4 + te_mult > 1 only) -----------------------------
    if has_te_callout:
        _box(
            ax, (1.45, 8.30), 1.70, 0.45,
            f"TE × {te_mult:g}  (dnaA only)",
            fc="#fef3c7", ec="#b45309", fontsize=7, color="#78350f",
        )

    # CHANGES sidebar (dnaa-4 only, uses extended canvas) ----------------
    if variant == "dnaa4":
        sb_x = 14.3
        sb_w = 2.5
        # Header
        _box(ax, (sb_x, 8.0), sb_w, 0.7,
             "DNAA-4 CHANGES",
             fc="#1f2937", ec="#0f172a", fontsize=10, color="#f9fafb")
        # Each modification as its own row. Only items NEW or MODIFIED in
        # this dnaa-4 study — the box-binding mechanism, fork-passage release,
        # and replication_data listener were inherited from dnaa-3.
        lines = []
        lines.append(("NEW", "Dynamic dnaA autoreg",
                      ("linear: prob ×= (1 − s·f)\n"
                       f"s = {linear_s}") if is_linear
                      else "Hill: prob ×= (1 − s·f^n/(K^n+f^n))\ns=0.8, n=4, K=0.5",
                      "#dcfce7", "#166534"))
        lines.append(("MOD", "K_d (high-affinity)",
                      "3 nM   [was 1 nM]", "#fef3c7", "#b45309"))
        lines.append(("MOD", "k_h (intrinsic hydrolysis)",
                      "0.025/min   [was 0.046]", "#fef3c7", "#b45309"))
        if has_te_callout:
            lines.append(("MOD", "dnaA translation efficiency",
                          f"× {te_mult:g}", "#fef3c7", "#b45309"))
        # Render rows top-down
        row_h = 1.0
        gap = 0.08
        y_cursor = 7.85
        for badge, title_text, body, body_fc, body_ec in lines:
            y_cursor -= (row_h + gap)
            # Badge pill (left, vertically centered)
            badge_fc = "#16a34a" if badge == "NEW" else "#d97706"
            _box(ax, (sb_x + 0.05, y_cursor + row_h - 0.40), 0.55, 0.32,
                 badge, fc=badge_fc, ec=badge_fc, fontsize=7, color="white")
            # Title + body inside one rounded box
            _box(ax, (sb_x + 0.05, y_cursor), sb_w - 0.10, row_h - 0.45,
                 body, fc=body_fc, ec=body_ec, fontsize=6.5, color="#1f2937")
            # Title text above body
            _label(ax, (sb_x + 0.70, y_cursor + row_h - 0.24),
                   title_text, fontsize=7.5, color="#0f172a", ha="left")

    # Initiation gate note (out of scope) -------------------------------
    if variant == "dnaa4":
        gate_caption = (
            "Initiation gate — mass-only (unchanged). "
            "dnaa-4 wires the autoregulation loop; Phase 3 will replace mass-only "
            "with oriC_low-occupancy.")
    else:
        gate_caption = (
            "Initiation gate — unchanged from Phase 1 (mass-only). "
            "Phase 2 is bookkeeping; Phase 3 will gate on oriC_low occupancy.")
    _label(
        ax, (7.0, 0.85), gate_caption,
        fontsize=8, color="#1f2937", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7ed",
                  edgecolor="#9a3412"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/figures/dnaa3_phase2_schematic.png")
    ap.add_argument("--variant", choices=("dnaa3", "dnaa4"), default="dnaa3",
                    help="dnaa3 = Phase 2 only (K_d=1 nM, k_h=0.046, no autoreg). "
                         "dnaa4 = adds self-autoregulation, K_d=3 nM, k_h=0.025.")
    ap.add_argument("--linear-s", type=float, default=None,
                    help="If set on dnaa-4 variant, use linear autoreg "
                         "(1 - s·f) with this s value instead of Hill.")
    ap.add_argument("--te-mult", type=float, default=1.0,
                    help="If >1 on dnaa-4 variant, add a TE multiplier callout "
                         "for dnaA (translation efficiency boost).")
    args = ap.parse_args()
    draw(Path(args.out), variant=args.variant,
         linear_s=args.linear_s, te_mult=args.te_mult)


if __name__ == "__main__":
    main()
