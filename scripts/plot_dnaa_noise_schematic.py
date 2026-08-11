"""Schematic explaining why DnaA-ADP is smoother than DnaA-ATP / total DnaA.

The core insight: DnaA-ATP is a flow variable (its level *reflects* the
balance between bursty production and steady hydrolysis), while DnaA-ADP is
the *time integral* of the hydrolysis flux. Integration is a low-pass
filter — Poisson translation noise gets averaged out before it reaches
the ADP pool.

Usage:
    python scripts/plot_dnaa_noise_schematic.py \\
        --out out/figures/dnaa_noise_schematic.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def _box(ax, xy, w, h, label, fc="#f1f5f9", ec="#334155", lw=1.4,
         fontsize=10, color="#0f172a"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center", fontsize=fontsize, color=color, zorder=3)


def _arrow(ax, xy_from, xy_to, color="#0f172a", lw=1.4,
           connectionstyle="arc3,rad=0.0", zorder=1, mutation_scale=14):
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=mutation_scale,
        color=color, lw=lw, connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(arrow)


def _label(ax, xy, text, fontsize=9, color="#475569",
           ha="left", va="center", weight="normal"):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color,
            ha=ha, va=va, weight=weight, zorder=4)


def _eq_arrow(ax, xy_from, xy_to, color="#0e7490", lw=1.6,
              connectionstyle="arc3,rad=0.0", zorder=2):
    """Bidirectional arrow for a fast equilibrium reaction."""
    arrow = mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="<|-|>",
        mutation_scale=12, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(arrow)


def _inset_trace(fig, bbox, t, y, color, title, ylim=None):
    """Add a small inset Axes with a sample trace."""
    ax = fig.add_axes(bbox)
    ax.plot(t, y, color=color, lw=1.4)
    ax.set_title(title, fontsize=9, color="#0f172a", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    if ylim:
        ax.set_ylim(*ylim)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#94a3b8")
        ax.spines[spine].set_linewidth(0.8)
    return ax


def _make_traces(seed=0):
    """Synthesize illustrative DnaA-ATP and DnaA-ADP traces.

    DnaA-ATP gets Poisson-like translation bursts; DnaA-ADP is the smoothed
    integral of the hydrolysis flux from ATP. Numbers chosen for visual
    clarity, not biological calibration.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 70, 700)               # 70 min cycle
    # Bursty transcription events (~29 events / cycle from measured data)
    n_events = rng.poisson(29)
    event_times = np.sort(rng.uniform(0, 70, n_events))
    event_sizes = rng.exponential(scale=8, size=n_events) + 4
    # Build production rate as sum of narrow Gaussians at event times
    production = np.zeros_like(t)
    for et, es in zip(event_times, event_sizes):
        production += es * np.exp(-((t - et) ** 2) / (2 * 0.6 ** 2))
    # Simple Euler for the two-pool system
    k_h = 0.025                                # /min on ATP
    atp = np.zeros_like(t)
    adp = np.zeros_like(t)
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        d_atp = production[i] - k_h * atp[i - 1]
        d_adp = k_h * atp[i - 1]
        atp[i] = max(atp[i - 1] + d_atp * dt, 0.0)
        adp[i] = adp[i - 1] + d_adp * dt
    return t, atp, adp


def draw(out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        "Why DnaA-ADP is smooth even though DnaA-ATP is noisy",
        fontsize=13, weight="bold", y=0.97,
    )
    ax.text(7.5, 7.25,
            "Integration acts as a low-pass filter — Poisson translation "
            "bursts hit DnaA-ATP directly, but get averaged out before "
            "reaching the DnaA-ADP integrator pool",
            ha="center", va="center", fontsize=10, color="#475569",
            style="italic")

    # ---------------------------------------------------------------------
    # Top row — translation source (bursty Poisson)
    # ---------------------------------------------------------------------
    # Transcription source
    _box(ax, (0.2, 5.30), 2.6, 1.25,
         "Transcription (dnaA)\n"
         "mean 1.7 mRNA / cell (0-7)\n"
         "~29 init events / cycle (15-42)\n"
         "Poisson noise floor ≈ 18%",
         fc="#fef3c7", ec="#b45309", fontsize=9, color="#78350f")

    # Translation arrow → apo DnaA
    _arrow(ax, (2.8, 5.92), (3.20, 5.92), color="#b45309", lw=2.0,
           mutation_scale=14)
    _label(ax, (3.0, 6.20), "translation", fontsize=8, color="#92400e",
           ha="center")

    # apo DnaA box (intermediate)
    _box(ax, (3.20, 5.30), 1.50, 1.25,
         "apo DnaA\n[PD03831]\nfast-cycling\nintermediate",
         fc="#fef9c3", ec="#a16207", fontsize=9, color="#713f12")

    # ----- DnaA-ATP recharge: FAST equilibrium --------------------------
    # apo + ATP ⇌ DnaA-ATP  — upper branch
    _eq_arrow(ax, (4.85, 6.30), (6.30, 6.90),
              color="#0e7490", lw=1.6,
              connectionstyle="arc3,rad=-0.20")
    _label(ax, (5.55, 7.05),
           "apo + ATP  ⇌  DnaA-ATP\n(FAST equilibrium, sub-tick)",
           fontsize=8, color="#0e7490", ha="center", weight="bold")

    # ----- DnaA-ADP recharge: SLOW kinetic equilibrium ------------------
    # apo + ADP ⇌ DnaA-ADP — bidirectional but integrated kinetically (not
    # driven to steady state). Reverse rate is small (~1e-7) but nonzero.
    _eq_arrow(ax, (4.85, 5.55), (6.30, 4.85),
              color="#991b1b", lw=1.6,
              connectionstyle="arc3,rad=0.20")
    _label(ax, (5.55, 4.60),
           "apo + ADP  ⇌  DnaA-ADP\n(SLOW kinetic eq, integrated over dt)",
           fontsize=8, color="#7f1d1d", ha="center", weight="bold")

    # DnaA-ATP box (upper pool)
    _box(ax, (6.30, 6.30), 4.10, 1.10,
         "DnaA-ATP   (flow variable)\n"
         r"$\frac{d[ATP]}{dt} = P_{ATP} - k_h \cdot [ATP] - \mathrm{drains}$"
         "\nτ_relax = 1/k_h ≈ 40 min",
         fc="#dcfce7", ec="#166534", fontsize=9, color="#14532d")

    # DnaA-ADP box (lower pool)
    _box(ax, (6.30, 4.10), 4.10, 1.10,
         "DnaA-ADP   (integrator)\n"
         r"$\frac{d[ADP]}{dt} = k_h[ATP] + k_r[apo][ADP_{bulk}] - \mathrm{dil}$"
         "\nτ_relax ≈ 100 min (dilution only)",
         fc="#fee2e2", ec="#991b1b", fontsize=9, color="#7f1d1d")

    # ----- Kinetic hydrolysis (slow, one-way) ---------------------------
    # DnaA-ATP → DnaA-ADP, vertical arrow on the right
    _arrow(ax, (9.30, 6.30), (9.30, 5.20),
           color="#991b1b", lw=2.4, mutation_scale=18)
    _label(ax, (9.55, 5.75),
           "k_h = 0.025/min\n(kinetic, slow)",
           fontsize=8.5, color="#7f1d1d", ha="left", weight="bold")

    # Side annotation: reaction-rate asymmetry
    _box(ax, (10.80, 5.30), 4.00, 1.25,
         "Key asymmetry (integrate_dt flag):\n"
         "• apo + ATP eq → driven to SS each tick (fast)\n"
         "• apo + ADP eq → integrated over dt (slow)\n"
         "• Hydrolysis → integrated over dt (slow)\n"
         "Fast ATP eq drains [apo] ≈ 0  →  the slow\n"
         "apo+ADP step barely fires in practice",
         fc="#f1f5f9", ec="#475569", fontsize=9, color="#0f172a")

    # ---------------------------------------------------------------------
    # Middle band — visual signal traces with arrows showing transmission
    # ---------------------------------------------------------------------
    # Make synthetic traces for the insets
    t, atp, adp = _make_traces(seed=1)
    # Production trace (the spiky input) — matches measured ~29 events/cycle
    rng = np.random.default_rng(1)
    n_events = rng.poisson(29)
    event_times = np.sort(rng.uniform(0, 70, n_events))
    event_sizes = rng.exponential(scale=8, size=n_events) + 4
    prod = np.zeros_like(t)
    for et, es in zip(event_times, event_sizes):
        prod += es * np.exp(-((t - et) ** 2) / (2 * 0.6 ** 2))

    # Inset 1: bursty production
    _inset_trace(
        fig, [0.04, 0.30, 0.20, 0.12],
        t, prod, "#b45309",
        "Translation events (Poisson)",
        ylim=(-2, prod.max() * 1.1),
    )

    # Inset 2: noisy ATP
    _inset_trace(
        fig, [0.36, 0.30, 0.22, 0.12],
        t, atp, "#166534",
        "DnaA-ATP  —  noisy",
        ylim=(0, atp.max() * 1.1),
    )

    # Inset 3: smooth ADP
    _inset_trace(
        fig, [0.66, 0.30, 0.22, 0.12],
        t, adp, "#991b1b",
        "DnaA-ADP  —  smooth",
        ylim=(0, adp.max() * 1.1),
    )

    # ---------------------------------------------------------------------
    # Bottom row — explanation panel
    # ---------------------------------------------------------------------
    _box(ax, (0.2, 0.40), 4.8, 1.85,
         "Why DnaA-ATP carries the noise\n\n"
         "• apo + ATP eq is FAST → ~all translation\n"
         "  bursts flow into DnaA-ATP within one tick\n"
         "• Loss is KINETIC (k_h ≈ 0.025/min)\n"
         "  → τ_ATP ≈ 40 min\n"
         "• Fast-in + slow-out = pool size reflects\n"
         "  the moment-to-moment balance\n"
         "• Pool size IS the running balance",
         fc="#ecfccb", ec="#3f6212", fontsize=9, color="#1a2e05")

    _box(ax, (5.10, 0.40), 4.8, 1.85,
         "Why integration smooths the signal\n\n"
         "• Each pool acts as a low-pass filter\n"
         "  with cutoff frequency 1/τ\n"
         "• τ_ATP ≈ 40 min,  τ_ADP ≈ 100 min\n"
         "  → ADP's cutoff is 2.5× lower\n"
         "  → ADP rejects ~2.5× more high-freq noise\n"
         "• Plus ADP's input is ALREADY low-pass-\n"
         "  filtered by ATP (cascade of two filters)",
         fc="#f1f5f9", ec="#475569", fontsize=9, color="#0f172a")

    _box(ax, (10.00, 0.40), 4.8, 1.85,
         "Why DnaA-ADP is smooth\n\n"
         "• NO fast input — both inputs are kinetic:\n"
         "    – apo + ADP → DnaA-ADP (slow)\n"
         "    – k_h × DnaA-ATP (slow + pre-filtered)\n"
         "• Loss only via dilution → τ_ADP ≈ 100 min\n"
         "• Pure slow integrator (long memory)\n"
         "• Binds 3 box pools (high-affinity only),\n"
         "  bound-ADP acts as a fixed offset",
         fc="#fef2f2", ec="#7f1d1d", fontsize=9, color="#450a0a")

    # ---------------------------------------------------------------------
    # Footnote
    # ---------------------------------------------------------------------
    ax.text(7.5, 0.40,
            "Total DnaA = ATP + ADP + apo  →  tracks ATP fluctuations "
            "(since ADP only adds a smooth offset)",
            ha="center", va="center", fontsize=10, color="#1f2937",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb",
                      edgecolor="#9ca3af"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="out/figures/dnaa_noise_schematic.png",
    )
    args = ap.parse_args()
    draw(Path(args.out))


if __name__ == "__main__":
    main()
