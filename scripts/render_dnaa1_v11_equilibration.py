"""dnaa-1 V=1.1e-3 EQUILIBRATION figure (15-generation run, 2026-06-09).

THE QUESTION: a 7-gen V=1.1e-3 run showed the DnaA monomer cycle-mean RAMPING
(gens 4-7: 484/573/804/842) with gens 6-7 creeping over 800, leaving it
unclear whether V=1.1e-3 EQUILIBRATES inside the accepted band [300,800] or
keeps DRIFTING over. The completed 15-generation run
`out/dnaa1_v1p1e-3_long` (V=1.1e-3 via --perturbation, seed 1, resume from the
dnaa-0 gen-3 dill) settles it.

DnaA monomer total per tick = sum of the 3 bulk forms
    PD03831[c] (apo) + MONOMER0-160[c] (ATP) + MONOMER0-4565[c] (ADP).
Canonical all-zeros daughter lineage (agent_id ^0+$); daughter-stub gens
(<5 min) dropped (gen 16 is an 18-tick stub).

This script consumes the per-generation stats already extracted with the
canonical `render_dnaa1_v12_multiseed.per_gen_stats` (NO simulation here — the
run is complete; light read of a stats JSON + matplotlib only) and draws the
DnaA cycle-mean (+ trough/peak band) vs generation 1-15 against the shaded
[300,800] band, with the equilibration verdict.

Usage:
    python scripts/render_dnaa1_v11_equilibration.py \
        --stats /tmp/dnaa1_long_stats.json \
        --out studies/dnaa-1-expression/charts
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

BAND_LO, BAND_HI = 300, 800
V_LABEL = "1.1e-3"
RUN_ID = "dnaa1_v1p1e-3_long"


def render(stats: list[dict], out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = sorted(stats, key=lambda s: s["gen"])
    gens = np.array([s["gen"] for s in stats])
    means = np.array([s["dnaa_mean"] for s in stats])
    troughs = np.array([s["dnaa_min"] for s in stats])
    peaks = np.array([s["dnaa_max"] for s in stats])

    # plateau / drift quantification on the late generations
    late = [s for s in stats if 10 <= s["gen"] <= 15]
    lg = np.array([s["gen"] for s in late])
    lm = np.array([s["dnaa_mean"] for s in late])
    late_slope = float(np.polyfit(lg, lm, 1)[0])
    late_mean = float(lm.mean())
    full_slope = float(np.polyfit(gens, means, 1)[0])
    peak_gen = int(gens[int(np.argmax(means))])
    peak_mean = float(means.max())
    final_mean = float(means[-1])

    # verdict: does it equilibrate in band (A) or drift/break out of band (B)?
    overshoots = bool((means > BAND_HI).any() or (peaks > BAND_HI).any())
    # A clean equilibration would settle with cycle-means in band; here the
    # pool OVERSHOOTS the band before coming back, so the 7-gen in-band-ness
    # was a pre-equilibration artifact.
    verdict_B = overshoots
    verdict_letter = "B" if verdict_B else "A"

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.axhspan(BAND_LO, BAND_HI, color="0.5", alpha=0.15, lw=0,
               label=f"accepted band [{BAND_LO}, {BAND_HI}]")

    # trough-peak envelope
    ax.fill_between(gens, troughs, peaks, color="#1f77b4", alpha=0.15,
                    label="per-gen trough–peak envelope", zorder=2)
    # cycle-mean trajectory
    cols = ["#2ca02c" if BAND_LO <= m <= BAND_HI else "#d62728" for m in means]
    ax.plot(gens, means, "-", color="#1f77b4", lw=1.8, zorder=3)
    ax.scatter(gens, means, c=cols, s=55, zorder=4,
               label="cycle-mean (green=in band, red=over)")
    for g, m in zip(gens, means):
        ax.annotate(f"{m:.0f}", (g, m), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5,
                    color=("#2ca02c" if BAND_LO <= m <= BAND_HI else "#d62728"))

    # mark the band-crossing and the peak
    ax.axhline(BAND_HI, color="0.4", lw=0.8, ls="--", alpha=0.7)
    ax.annotate(f"peak cycle-mean {peak_mean:.0f}\n(gen {peak_gen})",
                (peak_gen, peak_mean), textcoords="offset points",
                xytext=(8, 18), fontsize=8, color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

    ax.set_xlabel("generation (canonical all-zeros lineage)")
    ax.set_ylabel("DnaA monomer pool (counts, apo+ATP+ADP)")
    ax.set_xticks(gens)
    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(0, max(peaks) * 1.10)
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.93)

    if verdict_B:
        verdict = (
            f"VERDICT (B): V={V_LABEL} does NOT equilibrate IN band. The pool "
            f"RAMPS past 800 (cycle-means hit {peak_mean:.0f} at gen {peak_gen}, "
            f"peaks ~{int(peaks.max())}), then relaxes back to {final_mean:.0f} "
            f"by gen 15. The 7-gen apparent in-band-ness was a PRE-EQUILIBRATION "
            f"artifact; no single V closes band+rate — needs a different lever."
        )
        vcol = "#d62728"
    else:
        verdict = (
            f"VERDICT (A): V={V_LABEL} equilibrates in band (gens 10-15 mean "
            f"{late_mean:.0f}, slope {late_slope:+.0f}/gen) — sound steady state."
        )
        vcol = "#2ca02c"

    fig.suptitle(
        f"dnaa-1 — DnaA pool equilibration at V={V_LABEL} (15-gen run, seed 1)\n"
        "does the cycle-mean PLATEAU in [300,800] or DRIFT over?",
        fontsize=12, y=0.98)
    fig.text(0.5, 0.012, verdict, ha="center", va="bottom", fontsize=9,
             color=vcol, fontweight="bold", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa1_v11_equilibration")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "run_id": "dnaa1_v11_equilibration",
        "source_run_ids": [RUN_ID],
        "generation_id": None,
        "rendered_at": time.time(),
        "command": ("python scripts/render_dnaa1_v11_equilibration.py "
                    "--stats studies/dnaa-1-expression/charts/"
                    "dnaa1_v11_equilibration.stats.json "
                    f"--out {out_dir}"),
        "V": 1.1e-3,
        "v_label": V_LABEL,
        "band": [BAND_LO, BAND_HI],
        "seed": 1,
        "n_generations": int(len(stats)),
        "cycle_means": [float(m) for m in means],
        "troughs": [int(t) for t in troughs],
        "peaks": [int(p) for p in peaks],
        "peak_cycle_mean": peak_mean,
        "peak_gen": peak_gen,
        "final_cycle_mean": final_mean,
        "late_gens_10_15_mean": late_mean,
        "late_gens_10_15_slope_per_gen": late_slope,
        "full_slope_per_gen": full_slope,
        "overshoots_band": overshoots,
        "verdict": verdict_letter,
        "verdict_text": verdict,
        "note": ("DnaA total = bulk PD03831[c]+MONOMER0-160[c]+MONOMER0-4565[c] "
                 "(apo+ATP+ADP, counts); canonical all-zeros daughter lineage; "
                 "stub gens (<5 min, e.g. gen 16) dropped. All gens 1-15 "
                 "divided (mass ratio ~1.8-2.2) with oriC staying {1,2} (no "
                 "reinitiation to 4). FBA GLP_NOFEAS warnings peaked at gen 9 "
                 "(7) tracking the DnaA overshoot and vanished by gens 11-15."),
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"wrote {base}.svg / .png  (verdict {verdict_letter})")
    print(verdict)
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="studies/dnaa-1-expression/charts")
    a = ap.parse_args()
    with open(a.stats) as f:
        stats = json.load(f)
    render(stats, a.out)


if __name__ == "__main__":
    main()
