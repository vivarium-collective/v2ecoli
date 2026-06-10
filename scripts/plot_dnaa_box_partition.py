"""Four-panel DnaA-box partition figure for dnaa-3 Phase 2.

Panels (per spec ``dnaa-3-spec.md`` § Visualizations):

  1. DnaA-ATP partition across the cell (concentration = count / cell_mass)
       - bound to high-affinity boxes
         (chromosomal_high + oriC_high + promoter_high bound-ATP)
       - bound to low-affinity boxes (oriC_low bound)
       - free cytoplasmic DnaA-ATP (bulk pool)
       Sum equals total DnaA-ATP concentration.
  2. DnaA-ADP partition (no low-affinity trace; oriC_low is ATP-only)
       - bound to high-affinity boxes
       - free cytoplasmic DnaA-ADP
       Sum equals total DnaA-ADP concentration.
  3. Total DnaA boxes available — raw count (315 -> 630 across cycle).
  4. Total DnaA bound concentration — single trace.

Usage:
    python scripts/plot_dnaa_box_partition.py \\
        --exp-root out/dnaa3_phase2_seed1_parquet/dnaa3_phase2_seed1 \\
        --exp-id dnaa3_phase2_seed1 --lineage-seed 1 --gens 6 \\
        --title-extra "dnaa-3 Phase 2 — V=1e-3, burned-in, seed=1" \\
        --out out/figures/dnaa3_phase2_seed1_4panel.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"

POOL_LISTENER_COLS = [
    "listeners__replication_data__chromosomal_high_bound_atp",
    "listeners__replication_data__chromosomal_high_bound_adp",
    "listeners__replication_data__oriC_high_bound_atp",
    "listeners__replication_data__oriC_high_bound_adp",
    "listeners__replication_data__oriC_low_bound_atp",
    "listeners__replication_data__promoter_high_bound_atp",
    "listeners__replication_data__promoter_high_bound_adp",
    "listeners__replication_data__total_DnaA_boxes",
]

COLS = [
    "global_time",
    "bulk__id",
    "bulk__count",
    "listeners__mass__cell_mass",
] + POOL_LISTENER_COLS


def _lineage_agent(gen: int) -> str:
    return "0" * gen


def load_lineage(exp_root: str, exp_id: str, lineage_seed: int, n_gens: int) -> dict:
    times: list[np.ndarray] = []
    cell_mass: list[np.ndarray] = []
    bulk_atp: list[np.ndarray] = []
    bulk_adp: list[np.ndarray] = []
    bound_atp_high: list[np.ndarray] = []
    bound_atp_low: list[np.ndarray] = []
    bound_adp_high: list[np.ndarray] = []
    total_boxes: list[np.ndarray] = []
    gen_boundaries: list[float] = []
    durations: list[float] = []
    t_offset = 0.0

    for gen in range(1, n_gens + 1):
        agent = _lineage_agent(gen)
        pat = (f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
               f"lineage_seed={lineage_seed}/generation={gen}/agent_id={agent}/*.pq")
        files = sorted(glob.glob(pat),
                       key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
        if not files:
            print(f"  gen {gen}: NO DATA")
            continue
        t_g = []
        cm = []
        b_atp = []
        b_adp = []
        bnd_atp_high = []
        bnd_atp_low = []
        bnd_adp_high = []
        tot_boxes = []
        for f in files:
            tbl = pq.read_table(f, columns=COLS).to_pandas()
            for _, r in tbl.iterrows():
                ids = list(r["bulk__id"])
                cnts = r["bulk__count"]
                lookup = {n: int(cnts[i]) for i, n in enumerate(ids)}
                t_g.append(float(r["global_time"]))
                cm.append(float(r["listeners__mass__cell_mass"]))
                b_atp.append(lookup.get(DNAA_ATP_ID, 0))
                b_adp.append(lookup.get(DNAA_ADP_ID, 0))
                high_atp = (
                    int(r["listeners__replication_data__chromosomal_high_bound_atp"])
                    + int(r["listeners__replication_data__oriC_high_bound_atp"])
                    + int(r["listeners__replication_data__promoter_high_bound_atp"])
                )
                low_atp = int(r["listeners__replication_data__oriC_low_bound_atp"])
                high_adp = (
                    int(r["listeners__replication_data__chromosomal_high_bound_adp"])
                    + int(r["listeners__replication_data__oriC_high_bound_adp"])
                    + int(r["listeners__replication_data__promoter_high_bound_adp"])
                )
                bnd_atp_high.append(high_atp)
                bnd_atp_low.append(low_atp)
                bnd_adp_high.append(high_adp)
                tot_boxes.append(int(r["listeners__replication_data__total_DnaA_boxes"]))
        t_g = np.asarray(t_g)
        order = np.argsort(t_g)
        t_g = t_g[order]
        cm = np.asarray(cm)[order]
        b_atp = np.asarray(b_atp)[order]
        b_adp = np.asarray(b_adp)[order]
        bnd_atp_high = np.asarray(bnd_atp_high)[order]
        bnd_atp_low = np.asarray(bnd_atp_low)[order]
        bnd_adp_high = np.asarray(bnd_adp_high)[order]
        tot_boxes = np.asarray(tot_boxes)[order]

        dur = float(t_g[-1] - t_g[0])
        times.append(t_g + t_offset)
        cell_mass.append(cm)
        bulk_atp.append(b_atp)
        bulk_adp.append(b_adp)
        bound_atp_high.append(bnd_atp_high)
        bound_atp_low.append(bnd_atp_low)
        bound_adp_high.append(bnd_adp_high)
        total_boxes.append(tot_boxes)
        durations.append(dur)
        t_offset = float(times[-1][-1])
        gen_boundaries.append(t_offset)
        print(f"  gen {gen}: tau={dur/60:.1f} min  ticks={len(t_g)}  "
              f"end bound_atp_high={bnd_atp_high[-1]} bound_atp_low={bnd_atp_low[-1]} "
              f"bound_adp_high={bnd_adp_high[-1]} total_boxes={tot_boxes[-1]}")

    return {
        "t": np.concatenate(times) if times else np.array([]),
        "cell_mass": np.concatenate(cell_mass) if cell_mass else np.array([]),
        "bulk_atp": np.concatenate(bulk_atp) if bulk_atp else np.array([]),
        "bulk_adp": np.concatenate(bulk_adp) if bulk_adp else np.array([]),
        "bound_atp_high": np.concatenate(bound_atp_high) if bound_atp_high else np.array([]),
        "bound_atp_low": np.concatenate(bound_atp_low) if bound_atp_low else np.array([]),
        "bound_adp_high": np.concatenate(bound_adp_high) if bound_adp_high else np.array([]),
        "total_boxes": np.concatenate(total_boxes) if total_boxes else np.array([]),
        "gen_boundaries": gen_boundaries[:-1] if gen_boundaries else [],
        "durations_min": [d / 60 for d in durations],
    }


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean over `window` ticks. NaN-aware."""
    if window <= 1 or x.size == 0:
        return x
    x = np.asarray(x, dtype=np.float64)
    finite = np.where(np.isfinite(x), x, 0.0)
    mask = np.isfinite(x).astype(np.float64)
    kernel = np.ones(window) / window
    num = np.convolve(finite, kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def plot(d: dict, out_path: Path, title_extra: str,
         smooth_window: int = 60) -> None:
    t_min = d["t"] / 60
    boundaries_min = [b / 60 for b in d["gen_boundaries"]]
    cm = d["cell_mass"]

    with np.errstate(divide="ignore", invalid="ignore"):
        atp_high_conc = np.where(cm > 0, d["bound_atp_high"] / cm, np.nan)
        atp_low_conc = np.where(cm > 0, d["bound_atp_low"] / cm, np.nan)
        atp_free_conc = np.where(cm > 0, d["bulk_atp"] / cm, np.nan)
        adp_high_conc = np.where(cm > 0, d["bound_adp_high"] / cm, np.nan)
        adp_free_conc = np.where(cm > 0, d["bulk_adp"] / cm, np.nan)
        total_bound = (d["bound_atp_high"] + d["bound_atp_low"]
                       + d["bound_adp_high"])
        total_bound_conc = np.where(cm > 0, total_bound / cm, np.nan)

    # Apply rolling-mean smoothing to remove per-tick stochastic noise. The
    # window is in TICKS (1 tick = 1 second), so 60 ≈ 1-minute smoothing.
    atp_high_s = _rolling_mean(atp_high_conc, smooth_window)
    atp_low_s = _rolling_mean(atp_low_conc, smooth_window)
    atp_free_s = _rolling_mean(atp_free_conc, smooth_window)
    adp_high_s = _rolling_mean(adp_high_conc, smooth_window)
    adp_free_s = _rolling_mean(adp_free_conc, smooth_window)
    total_bound_s = _rolling_mean(total_bound_conc, smooth_window)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(
        f"{title_extra}\n"
        "dnaa-3 Phase 2 — DnaA-box equilibrium occupancy "
        "(K_d_high=1nM, K_d_low=100nM)",
        fontsize=11, y=0.995,
    )

    def vlines(ax):
        for b in boundaries_min:
            ax.axvline(b, color="#94a3b8", lw=1.0, ls="--", alpha=0.6, zorder=0)

    # Panel 1: DnaA-ATP partition (smoothed trend only)
    ax = axes[0]
    ax.plot(t_min, atp_high_s, color="#16a34a", lw=1.6,
            label="bound to high-aff boxes (chrom+oriC+promoter)")
    ax.plot(t_min, atp_low_s, color="#0ea5e9", lw=1.4,
            label="bound to low-aff boxes (oriC_low)")
    ax.plot(t_min, atp_free_s, color="#f97316", lw=1.4,
            label="free cytoplasmic")
    vlines(ax)
    ax.set_ylabel("DnaA-ATP partition\n(count / cell_mass)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Panel 2: DnaA-ADP partition (smoothed trend only)
    ax = axes[1]
    ax.plot(t_min, adp_high_s, color="#dc2626", lw=1.6,
            label="bound to high-aff boxes")
    ax.plot(t_min, adp_free_s, color="#2563eb", lw=1.4,
            label="free cytoplasmic")
    vlines(ax)
    ax.set_ylabel("DnaA-ADP partition\n(count / cell_mass)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Panel 3: Total DnaA boxes (raw count)
    ax = axes[2]
    ax.plot(t_min, d["total_boxes"], color="#7c3aed", lw=1.4)
    vlines(ax)
    ax.axhline(315, color="#a3a3a3", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(630, color="#a3a3a3", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Total DnaA boxes\n(count)")
    # Annotate the reference lines.
    if t_min.size:
        ax.text(t_min[-1], 315, "  315 (gen birth)", fontsize=7,
                color="#525252", va="center")
        ax.text(t_min[-1], 630, "  630 (post replication)", fontsize=7,
                color="#525252", va="center")

    # Panel 4: Total DnaA bound concentration (smoothed trend only)
    ax = axes[3]
    ax.plot(t_min, total_bound_s, color="#0f172a", lw=1.6,
            label="all bound / cell_mass")
    vlines(ax)
    ax.set_ylabel("Total DnaA bound\n(count / cell_mass)")
    ax.set_xlabel("Time (min)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes:
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True,
                    help="e.g. out/dnaa3_phase2_seed1_parquet/dnaa3_phase2_seed1")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--smooth-window", type=int, default=60,
                    help="Rolling-mean window in seconds (default 60 = 1 min). "
                         "Set to 1 to disable smoothing.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load_lineage(args.exp_root, args.exp_id, args.lineage_seed, args.gens)
    if d["t"].size == 0:
        print("No data loaded — aborting.")
        return
    plot(d, Path(args.out), args.title_extra, smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()
