"""Six-panel DnaA / oriC / cell-mass trajectory plot across a multigen lineage.

Mirrors the reference figure layout:
  1. Cell mass (fg)
  2. oriC count (step)
  3. DnaA forms (count): total, DnaA-ATP, DnaA-ADP, apo + PDF target band [300, 800]
  4. DnaA total / cell mass
  5. DnaA-form fractions: %ATP, %ADP, %apo + target band [0.2, 0.5] for %ATP
  6. bulk ATP / ADP small-molecule counts (log scale)

Usage:
  python scripts/plot_dnaa_traj.py \
    --exp-root out/dnaa2_seed0_v8e4_parquet/dnaa2_seed0_v8e4 \
    --exp-id dnaa2_seed0_v8e4 \
    --lineage-seed 0 \
    --gens 6 \
    --title-extra "V=8e-4, burned-in (gen5 dill), seed=0" \
    --out out/figures/dnaa2_seed0_v8e4_6gen_traj.png
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
DNAA_APO_ID = "PD03831[c]"
ATP_ID = "ATP[c]"
ADP_ID = "ADP[c]"

COLS = [
    "global_time",
    "bulk__id",
    "bulk__count",
    "listeners__mass__cell_mass",
    "listeners__replication_data__number_of_oric",
]

# Optional listener fields — present only on dnaa-3 Phase 2 runs that wire
# the box-binding listener. If missing the script falls back to bulk-only.
BOUND_FIELDS = [
    "listeners__replication_data__chromosomal_high_bound_atp",
    "listeners__replication_data__chromosomal_high_bound_adp",
    "listeners__replication_data__oriC_high_bound_atp",
    "listeners__replication_data__oriC_high_bound_adp",
    "listeners__replication_data__oriC_low_bound_atp",
    "listeners__replication_data__promoter_high_bound_atp",
    "listeners__replication_data__promoter_high_bound_adp",
]


def _lineage_agent(gen: int) -> str:
    return "0" * gen


def load_lineage(exp_root: str, exp_id: str, lineage_seed: int, n_gens: int) -> dict:
    times: list[np.ndarray] = []
    cell_mass: list[np.ndarray] = []
    n_oric: list[np.ndarray] = []
    oric_low_bound: list[np.ndarray] = []
    atp: list[np.ndarray] = []
    adp: list[np.ndarray] = []
    apo: list[np.ndarray] = []
    bulk_atp: list[np.ndarray] = []
    bulk_adp: list[np.ndarray] = []
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
        oc = []
        ol = []  # oriC_low_bound_atp count
        a_atp = []
        a_adp = []
        a_apo = []
        b_atp = []
        b_adp = []
        # Detect Phase 2 listener columns once per gen.
        schema = pq.read_schema(files[0])
        has_bound = all(c in schema.names for c in BOUND_FIELDS)
        cols_to_read = COLS + (BOUND_FIELDS if has_bound else [])
        for f in files:
            tbl = pq.read_table(f, columns=cols_to_read).to_pandas()
            for _, r in tbl.iterrows():
                ids = list(r["bulk__id"])
                cnts = r["bulk__count"]
                lookup = {n: int(cnts[i]) for i, n in enumerate(ids)}
                t_g.append(float(r["global_time"]))
                cm.append(float(r["listeners__mass__cell_mass"]))
                oc.append(int(r["listeners__replication_data__number_of_oric"]))
                ol.append(int(r["listeners__replication_data__oriC_low_bound_atp"])
                          if has_bound else 0)
                # Bulk free pool
                free_atp = lookup.get(DNAA_ATP_ID, 0)
                free_adp = lookup.get(DNAA_ADP_ID, 0)
                # Bound pool (Phase 2 only; otherwise 0)
                bnd_atp = 0
                bnd_adp = 0
                if has_bound:
                    bnd_atp = (
                        int(r["listeners__replication_data__chromosomal_high_bound_atp"]) +
                        int(r["listeners__replication_data__oriC_high_bound_atp"]) +
                        int(r["listeners__replication_data__oriC_low_bound_atp"]) +
                        int(r["listeners__replication_data__promoter_high_bound_atp"])
                    )
                    bnd_adp = (
                        int(r["listeners__replication_data__chromosomal_high_bound_adp"]) +
                        int(r["listeners__replication_data__oriC_high_bound_adp"]) +
                        int(r["listeners__replication_data__promoter_high_bound_adp"])
                    )
                # Total ATP/ADP across cell (free + bound)
                a_atp.append(free_atp + bnd_atp)
                a_adp.append(free_adp + bnd_adp)
                a_apo.append(lookup.get(DNAA_APO_ID, 0))
                b_atp.append(lookup.get(ATP_ID, 0))
                b_adp.append(lookup.get(ADP_ID, 0))
        t_g = np.asarray(t_g)
        order = np.argsort(t_g)
        t_g = t_g[order]
        cm = np.asarray(cm)[order]
        oc = np.asarray(oc)[order]
        ol = np.asarray(ol)[order]
        a_atp = np.asarray(a_atp)[order]
        a_adp = np.asarray(a_adp)[order]
        a_apo = np.asarray(a_apo)[order]
        b_atp = np.asarray(b_atp)[order]
        b_adp = np.asarray(b_adp)[order]

        dur = float(t_g[-1] - t_g[0])
        times.append(t_g + t_offset)
        cell_mass.append(cm)
        n_oric.append(oc)
        oric_low_bound.append(ol)
        atp.append(a_atp)
        adp.append(a_adp)
        apo.append(a_apo)
        bulk_atp.append(b_atp)
        bulk_adp.append(b_adp)
        durations.append(dur)
        t_offset = float(times[-1][-1])
        gen_boundaries.append(t_offset)
        print(f"  gen {gen}: tau={dur/60:.1f} min  ticks={len(t_g)}  "
              f"end DnaA(ATP+ADP+apo)={a_atp[-1]+a_adp[-1]+a_apo[-1]}  oriC={oc[-1]}")

    return {
        "t": np.concatenate(times) if times else np.array([]),
        "cell_mass": np.concatenate(cell_mass) if cell_mass else np.array([]),
        "n_oric": np.concatenate(n_oric) if n_oric else np.array([]),
        "oric_low_bound": np.concatenate(oric_low_bound) if oric_low_bound else np.array([]),
        "atp": np.concatenate(atp) if atp else np.array([]),
        "adp": np.concatenate(adp) if adp else np.array([]),
        "apo": np.concatenate(apo) if apo else np.array([]),
        "bulk_atp": np.concatenate(bulk_atp) if bulk_atp else np.array([]),
        "bulk_adp": np.concatenate(bulk_adp) if bulk_adp else np.array([]),
        "gen_boundaries": gen_boundaries[:-1] if gen_boundaries else [],
        "durations_min": [d / 60 for d in durations],
    }


def plot(d: dict, out_path: Path, title_extra: str) -> None:
    t_min = d["t"] / 60
    boundaries_min = [b / 60 for b in d["gen_boundaries"]]
    total = d["atp"] + d["adp"] + d["apo"]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_mass = np.where(d["cell_mass"] > 0, total / d["cell_mass"], np.nan)
        frac_atp = np.where(total > 0, d["atp"] / total, np.nan)
        frac_adp = np.where(total > 0, d["adp"] / total, np.nan)
        frac_apo = np.where(total > 0, d["apo"] / total, np.nan)

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"{title_extra}\n"
        "DnaA-ATP intrinsic hydrolysis wired into equilibrium (k = 0.046/min)\n"
        "K_d(ATP) = 29 nM (SS),  K_d(ADP) = 100 nM (kinetic)",
        fontsize=11, y=0.995,
    )

    def vlines(ax):
        for b in boundaries_min:
            ax.axvline(b, color="#94a3b8", lw=1.0, ls="--", alpha=0.6, zorder=0)

    # 1. Cell mass
    ax = axes[0]
    ax.plot(t_min, d["cell_mass"], color="#2563eb", lw=1.4)
    vlines(ax)
    ax.set_ylabel("Cell mass (fg)")

    # 2. oriC count
    ax = axes[1]
    ax.step(t_min, d["n_oric"], color="#7c3aed", lw=1.6, where="post")
    vlines(ax)
    ax.set_ylim(-0.3, max(int(d["n_oric"].max()) + 1, 4))
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("oriC count")

    # 3. DnaA forms
    ax = axes[2]
    ax.axhspan(300, 800, color="#94a3b8", alpha=0.18, zorder=0, label="PDF target [300, 800]")
    ax.plot(t_min, total, color="black", lw=1.4, label="total")
    ax.plot(t_min, d["atp"], color="#16a34a", lw=1.0, label="DnaA-ATP")
    ax.plot(t_min, d["adp"], color="#dc2626", lw=1.0, label="DnaA-ADP")
    ax.plot(t_min, d["apo"], color="#475569", lw=0.8, label="apo")
    vlines(ax)
    ax.set_ylabel("DnaA forms (count)")
    ax.legend(loc="upper left", fontsize=8, ncol=5, frameon=False)

    # 4. DnaA total / cell mass
    ax = axes[3]
    # Per-gen CV ±10% bands centred on each gen's own mean. Gen 1 is
    # transient (resume-dill warm-in) so we still draw a band but flag
    # it visually with a dashed mean line.
    gen_edges = [t_min[0]] + list(boundaries_min) + [t_min[-1]]
    for gi in range(len(gen_edges) - 1):
        t0, t1 = gen_edges[gi], gen_edges[gi + 1]
        mask = (t_min >= t0) & (t_min < t1)
        vals = per_mass[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        m = float(np.mean(vals))
        ax.fill_between([t0, t1], m * 0.9, m * 1.1,
                        color="#fbcfe8", alpha=0.45, zorder=0)
        ax.hlines(m, t0, t1, color="#9d174d", lw=0.8, ls="--",
                  alpha=0.75, zorder=1)
    ax.plot(t_min, per_mass, color="#ec4899", lw=1.4, zorder=2)
    vlines(ax)
    ax.set_ylabel("DnaA total /\ncell mass")
    ax.text(0.99, 0.02, "Per-gen CV ±10% band, dashed = gen mean",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#9d174d")

    # 5. DnaA-form fractions
    ax = axes[4]
    ax.axhspan(0.2, 0.5, color="#86efac", alpha=0.3, zorder=0, label="target [0.2, 0.5] for DnaA-ATP")
    ax.plot(t_min, frac_atp, color="#16a34a", lw=1.0, label="%DnaA-ATP")
    ax.plot(t_min, frac_adp, color="#dc2626", lw=1.0, label="%DnaA-ADP")
    ax.plot(t_min, frac_apo, color="#475569", lw=0.8, label="%apo")
    vlines(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("DnaA-form fraction")
    ax.legend(loc="center right", fontsize=8, frameon=False)

    # 6. bulk ATP / ADP (small molecules)
    ax = axes[5]
    ax.semilogy(t_min, d["bulk_atp"], color="#f97316", lw=1.2, label="bulk ATP")
    ax.semilogy(t_min, d["bulk_adp"], color="#2563eb", lw=1.2, label="bulk ADP")
    vlines(ax)
    ax.set_ylabel("bulk count")
    ax.set_xlabel("Time (min)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

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
                    help="e.g. out/dnaa2_seed0_v8e4_parquet/dnaa2_seed0_v8e4")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load_lineage(args.exp_root, args.exp_id, args.lineage_seed, args.gens)
    if d["t"].size == 0:
        print("No data loaded — aborting.")
        return
    plot(d, Path(args.out), args.title_extra)


if __name__ == "__main__":
    main()
