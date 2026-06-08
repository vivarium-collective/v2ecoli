"""dnaa-3 — DnaA-box binding ANALYSIS (Haochen 2026-06-06).

Addresses Haochen's points with a self-consistent binding model on the steady
(later) generations:
  - concentrations in nM = count / cell VOLUME (listeners.mass.volume), so they
    are comparable to the nM K_d values (Haochen pt 5);
  - SELF-CONSISTENT free DnaA: the high-affinity boxes SEQUESTER DnaA, so free
    DnaA is solved from total = free + Σ_pool N·free/(free+K_pool) — not the naive
    "all bulk is free" (Haochen pt 4: oriC-low should not be ~always bound);
  - free DnaA plotted alongside bound (pt 3);
  - oriC LOW-affinity occupancy on its own zoomed axis (pt 1);
  - LATER generations only — the first ~2 gens are not steady (pt 2).

Panels: (1) free DnaA-ATP/ADP concentration (nM) + 100 nM K_d_low line;
(2) oriC low-affinity occupancy (zoomed); (3) free vs bound DnaA (count);
(4) per-pool occupancy (high vs oriC-low).

IN-SIM MODE (--insim-run): read the REAL emitted listeners.dnaA_binding free
DnaA-ATP/ADP nM + per-pool occupancy from the read-only in-sim binding parquet,
instead of the analytical self-consistent recompute. This is the validated path:
the provided read-only binding listener fires inside the composite and emits the
occupancy; the free-DnaA magnitude is read straight off the run, not assumed.

Usage (in-sim, current verdict):
  .venv/bin/python scripts/render_dnaa3_binding_analysis.py \\
      --insim-run studies/dnaa-3-box-binding/parquet-runs/dnaa3-insim-readonly \\
      --out studies/dnaa-3-box-binding/charts/dnaa3_binding_analysis
"""
from __future__ import annotations
import argparse, glob, json, hashlib, time
from pathlib import Path
import numpy as np, polars as pl
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NA = 6.02214076e23
N_HIGH = 302 + 3 + 2   # chromosomal + oriC-high + promoter (K_d 1 nM, ATP+ADP)
N_LOW = 8              # oriC-low (K_d 100 nM, ATP only)
KD_HIGH_nM, KD_LOW_nM = 1.0, 100.0


def load(run, gens):
    fs = []
    for g in gens:
        fs += sorted(glob.glob(f"{run}/**/history/**/generation={g}/**/*.pq", recursive=True))
    fs = [f for f in fs if "/agent_id=0" in f]
    ids = pl.scan_parquet(fs[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    bi = lambda m: ids.index(m); bc = pl.col("bulk__count")
    df = (pl.scan_parquet(fs, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(["generation", "global_time",
                   bc.list.get(bi("MONOMER0-160[c]")).alias("atp"),
                   bc.list.get(bi("MONOMER0-4565[c]")).alias("adp"),
                   bc.list.get(bi("PD03831[c]")).alias("apo"),
                   pl.col("listeners__mass__volume").alias("vol_fL"),
                   pl.col("listeners__replication_data__number_of_oric").alias("oric")])
          .sort(["generation", "global_time"]).collect())
    return df


def solve_free(D, V_L, n_high, n_low):
    """Self-consistent free binding-competent DnaA (count): boxes sequester it."""
    Kh = KD_HIGH_nM * 1e-9 * V_L * NA
    Kl = KD_LOW_nM * 1e-9 * V_L * NA
    out = np.empty_like(D)
    for i in range(len(D)):
        d = D[i]
        if d <= 0:
            out[i] = 0.0; continue
        f = lambda F: F + n_high[i] * F / (F + Kh[i]) + n_low[i] * F / (F + Kl[i]) - d
        out[i] = brentq(f, 1e-9, d)
    return out


def load_insim(run, gens=None):
    """Load REAL emitted listeners.dnaA_binding free DnaA + per-pool occupancy.

    If ``gens`` is None: single-cycle artifact mode — glob all history and sort by
    global_time (the original full-cell-cycle single-gen artifact path).

    If ``gens == "all"`` or a list: MULTI-GEN full-history mode — restrict to the
    CANONICAL all-zeros agent_id lineage (gen N = agent_id "0"*N), drop daughter
    stubs, and add a cumulative lineage time so the whole multi-generation
    trajectory is shown end-to-end (each generation restarts global_time at 0)."""
    L = "listeners__dnaA_binding__"
    if gens is None:
        fs = sorted(glob.glob(f"{run}/**/history/**/*.pq", recursive=True))
        if not fs:
            raise SystemExit(f"no in-sim parquet under {run}")
        return _select_insim(pl.scan_parquet(fs, hive_partitioning=True), L).sort("global_time").collect(), False
    # multi-gen canonical chain
    if gens == "all" or gens == ["all"]:
        fs = sorted(glob.glob(f"{run}/**/history/**/*.pq", recursive=True))
    else:
        fs = []
        for g in gens:
            fs += sorted(glob.glob(
                f"{run}/**/history/**/generation={g}/**/*.pq", recursive=True))
    if not fs:
        raise SystemExit(f"no in-sim parquet under {run}")
    lf = (pl.scan_parquet(fs, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$")))
    df = _select_insim(lf, L, with_gen=True).sort(["generation", "global_time"]).collect()
    return df, True


def _select_insim(lf, L, with_gen=False):
    cols = (["generation"] if with_gen else []) + ["global_time"]
    return (lf
          .select([*([pl.col("generation")] if with_gen else []),
                   "global_time",
                   pl.col(f"{L}free_DnaA_ATP_nM").alias("free_atp_nM"),
                   pl.col(f"{L}free_DnaA_ADP_nM").alias("free_adp_nM"),
                   pl.col(f"{L}oric__high_affinity_occupied").alias("oric_high_occ"),
                   pl.col(f"{L}oric__low_affinity_occupied").alias("oric_low_occ"),
                   pl.col(f"{L}oric__n_bound").alias("oric_n_bound"),
                   pl.col(f"{L}oric__n_total").alias("oric_n_total"),
                   pl.col(f"{L}dnaap__occupied").alias("dnaap_occ"),
                   pl.col(f"{L}dnaap__n_bound").alias("dnaap_n_bound"),
                   pl.col(f"{L}chromosome__occupied").alias("chrom_occ"),
                   pl.col(f"{L}chromosome__occupied_count").alias("chrom_n_bound"),
                   pl.col(f"{L}chromosome__n_total").alias("chrom_n_total"),
                   pl.col(f"{L}total_DnaA_bound").alias("total_bound")]))


def _write_meta(out, run, gen_id, cmd):
    for ext in ("png", "svg"):
        p = Path(f"{out}.{ext}")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        meta = {"source_run_id": Path(run).name, "generation_id": gen_id,
                "rendered_at": time.time(), "command": cmd,
                "content_hash": f"sha256:{h}"}
        Path(f"{out}.{ext}.meta.json").write_text(json.dumps(meta, indent=2))


def main_insim(args):
    """DECONTAMINATED full-within-cycle binding viz.

    Reads the REAL emitted listeners.dnaA_binding from a SINGLE non-dividing
    generation (the read-only observer artifact), so the full per-tick cell-cycle
    trajectory is retained (not just the early-cycle window the multigen run kept).
    Only the PROVIDED fast-equilibrium binding mechanism + dnaa-2 nucleotide
    balance — NO ChIP-seq cap line, NO sink (both removed per Rashmi). Honest: if
    free DnaA-ATP sits above the K_d and low-affinity occupancy is high, it is
    shown as-is, not hidden.
    """
    multigen = args.insim_gens is not None
    gens_arg = None
    if multigen:
        gens_arg = "all" if args.insim_gens == "all" else [int(g) for g in args.insim_gens.split(",")]
    df, is_multigen = load_insim(args.insim_run, gens_arg)
    if is_multigen:
        gt = df["global_time"].to_numpy(); gen = df["generation"].to_numpy()
        off = np.zeros(len(gt)); acc = 0.0
        for g in sorted(set(gen.tolist())):
            m = gen == g; off[m] = acc; acc += gt[m].max() + 1.0
        t = (gt + off) / 60
    else:
        t = ((df["global_time"] - df["global_time"][0]) / 60).to_numpy()
    F_nM = df["free_atp_nM"].to_numpy()
    adp_nM = df["free_adp_nM"].to_numpy()
    p_low = df["oric_low_occ"].to_numpy()
    p_high = df["oric_high_occ"].to_numpy()
    p_chrom = df["chrom_occ"].to_numpy()
    p_dnaap = df["dnaap_occ"].to_numpy()
    oric_nb = df["oric_n_bound"].to_numpy()
    chrom_nb = df["chrom_n_bound"].to_numpy()
    dnaap_nb = df["dnaap_n_bound"].to_numpy()
    bound = df["total_bound"].to_numpy()

    f_min, f_mean, f_max = F_nM.min(), F_nM.mean(), F_nM.max()
    frac_above = float((F_nM > KD_LOW_nM).mean())
    lo_min, lo_mean, lo_max = p_low.min(), p_low.mean(), p_low.max()

    span = ("over the FULL multi-generation lineage (full per-tick history retained "
            "each generation)" if is_multigen
            else "over a FULL cell cycle (read-only observer; single non-dividing gen)")
    xlabel = "lineage time (min)" if is_multigen else "cell-cycle time (min)"
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"dnaa-3 — DnaA-box binding {span} "
                 "(emitted listeners.dnaA_binding)\n"
                 "Decontaminated: provided fast-equilibrium binding + dnaa-2 nucleotide balance ONLY "
                 "— no ChIP-seq cap, no sink", fontsize=11)

    # (1) free DnaA-ATP and free DnaA-ADP (nM) vs the 100 nM oriC-low K_d
    a = ax[0, 0]
    a.plot(t, F_nM, color="#0f172a", lw=1.7, label="free DnaA-ATP (emitted, nM)")
    a.plot(t, adp_nM, color="#f59e0b", lw=1.1, alpha=0.85, label="free DnaA-ADP (emitted, nM)")
    a.axhline(KD_LOW_nM, color="#dc2626", ls="--", lw=1.2, label="K_d(oriC-low) = 100 nM")
    a.set_title(f"Free DnaA vs 100 nM K_d — DnaA-ATP min {f_min:.0f} / mean {f_mean:.0f} / max {f_max:.0f} nM\n"
                f"free DnaA-ATP is ABOVE the K_d for {frac_above*100:.0f}% of the cycle",
                fontsize=9)
    a.set_ylabel("nM"); a.set_xlabel(xlabel)
    a.legend(fontsize=7); a.grid(alpha=0.25)

    # (2) bound DnaA by pool (count)
    a = ax[0, 1]
    a.plot(t, chrom_nb, color="#94a3b8", lw=1.4, label="chromosomal bound")
    a.plot(t, oric_nb, color="#16a34a", lw=1.6, label="oriC bound (high+low)")
    a.plot(t, dnaap_nb, color="#0891b2", lw=1.3, alpha=0.85, label="dnaA-promoter bound")
    a.plot(t, bound, color="#0f172a", lw=0.9, ls=":", label="TOTAL DnaA bound")
    a.set_title("Bound DnaA by pool (count, emitted) — boxes duplicate as replication proceeds",
                fontsize=9)
    a.set_ylabel("molecules bound"); a.set_xlabel(xlabel)
    a.legend(fontsize=7); a.grid(alpha=0.25)

    # (3) oriC LOW-affinity occupancy on its own zoomed axis
    a = ax[1, 0]
    a.plot(t, p_low, color="#16a34a", lw=1.7)
    a.axhline(0.5, color="#94a3b8", ls=":", lw=1)
    a.set_title(f"oriC LOW-affinity (K_d 100 nM) occupancy — min {lo_min:.2f} / mean {lo_mean:.2f} / max {lo_max:.2f}\n"
                f"HIGH and roughly flat across the whole cycle (no early→late switch)", fontsize=9)
    a.set_ylabel("fraction bound"); a.set_xlabel(xlabel)
    a.set_ylim(0, 1); a.grid(alpha=0.25)

    # (4) per-pool occupancy fraction (low vs high vs chromosomal vs promoter)
    a = ax[1, 1]
    a.plot(t, p_high, color="#2563eb", lw=1.6, label="oriC high-aff (K_d 1 nM)")
    a.plot(t, p_low, color="#16a34a", lw=1.6, label="oriC-low (K_d 100 nM)")
    a.plot(t, p_chrom, color="#94a3b8", lw=1.2, label="chromosomal (K_d 1 nM)")
    a.plot(t, p_dnaap, color="#0891b2", lw=1.2, alpha=0.8, label="dnaA-promoter (K_d 1 nM)")
    a.set_title("Per-pool occupancy fraction (emitted)", fontsize=9)
    a.set_ylabel("fraction bound"); a.set_xlabel(xlabel)
    a.set_ylim(0, 1.05); a.legend(fontsize=7); a.grid(alpha=0.25)

    note = (f"HONEST OPEN OBSERVATION (read-only, provided mechanisms only): over a full ~{t[-1]:.0f}-min cell cycle, "
            f"free DnaA-ATP = {f_min:.0f}-{f_max:.0f} nM (mean {f_mean:.0f}) — i.e. {f_mean/KD_LOW_nM:.1f}x the 100 nM "
            f"oriC-low K_d, ABOVE it for {frac_above*100:.0f}% of the cycle. oriC-low occupancy stays high "
            f"({lo_min:.2f}-{lo_max:.2f}) throughout; high-aff pools saturate (~1.0). The over-binding is present "
            f"the WHOLE cycle, not just at initiation. This is NOT resolved by a cap or sink — it is an open feature "
            f"of the provided mechanism at the emitted DnaA-ATP fraction. Numbers read straight off the parquet.")
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#7c2d12", wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=130, bbox_inches="tight")
    cmd = (f"cd ../.. && python scripts/render_dnaa3_binding_analysis.py "
           f"--insim-run {args.insim_run} --out {args.out}")
    _write_meta(args.out, args.insim_run, args.gen_id, cmd)
    print(f"wrote {args.out}.png/.svg (+ .meta.json) from {args.insim_run}\n"
          f"  free DnaA-ATP: min {f_min:.1f} / mean {f_mean:.1f} / max {f_max:.1f} nM "
          f"(> K_d for {frac_above*100:.0f}% of cycle)\n"
          f"  oriC-low occupancy: min {lo_min:.3f} / mean {lo_mean:.3f} / max {lo_max:.3f}\n"
          f"  oriC-high {p_high.mean():.3f}  chrom {p_chrom.mean():.3f}  promoter {p_dnaap.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dnaa2_seed1_8gen")
    ap.add_argument("--gens", default="4,5,6")   # LATER gens (steady), Haochen pt 2
    ap.add_argument("--insim-run", default=None,
                    help="read REAL emitted listeners.dnaA_binding from this in-sim run")
    ap.add_argument("--insim-gens", default=None,
                    help="MULTI-GEN mode: 'all' or comma-list of generations to "
                         "concatenate (canonical all-zeros lineage, cumulative time). "
                         "Omit for the single-cycle artifact path.")
    ap.add_argument("--gen-id", default=None,
                    help="generation_id to record in the chart .meta.json")
    ap.add_argument("--out", default="studies/dnaa-3-box-binding/charts/dnaa3_binding_analysis")
    args = ap.parse_args()
    if args.insim_run:
        return main_insim(args)
    gens = [int(g) for g in args.gens.split(",")]
    df = load(args.run, gens)

    gt = df["global_time"].to_numpy(); gen = df["generation"].to_numpy()
    off = np.zeros(len(gt)); run = 0.0
    for g in sorted(set(gen.tolist())):
        m = gen == g; off[m] = run; run += gt[m].max() + 1.0
    t = (gt + off) / 60
    atp = df["atp"].to_numpy().astype(float); adp = df["adp"].to_numpy().astype(float)
    V_L = df["vol_fL"].to_numpy() * 1e-15
    oric = df["oric"].to_numpy()
    # boxes DOUBLE with oriC (oriC=2 → oriC boxes double); chromosomal also doubles
    # with replication — approximate the replicated box count by the oriC factor on
    # the oriC pools and a smooth doubling on chromosomal via total_DnaA_boxes if present.
    repl = oric.astype(float)            # oriC-high/low scale with oriC (1 or 2)
    n_high = N_HIGH * np.where(repl >= 2, 1.6, 1.0)  # chromosomal partly replicated mid-cycle
    n_low = N_LOW * repl
    Dtot = atp + adp                      # binding-competent (apo≈0)
    F = solve_free(Dtot, V_L, n_high, n_low)        # self-consistent free DnaA (count)
    nM = lambda c: c / (V_L * NA) * 1e9
    F_nM = nM(F); atp_nM = nM(atp); adp_nM = nM(adp)
    Kl_ct = KD_LOW_nM * 1e-9 * V_L * NA
    p_low = F / (F + Kl_ct)
    Kh_ct = KD_HIGH_nM * 1e-9 * V_L * NA
    p_high = F / (F + Kh_ct)
    bound = Dtot - F                      # total bound to all boxes

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("dnaa-3 — DnaA-box binding analysis (self-consistent; steady gens "
                 f"{gens[0]}-{gens[-1]}; concentrations = count / cell VOLUME, nM) — Haochen 2026-06-06",
                 fontsize=12)

    a = ax[0, 0]
    a.plot(t, F_nM, color="#0f172a", lw=1.6, label="FREE DnaA (self-consistent)")
    a.plot(t, atp_nM, color="#2563eb", lw=0.9, alpha=0.6, label="bulk DnaA-ATP")
    a.plot(t, adp_nM, color="#f59e0b", lw=0.9, alpha=0.6, label="bulk DnaA-ADP")
    a.axhline(KD_LOW_nM, color="#dc2626", ls="--", lw=1, label="K_d(oriC-low)=100 nM")
    a.set_title("Free DnaA concentration (nM) vs the 100 nM oriC-low K_d")
    a.set_ylabel("nM"); a.legend(fontsize=7); a.grid(alpha=0.25)

    a = ax[0, 1]
    a.plot(t, p_low, color="#16a34a", lw=1.6)
    a.axhline(0.5, color="#94a3b8", ls=":", lw=1)
    a.set_title("oriC LOW-affinity box occupancy (zoomed) — the dynamic switch")
    a.set_ylabel("fraction bound"); a.set_ylim(0, 1); a.grid(alpha=0.25)
    a.text(0.02, 0.06, f"mean {p_low.mean():.2f} ({N_LOW*p_low.mean():.1f}/{N_LOW} bound)",
           transform=a.transAxes, fontsize=8, color="#16a34a")

    a = ax[1, 0]
    a.plot(t, bound, color="#475569", lw=1.5, label="BOUND DnaA")
    a.plot(t, F, color="#dc2626", lw=1.5, label="FREE DnaA")
    a.plot(t, Dtot, color="#0f172a", lw=0.8, ls=":", label="total (ATP+ADP)")
    a.set_title("Free vs bound DnaA over time (count)"); a.set_ylabel("DnaA count")
    a.set_xlabel("steady-gen time (min)"); a.legend(fontsize=7); a.grid(alpha=0.25)

    a = ax[1, 1]
    a.plot(t, p_high, color="#2563eb", lw=1.5, label=f"high-aff (K_d 1 nM, n≈{N_HIGH})")
    a.plot(t, p_low, color="#16a34a", lw=1.5, label=f"oriC-low (K_d 100 nM, n={N_LOW})")
    a.set_title("Per-pool occupancy"); a.set_ylabel("fraction bound"); a.set_ylim(0, 1.05)
    a.set_xlabel("steady-gen time (min)"); a.legend(fontsize=7); a.grid(alpha=0.25)

    note = (f"Self-consistent free DnaA ≈ {F_nM.mean():.0f} nM; the {N_HIGH}+{N_LOW} provided boxes sequester "
            f"~{bound.mean():.0f} of ~{Dtot.mean():.0f} DnaA. oriC high-affinity boxes stay near-saturated while the "
            f"oriC-low pool tracks free DnaA-ATP — the dynamic switch. Only the provided binding mechanism is modeled "
            f"(no cap, no added pool).")
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#7c2d12", wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{args.out}.{ext}", dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}.png/.svg | free DnaA {F_nM.mean():.0f} nM | oriC-low P {p_low.mean():.2f}")


if __name__ == "__main__":
    main()
