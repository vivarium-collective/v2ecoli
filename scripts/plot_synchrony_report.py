"""Generate a multi-page PDF report of the mother-cycle Δt synchrony analysis.

Reads out/synchrony_analysis/lineage_seed*.json and out/synchrony_analysis/synchrony_summary.json.

Pages:
  1. Δt histogram (full + zoomed) with summary stats box.
  2. Per-seed breakdown table + strip plot of all pairs by seed.
  3. Δt vs cell mass at initiation (context for outliers).
"""
import argparse
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_data(analysis_dir):
    summary = json.load(open(f"{analysis_dir}/synchrony_summary.json"))
    lineages = [json.load(open(p)) for p in sorted(glob.glob(f"{analysis_dir}/lineage_seed*.json"))]
    return summary, lineages


def compute_stats(dts, tau_mean_s):
    n = len(dts)
    if n < 2:
        return dict(n=n, mean=float("nan"), std=float("nan"), median=float("nan"), min_=float("nan"), max_=float("nan"), cv=float("nan"))
    return dict(
        n=n,
        mean=float(dts.mean()),
        std=float(dts.std(ddof=1)),
        median=float(np.median(dts)),
        min_=float(dts.min()),
        max_=float(dts.max()),
        cv=float(dts.std(ddof=1) / tau_mean_s),
    )


def page_histogram(pdf, summary):
    # Strict filter: pair must come from a gen born at 2 oriC where the 2nd
    # mother-cycle event brought oriC to 4 (i.e., 2->4 direct or 2->3->4).
    strict_pairs = [p for p in summary["pairs"]
                    if p.get("birth_oric") == 2 and p.get("e2_after", 0) >= 4]
    dts = np.array([p["delta_t_s"] for p in strict_pairs])
    tau_mean_min = summary["tau_mean_min"]
    tau_mean_s = tau_mean_min * 60
    # Also compute mean τ using only strict mother-cycle gens
    tau_mother_min = float(np.mean([p["tau_min"] for p in strict_pairs]))
    tau_mother_s = tau_mother_min * 60
    OUTLIER_THRESH_S = 600  # 10 min — matches Haochen imaging window

    all_stats = compute_stats(dts, tau_mean_s)
    all_stats_mother = compute_stats(dts, tau_mother_s)
    nonout = dts[dts <= OUTLIER_THRESH_S]
    nonout_stats = compute_stats(nonout, tau_mean_s)
    nonout_stats_mother = compute_stats(nonout, tau_mother_s)
    n_outliers = int((dts > OUTLIER_THRESH_S).sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 10))
    fig.suptitle("Mother-cycle initiation synchrony (Δt between the two initiations in an overlapping cell cycle)",
                 fontsize=12, y=0.985)
    if summary.get("_config_note"):
        fig.text(0.5, 0.958, summary["_config_note"], ha="center", va="top",
                 fontsize=9, style="italic", color="#0369a1")

    # Top: full histogram with outliers shown, log x scale so outliers visible
    dts_plot = np.where(dts == 0, 0.5, dts)  # nudge zeros for log axis
    bins = np.logspace(np.log10(0.5), np.log10(max(dts.max(), 10) * 1.5), 40)
    ax1.hist(dts_plot, bins=bins, color="#334155", edgecolor="white")
    ax1.axvline(OUTLIER_THRESH_S, ls="--", color="#dc2626", lw=1, alpha=0.7,
                label=f"outlier threshold = {OUTLIER_THRESH_S} s")
    ax1.set_xscale("log")
    ax1.set_xlabel("Δt (s, log scale)  — Δt = 0 shown at 0.5 s")
    ax1.set_ylabel("count")
    ax1.set_title(f"Full distribution (n={all_stats['n']}), log scale")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    # Bottom: zoomed histogram, linear, non-outliers only
    if len(nonout) > 0:
        bins2 = np.linspace(0, min(OUTLIER_THRESH_S, nonout.max() + 5), 40)
        ax2.hist(nonout, bins=bins2, color="#0369a1", edgecolor="white")
        ax2.set_xlabel("Δt (s)")
        ax2.set_ylabel("count")
        ax2.set_title(f"Zoomed: pairs with Δt ≤ {OUTLIER_THRESH_S} s (n={nonout_stats['n']}, {n_outliers} outliers excluded)")
        ax2.grid(alpha=0.3)

    fig.subplots_adjust(top=0.92, hspace=0.32, bottom=0.36)

    # Cell-size at initiation stats from every gen with a first-mother init
    init_masses = [p.get("first_mass") for p in summary["pairs"] if p.get("first_mass") is not None]
    init_vols   = [p.get("first_vol")  for p in summary["pairs"] if p.get("first_vol")  is not None]
    mass_line = ""
    if init_masses:
        m = np.array(init_masses)
        mass_line = f"initiation mass:   mean = {m.mean():.0f} fg   std = {m.std(ddof=1):.0f}   n = {len(m)}\n"
    vol_line = ""
    if init_vols:
        v = np.array(init_vols)
        vol_line = f"initiation volume: mean = {v.mean():.3f} fL  std = {v.std(ddof=1):.3f}  n = {len(v)}\n"

    footer = (
        f"n seeds = {summary['n_seeds']}   |   n generations (all) = {summary['n_gens_total']}   |   n mother-cycle gens (produce pair) = {all_stats['n']}\n"
        f"mean τ (all gens) = {tau_mean_min:.1f} min   |   mean τ (mother-cycle gens only) = {tau_mother_min:.1f} min\n"
        f"{mass_line}{vol_line}"
        f"\n"
        f"ALL {all_stats['n']} pairs:\n"
        f"   Δt mean = {all_stats['mean']:.1f} s    std = {all_stats['std']:.1f} s    median = {all_stats['median']:.1f} s    range [{all_stats['min_']:.0f}, {all_stats['max_']:.0f}] s\n"
        f"   intrinsic CV using population mean τ ({tau_mean_min:.1f} min)     = {all_stats['cv']:.4f}\n"
        f"   intrinsic CV using mother-cycle mean τ ({tau_mother_min:.1f} min) = {all_stats_mother['cv']:.4f}\n"
        f"\n"
        f"WITHOUT {n_outliers} outliers (Δt > {OUTLIER_THRESH_S} s):\n"
        f"   n = {nonout_stats['n']}    mean = {nonout_stats['mean']:.1f} s    std = {nonout_stats['std']:.1f} s\n"
        f"   intrinsic CV using population mean τ     = {nonout_stats['cv']:.4f}\n"
        f"   intrinsic CV using mother-cycle mean τ   = {nonout_stats_mother['cv']:.4f}"
    )
    # box the footer for prominence
    fig.text(0.5, 0.02, footer, family="monospace", fontsize=10, va="bottom", ha="center",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8))

    pdf.savefig(fig)
    plt.close(fig)


def page_per_seed_table(pdf, summary, lineages):
    OUTLIER_THRESH_S = 600
    fig, ax_tbl = plt.subplots(1, 1, figsize=(8.5, 11))
    fig.suptitle("Per-seed summary", fontsize=13, y=0.98)
    if summary.get("_config_note"):
        fig.text(0.5, 0.955, summary["_config_note"], ha="center", va="top",
                 fontsize=9, style="italic", color="#0369a1")
    fig.text(0.5, 0.905,
             "n pairs = mother-cycle Δt pairs (cell born with 2 oriC, cycle goes 2→4 direct or 2→3→4 sequential).",
             ha="center", fontsize=9, style="italic", color="#334155")

    per_seed = {}
    for lin in lineages:
        s = lin["seed"]
        # STRICT filter: pairs from gens born at 2 oriC that reach 4 oriC
        # (2→4 direct or 2→3→4 sequential). Matches page 2 and page 3.
        pairs = [p for p in summary["pairs"]
                 if p["seed"] == s and p.get("birth_oric") == 2 and p.get("e2_after", 0) >= 4]
        dts = np.array([p["delta_t_s"] for p in pairs]) if pairs else np.array([])
        taus = [g["tau_min"] for g in lin["generations"]]
        per_seed[s] = {
            "n_gens": sum(1 for g in lin["generations"] if g.get("tau_min", 0) >= 5),
            "n_pairs": len(pairs),
            "n_outliers": int((dts > OUTLIER_THRESH_S).sum()) if len(dts) else 0,
            "tau_mean": float(np.mean(taus)) if taus else float("nan"),
            "dt_mean": float(dts.mean()) if len(dts) else float("nan"),
            "dt_max": float(dts.max()) if len(dts) else float("nan"),
        }

    ax_tbl.axis("off")
    header = ["seed", "gens", "τ mean\n(min)", "n pairs", "Δt mean\n(s)", "Δt max\n(s)", "outliers\n(Δt>10min)"]
    cell_text = []
    for s, r in sorted(per_seed.items()):
        cell_text.append([
            str(s), str(r["n_gens"]),
            f"{r['tau_mean']:.1f}", str(r["n_pairs"]),
            f"{r['dt_mean']:.1f}" if not np.isnan(r["dt_mean"]) else "-",
            f"{r['dt_max']:.0f}" if not np.isnan(r["dt_max"]) else "-",
            str(r["n_outliers"]),
        ])
    total_gens = sum(r["n_gens"] for r in per_seed.values())
    total_pairs = sum(r["n_pairs"] for r in per_seed.values())
    total_outs = sum(r["n_outliers"] for r in per_seed.values())
    cell_text.append(["TOTAL", str(total_gens), f"{summary['tau_mean_min']:.1f}", str(total_pairs), "-", "-", str(total_outs)])
    tbl = ax_tbl.table(cellText=cell_text, colLabels=header, loc="upper center",
                       cellLoc="center", colWidths=[0.09, 0.09, 0.13, 0.11, 0.13, 0.11, 0.15])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for j in range(len(header)):
        tbl[(0, j)].set_facecolor("#e2e8f0")
        tbl[(0, j)].set_text_props(weight="bold")
    for j in range(len(header)):
        tbl[(len(cell_text), j)].set_facecolor("#fef3c7")
        tbl[(len(cell_text), j)].set_text_props(weight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def page_mother_cycle_gens(pdf, summary, lineages):
    """Table of every gen that showed overlapping rounds of replication (>= 2 mother-cycle inits)."""
    fig, ax_tbl = plt.subplots(1, 1, figsize=(8.5, 11))
    fig.suptitle("Generations with overlapping rounds of replication (mother-cycle gens)",
                 fontsize=12, y=0.97)
    fig.text(0.5, 0.935,
             "Cell born with 2 oriC that fires the next initiation round in the same cycle "
             "(oriC transitions 2→4 direct or 2→3→4 sequential).",
             ha="center", fontsize=9, style="italic", color="#334155")

    OUTLIER_THRESH_S = 600
    # Restrict to gens born at 2 oriC that reach 4 oriC (2->4 or 2->3->4).
    per_seed = {}
    for lin in lineages:
        s = lin["seed"]
        per_seed[s] = {"n_gens": 0, "taus": [], "dts": []}
        for g in lin["generations"]:
            if g["birth_oric"] != 2:
                continue
            me = [e for e in g["initiations"] if e["oric_before"] >= 2]
            if len(me) < 2 or me[1]["oric_after"] < 4:
                continue
            per_seed[s]["n_gens"] += 1
            per_seed[s]["taus"].append(g["tau_min"])
            per_seed[s]["dts"].append(me[1]["time_s_from_birth"] - me[0]["time_s_from_birth"])

    ax_tbl.axis("off")
    header = ["seed", "mother-cycle\ngens", "τ mean\n(min)", "Δt mean\n(s)", "Δt max\n(s)", "outliers\n(Δt>10min)"]
    cell_text = []
    for s in sorted(per_seed.keys()):
        r = per_seed[s]
        if r["n_gens"] == 0:
            cell_text.append([str(s), "0", "-", "-", "-", "0"])
            continue
        taus, dts = np.array(r["taus"]), np.array(r["dts"])
        cell_text.append([
            str(s), str(r["n_gens"]),
            f"{taus.mean():.1f}",
            f"{dts.mean():.1f}",
            f"{dts.max():.0f}",
            str(int((dts > OUTLIER_THRESH_S).sum())),
        ])
    all_taus = np.array([t for r in per_seed.values() for t in r["taus"]])
    all_dts = np.array([d for r in per_seed.values() for d in r["dts"]])
    total_gens = sum(r["n_gens"] for r in per_seed.values())
    total_outs = int((all_dts > OUTLIER_THRESH_S).sum())
    cell_text.append([
        "TOTAL", str(total_gens),
        f"{all_taus.mean():.1f}",
        f"{all_dts.mean():.1f}",
        f"{all_dts.max():.0f}",
        str(total_outs),
    ])

    tbl = ax_tbl.table(cellText=cell_text, colLabels=header, loc="upper center",
                       cellLoc="center", colWidths=[0.09, 0.16, 0.13, 0.13, 0.11, 0.15])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for j in range(len(header)):
        tbl[(0, j)].set_facecolor("#e2e8f0")
        tbl[(0, j)].set_text_props(weight="bold")
    for j in range(len(header)):
        tbl[(len(cell_text), j)].set_facecolor("#fef3c7")
        tbl[(len(cell_text), j)].set_text_props(weight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def page_strip(pdf, summary, lineages):
    OUTLIER_THRESH_S = 600
    fig, ax_strip = plt.subplots(1, 1, figsize=(8.5, 6))
    fig.suptitle("Δt per mother-cycle pair, per seed", fontsize=12, y=0.98)

    seeds_sorted = sorted({lin["seed"] for lin in lineages})
    for i, s in enumerate(seeds_sorted):
        pairs = [p for p in summary["pairs"] if p["seed"] == s]
        if not pairs: continue
        dts = np.array([p["delta_t_s"] for p in pairs])
        dts_plot = np.where(dts == 0, 0.5, dts)
        x = np.full_like(dts_plot, i, dtype=float) + np.random.RandomState(s).normal(0, 0.06, len(dts_plot))
        colors = ["#dc2626" if d > OUTLIER_THRESH_S else "#0369a1" for d in dts]
        ax_strip.scatter(x, dts_plot, c=colors, s=32, alpha=0.75, edgecolors="white", linewidths=0.5)

    ax_strip.axhline(OUTLIER_THRESH_S, ls="--", color="#dc2626", lw=1, alpha=0.7,
                     label=f"outlier threshold = {OUTLIER_THRESH_S} s")
    ax_strip.set_yscale("log")
    ax_strip.set_ylim(0.3, max(1e4, max(p["delta_t_s"] for p in summary["pairs"]) * 1.5))
    ax_strip.set_xticks(range(len(seeds_sorted)))
    ax_strip.set_xticklabels([f"seed {s}" for s in seeds_sorted], rotation=45, ha="right", fontsize=9)
    ax_strip.set_ylabel("Δt (s, log scale)")
    ax_strip.grid(alpha=0.3, which="both", axis="y")
    ax_strip.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(top=0.92, bottom=0.13)
    pdf.savefig(fig)
    plt.close(fig)


def page_mass_vs_dt(pdf, summary):
    OUTLIER_THRESH_S = 600
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Cell mass at the two initiation events (context for outliers)", fontsize=12, y=0.98)

    pairs = summary["pairs"]
    m1 = np.array([p["first_mass"] for p in pairs])
    m2 = np.array([p["second_mass"] for p in pairs])
    dts = np.array([p["delta_t_s"] for p in pairs])
    is_out = dts > OUTLIER_THRESH_S

    # Left: mass at first vs mass at second, colored by outlier
    ax1.scatter(m1[~is_out], m2[~is_out], c="#0369a1", s=30, alpha=0.7, edgecolors="white", linewidths=0.5, label=f"in-range ({(~is_out).sum()})")
    ax1.scatter(m1[is_out], m2[is_out], c="#dc2626", s=60, alpha=0.9, edgecolors="white", linewidths=1.0, label=f"outlier Δt > {OUTLIER_THRESH_S}s ({is_out.sum()})")
    lo = min(m1.min(), m2.min()) * 0.9
    hi = max(m1.max(), m2.max()) * 1.05
    ax1.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=0.8, alpha=0.5, label="mass_1 = mass_2")
    ax1.set_xlabel("cell mass at 1st initiation (fg)")
    ax1.set_ylabel("cell mass at 2nd initiation (fg)")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: Δt vs first-init cell mass (do large-mass cells stall more?)
    dts_plot = np.where(dts == 0, 0.5, dts)
    ax2.scatter(m1[~is_out], dts_plot[~is_out], c="#0369a1", s=30, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax2.scatter(m1[is_out], dts_plot[is_out], c="#dc2626", s=60, alpha=0.9, edgecolors="white", linewidths=1.0)
    ax2.axhline(OUTLIER_THRESH_S, ls="--", color="#dc2626", lw=1, alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_ylim(0.3, dts_plot.max() * 1.5)
    ax2.set_xlabel("cell mass at 1st initiation (fg)")
    ax2.set_ylabel("Δt (s, log scale)")
    ax2.grid(alpha=0.3, which="both", axis="y")

    fig.subplots_adjust(top=0.9, wspace=0.28, bottom=0.13)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="out/synchrony_analysis")
    ap.add_argument("--out", default="out/synchrony_analysis/synchrony_report.pdf")
    ap.add_argument("--config-note", default="",
                    help="One-line config note printed under each page's title")
    args = ap.parse_args()

    summary, lineages = load_data(args.analysis_dir)
    summary["_config_note"] = args.config_note

    with PdfPages(args.out) as pdf:
        page_per_seed_table(pdf, summary, lineages)
        page_mother_cycle_gens(pdf, summary, lineages)
        page_histogram(pdf, summary)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
