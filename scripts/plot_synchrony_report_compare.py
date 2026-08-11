"""Combined synchrony report comparing N conditions (e.g. baseline vs perturbations).

Reads N synchrony_summary.json files (one per condition) and generates a
single PDF with side-by-side Δt histograms, per-seed table, and cell-mass /
initiation-volume comparisons. Backwards compatible with the old two-
condition --summary-a / --summary-b CLI.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUTLIER_THRESH_S = 600  # 10 min — matches imaging window (overridable via --cutoff-s)

# Color palette cycled across columns (up to 6 conditions supported).
COLOR_CYCLE = ["#334155", "#dc2626", "#0369a1", "#b45309", "#059669", "#7c3aed"]


def load(summary_path):
    return json.load(open(summary_path))


def pair_stats(pairs):
    dts = np.array([p["delta_t_s"] for p in pairs])
    return {
        "n": len(dts),
        "mean": float(dts.mean()) if len(dts) else float("nan"),
        "std": float(dts.std(ddof=1)) if len(dts) > 1 else float("nan"),
        "median": float(np.median(dts)) if len(dts) else float("nan"),
        "min_": float(dts.min()) if len(dts) else float("nan"),
        "max_": float(dts.max()) if len(dts) else float("nan"),
    }


def strict(summary):
    return [p for p in summary["pairs"]
            if p.get("birth_oric") == 2 and p.get("e2_after", 0) >= 4]


def init_stats(pairs):
    masses = np.array([p["first_mass"] for p in pairs if p.get("first_mass") is not None])
    vols   = np.array([p["first_vol"]  for p in pairs if p.get("first_vol")  is not None])
    out = {}
    if len(masses):
        out["mass_mean"] = float(masses.mean()); out["mass_std"] = float(masses.std(ddof=1)); out["n_mass"] = len(masses)
    if len(vols):
        out["vol_mean"] = float(vols.mean());   out["vol_std"] = float(vols.std(ddof=1));   out["n_vol"] = len(vols)
    return out


def _pick_colors(n):
    return [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(n)]


def page_histogram(pdf, sums, labels, title_extra=""):
    n = len(sums)
    fig_w = max(11, 4 + 3.5 * n)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 10), squeeze=False)
    fig.suptitle(f"Mother-cycle initiation synchrony — condition comparison{title_extra}",
                 fontsize=13, y=0.98)

    colors = _pick_colors(n)

    # Shared bin edges so bar widths (and therefore probabilities) are comparable across conditions.
    all_dts = np.concatenate([np.array([p["delta_t_s"] for p in strict(s)]) for s in sums])
    log_bins = np.logspace(np.log10(0.5), np.log10(max(all_dts.max(), 10) * 1.5), 40)
    all_nonout = all_dts[all_dts <= OUTLIER_THRESH_S]
    zoom_edge = min(OUTLIER_THRESH_S, all_nonout.max() + 5) if len(all_nonout) else OUTLIER_THRESH_S
    lin_bins = np.linspace(0, zoom_edge, 40)

    for col, (sum_, lab, col_) in enumerate(zip(sums, labels, colors)):
        pairs = strict(sum_)
        dts = np.array([p["delta_t_s"] for p in pairs])
        if len(dts):
            dts_plot = np.where(dts == 0, 0.5, dts)
            w_full = np.ones(len(dts_plot)) / len(dts_plot)
            axes[0, col].hist(dts_plot, bins=log_bins, weights=w_full,
                              color=col_, edgecolor="white")
        axes[0, col].axvline(OUTLIER_THRESH_S, ls="--", color="#dc2626", lw=1, alpha=0.5,
                              label=f"outlier > {OUTLIER_THRESH_S} s")
        axes[0, col].set_xscale("log")
        axes[0, col].set_xlabel("Δt (s, log)")
        axes[0, col].set_ylabel("fraction of pairs")
        axes[0, col].set_title(f"{lab}  (n={len(dts)})")
        axes[0, col].grid(alpha=0.3, which="both")

        nonout = dts[dts <= OUTLIER_THRESH_S]
        if len(nonout):
            w_zoom = np.ones(len(nonout)) / len(nonout)
            axes[1, col].hist(nonout, bins=lin_bins, weights=w_zoom,
                              color=col_, edgecolor="white")
        axes[1, col].set_xlabel("Δt (s)")
        axes[1, col].set_ylabel("fraction of pairs")
        axes[1, col].set_title(f"Zoomed ≤ {OUTLIER_THRESH_S}s  (n={len(nonout)})")
        axes[1, col].grid(alpha=0.3)

    # Equalize y-limits across ALL columns in each row so conditions are directly comparable.
    for r in (0, 1):
        ymax = max(axes[r, c].get_ylim()[1] for c in range(n))
        for c in range(n):
            axes[r, c].set_ylim(0, ymax)

    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.28, bottom=0.34)

    # Footer with N-column stats.
    # Each non-baseline column shows "value (±X.X%)" relative to col-1 for pct rows.
    COL_W = 22
    rows = []
    hdr = f'{"metric":<32}  ' + '  '.join(f'{lab:>{COL_W}}' for lab in labels)
    rows.append(hdr)
    rows.append("-" * len(hdr))

    def _pct_suffix(a, b):
        if b is None or a is None or (isinstance(a, float) and np.isnan(a)) \
           or (isinstance(b, float) and np.isnan(b)) or a == 0:
            return ""
        return f" ({(b/a - 1) * 100:+.1f}%)"

    # Pull per-condition Δt arrays (all + cutoff-filtered) and τ mean.
    dts_all   = [np.array([p["delta_t_s"] for p in strict(s)]) for s in sums]
    dts_meas  = [a[a <= OUTLIER_THRESH_S] for a in dts_all]
    tau_means = [s["tau_mean_min"] for s in sums]

    def row(label, values, fmt="{:.3f}", delta_pct=True):
        cells = []
        base = values[0] if values else None
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cell = "-"
            else:
                try:
                    cell = fmt.format(v)
                except (ValueError, TypeError):
                    cell = str(v)
                if i > 0 and delta_pct:
                    cell = cell + _pct_suffix(base, v)
            cells.append(f'{cell:>{COL_W}}')
        rows.append(f"{label:<32}  " + '  '.join(cells))

    # Dynamic cutoff label — use minutes when ≥ 60 s and divisible, else seconds
    if OUTLIER_THRESH_S >= 60 and OUTLIER_THRESH_S % 60 == 0:
        cutoff_lbl = f"≤ {int(OUTLIER_THRESH_S/60)}-min window"
        sigma_lbl = f"σ_Δt (≤ {int(OUTLIER_THRESH_S/60)} min, min)"
        cv_lbl = f"intrinsic CV (≤ {int(OUTLIER_THRESH_S/60)} min)"
        med_lbl = f"median Δt (≤ {int(OUTLIER_THRESH_S/60)} min, s)"
    else:
        cutoff_lbl = f"≤ {OUTLIER_THRESH_S:g}-s window"
        sigma_lbl = f"σ_Δt (≤ {OUTLIER_THRESH_S:g} s, min)"
        cv_lbl = f"intrinsic CV (≤ {OUTLIER_THRESH_S:g} s)"
        med_lbl = f"median Δt (≤ {OUTLIER_THRESH_S:g} s, s)"

    row("n matched seeds", [s["n_seeds"] for s in sums], "{:d}", False)
    row("n real gens (gen1 excluded)", [s["n_gens_total"] for s in sums], "{:d}", False)
    row(f"n pairs ({cutoff_lbl})", [len(x) for x in dts_meas], "{:d}", False)
    row("mean τ (min)", tau_means, "{:.1f}")
    inits = [init_stats(strict(s)) for s in sums]
    row("mean initiation mass (fg)", [i.get("mass_mean") for i in inits], "{:.0f}")
    row("mean initiation volume (fL)", [i.get("vol_mean") for i in inits], "{:.3f}")
    row(sigma_lbl,
        [(x.std(ddof=1)/60 if len(x) > 1 else float("nan")) for x in dts_meas],
        "{:.2f}")
    row(med_lbl,
        [(float(np.median(x)) if len(x) else float("nan")) for x in dts_meas],
        "{:.1f}")
    row(cv_lbl,
        [(x.std(ddof=1)/(t*60) if len(x) > 1 and t else float("nan"))
         for x, t in zip(dts_meas, tau_means)],
        "{:.4f}")

    rows.append("")
    rows.append("(±X.X%) after each value = change vs col-1 (first condition = baseline)")
    footer = "\n".join(rows)
    # Shrink font a touch when N is large so the table still fits on the page.
    footer_fontsize = 9 if n <= 3 else 8 if n <= 5 else 7
    fig.text(0.5, 0.02, footer, family="monospace", fontsize=footer_fontsize, va="bottom", ha="center",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8))
    pdf.savefig(fig)
    plt.close(fig)


def page_dt_and_tau(pdf, sums, labels):
    """Δt histogram (unsigned) on top and τ histogram on bottom. One column per condition."""
    n = len(sums)
    fig_w = max(11, 4 + 3.5 * n)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 9), squeeze=False)
    fig.suptitle("Δt and τ distributions", fontsize=12, y=0.98)

    colors = _pick_colors(n)

    for col, (sum_, lab, col_) in enumerate(zip(sums, labels, colors)):
        pairs = strict(sum_)
        dts = np.array([p["delta_t_s"] for p in pairs])
        dts_meas_min = dts[dts <= OUTLIER_THRESH_S] / 60.0

        sigma_min = float(np.std(dts_meas_min, ddof=1)) if len(dts_meas_min) > 1 else float("nan")

        edge = max(dts_meas_min.max() * 1.05 if len(dts_meas_min) else 10, 10)
        bins = np.linspace(0, edge, 41)
        if len(dts_meas_min):
            axes[0, col].hist(dts_meas_min, bins=bins, color=col_, edgecolor="white")
        axes[0, col].set_xlabel("Δt (min)")
        axes[0, col].set_ylabel("count")
        axes[0, col].set_title(f"{lab}  (n = {len(dts_meas_min)})", fontsize=11)
        axes[0, col].text(0.97, 0.92, f"σ_Δt ≈ {sigma_min:.2f} min",
                          transform=axes[0, col].transAxes, ha="right",
                          fontsize=11, color=col_,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                    edgecolor=col_, alpha=0.9))
        axes[0, col].grid(alpha=0.3)

        # τ histogram
        taus = np.array([p["tau_min"] for p in pairs])
        if len(taus):
            axes[1, col].hist(taus, bins=25, color=col_, edgecolor="white", alpha=0.85)
            axes[1, col].axvline(taus.mean(), color="black", ls="--", lw=1, alpha=0.6)
            axes[1, col].text(0.97, 0.92, f"τ ≈ {taus.mean():.1f} min",
                              transform=axes[1, col].transAxes, ha="right",
                              fontsize=11, color=col_,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                        edgecolor=col_, alpha=0.9))
        axes[1, col].set_xlabel("generation time τ (min)")
        axes[1, col].set_ylabel("count")
        axes[1, col].set_xlim(0, 120)
        axes[1, col].grid(alpha=0.3)

    fig.subplots_adjust(top=0.92, hspace=0.35, wspace=0.25, bottom=0.08)
    pdf.savefig(fig)
    plt.close(fig)


def page_mass_vol_compare(pdf, sums, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Initiation cell mass / volume distribution — condition comparison",
                 fontsize=12, y=0.98)

    colors = _pick_colors(len(sums))
    for sum_, lab, col_ in zip(sums, labels, colors):
        pairs = strict(sum_)
        masses = np.array([p["first_mass"] for p in pairs if p.get("first_mass") is not None])
        vols   = np.array([p["first_vol"]  for p in pairs if p.get("first_vol")  is not None])
        if len(masses):
            ax1.hist(masses, bins=25, color=col_, alpha=0.5, edgecolor="white",
                     label=f"{lab}  mean={masses.mean():.0f}, n={len(masses)}")
        if len(vols):
            ax2.hist(vols, bins=25, color=col_, alpha=0.5, edgecolor="white",
                     label=f"{lab}  mean={vols.mean():.3f}, n={len(vols)}")

    ax1.set_xlabel("initiation cell mass (fg)")
    ax1.set_ylabel("count")
    ax1.set_title("Initiation mass")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("initiation cell volume (fL)")
    ax2.set_ylabel("count")
    ax2.set_title("Initiation volume")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    global OUTLIER_THRESH_S
    ap = argparse.ArgumentParser()
    # N-way flags (preferred). Repeatable.
    ap.add_argument("--summary", action="append", default=[],
                    help="path to a synchrony_summary.json (repeat for each condition, 1–6)")
    ap.add_argument("--label", action="append", default=[],
                    help="label for a condition (repeat, one per --summary in same order)")
    # Backwards-compat 2-condition flags.
    ap.add_argument("--summary-a", default=None, help="[legacy] alias for the first --summary")
    ap.add_argument("--summary-b", default=None, help="[legacy] alias for the second --summary")
    ap.add_argument("--label-a", default=None, help="[legacy] alias for the first --label")
    ap.add_argument("--label-b", default=None, help="[legacy] alias for the second --label")

    ap.add_argument("--config-note", default="")
    ap.add_argument("--cutoff-s", type=float, default=OUTLIER_THRESH_S,
                    help="primary outlier cutoff in seconds for Δt (default 600 = 10 min)")
    ap.add_argument("--extra-cutoff-s", type=float, default=None,
                    help="optional secondary cutoff; if set, adds a second histogram+table page")
    ap.add_argument("--first-page-only", action="store_true",
                    help="emit only the histograms+table pages (skip mass/vol and τ pages)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Resolve summaries + labels from either N-way or legacy flags.
    summaries = list(args.summary)
    labels = list(args.label)
    if args.summary_a is not None:
        summaries.insert(0, args.summary_a)
        labels.insert(0, args.label_a or "Condition A")
    if args.summary_b is not None:
        summaries.append(args.summary_b)
        labels.append(args.label_b or "Condition B")

    if not summaries:
        ap.error("no summaries given: use --summary (repeatable) or legacy --summary-a/--summary-b")
    if len(summaries) != len(labels):
        # Auto-fill missing labels with 'Condition {i+1}'.
        while len(labels) < len(summaries):
            labels.append(f"Condition {len(labels) + 1}")
        labels = labels[:len(summaries)]
    if not (1 <= len(summaries) <= 6):
        ap.error(f"N conditions must be in [1, 6], got {len(summaries)}")

    sums = [load(p) for p in summaries]

    with PdfPages(args.out) as pdf:
        note_base = f"\n{args.config_note}" if args.config_note else ""

        OUTLIER_THRESH_S = args.cutoff_s
        page_histogram(pdf, sums, labels, title_extra=note_base)

        if args.extra_cutoff_s is not None:
            OUTLIER_THRESH_S = args.extra_cutoff_s
            page_histogram(pdf, sums, labels, title_extra=note_base)

        if not args.first_page_only:
            page_mass_vol_compare(pdf, sums, labels)
            page_dt_and_tau(pdf, sums, labels)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
