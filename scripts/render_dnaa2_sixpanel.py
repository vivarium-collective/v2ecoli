"""6-panel minutes-axis trajectory figure for the dnaa-2 hydrolysis study.

Per the hydrolysis-study spec's deliverable: a single trajectory figure with
x-axis in MINUTES (cumulative across the lineage, generation boundaries
marked) and the six panels:

  1. Cell mass (fg)
  2. oriC count
  3. DnaA forms: total / DnaA-ATP / DnaA-ADP / apo  (+ [300,800] total band)
  4. DnaA concentration (total DnaA / cell mass, counts/fg)
  5. DnaA-form fractions: %ATP / %ADP / %apo        (+ [0.2,0.5] ATP band)
  6. Cellular bulk ATP[c] and ADP[c] counts

Reads the workflow's hive-partitioned history parquet for one lineage_seed
(each generation is a fresh composite whose global_time resets, so we offset
each generation by the cumulative duration of prior generations). Renders
with matplotlib to SVG + PNG (the repo's chart convention).

    python scripts/render_dnaa2_sixpanel.py \
        --run studies/dnaa-2-atp-hydrolysis/parquet-runs/dnaa2-multiseed/parquet \
        --seed 0 --step 2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Worktree's v2ecoli must win over the editable install (which points at the
# main checkout and lacks this worktree's visualization classes).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import polars as pl

C = dict(total="#222222", atp="#1f77b4", adp="#d62728", apo="#2ca02c",
         oric="#9467bd", conc="#ff7f0e")


def _load_seed(run_dir: str, seed: int):
    files = glob.glob(os.path.join(
        run_dir, "**", "history", "**", f"lineage_seed={seed}", "**", "*.pq"),
        recursive=True)
    if not files:
        raise SystemExit(f"no history parquet for seed {seed} under {run_dir}")
    ids = pl.scan_parquet(files[0]).select("bulk__id").head(1).collect()["bulk__id"][0].to_list()
    atp_i, adp_i = ids.index("ATP[c]"), ids.index("ADP[c]")
    keep = ["generation", "global_time",
            "listeners__mass__cell_mass",
            "listeners__replication_data__number_of_oric",
            "listeners__dnaA_cycle__total", "listeners__dnaA_cycle__atp_count",
            "listeners__dnaA_cycle__adp_count", "listeners__dnaA_cycle__apo_count",
            "listeners__dnaA_cycle__atp_fraction",
            "listeners__dnaA_cycle__adp_fraction",
            "listeners__dnaA_cycle__apo_fraction"]
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .select(keep + [pl.col("bulk__count").list.get(atp_i).alias("bulk_atp"),
                          pl.col("bulk__count").list.get(adp_i).alias("bulk_adp")])
          .sort(["generation", "global_time"]).collect())
    offset, cum, bounds = 0.0, [], []
    for gen in sorted(df["generation"].unique().to_list()):
        t = df.filter(pl.col("generation") == gen)["global_time"].to_numpy()
        cum.extend((t + offset) / 60.0)
        offset += float(t.max())
        bounds.append(offset / 60.0)
    df = df.with_columns(pl.Series("t_min", cum))
    n = df.height
    if n > 4000:
        df = df.gather_every(max(1, n // 4000))
    return df, bounds[:-1]


def render(run_dir: str, seed: int, step: int, out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, bounds = _load_seed(run_dir, seed)
    x = df["t_min"].to_numpy()
    ngen = len(bounds) + 1
    fig, ax = plt.subplots(6, 1, figsize=(11, 15), sharex=True)

    ax[0].plot(x, df["listeners__mass__cell_mass"], color=C["total"], lw=1.3)
    ax[0].set_ylabel("cell mass\n(fg)")

    ax[1].step(x, df["listeners__replication_data__number_of_oric"], color=C["oric"], lw=1.3, where="post")
    ax[1].set_ylabel("oriC\ncount")
    ax[1].set_ylim(bottom=0)

    ax[2].axhspan(300, 800, color="0.5", alpha=0.10, lw=0)
    ax[2].plot(x, df["listeners__dnaA_cycle__total"], color=C["total"], lw=1.8, label="total DnaA")
    ax[2].plot(x, df["listeners__dnaA_cycle__atp_count"], color=C["atp"], lw=1.1, label="DnaA-ATP")
    ax[2].plot(x, df["listeners__dnaA_cycle__adp_count"], color=C["adp"], lw=1.1, label="DnaA-ADP")
    ax[2].plot(x, df["listeners__dnaA_cycle__apo_count"], color=C["apo"], lw=1.1, label="apo-DnaA")
    ax[2].set_ylabel("DnaA forms\n(counts)")
    ax[2].legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.9)
    ax[2].text(0.995, 0.06, "band = total [300, 800]", transform=ax[2].transAxes,
               ha="right", va="bottom", fontsize=7, color="0.4")

    conc = (df["listeners__dnaA_cycle__total"] / df["listeners__mass__cell_mass"]).to_numpy()
    ax[3].plot(x, conc, color=C["conc"], lw=1.3)
    ax[3].set_ylabel("DnaA conc\n(counts/fg)")

    ax[4].axhspan(0.2, 0.5, color=C["atp"], alpha=0.10, lw=0)
    ax[4].plot(x, df["listeners__dnaA_cycle__atp_fraction"], color=C["atp"], lw=1.8, label="%DnaA-ATP")
    ax[4].plot(x, df["listeners__dnaA_cycle__adp_fraction"], color=C["adp"], lw=1.1, label="%DnaA-ADP")
    ax[4].plot(x, df["listeners__dnaA_cycle__apo_fraction"], color=C["apo"], lw=1.1, label="%apo")
    ax[4].set_ylabel("form\nfractions")
    ax[4].set_ylim(-0.03, 1.06)
    ax[4].legend(loc="center left", fontsize=7, ncol=3, framealpha=0.9)
    ax[4].text(0.995, 0.06, "band = DnaA-ATP [0.2, 0.5]", transform=ax[4].transAxes,
               ha="right", va="bottom", fontsize=7, color="0.4")

    ax[5].plot(x, df["bulk_atp"], color=C["atp"], lw=1.2, label="ATP[c]")
    ax[5].plot(x, df["bulk_adp"], color=C["adp"], lw=1.2, label="ADP[c]")
    ax[5].set_ylabel("bulk pool\n(counts)")
    ax[5].legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    ax[5].set_xlabel("lineage time (min, cumulative across generations)")

    for a in ax:
        for b in bounds:
            a.axvline(b, color="0.0", lw=0.8, ls=":", alpha=0.3)
        a.grid(True, alpha=0.15)
    mech = "ON" if step >= 2 else "OFF"
    fig.suptitle(f"dnaa-2 Step {step} — DnaA nucleotide-state trajectory "
                 f"(succinate, seed {seed}, mechanism {mech}; "
                 f"{ngen} generations, dotted = division boundaries)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"dnaa2_step{step}_sixpanel_seed{seed}")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)
    print(f"wrote {base}.svg / .png  ({df.height} pts, {ngen} generations)")
    return base + ".svg"


def render_html(run_dir: str, seed: int, step: int, out_path: str,
                title: str | None = None) -> str:
    """Interactive Plotly 6-panel via DnaaSixPanelVisualization → HTML file
    (the dashboard surfaces reports/figures/<study>/<name>.html by name)."""
    from v2ecoli.visualizations.dnaa_succinate import DnaaSixPanelVisualization
    from v2ecoli.core import build_core

    df, bounds = _load_seed(run_dir, seed)
    col = lambda c: [float(v) for v in df[c].to_list()]
    conc = (df["listeners__dnaA_cycle__total"] / df["listeners__mass__cell_mass"]).to_list()
    mech = "ON" if step >= 2 else "OFF"
    state = {
        "t": col("t_min"),
        "cell_mass": col("listeners__mass__cell_mass"),
        "oric": col("listeners__replication_data__number_of_oric"),
        "total": col("listeners__dnaA_cycle__total"),
        "atp": col("listeners__dnaA_cycle__atp_count"),
        "adp": col("listeners__dnaA_cycle__adp_count"),
        "apo": col("listeners__dnaA_cycle__apo_count"),
        "conc": [float(v) for v in conc],
        "atp_frac": col("listeners__dnaA_cycle__atp_fraction"),
        "adp_frac": col("listeners__dnaA_cycle__adp_fraction"),
        "apo_frac": col("listeners__dnaA_cycle__apo_fraction"),
        "atp_pool": col("bulk_atp"),
        "adp_pool": col("bulk_adp"),
        "division_times": bounds,
        "_caption": (f"succinate, seed {seed}, mechanism {mech} "
                     f"(k={'0.046' if step >= 2 else '0'} /min); "
                     f"{len(bounds) + 1} generations; dashed = division boundaries"),
    }
    core = build_core()
    viz = DnaaSixPanelVisualization(config={
        "title": title or f"dnaa-2 Step {step} — DnaA nucleotide-state trajectory (seed {seed})",
    }, core=core)
    html = viz.update(state).get("html", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path}  ({len(html)/1024:.0f} KB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out-dir", default="studies/dnaa-2-atp-hydrolysis/charts")
    ap.add_argument("--format", choices=["svg", "html", "both"], default="both")
    ap.add_argument("--html-out", default=None,
                    help="HTML output path (default: reports/figures/"
                         "dnaa-2-atp-hydrolysis/dnaa2_step<N>_sixpanel_seed<S>.html)")
    a = ap.parse_args()
    if a.format in ("svg", "both"):
        render(a.run, a.seed, a.step, a.out_dir)
    if a.format in ("html", "both"):
        html_out = a.html_out or (f"reports/figures/dnaa-2-atp-hydrolysis/"
                                  f"dnaa2_step{a.step}_sixpanel_seed{a.seed}.html")
        render_html(a.run, a.seed, a.step, html_out)


if __name__ == "__main__":
    main()
