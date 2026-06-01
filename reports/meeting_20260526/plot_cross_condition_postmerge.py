"""Cross-condition comparison using the POST-MERGE parquet outputs.

Companion to plot_cross_condition.py (which reads pre-merge apr24 SQLite +
vEcoli parquet). This one reads the v2ecoli parquet outputs from
out/<condition>_postmerge_parquet/ and the same vEcoli parquet trees, so
the comparison reflects what the canonical_baseline.html table now shows.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT_DIR = Path("reports/meeting_20260526")

# (condition, declared tau, v2ecoli parquet root, vEcoli parquet dir)
CONDITIONS = [
    ("basal",     44.0,  "out/basal_postmerge_parquet/basal_postmerge",         "out/vecoli_basal_compare_parquet"),
    ("with_aa",   25.0,  "out/with_aa_postmerge_parquet/with_aa_postmerge",     "out/vecoli_with_aa_compare_parquet"),
    ("acetate",  136.0,  "out/acetate_postmerge_parquet/acetate_postmerge",     "out/vecoli_acetate_compare_parquet"),
    ("succinate", 82.0,  "out/succinate_postmerge_parquet/succinate_postmerge", "out/vecoli_succinate_compare_parquet"),
    ("no_oxygen",100.0,  "out/no_oxygen_postmerge_parquet/no_oxygen_postmerge", "out/vecoli_no_oxygen_compare_parquet"),
]

V2_COLS = [
    "global_time",
    "listeners__mass__cell_mass",
    "listeners__mass__dry_mass",
    "listeners__mass__dna_mass",
    "listeners__replication_data__number_of_oric",
    "listeners__replication_data__fork_coordinates",
]


def load_v2(parquet_root: str, max_points: int = 400) -> dict | None:
    """Read v2ecoli post-merge parquet for one condition.

    The history files live under
    ``<root>/history/experiment_id=.../variant=0/lineage_seed=0/generation=1/agent_id=0/*.pq``.
    Each file is one batch; concat them and downsample to max_points rows.
    """
    files = sorted(
        glob.glob(f"{parquet_root}/history/**/*.pq", recursive=True),
        key=lambda p: int(Path(p).stem),
    )
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f, columns=V2_COLS) for f in files]).sort("global_time")
    df = df.with_columns(
        pl.col("listeners__replication_data__fork_coordinates").list.len().alias("n_forks")
    )
    every = max(1, df.height // max_points)
    df = df.gather_every(every)
    return {
        "t": (df["global_time"] / 60.0).to_numpy(),
        "cell_mass": df["listeners__mass__cell_mass"].to_numpy(),
        "dry_mass":  df["listeners__mass__dry_mass"].to_numpy(),
        "dna_mass":  df["listeners__mass__dna_mass"].to_numpy(),
        "n_oric":     df["listeners__replication_data__number_of_oric"].to_numpy(),
        "n_replisomes": df["n_forks"].to_numpy(),
    }


def load_vecoli(parquet_root: str, max_points: int = 400) -> dict | None:
    import pyarrow.parquet as pq
    hist_dirs = glob.glob(f"{parquet_root}/*/history/*/*/*/*/*")
    if not hist_dirs:
        hist_dirs = glob.glob(f"{parquet_root}/history/*/*/*/*/*")
    if not hist_dirs:
        return None
    files = sorted(
        glob.glob(f"{hist_dirs[0]}/*.pq"),
        key=lambda p: int(Path(p).stem),
    )
    if not files:
        return None
    cols = [
        "time",
        "listeners__mass__cell_mass",
        "listeners__mass__dry_mass",
        "listeners__mass__dna_mass",
        "listeners__replication_data__number_of_oric",
        "listeners__unique_molecule_counts__active_replisome",
    ]
    dfs = []
    for f in files:
        try:
            t = pq.read_table(f, columns=cols).to_pandas()
        except Exception:
            t = pq.read_table(f).to_pandas()
            t = t[[c for c in cols if c in t.columns]]
        dfs.append(t)
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    every = max(1, len(df) // max_points)
    df = df.iloc[::every].reset_index(drop=True)
    def col_int(name):
        return df[name].astype(int).to_numpy() if name in df.columns else np.zeros(len(df), dtype=int)
    def col_float(name):
        return df[name].astype(float).to_numpy() if name in df.columns else np.zeros(len(df))
    return {
        "t": (df["time"].astype(float) / 60.0).to_numpy(),
        "cell_mass": col_float("listeners__mass__cell_mass"),
        "dry_mass":  col_float("listeners__mass__dry_mass"),
        "dna_mass":  col_float("listeners__mass__dna_mass"),
        "n_oric":     col_int("listeners__replication_data__number_of_oric"),
        "n_replisomes": col_int("listeners__unique_molecule_counts__active_replisome"),
    }


def cycle_time(d: dict | None) -> float:
    if d is None or len(d["t"]) == 0:
        return float("nan")
    return float(d["t"][-1])


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    data: dict[str, dict] = {}
    for cond, tau, v2_root, ve_dir in CONDITIONS:
        v2 = load_v2(v2_root)
        ve = load_vecoli(ve_dir) if Path(ve_dir).exists() else None
        data[cond] = {"tau": tau, "v2": v2, "ve": ve}
        v2_t = cycle_time(v2)
        ve_t = cycle_time(ve)
        print(f"  {cond:>10}: tau={tau:5.1f}min  v2={v2_t:6.1f}  vEcoli={ve_t:6.1f}")

    # Figure 1 — cycle time bar chart
    conds = [c for c, _, _, _ in CONDITIONS]
    taus = np.array([data[c]["tau"] for c in conds])
    v2s = np.array([cycle_time(data[c]["v2"]) for c in conds])
    ves = np.array([cycle_time(data[c]["ve"]) for c in conds])

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(conds))
    width = 0.27
    ax.bar(x - width, taus, width, color="#57606a", label="declared τ", alpha=0.85)
    ax.bar(x, v2s, width, color="#2470d6", label="v2ecoli observed (post-merge)", alpha=0.95)
    ax.bar(x + width, ves, width, color="#bf8700", label="vEcoli observed", alpha=0.95)
    for xi, v in zip(x - width, taus):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8, color="#1f2328")
    for xi, v in zip(x, v2s):
        if not np.isnan(v):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8, color="#2470d6")
    for xi, v in zip(x + width, ves):
        if not np.isnan(v):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8, color="#bf8700")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("cell cycle time (min)", fontsize=10)
    ax.set_title("Cell cycle time — declared τ vs observed (v2ecoli post-merge, vEcoli)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_condition_cycle_time.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'cross_condition_cycle_time.png'}")

    # Figure 2 — per-condition trajectories
    fig, axes = plt.subplots(len(CONDITIONS), 4, figsize=(14, 2.0 * len(CONDITIONS)), sharex=False)
    fig.suptitle(
        "Chromosome / mass dynamics — v2ecoli post-merge vs vEcoli",
        fontsize=11, y=0.995,
    )
    for i, (cond, tau, _, _) in enumerate(CONDITIONS):
        v2 = data[cond]["v2"]
        ve = data[cond]["ve"]

        ax = axes[i, 0]
        if v2 is not None:
            ax.plot(v2["t"], v2["dry_mass"], color="#2470d6", lw=1.4, label="v2ecoli")
        if ve is not None:
            ax.plot(ve["t"], ve["dry_mass"], color="#bf8700", lw=1.4, label="vEcoli")
        ax.set_ylabel(f"{cond}\ndry mass (fg)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("dry mass", fontsize=10)
            ax.legend(fontsize=8, loc="upper left", frameon=False)

        ax = axes[i, 1]
        if v2 is not None:
            ax.step(v2["t"], v2["n_oric"], color="#2470d6", lw=1.4, where="post")
        if ve is not None:
            ax.step(ve["t"], ve["n_oric"], color="#bf8700", lw=1.4, where="post")
        ax.set_ylabel("n oriC", fontsize=9)
        ax.grid(True, alpha=0.3)
        ymax = 8
        if v2 is not None and len(v2["n_oric"]):
            ymax = max(ymax, int(v2["n_oric"].max()))
        if ve is not None and len(ve["n_oric"]):
            ymax = max(ymax, int(ve["n_oric"].max()))
        ax.set_ylim(-0.3, ymax + 0.5)
        if i == 0:
            ax.set_title("oriC count", fontsize=10)

        ax = axes[i, 2]
        if v2 is not None:
            ax.step(v2["t"], v2["n_replisomes"], color="#2470d6", lw=1.4, where="post")
        if ve is not None:
            ax.step(ve["t"], ve["n_replisomes"], color="#bf8700", lw=1.4, where="post")
        ax.set_ylabel("n active replisomes", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("active replisomes (= forks)", fontsize=10)

        ax = axes[i, 3]
        if v2 is not None:
            ax.plot(v2["t"], v2["dna_mass"], color="#2470d6", lw=1.4)
        if ve is not None:
            ax.plot(ve["t"], ve["dna_mass"], color="#bf8700", lw=1.4)
        ax.set_ylabel("DNA mass (fg)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("DNA mass", fontsize=10)

        v2_c = cycle_time(v2)
        ve_c = cycle_time(ve)
        msg = f"τ={tau:.0f}  |  v2={v2_c:.0f}  vE={ve_c:.0f}"
        axes[i, 0].text(0.98, 0.05, msg, ha="right", va="bottom",
                        transform=axes[i, 0].transAxes, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d7de", alpha=0.85))
        if i == len(CONDITIONS) - 1:
            for j in range(4):
                axes[i, j].set_xlabel("time (min)", fontsize=9)

    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_DIR / "cross_condition_dynamics.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'cross_condition_dynamics.png'}")


if __name__ == "__main__":
    main()
