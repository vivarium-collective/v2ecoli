"""Cross-condition comparison: v2ecoli vs vEcoli on all 5 canonical media.

Reads v2ecoli SQLite + vEcoli parquet, produces:
  1. Bar chart: declared tau vs observed cycle time per simulator
  2. Per-condition trajectory grid (5 rows x 3 cols): mass / n_forks / dry_mass
     v2ecoli vs vEcoli overlaid in each panel
"""
from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("reports/meeting_20260526")

# (condition, declared tau, v2ecoli sqlite db, vEcoli parquet dir)
CONDITIONS = [
    ("basal",     44.0,  "out/basal_apr24_run.db",      "out/vecoli_basal_compare_parquet"),
    ("with_aa",   25.0,  "out/with_aa_apr24_run.db",    "out/vecoli_with_aa_compare_parquet"),
    ("acetate",  136.0,  "out/acetate_apr24_run.db",    "out/vecoli_acetate_compare_parquet"),
    ("succinate", 82.0,  "out/succinate_apr24_run.db",  "out/vecoli_succinate_compare_parquet"),
    ("no_oxygen",100.0,  "out/no_oxygen_apr24_run.db",  "out/vecoli_no_oxygen_compare_parquet"),
]


def load_v2ecoli(db_path: str, max_points: int = 400) -> dict:
    """Load t, cell_mass, dry_mass, n_oriC, n_forks from a v2ecoli SQLite."""
    c = sqlite3.connect(db_path)
    sid = c.execute("SELECT simulation_id FROM simulations LIMIT 1").fetchone()[0]
    rows = c.execute(
        "SELECT step, state FROM history WHERE simulation_id=? ORDER BY step",
        (sid,),
    ).fetchall()
    if not rows:
        return None
    every = max(1, len(rows) // max_points)
    # Unique-molecule structured arrays serialize as lists of rows whose
    # _entryState column position depends on the molecule's dtype:
    #   full_chromosome:   14 cols, _entryState at index 12
    #   active_replisome:  14 cols, _entryState at index 12
    ES_FULL_CHROM = 12
    ES_REPLISOME = 12

    def count_active(rows_list, es_col):
        return sum(
            1 for r in rows_list
            if isinstance(r, list) and len(r) > es_col and r[es_col]
        )

    out = {"t": [], "cell_mass": [], "dry_mass": [], "dna_mass": [],
           "n_oric": [], "n_replisomes": [], "n_full_chrom": []}
    for i, (step, st) in enumerate(rows):
        if i % every != 0 and i != len(rows) - 1:
            continue
        s = json.loads(st)
        m = s.get("listeners", {}).get("mass", {}) or {}
        rd = s.get("listeners", {}).get("replication_data", {}) or {}
        out["t"].append(float(s.get("time", step)) / 60.0)
        out["cell_mass"].append(float(m.get("cell_mass", 0.0) or 0.0))
        out["dry_mass"].append(float(m.get("dry_mass", 0.0) or 0.0))
        out["dna_mass"].append(float(m.get("dna_mass", 0.0) or 0.0))
        out["n_oric"].append(int(rd.get("number_of_oric", 0) or 0))
        out["n_replisomes"].append(
            count_active(s.get("active_replisome", []) or [], ES_REPLISOME)
        )
        out["n_full_chrom"].append(
            count_active(s.get("full_chromosome", []) or [], ES_FULL_CHROM)
        )
    return {k: np.array(v) for k, v in out.items()}


def load_vecoli(parquet_root: str, max_points: int = 400) -> dict:
    """Load t, cell_mass, dry_mass, n_oriC, n_forks from a vEcoli parquet tree."""
    import pyarrow.parquet as pq

    # find the history dir — handle both layouts:
    #   <root>/<expt>/history/exp_id=.../variant=0/lineage_seed=0/generation=1/agent_id=0
    #   <root>/history/exp_id=.../variant=0/lineage_seed=0/generation=1/agent_id=0
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
        "listeners__unique_molecule_counts__full_chromosome",
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
    out = {
        "t": (df["time"].astype(float) / 60.0).to_numpy(),
        "cell_mass": col_float("listeners__mass__cell_mass"),
        "dry_mass": col_float("listeners__mass__dry_mass"),
        "dna_mass": col_float("listeners__mass__dna_mass"),
        "n_oric": col_int("listeners__replication_data__number_of_oric"),
        "n_replisomes": col_int("listeners__unique_molecule_counts__active_replisome"),
        "n_full_chrom": col_int("listeners__unique_molecule_counts__full_chromosome"),
    }
    return out


def cycle_time(d: dict, name: str = "") -> float:
    """Cycle time in minutes — last timestep (= division time, since the
    sim halts at division for all 5 canonical conditions)."""
    if d is None or len(d.get("t", [])) == 0:
        return float("nan")
    return float(d["t"][-1])


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    data = {}
    for cond, tau, v2_db, ve_dir in CONDITIONS:
        v2 = None
        ve = None
        if Path(v2_db).exists():
            try:
                v2 = load_v2ecoli(v2_db)
            except Exception as e:
                print(f"  v2 {cond}: load error: {e}")
        if Path(ve_dir).exists():
            try:
                ve = load_vecoli(ve_dir)
            except Exception as e:
                print(f"  vEcoli {cond}: load error: {e}")
        data[cond] = {"tau": tau, "v2": v2, "ve": ve}
        v2_t = cycle_time(v2, cond)
        ve_t = cycle_time(ve, cond)
        print(f"  {cond:>10}: tau={tau:5.1f}min  v2={v2_t:6.1f}  vEcoli={ve_t:6.1f}")

    # =========================================================================
    # Figure 1 — cycle time bar chart
    # =========================================================================
    conds = [c for c, _, _, _ in CONDITIONS]
    taus = np.array([data[c]["tau"] for c in conds])
    v2s  = np.array([cycle_time(data[c]["v2"], c) for c in conds])
    ves  = np.array([cycle_time(data[c]["ve"], c) for c in conds])

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(conds))
    width = 0.27
    ax.bar(x - width, taus, width, color="#57606a", label="declared τ", alpha=0.85)
    ax.bar(x,         v2s,  width, color="#2470d6", label="v2ecoli observed", alpha=0.95)
    ax.bar(x + width, ves,  width, color="#bf8700", label="vEcoli observed", alpha=0.95)
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
    ax.set_title("Cell cycle time across canonical media — declared τ vs observed (v2ecoli, vEcoli)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_condition_cycle_time.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'cross_condition_cycle_time.png'}")

    # =========================================================================
    # Figure 2 — per-condition trajectories
    # =========================================================================
    fig, axes = plt.subplots(len(CONDITIONS), 4, figsize=(14, 2.0 * len(CONDITIONS)), sharex=False)
    fig.suptitle(
        "Chromosome / mass dynamics across canonical media — v2ecoli vs vEcoli  "
        "(canonical condition_defs)",
        fontsize=11, y=0.995,
    )

    for i, (cond, tau, _, _) in enumerate(CONDITIONS):
        v2 = data[cond]["v2"]
        ve = data[cond]["ve"]

        # mass
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

        # n_oriC
        ax = axes[i, 1]
        if v2 is not None:
            ax.step(v2["t"], v2["n_oric"], color="#2470d6", lw=1.4, where="post")
        if ve is not None:
            ax.step(ve["t"], ve["n_oric"], color="#bf8700", lw=1.4, where="post")
        ax.set_ylabel("n oriC", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.3, max(8, max((v2 or {}).get("n_oric", [0]).max() if v2 is not None else 0,
                                     (ve or {}).get("n_oric", [0]).max() if ve is not None else 0) + 0.5))
        if i == 0:
            ax.set_title("oriC count", fontsize=10)

        # n_replisomes (= n active forks, since each replisome rides 1 fork)
        ax = axes[i, 2]
        if v2 is not None:
            ax.step(v2["t"], v2["n_replisomes"], color="#2470d6", lw=1.4, where="post")
        if ve is not None:
            ax.step(ve["t"], ve["n_replisomes"], color="#bf8700", lw=1.4, where="post")
        ax.set_ylabel("n active replisomes", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("active replisomes (= forks)", fontsize=10)

        # n_full_chromosomes
        ax = axes[i, 3]
        if v2 is not None and "n_full_chrom" in v2:
            ax.step(v2["t"], v2["n_full_chrom"], color="#2470d6", lw=1.4, where="post")
        if ve is not None and "n_full_chrom" in ve:
            ax.step(ve["t"], ve["n_full_chrom"], color="#bf8700", lw=1.4, where="post")
        ax.set_ylabel("n full chromosomes", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("full chromosomes", fontsize=10)

        # cycle-time annotation in mass panel
        v2_c = cycle_time(v2, cond)
        ve_c = cycle_time(ve, cond)
        msg = f"τ={tau:.0f}  |  v2={v2_c:.0f}  vE={ve_c:.0f}"
        axes[i, 0].text(0.98, 0.05, msg, ha="right", va="bottom",
                        transform=axes[i, 0].transAxes, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d7de", alpha=0.85))

        # x label on last row
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
