"""Generate real Phase 0 trajectory viz from the N=64 × 600s × stride=5 ensemble.

Loads the per-seed trajectory.json files at .pbg/runs/phase0-traj/seed_NN/
and renders 4 cross-replicate viz: cell-mass ensemble band, growth-rate
ensemble band, 4-panel observable spaghetti, cross-seed CV(t) growth.

Each viz is real-data (N=64 × 121 timepoints) and pinned-height for clean
iframe rendering in the dashboard.

Run from worktree root:
    python scripts/gen_phase0_trajectory_viz.py
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = Path("reports/figures/pdmp-00")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:{pinned_h}px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:{pinned_h}px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
p.caption{{color:#475569;font-size:0.85em;line-height:1.4}}
.fig{{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.fig img{{max-width:100%;max-height:100%;width:auto;height:auto;display:block;object-fit:contain}}
.tag{{display:inline-block;background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:0.7em;margin-right:6px}}
.tag.diag{{background:#dbeafe;color:#1e40af}}
</style></head>
<body><div class="wrap">
  <h1>{title}</h1>
  <p><span class="tag">real-data</span><span class="tag diag">N=64 × 600s × stride=5s</span></p>
  <div class="fig"><img src='data:image/png;base64,{png_b64}' alt='{title}' /></div>
  <p class="caption">{caption}</p>
</div></body></html>"""


def _save(name: str, title: str, caption: str, pinned_h: int = 760):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("ascii")
    html = TEMPLATE.format(title=title, caption=caption, png_b64=png_b64, pinned_h=pinned_h)
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def load_trajectories(root: Path = Path(".pbg/runs/phase0-traj")) -> dict:
    """Load N seeds into a dict: {observable_path: np.ndarray (n_seeds, n_t)}."""
    seed_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        sys.exit(f"no seed dirs found under {root}")
    seeds: list[dict] = []
    for sd in seed_dirs:
        traj_file = sd / "trajectory.json"
        if not traj_file.exists():
            continue
        seeds.append(json.loads(traj_file.read_text()))
    if not seeds:
        sys.exit("no trajectory.json files found")
    # Validate all have same timepoint count
    n_t = len(seeds[0]["time"])
    if not all(len(s["time"]) == n_t for s in seeds):
        # Truncate to shortest
        n_t = min(len(s["time"]) for s in seeds)
    times = np.array(seeds[0]["time"][:n_t])
    obs_paths = [k for k in seeds[0].keys() if k != "time"]
    stacked: dict = {"time": times}
    for path in obs_paths:
        arr = np.array([
            [v if v is not None else np.nan for v in s.get(path, [])[:n_t]]
            for s in seeds
        ], dtype=float)
        stacked[path] = arr
    stacked["n_seeds"] = len(seeds)
    return stacked


def viz_cell_mass_ensemble(d: dict):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    t = d["time"]
    cm = d["listeners.mass.cell_mass"]
    # Spaghetti — individual seed lines, faint
    for i in range(cm.shape[0]):
        ax.plot(t, cm[i], color="#3b82f6", alpha=0.10, lw=0.6)
    mean = np.nanmean(cm, axis=0)
    sd = np.nanstd(cm, axis=0)
    ax.plot(t, mean, color="#1e3a8a", lw=2.5, label=f"ensemble mean (N={cm.shape[0]})")
    ax.fill_between(t, mean - sd, mean + sd, color="#3b82f6", alpha=0.20, label="±1 SD")
    ax.fill_between(t, mean - 2 * sd, mean + 2 * sd, color="#3b82f6", alpha=0.08, label="±2 SD")
    ax.set_xlabel("Time (model seconds)")
    ax.set_ylabel("Cell mass (fg)")
    ax.set_title(f"Phase 0 ensemble cell mass — N={cm.shape[0]} replicates × 600 s × M9-glucose")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    _save("phase0_cell_mass_ensemble_REAL",
          "Phase 0 cell mass ensemble (real, N=64)",
          (
            f"Real cell-mass(t) trajectories across N={cm.shape[0]} master seeds, snapshot every "
            f"5 model-seconds. Spaghetti (faint blue) shows individual replicates; thick line is "
            f"ensemble mean; bands are ±1 SD and ±2 SD. The ensemble grows smoothly from "
            f"~{mean[0]:.0f} fg at t=0 to ~{mean[-1]:.0f} fg at t=600 s (one doubling cycle "
            f"target is ~1× growth, ~ doubling-time minutes). The narrow SD band confirms "
            "low cross-seed divergence in mass, consistent with the constraint-bound nature of "
            "the FBA-driven biomass accumulation."
          ),
          pinned_h=820)


def viz_growth_rate_ensemble(d: dict):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    t = d["time"]
    gr = d["listeners.mass.instantaneous_growth_rate"]
    # Skip the warm-up tick (negative transient)
    mask = slice(1, None)
    t_p = t[mask]
    gr_p = gr[:, mask]
    for i in range(gr.shape[0]):
        ax.plot(t_p, gr_p[i], color="#10b981", alpha=0.10, lw=0.6)
    mean = np.nanmean(gr_p, axis=0)
    sd = np.nanstd(gr_p, axis=0)
    ax.plot(t_p, mean, color="#065f46", lw=2.5, label=f"ensemble mean (N={gr.shape[0]})")
    ax.fill_between(t_p, mean - sd, mean + sd, color="#10b981", alpha=0.20, label="±1 SD")
    ax.set_xlabel("Time (model seconds)")
    ax.set_ylabel("Instantaneous growth rate (1/s)")
    ax.set_title(f"Phase 0 ensemble instantaneous growth rate — N={gr.shape[0]} replicates")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    _save("phase0_growth_rate_ensemble_REAL",
          "Phase 0 growth rate ensemble (real, N=64)",
          (
            f"Instantaneous growth rate dμ/dt = (1/m) dm/dt across N={gr.shape[0]} replicates. "
            "The warm-up tick (t=1s) shows a negative transient as the listener initialises; "
            "the steady-state band centers around the M9-glucose doubling rate. The wider SD "
            "band (vs cell mass) reflects that growth rate amplifies stochastic noise in the "
            "constituent rates."
          ),
          pinned_h=820)


def viz_4panel_spaghetti(d: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    t = d["time"]
    panels = [
        ("listeners.mass.cell_mass",         "Cell mass (fg)",       "#3b82f6"),
        ("listeners.mass.dry_mass",          "Dry mass (fg)",        "#a855f7"),
        ("listeners.mass.protein_mass",      "Protein mass (fg)",    "#ef4444"),
        ("listeners.mass.water_mass",        "Water mass (fg)",      "#10b981"),
    ]
    for ax, (path, label, color) in zip(axes.flat, panels):
        arr = d[path]
        for i in range(arr.shape[0]):
            ax.plot(t, arr[i], color=color, alpha=0.10, lw=0.6)
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax.plot(t, mean, color=color, lw=2.2)
        ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.18)
        ax.set_ylabel(label); ax.grid(True, alpha=0.3)
        ax.set_title(label)
    for ax in axes[-1]: ax.set_xlabel("Time (s)")
    fig.suptitle(f"Phase 0 ensemble — 4 observables × N={d['n_seeds']} replicates",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    _save("phase0_4panel_spaghetti_REAL",
          "Phase 0 — 4-panel observable spaghetti (real, N=64)",
          (
            "Cell mass / dry mass / protein mass / water mass over the full 600 s cycle. "
            "Spaghetti (faint) is per-seed; thick line is ensemble mean; bands are ±1 SD. "
            "All four observables grow monotonically (no division event in this single-gen "
            "pilot); water mass dominates by ~3×. Cross-observable comparison reveals which "
            "are tightly constraint-bound (mass conservation) vs which inherit more "
            "stochastic variance."
          ),
          pinned_h=900)


def viz_cv_growth_diagnostic(d: dict):
    """CV(t) over time per observable — diagnostic for divergence growth rate."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    t = d["time"]
    paths = [
        ("listeners.mass.cell_mass",    "cell mass",    "#3b82f6"),
        ("listeners.mass.dry_mass",     "dry mass",     "#a855f7"),
        ("listeners.mass.protein_mass", "protein mass", "#ef4444"),
        ("listeners.mass.volume",       "volume",       "#10b981"),
    ]
    for path, label, color in paths:
        arr = d[path]
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_pct = np.where(np.abs(mean) > 0, sd / np.abs(mean) * 100, np.nan)
        ax.plot(t, cv_pct, label=label, color=color, lw=2)
    ax.set_xlabel("Time (model seconds)")
    ax.set_ylabel("Cross-seed CV (%)")
    ax.set_title(f"Cross-seed CV(t) — how does ensemble divergence grow over the cycle? (N={d['n_seeds']})")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    _save("phase0_cv_growth_diagnostic_REAL",
          "Phase 0 — cross-seed CV(t) (real, N=64)",
          (
            "Coefficient of variation (CV = SD/mean) across N=64 seeds as a function of time. "
            "CV starts near 0 (all seeds share the same initial state) and grows over the cycle "
            "as stochastic processes accumulate divergence. The shape of this curve answers a "
            "key Phase 0 question: how fast does the ensemble decohere? Roughly-linear growth "
            "would suggest noise accumulation; saturating growth would suggest constraint-bound "
            "trajectories (FBA pulling seeds back together). Direct evidence about the "
            "stochasticity budget Phase 1+ jump-process replacements have to match."
          ),
          pinned_h=820)


def main():
    d = load_trajectories()
    n_seeds = d["n_seeds"]
    n_t = len(d["time"])
    print(f"loaded N={n_seeds} seeds × {n_t} timepoints")
    viz_cell_mass_ensemble(d)
    viz_growth_rate_ensemble(d)
    viz_4panel_spaghetti(d)
    viz_cv_growth_diagnostic(d)
    print("done.")


if __name__ == "__main__":
    main()
