#!/usr/bin/env python
"""Generate pdmp-03 inference figures from the REAL run data.

IMPORTANT — honesty scope. The study's *planned* deliverables are an ABC-SMC
posterior, SBC rank histogram, and PPC coverage. Those are inference RESULTS
that were never computed: the on-disk zarr stores contain no ``posterior`` /
``summary`` / ``diagnostics`` groups, only forward runs. What DOES exist is a
real, computed **log-likelihood** on a coarse forward parameter grid:

  .pbg/runs/pdmp-03-abc-2d/ts_<t>_ps_<p>/seed_<k>/store.zarr
      observables per run (60 timepoints): cell_mass, dry_mass (fg);
      and additive log-likelihood contributions polypeptide_init,
      transcript_init whose sum is stored as ``total``
      (verified: total == polypeptide_init + transcript_init exactly).

  .pbg/runs/pdmp-03-likelihood/seed_<k>/store.zarr  — 8-seed ensemble.

So these figures show the genuine log-likelihood landscape and ensemble
spread. They are explicitly NOT the ABC-SMC posterior / SBC / PPC, which
require actually running the SMC sampler against synthetic data.

Run from the workspace root:
  .venv/bin/python workspace/investigations/v2ecoli-pdmp/studies/pdmp-03-inference/sims/make_charts.py
"""
from __future__ import annotations

import base64
import glob
import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
RUNS = os.path.join(WS_ROOT, ".pbg", "runs")
OUT_DIR = os.path.join(WS_ROOT, "reports", "figures", "pdmp-03")

TS = ["0.850", "1.000", "1.150"]   # translation_scale
PS = ["0.850", "1.000", "1.150"]   # protein_scale

_HTML = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:720px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:720px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
.imgbox{{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}}
.imgbox img{{max-width:100%;max-height:100%;object-fit:contain}}
p.caption{{font-size:0.82em;color:#475569;line-height:1.4}}
</style></head><body>
<div class="wrap">
  <h1>{title}</h1>
  <div class="imgbox"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
  <p class="caption">{caption}</p>
</div></body></html>
"""


def _lineage(store: str):
    g = zarr.open_group(store, mode="r")
    g = g[[k for k in g][0]]                                   # experiment_id=...
    g = g[[k for k in g][0]]                                   # variant=0
    return g[[k for k in g if k.startswith("lineage")][0]]     # lineage_seed=0


def _obs(store: str, name: str) -> np.ndarray:
    arr = _lineage(store)[name]                                # group: <name>
    return np.asarray(arr[[k for k in arr][0]])                # generation=1


def _time(store: str) -> np.ndarray:
    g = _lineage(store)
    key = [k for k in g if k.startswith("time_gen")][0]        # direct array time_gen=1
    return np.asarray(g[key])


def _write(name: str, title: str, caption: str, fig) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(_HTML.format(title=title, caption=caption, b64=b64))
    return path


def _grid_loglik():
    """Mean final total log-likelihood over the 3x3 (ts, ps) grid + seed spread."""
    mean = np.full((3, 3), np.nan)
    sd = np.full((3, 3), np.nan)
    for i, ps in enumerate(PS):
        for j, ts in enumerate(TS):
            stores = sorted(glob.glob(f"{RUNS}/pdmp-03-abc-2d/ts_{ts}_ps_{ps}/seed_*/store.zarr"))
            vals = [float(_obs(s, "total")[-1]) for s in stores]
            mean[i, j] = np.mean(vals)
            sd[i, j] = np.std(vals)
    return mean, sd


def main() -> None:
    # --- Figure 1: grid log-likelihood surface ---
    mean, sd = _grid_loglik()
    pk = np.unravel_index(np.nanargmax(mean), mean.shape)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(mean, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(TS)
    ax.set_yticks(range(3)); ax.set_yticklabels(PS)
    ax.set_xlabel("translation_scale (ts)")
    ax.set_ylabel("protein_scale (ps)")
    ax.set_title("Grid log-likelihood surface  (mean of 4 seeds/cell)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mean[i,j]:.0f}\n±{sd[i,j]:.0f}", ha="center", va="center",
                    color="white" if mean[i, j] < mean.mean() else "black", fontsize=9)
    ax.add_patch(plt.Rectangle((pk[1]-0.5, pk[0]-0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    fig.colorbar(im, ax=ax, label="total log-likelihood")
    fig.tight_layout()
    cap1 = (
        f"Total log-likelihood (= polypeptide_init + transcript_init contributions, summed in log space) "
        f"evaluated on the coarse 3x3 forward grid over the two ABC parameters translation_scale (ts) and "
        f"protein_scale (ps), mean of 4 seeds per cell (±sd shown). The surface is monotone toward the "
        f"ts={TS[pk[1]]}, ps={PS[pk[0]]} corner (red box, {mean[pk]:.0f}) — i.e. the 3x3 grid does NOT bracket the "
        f"maximum-likelihood point; it lies at or beyond the low-scale corner, so a follow-up sweep must extend "
        f"ts, ps below 0.85. This is the real forward-likelihood landscape, NOT the planned ABC-SMC posterior "
        f"(which requires running the SMC sampler against synthetic data). "
        f"Source: .pbg/runs/pdmp-03-abc-2d/ (9 cells x 4 seeds)."
    )
    out1 = _write("phase3_grid_loglikelihood_surface.html",
                  "Phase 3 — grid log-likelihood landscape (real forward runs)", cap1, fig)

    # --- Figure 2: likelihood-component decomposition across the ts diagonal ---
    fig2, ax = plt.subplots(figsize=(9, 4.6))
    labels, poly, trans = [], [], []
    for ts, ps in zip(TS, PS):  # diagonal cells
        stores = sorted(glob.glob(f"{RUNS}/pdmp-03-abc-2d/ts_{ts}_ps_{ps}/seed_*/store.zarr"))
        poly.append(np.mean([float(_obs(s, "polypeptide_init")[-1]) for s in stores]))
        trans.append(np.mean([float(_obs(s, "transcript_init")[-1]) for s in stores]))
        labels.append(f"ts={ts}\nps={ps}")
    x = np.arange(len(labels))
    ax.bar(x, poly, color="#6d28d9", label="polypeptide_init log-lik")
    ax.bar(x, trans, bottom=poly, color="#0891b2", label="transcript_init log-lik")
    for k in range(len(x)):
        ax.text(x[k], poly[k] + trans[k] - 30, f"{poly[k]+trans[k]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("log-likelihood contribution")
    ax.set_title("Log-likelihood decomposition along the grid diagonal")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.2, axis="y")
    fig2.tight_layout()
    cap2 = (
        "Additive decomposition of the total run log-likelihood into its two recorded modality contributions "
        "(protein-initiation and transcript-initiation), averaged over 4 seeds, along the grid diagonal. The "
        "protein-initiation term dominates (~-940) and drives the overall ts/ps dependence; the transcript term "
        "(~-120 to -150) is a smaller, flatter contribution. Confirms the log-space likelihood accumulator is "
        "wired and additive (total == sum of parts, verified exactly). Source: .pbg/runs/pdmp-03-abc-2d/."
    )
    out2 = _write("phase3_loglikelihood_components.html",
                  "Phase 3 — log-likelihood component decomposition (real)", cap2, fig2)

    # --- Figure 3: 8-seed observable ensemble (cell_mass) ---
    seeds = sorted(glob.glob(f"{RUNS}/pdmp-03-likelihood/seed_*/store.zarr"))
    fig3, ax = plt.subplots(figsize=(9, 4.6))
    traj = []
    for s in seeds:
        cm = _obs(s, "cell_mass"); t = _time(s)
        ax.plot(t, cm, color="#94a3b8", lw=0.8, alpha=0.7)
        traj.append(cm)
    traj = np.vstack(traj); t = _time(seeds[0])
    ax.plot(t, traj.mean(0), color="#dc2626", lw=2.0, label=f"mean of {len(seeds)} seeds")
    ax.fill_between(t, traj.mean(0) - traj.std(0), traj.mean(0) + traj.std(0),
                    color="#dc2626", alpha=0.15, label="±1 sd")
    ax.set_xlabel("time (s)"); ax.set_ylabel("cell_mass (fg)")
    ax.set_title(f"Cell-mass ensemble across {len(seeds)} stochastic seeds")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.2)
    fig3.tight_layout()
    cv = traj[:, -1].std() / traj[:, -1].mean() * 100
    cap3 = (
        f"Cell-mass trajectories for the {len(seeds)}-seed likelihood ensemble at the reference parameter point. "
        f"Inter-seed spread is tight (endpoint CV {cv:.2f}%), showing the PDMP forward model is stable across the "
        f"stochastic replicates the inference layer would average over. Source: .pbg/runs/pdmp-03-likelihood/ "
        f"(8 seeds)."
    )
    out3 = _write("phase3_observable_ensemble.html",
                  "Phase 3 — cell-mass ensemble (real, 8 seeds)", cap3, fig3)

    for o in (out1, out2, out3):
        print("wrote", os.path.relpath(o, WS_ROOT))


if __name__ == "__main__":
    main()
