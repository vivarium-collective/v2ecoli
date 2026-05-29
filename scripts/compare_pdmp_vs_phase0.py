"""Compare PDMP-Phase-1 single-run trajectory against the Phase-0 kFBA ensemble.

The pdmp-01 acceptance gate is a W₂-distance comparison of interface
statistics from the PDMP-modified composite vs. the Phase-0 reference
ensemble (kFBA-Metabolism baseline, 3 conditions × N=64 replicates).
This script lands the first half of that gate for the M9-glucose
condition: it pulls every committed Phase-0 trajectory under
`.pbg/runs/phase0-traj/seed_*`, computes per-timepoint mean ± std of
cell_mass and dry_mass, runs one fresh PDMP composite for the same
duration, and saves a 4-panel matplotlib chart + summary JSON.

Usage:
    .venv/bin/python scripts/compare_pdmp_vs_phase0.py
        [--duration 600] [--sample-every 5]

Writes:
    .pbg/runs/pdmp-vs-phase0/summary.json
    reports/figures/pdmp-01/pdmp_vs_phase0.html
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys
import time
import warnings
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")


# -------------------------- Phase-0 ensemble load ---------------------------

def load_phase0_ensemble(root: Path = Path(".pbg/runs/phase0-traj")):
    """Return (time, cell_mass[N,T], dry_mass[N,T])."""
    seeds = sorted(root.glob("seed_*"))
    if not seeds:
        raise FileNotFoundError(f"No Phase-0 trajectories under {root}")
    rows_cell, rows_dry, t_ref = [], [], None
    for seed_dir in seeds:
        traj_path = seed_dir / "trajectory.json"
        if not traj_path.is_file():
            continue
        traj = json.loads(traj_path.read_text())
        t = traj.get("time")
        cm = traj.get("listeners.mass.cell_mass")
        dm = traj.get("listeners.mass.dry_mass")
        if not (t and cm and dm):
            continue
        if t_ref is None:
            t_ref = np.asarray(t, dtype=float)
            ref_len = len(t_ref)
        if len(cm) != ref_len:
            continue   # don't try to align variable-length trajectories
        rows_cell.append(cm)
        rows_dry.append(dm)
    if not rows_cell:
        raise RuntimeError(f"No usable Phase-0 trajectories in {root}")
    return (
        t_ref,
        np.asarray(rows_cell, dtype=float),
        np.asarray(rows_dry, dtype=float),
    )


# -------------------------- PDMP run + capture ------------------------------

def run_pdmp(
    duration_s: int,
    sample_every_s: int,
    with_ref_growth: bool = False,
    ref_growth_flux_source: str = "proportional",
):
    """Run the millard_pdmp_baseline composite and sample at sample_every_s."""
    from v2ecoli import build_composite

    t0 = time.perf_counter()
    c = build_composite(
        "millard_pdmp_baseline",
        with_ref_growth=with_ref_growth,
        ref_growth_flux_source=ref_growth_flux_source,
    )
    build_wall = time.perf_counter() - t0
    print(f"  PDMP build: {build_wall:.1f}s", flush=True)

    times, cell_masses, dry_masses = [], [], []
    sim_time = 0
    t_run = time.perf_counter()
    while sim_time < duration_s:
        c.run(sample_every_s)
        sim_time += sample_every_s
        mass = (c.state["agents"]["0"].get("listeners") or {}).get("mass") or {}
        times.append(sim_time)
        cell_masses.append(float(mass.get("cell_mass") or 0.0))
        dry_masses.append(float(mass.get("dry_mass") or 0.0))
    run_wall = time.perf_counter() - t_run
    print(f"  PDMP run wall: {run_wall:.1f}s for {duration_s}s sim", flush=True)
    return (
        np.asarray(times, dtype=float),
        np.asarray(cell_masses, dtype=float),
        np.asarray(dry_masses, dtype=float),
        build_wall,
        run_wall,
    )


# -------------------------- Viz + summary -----------------------------------

def fig_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_viz(t_p0, cm_p0, dm_p0, t_pdmp, cm_pdmp, dm_pdmp,
             build_wall, run_wall, n_p0):
    """4-panel matplotlib comparison + z-score panel + summary."""
    # Align Phase-0 time axis (assumes uniform) to nearest PDMP samples for
    # the z-score panel; for the mass panels we just overlay.
    cm_mean = cm_p0.mean(axis=0)
    cm_std = cm_p0.std(axis=0)
    dm_mean = dm_p0.mean(axis=0)
    dm_std = dm_p0.std(axis=0)

    # For z-score interpolate Phase-0 stats to PDMP sample times.
    cm_p0_mean_at_pdmp = np.interp(t_pdmp, t_p0, cm_mean)
    cm_p0_std_at_pdmp = np.interp(t_pdmp, t_p0, cm_std)
    dm_p0_mean_at_pdmp = np.interp(t_pdmp, t_p0, dm_mean)
    dm_p0_std_at_pdmp = np.interp(t_pdmp, t_p0, dm_std)

    cm_z = (cm_pdmp - cm_p0_mean_at_pdmp) / np.where(
        cm_p0_std_at_pdmp > 0, cm_p0_std_at_pdmp, np.inf)
    dm_z = (dm_pdmp - dm_p0_mean_at_pdmp) / np.where(
        dm_p0_std_at_pdmp > 0, dm_p0_std_at_pdmp, np.inf)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.rcParams.update({"font.size": 10})

    # 1) cell_mass overlay
    ax = axes[0, 0]
    ax.fill_between(t_p0, cm_mean - cm_std, cm_mean + cm_std,
                    alpha=0.25, color="#1f77b4",
                    label=f"Phase-0 kFBA (N={n_p0}, ±1σ)")
    ax.plot(t_p0, cm_mean, color="#1f77b4", lw=1.2)
    ax.plot(t_pdmp, cm_pdmp, color="#d62728", lw=1.5,
            label="PDMP-Phase-1 (single rep)")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("cell_mass (fg)")
    ax.set_title("cell_mass(t) — PDMP vs Phase-0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2) dry_mass overlay
    ax = axes[0, 1]
    ax.fill_between(t_p0, dm_mean - dm_std, dm_mean + dm_std,
                    alpha=0.25, color="#1f77b4",
                    label=f"Phase-0 kFBA (N={n_p0}, ±1σ)")
    ax.plot(t_p0, dm_mean, color="#1f77b4", lw=1.2)
    ax.plot(t_pdmp, dm_pdmp, color="#d62728", lw=1.5,
            label="PDMP-Phase-1 (single rep)")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("dry_mass (fg)")
    ax.set_title("dry_mass(t) — PDMP vs Phase-0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3) cell_mass z-score over time
    ax = axes[1, 0]
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="gray", lw=0.5, ls="--")
    ax.axhline(-1, color="gray", lw=0.5, ls="--")
    ax.axhline(2, color="orange", lw=0.5, ls="--")
    ax.axhline(-2, color="orange", lw=0.5, ls="--")
    ax.plot(t_pdmp, cm_z, color="#d62728", lw=1.2)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("z-score (cell_mass)")
    ax.set_title("cell_mass z-score vs Phase-0 ensemble")
    ax.grid(alpha=0.3)

    # 4) dry_mass z-score over time
    ax = axes[1, 1]
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="gray", lw=0.5, ls="--")
    ax.axhline(-1, color="gray", lw=0.5, ls="--")
    ax.axhline(2, color="orange", lw=0.5, ls="--")
    ax.axhline(-2, color="orange", lw=0.5, ls="--")
    ax.plot(t_pdmp, dm_z, color="#d62728", lw=1.2)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("z-score (dry_mass)")
    ax.set_title("dry_mass z-score vs Phase-0 ensemble")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"PDMP-Phase-1 vs Phase-0 kFBA ensemble  ·  "
        f"M9-glucose, single replicate  ·  "
        f"build={build_wall:.1f}s  run={run_wall:.1f}s",
        fontsize=11
    )
    fig.tight_layout()

    cm_max_abs_z = float(np.max(np.abs(cm_z[np.isfinite(cm_z)]))) if cm_z.size else float("nan")
    dm_max_abs_z = float(np.max(np.abs(dm_z[np.isfinite(dm_z)]))) if dm_z.size else float("nan")
    return fig, {
        "n_phase0": int(n_p0),
        "t_pdmp_first": float(t_pdmp[0]) if t_pdmp.size else None,
        "t_pdmp_last": float(t_pdmp[-1]) if t_pdmp.size else None,
        "cm_max_abs_z": cm_max_abs_z,
        "dm_max_abs_z": dm_max_abs_z,
        "cm_final_pdmp": float(cm_pdmp[-1]) if cm_pdmp.size else None,
        "cm_final_p0_mean": float(cm_mean[-1]) if cm_mean.size else None,
        "cm_final_p0_std": float(cm_std[-1]) if cm_std.size else None,
        "dm_final_pdmp": float(dm_pdmp[-1]) if dm_pdmp.size else None,
        "dm_final_p0_mean": float(dm_mean[-1]) if dm_mean.size else None,
        "dm_final_p0_std": float(dm_std[-1]) if dm_std.size else None,
    }


def write_html(out_path: Path, fig_data_uri: str, summary: dict,
               build_wall: float, run_wall: float):
    cm_z_color = "var(--string)" if summary["cm_max_abs_z"] <= 2.0 else "#b91c1c"
    dm_z_color = "var(--string)" if summary["dm_max_abs_z"] <= 2.0 else "#b91c1c"
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>PDMP-Phase-1 vs Phase-0 kFBA ensemble</title>
<link rel="stylesheet" href="../../assets/style.css">
<style>
  body {{ margin: 0; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 22px; }}
  table.cmp {{ width: 100%; border-collapse: collapse; margin: 14px 0; }}
  table.cmp th, table.cmp td {{ padding: 7px 12px; border: 1px solid var(--border); }}
  table.cmp th {{ background: rgba(0,0,0,0.03); }}
  table.cmp td.num {{ text-align: right; font-variant-numeric: tabular-nums;
                       font-family: ui-monospace, Menlo, monospace; }}
  img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 4px; }}
</style></head>
<body>
<header>
  <h1>PDMP-Phase-1 vs Phase-0 kFBA ensemble</h1>
  <p class="subtitle">M9-glucose · single PDMP replicate vs Phase-0 N={summary["n_phase0"]} ·
     auto-generated {time.strftime("%Y-%m-%d")}</p>
</header>
<main>
<p>
This is the pdmp-01 acceptance-gate viz. PDMP-Phase-1 substitutes
v2ecoli's kFBA Metabolism (multi-objective LP via
<code>wholecell.utils.modular_fba</code>) with the Millard 2017 kinetic
ODE wrapped in <code>MillardPDMPMetabolism</code> (this PR). The chart
overlays one PDMP-Phase-1 replicate on the Phase-0 kFBA reference
ensemble for cell_mass(t) and dry_mass(t), then reports a per-timepoint
z-score for each. |z| ≤ 1 = within ensemble's 68% band; |z| ≤ 2 = within
95%; |z| &gt; 2 flags drift.
</p>
<table class="cmp">
  <tr><th>quantity</th><th>PDMP final</th><th>Phase-0 mean ± σ (final)</th><th>max |z| over run</th></tr>
  <tr><td>cell_mass (fg)</td>
      <td class="num">{summary["cm_final_pdmp"]:.2f}</td>
      <td class="num">{summary["cm_final_p0_mean"]:.2f} ± {summary["cm_final_p0_std"]:.2f}</td>
      <td class="num" style="color:{cm_z_color};font-weight:600">{summary["cm_max_abs_z"]:.2f}</td></tr>
  <tr><td>dry_mass (fg)</td>
      <td class="num">{summary["dm_final_pdmp"]:.2f}</td>
      <td class="num">{summary["dm_final_p0_mean"]:.2f} ± {summary["dm_final_p0_std"]:.2f}</td>
      <td class="num" style="color:{dm_z_color};font-weight:600">{summary["dm_max_abs_z"]:.2f}</td></tr>
</table>
<p class="ref">PDMP run: build wall {build_wall:.1f}s, run wall {run_wall:.1f}s for
{int(summary["t_pdmp_last"])} simulated seconds. Phase-0 ensemble loaded from
<code>.pbg/runs/phase0-traj/seed_*</code>.</p>
<img src="{fig_data_uri}" alt="PDMP vs Phase-0 comparison panel">

<p class="ref" style="margin-top:16px">
Caveat: this is a single PDMP replicate vs. an N={summary["n_phase0"]} reference. A proper
W₂ gate requires PDMP_N ≥ a few replicates; once the JAX backend (task #17/#20)
or a parallelised basico runner lands, this script extends to a real
Wasserstein-2 computation per observable.
</p>
</main>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


# -------------------------- main --------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=600,
                   help="Simulated seconds for the PDMP run.")
    p.add_argument("--sample-every", type=int, default=5,
                   help="PDMP sampling interval in sim-seconds.")
    p.add_argument("--phase0-root", default=".pbg/runs/phase0-traj")
    p.add_argument("--out-dir", default=".pbg/runs/pdmp-vs-phase0")
    p.add_argument("--with-ref-growth", action="store_true",
                   help="Enable the reference growth driver in the PDMP composite.")
    p.add_argument(
        "--ref-growth-flux-source", default="proportional",
        choices=["proportional", "measured_kfba", "consumption_matched"],
        help=("Driver flux mode (only used with --with-ref-growth). "
              "Default 'proportional' for backward compatibility."),
    )
    p.add_argument(
        "--viz-out",
        default="reports/figures/pdmp-01/pdmp_vs_phase0.html")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_out = REPO_ROOT / args.viz_out

    print(f"Loading Phase-0 reference from {args.phase0_root}...")
    t_p0, cm_p0, dm_p0 = load_phase0_ensemble(Path(args.phase0_root))
    print(f"  Phase-0: N={cm_p0.shape[0]} replicates × {cm_p0.shape[1]} timepoints "
          f"({t_p0[0]:.0f}..{t_p0[-1]:.0f}s)")

    feat_tag = ""
    if args.with_ref_growth:
        feat_tag = f" (+ref_growth_driver:{args.ref_growth_flux_source})"
    print(f"\nRunning PDMP{feat_tag} for {args.duration}s, "
          f"sample every {args.sample_every}s...")
    t_pdmp, cm_pdmp, dm_pdmp, build_wall, run_wall = run_pdmp(
        args.duration, args.sample_every,
        with_ref_growth=args.with_ref_growth,
        ref_growth_flux_source=args.ref_growth_flux_source,
    )

    fig, summary = make_viz(
        t_p0, cm_p0, dm_p0, t_pdmp, cm_pdmp, dm_pdmp,
        build_wall, run_wall, cm_p0.shape[0])

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        **summary,
        "build_wall_s": build_wall,
        "run_wall_s": run_wall,
        "duration_s": args.duration,
        "sample_every_s": args.sample_every,
    }, indent=2))
    print(f"\nWrote {summary_path}")

    data_uri = fig_to_data_uri(fig)
    viz_out.parent.mkdir(parents=True, exist_ok=True)
    write_html(viz_out, data_uri, summary, build_wall, run_wall)
    print(f"Wrote {viz_out}")
    print(f"\nSummary: cell_max|z| = {summary['cm_max_abs_z']:.2f}, "
          f"dry_max|z| = {summary['dm_max_abs_z']:.2f}")


if __name__ == "__main__":
    main()
