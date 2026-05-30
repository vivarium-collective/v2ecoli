"""Phase-2 acceptance gate — ensemble W₂ validation.

Runs the v2ecoli baseline (kFBA) composite and the PDMP+consumption_matched
composite for N replicates each, in both ``discrete`` (legacy multinomial)
and ``poisson`` (Phase-2 jump-process) initiation-sampler modes. Computes
per-timepoint Wasserstein-2 distance between the PDMP and baseline
ensembles for each mode, and renders one HTML asking the acceptance-gate
question:

  Does the Phase-2 Poisson sampler bring the PDMP ensemble closer to
  the baseline (kFBA) reference ensemble than the legacy discrete
  sampler did?

If yes (W₂_poisson < W₂_discrete) the jump-process kinetics are not just
architecturally cleaner for Phase-3 likelihoods but quantitatively better
estimators of the reference trajectory shape.

Usage:
    .venv/bin/python scripts/ensemble_validation.py
        [--n 8] [--duration 600]
        [--out reports/figures/pdmp-02/ensemble_validation.html]
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

from scripts.compare_pdmp_vs_baseline import collect_provenance  # noqa: E402


def run_one(
    composite_name: str,
    seed: int,
    duration_s: int,
    sample_every_s: int,
    **build_kwargs,
) -> dict:
    """Run a single replicate, sample cell_mass + dry_mass per chunk."""
    from v2ecoli import build_composite

    c = build_composite(composite_name, seed=seed, **build_kwargs)
    t0 = time.perf_counter()
    cm: list[float] = []
    dm: list[float] = []
    t_axis: list[int] = []
    sim_t = 0
    while sim_t < duration_s:
        chunk = min(sample_every_s, duration_s - sim_t)
        c.run(chunk)
        sim_t += chunk
        mass = (
            c.state.get("agents", {}).get("0", {})
            .get("listeners", {}).get("mass", {})
        )
        cm.append(float(mass.get("cell_mass", 0.0)))
        dm.append(float(mass.get("dry_mass", 0.0)))
        t_axis.append(sim_t)
    wall = time.perf_counter() - t0
    return {
        "seed": seed,
        "wall_s": wall,
        "t": t_axis,
        "cell_mass": cm,
        "dry_mass": dm,
    }


def run_ensemble(
    composite_name: str,
    n: int,
    duration_s: int,
    sample_every_s: int,
    label: str,
    **build_kwargs,
) -> dict:
    print(f"  ensemble {label}: N={n} sims of {duration_s} s...", flush=True)
    reps = []
    t_start = time.perf_counter()
    for k in range(n):
        rep = run_one(composite_name, seed=k,
                      duration_s=duration_s,
                      sample_every_s=sample_every_s,
                      **build_kwargs)
        reps.append(rep)
        print(f"    seed {k}: cm[-1]={rep['cell_mass'][-1]:.2f}, "
              f"dm[-1]={rep['dry_mass'][-1]:.2f}, wall={rep['wall_s']:.1f}s",
              flush=True)
    wall_total = time.perf_counter() - t_start
    cm_grid = np.asarray([r["cell_mass"] for r in reps])  # shape (N, T)
    dm_grid = np.asarray([r["dry_mass"] for r in reps])
    t_axis = np.asarray(reps[0]["t"])
    return {
        "label": label,
        "composite": composite_name,
        "build_kwargs": build_kwargs,
        "n": n,
        "duration_s": duration_s,
        "sample_every_s": sample_every_s,
        "t": t_axis,
        "cell_mass_grid": cm_grid,
        "dry_mass_grid": dm_grid,
        "wall_total_s": wall_total,
    }


def per_timepoint_w2(grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
    """Compute Wasserstein-2 distance per timepoint between two ensembles.

    SciPy's ``wasserstein_distance`` is W₁. We compute W₂ as the
    Euclidean distance between sorted samples (closed-form 1D), since
    that's what the Phase-1 pdmp-vs-Phase-0 report tracked.
    """
    n_t = grid_a.shape[1]
    w2 = np.zeros(n_t)
    for t in range(n_t):
        a = np.sort(grid_a[:, t])
        b = np.sort(grid_b[:, t])
        # equal-N sorted-samples form
        w2[t] = float(np.sqrt(np.mean((a - b) ** 2)))
    return w2


def make_viz(results: dict, w2_discrete_cm, w2_poisson_cm,
             w2_discrete_dm, w2_poisson_dm):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        "Phase-2 ensemble validation — Poisson tau-leap vs legacy discrete",
        fontsize=14, fontweight="bold",
    )

    # ----- Panel: cell_mass ensembles, discrete -----
    ax = axes[0, 0]
    _plot_ensemble(ax, results["baseline_discrete"], "baseline kFBA",
                   "#3b82f6", "cell_mass_grid")
    _plot_ensemble(ax, results["pdmp_discrete"], "PDMP+cm",
                   "#a855f7", "cell_mass_grid")
    ax.set_title("Discrete (legacy multinomial)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cell_mass (fg)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # ----- Panel: cell_mass ensembles, poisson -----
    ax = axes[0, 1]
    _plot_ensemble(ax, results["baseline_poisson"], "baseline kFBA",
                   "#3b82f6", "cell_mass_grid")
    _plot_ensemble(ax, results["pdmp_poisson"], "PDMP+cm",
                   "#a855f7", "cell_mass_grid")
    ax.set_title("Poisson (Phase-2 tau-leap)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cell_mass (fg)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # ----- Panel: W₂(t) cell_mass -----
    ax = axes[1, 0]
    t = results["baseline_discrete"]["t"]
    ax.plot(t, w2_discrete_cm, color="#1e3a8a", lw=1.6,
            label=f"discrete (peak {w2_discrete_cm.max():.1f})")
    ax.plot(t, w2_poisson_cm, color="#7c3aed", lw=1.6,
            label=f"poisson (peak {w2_poisson_cm.max():.1f})")
    ax.fill_between(t, 0, w2_discrete_cm, color="#1e3a8a", alpha=0.1)
    ax.fill_between(t, 0, w2_poisson_cm, color="#7c3aed", alpha=0.1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("W₂ (cell_mass) [fg]")
    ax.set_title("W₂(PDMP, baseline) — cell_mass")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # ----- Panel: W₂(t) dry_mass -----
    ax = axes[1, 1]
    ax.plot(t, w2_discrete_dm, color="#1e3a8a", lw=1.6,
            label=f"discrete (peak {w2_discrete_dm.max():.1f})")
    ax.plot(t, w2_poisson_dm, color="#7c3aed", lw=1.6,
            label=f"poisson (peak {w2_poisson_dm.max():.1f})")
    ax.fill_between(t, 0, w2_discrete_dm, color="#1e3a8a", alpha=0.1)
    ax.fill_between(t, 0, w2_poisson_dm, color="#7c3aed", alpha=0.1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("W₂ (dry_mass) [fg]")
    ax.set_title("W₂(PDMP, baseline) — dry_mass")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig


def _plot_ensemble(ax, ens, label, color, field):
    grid = ens[field]
    t = ens["t"]
    mean = grid.mean(axis=0)
    std = grid.std(axis=0)
    for i in range(grid.shape[0]):
        ax.plot(t, grid[i], color=color, alpha=0.15, lw=0.5)
    ax.plot(t, mean, color=color, lw=1.8, label=f"{label} (mean ± σ, N={grid.shape[0]})")
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(out_path: Path, data_uri: str,
               results: dict,
               w2_discrete_cm, w2_poisson_cm,
               w2_discrete_dm, w2_poisson_dm,
               provenance: dict) -> None:
    cm_dec = w2_discrete_cm[-1] - w2_poisson_cm[-1]
    dm_dec = w2_discrete_dm[-1] - w2_poisson_dm[-1]
    cm_pct = 100.0 * cm_dec / max(1e-9, w2_discrete_cm[-1])
    dm_pct = 100.0 * dm_dec / max(1e-9, w2_discrete_dm[-1])

    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if provenance.get("git_dirty") else ""
    )
    short = provenance.get("git_short", "")
    full_sha = provenance.get("git_sha", "")
    n = results["baseline_discrete"]["n"]
    dur = results["baseline_discrete"]["duration_s"]
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase-2 ensemble validation — {short or 'report'}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1500px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ margin: 0 0 6px 0; }}
  .meta {{ color: #6b7280; font-size: 0.9em; }}
  .meta code {{ background: rgba(0,0,0,0.04); padding: 1px 5px; border-radius: 3px; }}
  .provenance {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                 padding:10px 14px; margin:14px 0 20px; font-size:0.85em;
                 line-height:1.55; }}
  .provenance dt {{ display:inline-block; min-width:120px; color:#475569;
                    font-weight:600; }}
  .provenance dd {{ display:inline; margin:0; font-family: ui-monospace, Menlo, monospace; }}
  .provenance .row {{ margin: 1px 0; }}
  table.summary {{ border-collapse: collapse; margin: 14px 0; font-size: 0.95em; }}
  table.summary th, table.summary td {{ padding: 6px 12px; border: 1px solid #e5e7eb; }}
  table.summary th {{ background: #f3f4f6; font-weight: 600; text-align: left; }}
  table.summary td.num {{ text-align: right; font-variant-numeric: tabular-nums;
                          font-family: ui-monospace, Menlo, monospace; }}
  .verdict {{ background:#ecfeff; border-left:4px solid #06b6d4; padding:12px 16px;
              margin:14px 0; font-size:0.95em; }}
  img {{ max-width: 100%; }}
</style>
<h1>Phase-2 ensemble validation: Poisson tau-leap vs legacy discrete</h1>
<div class="meta">
  N={n} replicates each, {dur} s sim, sample every
  {results['baseline_discrete']['sample_every_s']} s.
</div>

<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{provenance.get('generated_at','')}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="https://github.com/vivarium-collective/v2ecoli/commit/{full_sha}"
        style="color:#0369a1;text-decoration:none">{short}</a> &nbsp;<code>{full_sha}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{provenance.get('git_branch','')}</dd></div>
  <div class="row"><dt>last commit</dt><dd>{provenance.get('git_last_commit_msg','')} — {provenance.get('git_last_commit_author','')} ({provenance.get('git_last_commit_when','')})</dd></div>
  <div class="row"><dt>script</dt><dd>{provenance.get('script','')}</dd></div>
  <div class="row"><dt>host</dt><dd>{provenance.get('host','')} &nbsp; <span style="color:#94a3b8">{provenance.get('platform','')}, Python {provenance.get('python','')}</span></dd></div>
</div>

<table class="summary">
  <tr><th>mode</th>
      <th>baseline cm @ t={dur} (mean ± σ)</th>
      <th>PDMP+cm cm @ t={dur} (mean ± σ)</th>
      <th>W₂(PDMP, baseline) @ t={dur} — cell_mass</th>
      <th>W₂(PDMP, baseline) @ t={dur} — dry_mass</th></tr>
  <tr><td>discrete (legacy)</td>
      <td class="num">{results['baseline_discrete']['cell_mass_grid'][:,-1].mean():.2f} ± {results['baseline_discrete']['cell_mass_grid'][:,-1].std():.2f}</td>
      <td class="num">{results['pdmp_discrete']['cell_mass_grid'][:,-1].mean():.2f} ± {results['pdmp_discrete']['cell_mass_grid'][:,-1].std():.2f}</td>
      <td class="num">{w2_discrete_cm[-1]:.2f} fg</td>
      <td class="num">{w2_discrete_dm[-1]:.2f} fg</td></tr>
  <tr><td>poisson (Phase-2)</td>
      <td class="num">{results['baseline_poisson']['cell_mass_grid'][:,-1].mean():.2f} ± {results['baseline_poisson']['cell_mass_grid'][:,-1].std():.2f}</td>
      <td class="num">{results['pdmp_poisson']['cell_mass_grid'][:,-1].mean():.2f} ± {results['pdmp_poisson']['cell_mass_grid'][:,-1].std():.2f}</td>
      <td class="num">{w2_poisson_cm[-1]:.2f} fg</td>
      <td class="num">{w2_poisson_dm[-1]:.2f} fg</td></tr>
  <tr><td>Δ (poisson − discrete)</td>
      <td class="num"></td>
      <td class="num"></td>
      <td class="num">{-cm_dec:+.2f} fg ({-cm_pct:+.1f}%)</td>
      <td class="num">{-dm_dec:+.2f} fg ({-dm_pct:+.1f}%)</td></tr>
</table>

<div class="verdict">
  <strong>Phase-2 acceptance-gate verdict:</strong>
  Poisson tau-leap {'reduces' if cm_dec > 0 else 'increases'} W₂(PDMP, baseline)
  by {abs(cm_pct):.1f}% on cell_mass and {abs(dm_pct):.1f}% on dry_mass.
  {'This is the quantitative validation that Phase-2 jump-process kinetics aren&apos;t just architecturally cleaner for Phase-3 likelihoods — they also bring the PDMP ensemble closer to the kFBA-reference ensemble.' if cm_dec > 0 else 'The Poisson sampler doesn&apos;t close the ensemble W₂ gap at this N — either N=' + str(n) + ' is too small to see the signal, or the dominant divergence source is elsewhere (e.g. the consumption_matched water rate, the Riccati-failing LQR).'}
</div>

<h2 style="margin-top:24px;">Ensembles + W₂(t)</h2>
<img src="{data_uri}" alt="ensemble validation panels">
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8,
                   help="Replicates per ensemble (default 8).")
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--sample-every", type=int, default=30)
    p.add_argument("--feedback-tau-s", type=float, default=1.0,
                   help="ref_growth_driver feedback smoothing time constant "
                        "(see ref_growth_driver.py:feedback_tau_s). Default "
                        "1.0 reproduces the legacy tight per-tick controller. "
                        "Try 60.0 to let per-tick jump-process variance "
                        "manifest in the PDMP ensemble.")
    p.add_argument("--out",
                   default="reports/figures/pdmp-02/ensemble_validation.html")
    args = p.parse_args()

    results: dict[str, dict] = {}

    for mode in ("discrete", "poisson"):
        print(f"\n=== {mode} samplers ===", flush=True)
        results[f"baseline_{mode}"] = run_ensemble(
            "baseline", args.n, args.duration, args.sample_every,
            f"baseline_{mode}",
            transcript_initiation_mode=mode,
            polypeptide_initiation_mode=mode,
        )
        results[f"pdmp_{mode}"] = run_ensemble(
            "millard_pdmp_baseline", args.n, args.duration, args.sample_every,
            f"pdmp_{mode}",
            with_ref_growth=True,
            ref_growth_flux_source="consumption_matched",
            ref_growth_feedback_tau_s=args.feedback_tau_s,
            transcript_initiation_mode=mode,
            polypeptide_initiation_mode=mode,
        )

    w2_discrete_cm = per_timepoint_w2(
        results["pdmp_discrete"]["cell_mass_grid"],
        results["baseline_discrete"]["cell_mass_grid"],
    )
    w2_poisson_cm = per_timepoint_w2(
        results["pdmp_poisson"]["cell_mass_grid"],
        results["baseline_poisson"]["cell_mass_grid"],
    )
    w2_discrete_dm = per_timepoint_w2(
        results["pdmp_discrete"]["dry_mass_grid"],
        results["baseline_discrete"]["dry_mass_grid"],
    )
    w2_poisson_dm = per_timepoint_w2(
        results["pdmp_poisson"]["dry_mass_grid"],
        results["baseline_poisson"]["dry_mass_grid"],
    )

    provenance = collect_provenance(extra={
        "n_per_ensemble": args.n,
        "duration_s": args.duration,
        "sample_every_s": args.sample_every,
    })

    print("\nRendering...", flush=True)
    fig = make_viz(results, w2_discrete_cm, w2_poisson_cm,
                   w2_discrete_dm, w2_poisson_dm)
    data_uri = fig_to_data_uri(fig)
    write_html(Path(args.out), data_uri, results,
               w2_discrete_cm, w2_poisson_cm,
               w2_discrete_dm, w2_poisson_dm,
               provenance)
    print(f"Wrote viz {args.out}", flush=True)

    short = provenance.get("git_short") or "nogit"
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = Path(args.out).with_name(
        f"ensemble_validation_{stamp}_{short}.html"
    )
    archive.write_bytes(Path(args.out).read_bytes())
    print(f"Wrote archive {archive}", flush=True)


if __name__ == "__main__":
    main()
