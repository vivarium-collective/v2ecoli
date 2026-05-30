"""Phase 3 sprint 12 — count-observable ABC re-evaluation.

Sprint 11 found that the pipeline's emitted log-likelihood is
self-calibrated under the model's own rates, weakening identifiability.
Sprint 12 evaluates a count-based ABC distance instead: SSE between
proposed and observed per-tick TOTAL INITIATION COUNTS
(``total_rna_init`` and ``did_initialize``) — the actual observable
realizations of the jump processes.

If count-based distance distinguishes the truth more cleanly than the
log-likelihood distance from sprint 10, that confirms the
self-calibration hypothesis. If they're similar, the trajectory-level
RNG variance is what drives both.

Pipeline:

  1. Re-emit sprint-4's reference (N=8 at scale=1.0) with the new
     count paths in EMIT_PATHS, persisted at
     ``.pbg/runs/pdmp-03-v12-likelihood/seed_<NN>/store.zarr/``.
  2. Re-emit sprint-10's 3×3 grid (Ts, Ps) × N=4 with the same paths,
     persisted under ``.pbg/runs/pdmp-03-v12-abc-2d/``.
  3. Compute SSE per cell on the count observables AND on the
     log-likelihood observables.
  4. Compare: truth-vs-next-nearest ratios under each metric.

Output: ``reports/figures/pdmp-03/abc_count_metric.html``.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase3_abc_smc_stub import _drop_trailing_nan, load_ensemble_at
from scripts.phase3_likelihood_xarray_ensemble import run_one


V12_LIKELIHOOD_ROOT = Path(".pbg/runs/pdmp-03-v12-likelihood")
V12_ABC_2D_ROOT = Path(".pbg/runs/pdmp-03-v12-abc-2d")


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def collect_provenance():
    try:
        sha = _git("rev-parse", "HEAD")
    except Exception:
        sha = "(unknown)"
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:
        branch = "(unknown)"
    try:
        subprocess.run(["git", "update-index", "--really-refresh"],
                       check=False, capture_output=True)
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"],
                           capture_output=True)
        dirty = r.returncode != 0
    except Exception:
        dirty = False
    return {"sha": sha, "short": sha[:8] if sha != "(unknown)" else sha,
            "branch": branch, "dirty": dirty,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version()}


def run_ref_ensemble(n: int, duration: int, chunk: int):
    print(f"Running v12 reference: N={n} × {duration}s, scale=1.0",
          flush=True)
    V12_LIKELIHOOD_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in range(n):
        cached = V12_LIKELIHOOD_ROOT / f"seed_{seed:02d}" / "store.zarr"
        if cached.is_dir():
            print(f"  seed={seed:02d}: cached ✓", flush=True)
            continue
        s = run_one(seed, duration_s=duration, chunk=chunk,
                    transcript_scale=1.0, polypeptide_scale=1.0,
                    out_root=V12_LIKELIHOOD_ROOT)
        if "error" in s:
            print(f"  seed={seed:02d}: ERROR {s['error']}")
        else:
            print(f"  seed={seed:02d}: wall={s['wall_s']:.1f}s",
                  flush=True)


def run_grid(ts_grid, ps_grid, n: int, duration: int, chunk: int):
    print(f"\nRunning v12 2D grid: {len(ts_grid)}×{len(ps_grid)}, "
          f"N={n}/cell")
    for ts in ts_grid:
        for ps in ps_grid:
            out_dir = V12_ABC_2D_ROOT / f"ts_{ts:.3f}_ps_{ps:.3f}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"  (ts={ts:.3f}, ps={ps:.3f})", flush=True)
            for seed in range(n):
                cached = out_dir / f"seed_{seed:02d}" / "store.zarr"
                if cached.is_dir():
                    print(f"    seed={seed:02d}: cached ✓", flush=True)
                    continue
                s = run_one(seed, duration_s=duration, chunk=chunk,
                            transcript_scale=ts, polypeptide_scale=ps,
                            out_root=out_dir)
                if "error" in s:
                    print(f"    seed={seed:02d}: ERROR {s['error']}")
                else:
                    print(f"    seed={seed:02d}: wall={s['wall_s']:.1f}s",
                          flush=True)


def cell_distances(ts, ps, n, observed_means, t_observed, channels):
    out_dir = V12_ABC_2D_ROOT / f"ts_{ts:.3f}_ps_{ps:.3f}"
    if not out_dir.is_dir():
        return {ch: None for ch in channels}
    ds = _drop_trailing_nan(load_ensemble_at(out_dir, n))
    out = {}
    for ch in channels:
        if ch not in ds.data_vars or ch not in observed_means:
            out[ch] = None
            continue
        mean_traj = ds[ch].mean(dim="replicate").values
        t_s = ds["time"].values
        common_t = np.intersect1d(t_s, t_observed)
        obs_aligned = np.interp(common_t, t_observed,
                                observed_means[ch])
        mean_aligned = np.interp(common_t, t_s, mean_traj)
        out[ch] = float(np.sum((obs_aligned - mean_aligned) ** 2))
    return out


def make_figure(ts_grid, ps_grid, d_total, d_count_t, d_count_p,
                truth_ts=1.0, truth_ps=1.0) -> str:
    """Side-by-side heatmaps comparing the two metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    extent = [ps_grid[0] - 0.075, ps_grid[-1] + 0.075,
              ts_grid[0] - 0.075, ts_grid[-1] + 0.075]

    def render(ax, mat, title):
        vmax = mat.max() * 1.05
        im = ax.imshow(mat, cmap="viridis_r", origin="lower",
                       extent=extent, aspect="auto", vmin=0, vmax=vmax)
        n_ts, n_ps = mat.shape
        for i in range(n_ts):
            for j in range(n_ps):
                v = mat[i, j]
                ax.text(ps_grid[j], ts_grid[i], f"{v:.0f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > vmax * 0.5 else "black")
        ax.scatter([truth_ps], [truth_ts], marker="*", s=400,
                   color="#fde68a", edgecolors="k", linewidths=2,
                   zorder=10)
        ax.set_xticks(ps_grid)
        ax.set_yticks(ts_grid)
        ax.set_xlabel("polypeptide_init_prob_scale (Ps)")
        ax.set_ylabel("transcript_init_prob_scale (Ts)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="SSE")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

    render(axes[0], d_total,
           "Sprint 10 — log-likelihood total (aggregate)")
    render(axes[1], d_count_t,
           "Sprint 12 — count: total_rna_init")
    render(axes[2], d_count_p,
           "Sprint 12 — count: ribosome did_initialize")

    fig.suptitle(
        "Phase-3 sprint 12: count-based ABC distance vs "
        "log-likelihood-based",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path, ts_grid, ps_grid, distances, ratios,
               plot_uri) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    rows = []
    for i, ts in enumerate(ts_grid):
        for j, ps in enumerate(ps_grid):
            star = " ⭐" if ts == 1.0 and ps == 1.0 else ""
            rows.append(
                f"<tr><td class='num'>{ts:.3f}{star}</td>"
                f"<td class='num'>{ps:.3f}</td>"
                f"<td class='num'>{distances['total'][i, j]:.0f}</td>"
                f"<td class='num'>{distances['rna'][i, j]:.0f}</td>"
                f"<td class='num'>{distances['ribosome'][i, j]:.0f}</td></tr>")
    table_rows = "\n".join(rows)

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 3 sprint 12 — count vs log-likelihood ABC</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1600px; margin: 24px auto; padding: 0 18px;
         line-height: 1.55; }}
  h1 {{ margin: 0 0 6px 0; color:#1e3a8a; }}
  h2 {{ margin-top: 24px; border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px; color:#1e3a8a; }}
  .provenance {{ background:#f8fafc; border:1px solid #e2e8f0;
                 border-radius:8px; padding:10px 14px; margin:14px 0 20px;
                 font-size:0.85em; }}
  .provenance dt {{ display:inline-block; min-width:110px; color:#475569;
                    font-weight:600; }}
  .provenance dd {{ display:inline; margin:0;
                    font-family: ui-monospace, Menlo, monospace; }}
  .provenance .row {{ margin: 1px 0; }}
  table {{ border-collapse: collapse; margin: 12px 0; width: 100%;
           font-size: 0.92em; }}
  th, td {{ padding: 6px 12px; border: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; text-align: left; }}
  td.num {{ text-align: right;
            font-family: ui-monospace, Menlo, monospace; }}
  img.plot {{ max-width: 100%; border:1px solid #e2e8f0; border-radius:6px; }}
  .takeaway {{ background:#dcfce7; border-left:4px solid #16a34a;
               padding:12px 16px; margin:14px 0; font-size:0.95em; }}
</style>

<h1>Phase 3 sprint 12 — count-based ABC vs log-likelihood-based</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Tests the sprint-11 hypothesis: the pipeline's emitted log-likelihood
  is self-calibrated under the model's own rates, weakening
  identifiability. Switching to SSE on the actual per-tick INITIATION
  COUNTS (an ABC-correct observable) should sharpen it.
</p>

<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{prov['generated']}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="https://github.com/vivarium-collective/v2ecoli/commit/{prov['sha']}"
       style="color:#0369a1;text-decoration:none">{prov['short']}</a>
       &nbsp;<code>{prov['sha']}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{prov['branch']}</dd></div>
  <div class="row"><dt>host</dt><dd>{prov['host']} &nbsp;
    <span style="color:#94a3b8">Python {prov['python']}</span></dd></div>
</div>

<h2>Setup</h2>
<ul>
  <li><strong>Reference:</strong> N=8 sims at (1.0, 1.0) under v12
      EMIT_PATHS at <code>.pbg/runs/pdmp-03-v12-likelihood/</code>.
      The previous sprint-4 reference at
      <code>.pbg/runs/pdmp-03-likelihood/</code> didn't emit the count
      paths so we had to re-run.</li>
  <li><strong>2D grid:</strong> sprint-10's 3×3 (Ts, Ps) × N=4 re-run
      at <code>.pbg/runs/pdmp-03-v12-abc-2d/</code> with the new
      EMIT_PATHS.</li>
  <li><strong>New observables:</strong>
      <code>listeners.rna_synth_prob.total_rna_init</code> (per-tick
      total transcript init events) and
      <code>listeners.ribosome_data.did_initialize</code> (per-tick
      total ribosome activations).</li>
  <li><strong>Distance metric:</strong> SSE between
      proposed-ensemble-mean and observed-ensemble-mean per-tick count
      timeseries (same shape as sprint 7's log-likelihood metric, but
      on the count observables).</li>
</ul>

<h2>Side-by-side: log-likelihood vs count metrics</h2>
<img class="plot" src="{plot_uri}" alt="3 heatmaps: log-likelihood total, count rna_init, count did_initialize">

<h2>Truth identifiability ratios (next-nearest cell / truth)</h2>
<table>
  <tr><th>metric</th><th>truth d</th><th>next-nearest</th><th>ratio</th></tr>
  <tr><td>log-likelihood total (sprint 10)</td>
      <td class="num">{ratios['total']['truth']:.0f}</td>
      <td class="num">{ratios['total']['next']:.0f}</td>
      <td class="num">{ratios['total']['ratio']:.2f}×</td></tr>
  <tr><td>count: total_rna_init</td>
      <td class="num">{ratios['rna']['truth']:.0f}</td>
      <td class="num">{ratios['rna']['next']:.0f}</td>
      <td class="num">{ratios['rna']['ratio']:.2f}×</td></tr>
  <tr><td>count: ribosome did_initialize</td>
      <td class="num">{ratios['ribosome']['truth']:.0f}</td>
      <td class="num">{ratios['ribosome']['next']:.0f}</td>
      <td class="num">{ratios['ribosome']['ratio']:.2f}×</td></tr>
</table>

<h2>Per-cell distance comparison</h2>
<table>
  <tr><th>Ts</th><th>Ps</th>
      <th>log-likelihood d_total</th>
      <th>count d_rna_init</th>
      <th>count d_did_initialize</th></tr>
  {table_rows}
</table>

<h2>Implication</h2>
<p>
  Compare the truth-identifiability ratios above. If the count-based
  metric gives a much higher truth-vs-next-nearest ratio than the
  log-likelihood-based one, the self-calibration hypothesis (sprint 11)
  is confirmed and the architectural answer is "for ABC, distance on
  count observables, not on emitted log-likelihoods". If they're
  similar, the trajectory-level RNG variance dominates both metrics
  and the choice is less critical.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-grid", default="0.85,1.0,1.15")
    ap.add_argument("--ps-grid", default="0.85,1.0,1.15")
    ap.add_argument("--n-ref", type=int, default=8)
    ap.add_argument("--n-per-cell", type=int, default=4)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--skip-sims", action="store_true",
                    help="Skip running sims; load cached zarrs only.")
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-03/abc_count_metric.html"))
    args = ap.parse_args()

    ts_grid = [float(x) for x in args.ts_grid.split(",")]
    ps_grid = [float(x) for x in args.ps_grid.split(",")]

    if not args.skip_sims:
        run_ref_ensemble(args.n_ref, args.duration, args.chunk)
        run_grid(ts_grid, ps_grid, args.n_per_cell, args.duration,
                 args.chunk)

    # Load reference + observed means.
    ref_ds = _drop_trailing_nan(load_ensemble_at(V12_LIKELIHOOD_ROOT,
                                                 args.n_ref))
    print(f"\nRef vars: {list(ref_ds.data_vars)}")

    # Identify channel names. After view_from_emit_paths, the leaf
    # names should be: transcript_init, polypeptide_init, total,
    # total_rna_init, did_initialize, cell_mass, dry_mass.
    channels_to_score = {
        "total": "total",
        "rna": "total_rna_init",
        "ribosome": "did_initialize",
    }

    observed_means = {}
    t_observed = ref_ds["time"].values
    for name in channels_to_score.values():
        if name in ref_ds.data_vars:
            observed_means[name] = ref_ds[name].mean(dim="replicate").values
        else:
            print(f"WARN: channel {name!r} missing from reference")

    # Compute per-cell distances on every channel.
    n_ts, n_ps = len(ts_grid), len(ps_grid)
    dist_grids = {
        key: np.zeros((n_ts, n_ps))
        for key in channels_to_score
    }
    for i, ts in enumerate(ts_grid):
        for j, ps in enumerate(ps_grid):
            d = cell_distances(ts, ps, args.n_per_cell, observed_means,
                                t_observed, list(channels_to_score.values()))
            for key, name in channels_to_score.items():
                val = d.get(name)
                dist_grids[key][i, j] = (val if val is not None
                                         else np.nan)

    # Print + compute identifiability ratios.
    print("\nPer-cell distances:")
    for i, ts in enumerate(ts_grid):
        for j, ps in enumerate(ps_grid):
            tag = " ⭐" if ts == 1.0 and ps == 1.0 else ""
            row = ", ".join(f"{k}={dist_grids[k][i, j]:.0f}"
                            for k in dist_grids)
            print(f"  (ts={ts:.2f}, ps={ps:.2f}): {row}{tag}")

    # Truth is at (1.0, 1.0) = (i=1, j=1).
    ratios = {}
    for key, grid in dist_grids.items():
        truth = grid[1, 1]
        # Mask truth cell to find next-nearest.
        masked = grid.copy()
        masked[1, 1] = np.inf
        next_nearest = float(np.nanmin(masked))
        ratios[key] = {
            "truth": float(truth),
            "next": next_nearest,
            "ratio": next_nearest / truth if truth > 0 else float("inf"),
        }
        print(f"\n{key} metric: truth={truth:.0f}, "
              f"next-nearest={next_nearest:.0f}, "
              f"ratio={ratios[key]['ratio']:.2f}×")

    plot_uri = make_figure(ts_grid, ps_grid,
                           dist_grids["total"],
                           dist_grids["rna"],
                           dist_grids["ribosome"])
    write_html(args.out, ts_grid, ps_grid, dist_grids, ratios, plot_uri)


if __name__ == "__main__":
    main()
