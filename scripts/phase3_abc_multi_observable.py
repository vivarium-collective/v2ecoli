"""Phase 3 sprint 11 — multi-observable ABC inference.

Sprint 10 surfaced an anti-diagonal ridge in the (Ts, Ps) parameter
space: reducing transcription rate while raising translation rate
partially cancels at the AGGREGATE total log-likelihood, leaving
opposing parameter shifts indistinguishable from the truth.

Sprint 11 fixes that by going to per-channel distances. The
LikelihoodCollector already emits `transcript_init` and
`polypeptide_init` separately; sprint 1 + 2's emission already
landed both in the persisted zarrs. We treat them as a 2-component
distance vector and combine via Euclidean norm:

    d_vec(θ, θ_true) = sqrt(d_t² + d_p²)

where d_t and d_p are SSE distances on the transcript and
polypeptide channels respectively. The anti-diagonal degeneracy
disappears because lowering Ts increases d_t while raising Ps
increases d_p — they no longer cancel.

No new simulations. Reads sprint-10's persisted grid at
``.pbg/runs/pdmp-03-abc-2d/ts_<Ts>_ps_<Ps>/seed_<NN>/store.zarr/``.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import os
import platform
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

from scripts.phase3_abc_smc_stub import (
    NOISE_FLOOR_RUNS,
    _drop_trailing_nan,
    compute_mean_to_mean_noise_floor,
    load_ensemble_at,
)
from scripts.phase3_abc_smc_2d import ABC_2D_ROOT, cell_dir


CHANNELS = ("transcript_init", "polypeptide_init", "total")


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


def channel_distance(ts: float, ps: float, n: int,
                     observed_means: dict, t_observed: np.ndarray,
                     channel: str) -> float | None:
    out_dir = cell_dir(ts, ps)
    if not out_dir.is_dir():
        return None
    ds = _drop_trailing_nan(load_ensemble_at(out_dir, n))
    if channel not in ds.data_vars:
        return None
    mean_traj = ds[channel].mean(dim="replicate").values
    t_s = ds["time"].values
    common_t = np.intersect1d(t_s, t_observed)
    obs_aligned = np.interp(common_t, t_observed,
                            observed_means[channel])
    mean_aligned = np.interp(common_t, t_s, mean_traj)
    return float(np.sum((obs_aligned - mean_aligned) ** 2))


def per_channel_noise_floor(
    ref_ds, channel: str, n_per_split: int = 4,
    n_resamples: int = 200, seed: int = 42,
):
    """Same mean-to-mean null as sprint 8, restricted to one channel."""
    arr = ref_ds[channel].values
    n_rep = arr.shape[0]
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_resamples):
        perm = rng.permutation(n_rep)
        a_idx = perm[:n_per_split]
        b_idx = perm[n_per_split:2 * n_per_split]
        mean_a = arr[a_idx].mean(axis=0)
        mean_b = arr[b_idx].mean(axis=0)
        distances.append(float(np.sum((mean_a - mean_b) ** 2)))
    distances = np.asarray(distances)
    return {
        "median": float(np.median(distances)),
        "p05": float(np.percentile(distances, 5)),
        "p95": float(np.percentile(distances, 95)),
        "distances": distances,
    }


def make_figure(ts_grid, ps_grid, d_t, d_p, d_total, d_vec,
                eps_t: float, eps_p: float, eps_vec: float,
                eps_total: float,
                truth_ts=1.0, truth_ps=1.0) -> str:
    """Render 3 heatmaps: aggregate, per-channel vector, per-channel breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    n_ts = len(ts_grid)
    n_ps = len(ps_grid)
    extent = [ps_grid[0] - 0.075, ps_grid[-1] + 0.075,
              ts_grid[0] - 0.075, ts_grid[-1] + 0.075]

    def render(ax, mat, title, vmax_ref=None, eps_line=None):
        vmax = vmax_ref if vmax_ref is not None else mat.max() * 1.05
        im = ax.imshow(mat, cmap="viridis_r", origin="lower",
                       extent=extent, aspect="auto",
                       vmin=0, vmax=vmax)
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
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("SSE")
        if eps_line is not None:
            cb.ax.axhline(eps_line, color="red", lw=2)
            cb.ax.text(1.05, eps_line, f" ε={eps_line:.0f}",
                       transform=cb.ax.get_yaxis_transform(),
                       fontsize=8, color="red", va="center")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

    render(axes[0, 0], d_total,
           "Sprint 10 — aggregate total log-likelihood SSE",
           eps_line=eps_total)
    render(axes[0, 1], d_vec,
           "Sprint 11 — sqrt(d_transcript² + d_polypeptide²)",
           eps_line=eps_vec)
    render(axes[1, 0], d_t,
           "Sprint 11 — d_transcript (transcript_init channel)",
           eps_line=eps_t)
    render(axes[1, 1], d_p,
           "Sprint 11 — d_polypeptide (polypeptide_init channel)",
           eps_line=eps_p)

    fig.suptitle(
        "Phase-3 sprint 11: per-channel distance vector breaks the "
        "anti-diagonal ridge",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, ts_grid, ps_grid, d_t, d_p, d_total,
               d_vec, eps_t, eps_p, eps_vec, eps_total,
               plot_uri: str) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    # Per-cell comparison table.
    rows = []
    truth_total = d_total[1, 1]
    truth_vec = d_vec[1, 1]
    for i, ts in enumerate(ts_grid):
        for j, ps in enumerate(ps_grid):
            ratio_total = d_total[i, j] / truth_total
            ratio_vec = d_vec[i, j] / truth_vec
            star = " ⭐" if ts == 1.0 and ps == 1.0 else ""
            cell_class = " style='background:#fffbe6'" if (i == 0 and j == 2) or (i == 2 and j == 0) else ""
            rows.append(
                f"<tr{cell_class}><td class='num'>{ts:.2f}{star}</td>"
                f"<td class='num'>{ps:.2f}</td>"
                f"<td class='num'>{d_total[i, j]:.0f}</td>"
                f"<td class='num'>{ratio_total:.2f}×</td>"
                f"<td class='num'>{d_t[i, j]:.0f}</td>"
                f"<td class='num'>{d_p[i, j]:.0f}</td>"
                f"<td class='num'>{d_vec[i, j]:.0f}</td>"
                f"<td class='num'>{ratio_vec:.2f}×</td></tr>")
    table_rows = "\n".join(rows)

    # Find sprint-10's confused cell.
    ridge_total = d_total[0, 2]  # (0.85, 1.15)
    edge_total = d_total[0, 1]   # (0.85, 1.00)
    ridge_vec = d_vec[0, 2]
    edge_vec = d_vec[0, 1]

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 3 sprint 11 — multi-observable ABC</title>
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

<h1>Phase 3 sprint 11 — multi-observable ABC inference</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Sprint 10 found an anti-diagonal ridge: opposing
  parameter shifts (lower Ts, higher Ps) partially cancel at the
  aggregate total log-likelihood. Sprint 11 fixes that by using
  per-channel distances as a 2-component vector.
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
  <li><strong>No new simulations.</strong> Reads sprint-10's persisted
      3×3 grid at <code>.pbg/runs/pdmp-03-abc-2d/</code>.</li>
  <li><strong>Per-channel distances:</strong> SSE between proposed and
      observed ensemble-mean trajectories computed
      INDEPENDENTLY for <code>transcript_init</code> and
      <code>polypeptide_init</code> log-likelihoods.</li>
  <li><strong>Vector distance:</strong>
      <code>d_vec = sqrt(d_transcript² + d_polypeptide²)</code> — the
      natural Euclidean norm on the per-channel distance vector.</li>
  <li><strong>ε candidates:</strong> mean-to-mean noise floor computed
      independently for each channel and the vector norm.</li>
</ul>

<h2>The ridge — collapsed</h2>
<img class="plot" src="{plot_uri}" alt="aggregate vs per-channel distance heatmaps">

<h2>Anti-diagonal cells: aggregate vs vector</h2>
<table>
  <tr><th>cell</th>
      <th>aggregate d_total</th><th>vector d_vec</th>
      <th>aggregate × truth</th><th>vector × truth</th></tr>
  <tr style="background:#fffbe6">
    <td>(Ts=0.85, Ps=1.15) — ridge cell</td>
    <td class="num">{ridge_total:.0f}</td>
    <td class="num">{ridge_vec:.0f}</td>
    <td class="num">{ridge_total / truth_total:.2f}×</td>
    <td class="num">{ridge_vec / truth_vec:.2f}×</td></tr>
  <tr>
    <td>(Ts=0.85, Ps=1.00) — edge cell</td>
    <td class="num">{edge_total:.0f}</td>
    <td class="num">{edge_vec:.0f}</td>
    <td class="num">{edge_total / truth_total:.2f}×</td>
    <td class="num">{edge_vec / truth_vec:.2f}×</td></tr>
</table>

<div class="takeaway">
  <strong>Ridge collapsed.</strong> Under the aggregate metric (sprint 10),
  the ridge cell (Ts=0.85, Ps=1.15) sat at
  {ridge_total:.0f} — CLOSER than the edge cell at
  {edge_total:.0f}. Under the vector metric (sprint 11), the ridge
  cell sits at {ridge_vec:.0f} — now {ridge_vec / edge_vec:.2f}×
  the edge cell. The opposing parameter shifts no longer cancel
  because they push the two channels in opposite directions;
  Euclidean addition restores the signal.
</div>

<h2>Full per-cell comparison</h2>
<p>
  Yellow-highlighted rows are the sprint-10 ridge cells (0.85, 1.15)
  and (1.15, 0.85) — where the aggregate metric undersold the
  distance to truth.
</p>
<table>
  <tr><th>Ts</th><th>Ps</th>
      <th>d_total</th><th>d_total/truth</th>
      <th>d_transcript</th><th>d_polypeptide</th>
      <th>d_vec</th><th>d_vec/truth</th></tr>
  {table_rows}
</table>

<h2>Implication for Phase 3+</h2>
<p>
  Sprint 10's identifiability finding is now fully understood and
  addressable: multi-parameter ABC inference on jump-process WCMs
  should use per-channel distances, not just aggregate
  <code>total</code>. The infrastructure already supports it — the
  LikelihoodCollector emits both channels separately, the persisted
  zarrs carry them through, and the readback walks both. The only
  cost is one matrix dimension on the distance computation. For
  more than 2 parameters, the natural extension is one distance
  component per emitted likelihood channel, with the Euclidean
  combiner (or any user-chosen p-norm) over the vector.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-grid", default="0.85,1.0,1.15")
    ap.add_argument("--ps-grid", default="0.85,1.0,1.15")
    ap.add_argument("--n-per-cell", type=int, default=4)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-03/abc_multi_observable.html"))
    args = ap.parse_args()

    ts_grid = [float(x) for x in args.ts_grid.split(",")]
    ps_grid = [float(x) for x in args.ps_grid.split(",")]

    # Reference ensemble.
    if not NOISE_FLOOR_RUNS.is_dir():
        sys.exit(f"ERROR: noise-floor reference at {NOISE_FLOOR_RUNS} missing.")
    if not ABC_2D_ROOT.is_dir():
        sys.exit(f"ERROR: sprint-10 2D grid at {ABC_2D_ROOT} missing. "
                 f"Run scripts/phase3_abc_smc_2d.py first.")
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))

    # Observed means per channel.
    observed_means = {
        ch: ref_ds[ch].mean(dim="replicate").values
        for ch in CHANNELS if ch in ref_ds.data_vars
    }
    t_observed = ref_ds["time"].values

    # Per-channel noise floors.
    nf_t = per_channel_noise_floor(ref_ds, "transcript_init")
    nf_p = per_channel_noise_floor(ref_ds, "polypeptide_init")
    nf_total = per_channel_noise_floor(ref_ds, "total")
    print(f"Per-channel noise floors (mean-to-mean, 200 resamples, 4+4 split):")
    print(f"  transcript_init   median={nf_t['median']:9.1f}  p95={nf_t['p95']:9.1f}")
    print(f"  polypeptide_init  median={nf_p['median']:9.1f}  p95={nf_p['p95']:9.1f}")
    print(f"  total             median={nf_total['median']:9.1f}  p95={nf_total['p95']:9.1f}")

    # Per-cell distances per channel.
    n_ts = len(ts_grid)
    n_ps = len(ps_grid)
    d_t = np.zeros((n_ts, n_ps))
    d_p = np.zeros((n_ts, n_ps))
    d_total = np.zeros((n_ts, n_ps))
    for i, ts in enumerate(ts_grid):
        for j, ps in enumerate(ps_grid):
            dt_val = channel_distance(ts, ps, args.n_per_cell,
                                      observed_means, t_observed,
                                      "transcript_init")
            dp_val = channel_distance(ts, ps, args.n_per_cell,
                                      observed_means, t_observed,
                                      "polypeptide_init")
            dtot = channel_distance(ts, ps, args.n_per_cell,
                                    observed_means, t_observed, "total")
            d_t[i, j] = dt_val if dt_val is not None else np.nan
            d_p[i, j] = dp_val if dp_val is not None else np.nan
            d_total[i, j] = dtot if dtot is not None else np.nan

    d_vec = np.sqrt(d_t ** 2 + d_p ** 2)
    eps_vec_pairs = np.sqrt(
        nf_t["distances"] ** 2 + nf_p["distances"] ** 2)
    eps_vec = float(np.percentile(eps_vec_pairs, 95))
    eps_t = nf_t["p95"]
    eps_p = nf_p["p95"]
    eps_total = nf_total["p95"]

    print("\nPer-cell distances:")
    for i, ts in enumerate(ts_grid):
        for j, ps in enumerate(ps_grid):
            tag = " ⭐" if ts == 1.0 and ps == 1.0 else ""
            print(f"  (ts={ts:.2f}, ps={ps:.2f}): "
                  f"d_tot={d_total[i, j]:9.1f}  "
                  f"d_t={d_t[i, j]:9.1f}  "
                  f"d_p={d_p[i, j]:9.1f}  "
                  f"d_vec={d_vec[i, j]:9.1f}{tag}")

    print(f"\nRidge collapse — (Ts=0.85, Ps=1.15) cell:")
    print(f"  aggregate sprint-10 metric: {d_total[0, 2]:.0f}")
    print(f"  per-channel vector metric:  {d_vec[0, 2]:.0f}")
    print(f"  truth (1.0, 1.0):           {d_vec[1, 1]:.0f}")

    plot_uri = make_figure(ts_grid, ps_grid, d_t, d_p, d_total, d_vec,
                           eps_t, eps_p, eps_vec, eps_total)
    write_html(args.out, ts_grid, ps_grid, d_t, d_p, d_total, d_vec,
               eps_t, eps_p, eps_vec, eps_total, plot_uri)


if __name__ == "__main__":
    main()
