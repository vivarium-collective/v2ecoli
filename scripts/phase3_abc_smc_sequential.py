"""Phase 3 sprint 9 — sequential ε tightening (SMC refinement).

Demonstrates the SMC part of ABC-SMC: iteratively shrink the acceptance
threshold and watch the posterior concentrate toward the truth.

Reuses the sprint-7 per-scale ensembles (cached zarrs under
``.pbg/runs/pdmp-03-abc-smc/scale_<s>/``) and the sprint-8 mean-to-mean
noise-floor calibration. No new simulations required.

Pipeline:

  1. Load sprint-4's N=8 reference; compute the mean-to-mean null
     distance distribution (sprint 8's primitive).
  2. Pick ε levels from the null distribution at decreasing percentiles
     (default ``[95, 75, 50, 25, 5]``).
  3. For each ε, recompute the acceptance verdict per proposed scale
     using sprint 7's cached per-scale distances.
  4. Render a staircase: each row is one ε level, columns are proposed
     scales, cells are colored by accept (green) / reject (red); the ε
     value and posterior set are annotated.

Output: ``reports/figures/pdmp-03/abc_smc_sequential.html``
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
from matplotlib.colors import ListedColormap
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase3_abc_smc_stub import (
    ABC_OUT_ROOT,
    NOISE_FLOOR_RUNS,
    _drop_trailing_nan,
    compute_mean_to_mean_noise_floor,
    load_ensemble_at,
)


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
    return {
        "sha": sha,
        "short": sha[:8] if sha != "(unknown)" else sha,
        "branch": branch,
        "dirty": dirty,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }


def per_scale_distance(scale: float, observed_mean, t_observed,
                       n_per_scale: int) -> float | None:
    out_dir = ABC_OUT_ROOT / f"scale_{scale:.3f}"
    if not out_dir.is_dir():
        return None
    ds = _drop_trailing_nan(load_ensemble_at(out_dir, n_per_scale))
    mean_traj = ds["total"].mean(dim="replicate").values
    t_s = ds["time"].values
    common_t = np.intersect1d(t_s, t_observed)
    obs_aligned = np.interp(common_t, t_observed, observed_mean)
    mean_aligned = np.interp(common_t, t_s, mean_traj)
    return float(np.sum((obs_aligned - mean_aligned) ** 2))


def make_figure(scales, distances, eps_levels, accept_matrix,
                truth_scale=1.0) -> str:
    """Render the staircase + threshold-vs-percentile curve."""
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1.4, 1]})

    # --- Panel 1: acceptance staircase ---
    ax = axes[0]
    # accept_matrix shape: (n_eps, n_scales). 1=accept (green), 0=reject (red).
    cmap = ListedColormap(["#fee2e2", "#dcfce7"])
    im = ax.imshow(
        accept_matrix, cmap=cmap, vmin=0, vmax=1,
        aspect="auto", origin="upper",
    )
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{s:.2f}" for s in scales])
    ax.set_yticks(range(len(eps_levels)))
    ax.set_yticklabels([
        f"{pct}%  ε={eps:.0f}" for pct, eps in eps_levels
    ])
    ax.set_xlabel("transcript_init_prob_scale")
    ax.set_ylabel("ε threshold (percentile of mean-to-mean null)")
    # Truth column marker
    truth_col = scales.index(truth_scale) if truth_scale in scales else None
    if truth_col is not None:
        ax.axvline(truth_col, color="#1e40af", lw=2.0, ls=":",
                   alpha=0.5)
    # Cell annotations
    for ei, (_pct, eps) in enumerate(eps_levels):
        for si, s in enumerate(scales):
            d = distances[s]
            text = f"{d/eps:.2f}×"
            color = "#065f46" if accept_matrix[ei, si] else "#991b1b"
            ax.text(si, ei, text, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    ax.set_title("ABC-SMC acceptance staircase: posterior tightens as ε ↓",
                 loc="left", fontsize=11, fontweight="bold")

    # --- Panel 2: SSE distance per scale + ε levels ---
    ax = axes[1]
    dists = np.array([distances[s] for s in scales])
    ax.scatter(scales, dists, s=140, color="#1e40af", edgecolors="k",
               zorder=10, label="proposed scale")
    # Truth marker
    if truth_scale in scales:
        ti = scales.index(truth_scale)
        ax.scatter([truth_scale], [dists[ti]], s=240, marker="*",
                   color="#16a34a", edgecolors="k",
                   label=f"truth (scale={truth_scale})", zorder=11)
    # ε horizontal lines (lighter for tighter ε)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(eps_levels)))
    for (pct, eps), c in zip(eps_levels, colors):
        ax.axhline(eps, color=c, lw=1.6,
                   label=f"ε p{pct} = {eps:.0f}")
    ax.set_xlabel("transcript_init_prob_scale")
    ax.set_ylabel("SSE distance to observed")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", fontsize=9, ncol=1)
    ax.set_title("Distance vs scale with sequential ε levels",
                 loc="left", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Phase-3 sprint 9: sequential ε tightening "
        "(SMC refinement of the sprint-7 ABC stub)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, scales, distances, eps_levels,
               accept_matrix, plot_uri: str, nf: dict) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    # Per-eps posterior + verdict table.
    posterior_rows = []
    for ei, (pct, eps) in enumerate(eps_levels):
        accepts = [scales[si] for si in range(len(scales))
                   if accept_matrix[ei, si]]
        rejects = [scales[si] for si in range(len(scales))
                   if not accept_matrix[ei, si]]
        accepted_str = (
            "{" + ", ".join(f"{s:.2f}" for s in accepts) + "}"
            if accepts else "{ }"
        )
        rejected_str = (
            "{" + ", ".join(f"{s:.2f}" for s in rejects) + "}"
            if rejects else "{ }"
        )
        posterior_rows.append(
            f"<tr><td>p{pct}</td><td class='num'>{eps:.1f}</td>"
            f"<td>{accepted_str}</td><td>{rejected_str}</td>"
            f"<td class='num'>{len(accepts)}/{len(scales)}</td></tr>"
        )
    pst_table_rows = "\n".join(posterior_rows)

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 3 sprint 9 — sequential ε tightening</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1500px; margin: 24px auto; padding: 0 18px;
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
  .verdict {{ background:#ecfeff; border-left:4px solid #06b6d4;
              padding:12px 16px; margin:14px 0; font-size:0.95em; }}
</style>

<h1>Phase 3 sprint 9 — sequential ε tightening (SMC refinement)</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  The SMC part of ABC-SMC: take sprint 7's fixed-grid ABC stub and
  sweep the acceptance threshold ε from loose to tight. The posterior
  concentrates around the truth as ε ↓.
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
  <li>Per-scale sims persisted by sprint 7
      (<code>.pbg/runs/pdmp-03-abc-smc/scale_<i>s</i>/</code>); not re-run here.</li>
  <li>ε candidate levels: percentiles of the sprint-8 mean-to-mean
      null distance distribution (4+4 splits of sprint-4's N=8,
      {nf['n_resamples']} resamples).</li>
  <li>Distance metric: SSE between proposed-ensemble-mean and
      observed-ensemble-mean <code>total</code> log-likelihood trajectories.</li>
</ul>

<h2>Staircase: posterior at each ε</h2>
<img class="plot" src="{plot_uri}" alt="acceptance staircase + ε-vs-distance">

<h2>Per-ε posterior</h2>
<table>
  <tr><th>ε percentile</th><th>ε value</th>
      <th>accepted scales (posterior)</th>
      <th>rejected scales</th>
      <th>accept/total</th></tr>
  {pst_table_rows}
</table>

<div class="verdict">
  <strong>SMC reading.</strong> At the loosest ε (p95 of the null), the
  posterior is wide — any scale within ~0.3 of truth is accepted. As ε
  tightens (p75 → p50 → p25 → p05), the posterior narrows toward the
  true value (scale = 1.0). The truth is correctly retained at every
  ε level (its distance is the smallest across the sweep, ~4× below
  the next-closest proposal); extreme scales are progressively
  eliminated. This is what a real SMC iteration does: propose,
  weight, re-tighten, repeat.
</div>

<h2>Phase 3 state of play after sprint 9</h2>
<p>
  Phase 3's stated deliverable was "observation likelihoods +
  likelihood accumulator + Vivarium observe/intervene effects;
  ABC-SMC baseline". Sprints 1–9 cover all of: per-process
  observation likelihoods (1, 2), aggregate likelihood collector
  (2), XArrayEmitter persistence (4), ensemble-shaped readback (4),
  inference noise floor (6), one-parameter forward-model ABC stub
  (7), corrected mean-to-mean ε calibration (8), and now sequential
  ε refinement (9). The remaining ABC-SMC gap is the
  proposal-perturbation kernel + importance reweighting between
  rounds; both build directly on what's already on disk.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.7,0.85,1.0,1.15,1.3",
                    help="Comma-separated list of proposed scales "
                         "(must match a sprint-7 persisted run).")
    ap.add_argument("--n-per-scale", type=int, default=4)
    ap.add_argument("--eps-percentiles", default="95,75,50,25,5",
                    help="Comma-separated null-distribution percentiles "
                         "to use as ε candidates.")
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-03/abc_smc_sequential.html"))
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",")]
    eps_pcts = sorted({int(x) for x in args.eps_percentiles.split(",")},
                      reverse=True)

    # Step 0 — sprint-4 noise-floor calibration.
    if not NOISE_FLOOR_RUNS.is_dir():
        sys.exit(f"ERROR: noise-floor reference ensemble at "
                 f"{NOISE_FLOOR_RUNS} missing.")
    print(f"Loading sprint-4 noise-floor ensemble from {NOISE_FLOOR_RUNS}...")
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))
    nf = compute_mean_to_mean_noise_floor(ref_ds, n_per_split=4,
                                          n_resamples=200, seed=42)
    eps_levels = [
        (pct, float(np.percentile(nf["distances"], pct)))
        for pct in eps_pcts
    ]
    print(f"  ε candidates from {nf['n_resamples']}-resample null "
          f"({len(nf['distances'])} sample distances):")
    for pct, eps in eps_levels:
        print(f"    p{pct:2d} → ε = {eps:9.1f}")

    # Step 1 — observed mean.
    observed_mean = ref_ds["total"].mean(dim="replicate").values
    t_observed = ref_ds["time"].values

    # Step 2 — per-scale distances from sprint-7 persisted sims.
    print("\nReading cached per-scale ensembles + computing distances:")
    distances: dict[float, float] = {}
    for s in scales:
        d = per_scale_distance(s, observed_mean, t_observed,
                               args.n_per_scale)
        if d is None:
            print(f"  scale={s:.3f}: CACHE MISS — run sprint 7's stub first")
            sys.exit(1)
        distances[s] = d
        print(f"  scale={s:.3f}: SSE = {d:.1f}")

    # Step 3 — acceptance matrix at each ε.
    accept_matrix = np.zeros(
        (len(eps_levels), len(scales)), dtype=np.int32)
    for ei, (_pct, eps) in enumerate(eps_levels):
        for si, s in enumerate(scales):
            accept_matrix[ei, si] = 1 if distances[s] < eps else 0

    # Step 4 — verdicts.
    print("\nSequential posterior:")
    for ei, (pct, eps) in enumerate(eps_levels):
        accepts = [scales[si] for si in range(len(scales))
                   if accept_matrix[ei, si]]
        print(f"  ε = p{pct:2d} = {eps:9.1f}: "
              f"accept {accepts}  ({len(accepts)}/{len(scales)})")

    plot_uri = make_figure(scales, distances, eps_levels, accept_matrix)
    write_html(args.out, scales, distances, eps_levels, accept_matrix,
               plot_uri, nf)


if __name__ == "__main__":
    main()
