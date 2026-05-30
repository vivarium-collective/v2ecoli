"""Phase 3 sprint 7 — ABC-SMC stub on the transcript_init_prob_scale parameter.

Demonstrates the Phase-3 pipeline closing the loop on a simple one-parameter
inference problem:

  1. Define "observed" as the cross-replicate mean `total` log-likelihood
     timeseries from an ensemble run at the **true** parameter
     (transcript_init_prob_scale = 1.0).
  2. For each proposed scale in a sweep, run a small ensemble of forward
     sims and compute the cross-replicate mean log-likelihood timeseries.
  3. Compute the SSE distance d(observed_mean, proposed_mean) and compare
     against the sprint-6 noise floor (p05–p95 within-ensemble band).
  4. ABC acceptance: a proposed scale is accepted iff d < ε, where ε is
     the noise floor p95.

Outputs:

- Per-(scale, seed) zarr stores under
  ``.pbg/runs/pdmp-03-abc-smc/scale_<s>/seed_<NN>/store.zarr/`` (the
  forward-sim "particles" of the SMC).
- A summary HTML at ``reports/figures/pdmp-03/abc_smc_stub.html``
  with the scale-vs-distance curve, noise-floor band overlay, and
  per-scale accept/reject verdict.

Usage::

    .venv/bin/python scripts/phase3_abc_smc_stub.py \\
        --scales 0.7,0.85,1.0,1.15,1.3 --n-per-scale 4 --duration 60
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

from scripts.phase3_likelihood_xarray_ensemble import (
    load_ensemble,
    run_one,
)


ABC_OUT_ROOT = Path(".pbg/runs/pdmp-03-abc-smc")
NOISE_FLOOR_RUNS = Path(".pbg/runs/pdmp-03-likelihood")  # sprint 4's N=8


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


def load_ensemble_at(root: Path, n: int):
    """Adapt sprint-4's load_ensemble to a configurable root."""
    # The script-level OUT_ROOT is a module-level constant. We
    # temporarily rebind it to the desired root.
    import scripts.phase3_likelihood_xarray_ensemble as ens
    original = ens.OUT_ROOT
    try:
        ens.OUT_ROOT = root
        return load_ensemble(n)
    finally:
        ens.OUT_ROOT = original


def compute_noise_floor_max(ds):
    """Pairwise SSE on `total` — copies sprint 6's compute_noise_floor."""
    arr = ds["total"].values
    n_rep = arr.shape[0]
    iu = np.triu_indices(n_rep, k=1)
    pairs = []
    for i, j in zip(*iu):
        pairs.append(float(np.sum((arr[i] - arr[j]) ** 2)))
    pairs = np.asarray(pairs)
    return {
        "median": float(np.median(pairs)),
        "p05": float(np.percentile(pairs, 5)),
        "p95": float(np.percentile(pairs, 95)),
        "max": float(pairs.max()),
    }


def run_scale_ensemble(scale: float, n: int, duration: int, chunk: int):
    """Run N forward sims at the given scale; persist to per-scale dir."""
    out_dir = ABC_OUT_ROOT / f"scale_{scale:.3f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  scale={scale:.3f}: N={n} sims of {duration}s...", flush=True)
    for seed in range(n):
        s = run_one(seed, duration_s=duration, chunk=chunk,
                    transcript_scale=scale, out_root=out_dir)
        if "error" in s:
            print(f"    seed={seed:02d}: ERROR {s['error']}")
        else:
            print(f"    seed={seed:02d}: wall={s['wall_s']:.1f}s", flush=True)
    return out_dir


def _drop_trailing_nan(ds):
    valid_mask = ~np.isnan(ds["total"]).any(dim="replicate")
    return ds.where(valid_mask, drop=True)


def make_figure(results: list[dict], noise_floor: dict,
                eps: float) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [1, 1.1]})

    # Panel 1: scale-vs-distance acceptance curve.
    ax = axes[0]
    scales = np.array([r["scale"] for r in results])
    dists = np.array([r["distance"] for r in results])
    colors = ["#16a34a" if r["accept"] else "#dc2626" for r in results]
    ax.scatter(scales, dists, c=colors, s=120, edgecolors="k", zorder=10)
    # Noise floor band.
    ax.axhspan(noise_floor["p05"], noise_floor["p95"], color="#7c3aed",
               alpha=0.18, label=f"noise floor p05-p95")
    ax.axhline(eps, color="#7c3aed", lw=1.5, ls="--",
               label=f"acceptance ε = {eps:.0f}")
    ax.axhline(noise_floor["median"], color="#7c3aed", lw=1.0, ls=":",
               label=f"noise median = {noise_floor['median']:.0f}")
    # Truth marker.
    ax.axvline(1.0, color="#16a34a", lw=1.0, ls=":", alpha=0.6,
               label="true scale = 1.0")
    ax.set_xlabel("transcript_init_prob_scale")
    ax.set_ylabel("SSE distance to observed (mean trajectory)")
    ax.set_title("ABC acceptance vs proposed scale",
                 fontsize=11, loc="left", fontweight="bold")
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_yscale("symlog")
    # Annotate each point.
    for s, d, accept in zip(scales, dists, [r["accept"] for r in results]):
        label = "✓" if accept else "✗"
        ax.annotate(f" {label} {s:.2f}", (s, d), fontsize=10,
                    xytext=(8, 0), textcoords="offset points")

    # Panel 2: per-scale mean trajectory plot.
    ax = axes[1]
    for r in results:
        ax.plot(r["t"], r["mean_traj"],
                color="#dc2626" if not r["accept"] else "#16a34a",
                alpha=0.7, lw=1.2,
                label=f"scale={r['scale']:.2f} (d={r['distance']:.0f})")
    if results:
        # The "observed" mean trajectory.
        obs = results[0]["observed_mean"]
        t_obs = results[0]["t_observed"]
        ax.plot(t_obs, obs, color="black", lw=2.5,
                label="observed (scale=1.0 truth)", zorder=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total log-likelihood per tick")
    ax.set_title("Mean trajectory by proposed scale",
                 fontsize=11, loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Phase-3 sprint 7: ABC-SMC stub on transcript_init_prob_scale",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, results: list[dict],
               noise_floor: dict, eps: float, plot_uri: str) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    row_lines = []
    for r in sorted(results, key=lambda x: x["scale"]):
        verdict = ("<span style='color:#16a34a;font-weight:600'>ACCEPT</span>"
                   if r["accept"] else
                   "<span style='color:#dc2626;font-weight:600'>REJECT</span>")
        row_lines.append(
            f"<tr><td class='num'>{r['scale']:.3f}</td>"
            f"<td class='num'>{r['distance']:.1f}</td>"
            f"<td class='num'>{r['distance'] / eps:.2f}×</td>"
            f"<td>{verdict}</td></tr>")
    rows_html = "\n".join(row_lines)

    accepted = [r["scale"] for r in results if r["accept"]]
    posterior_summary = (
        f"Posterior: {len(accepted)} of {len(results)} proposed scales "
        f"accepted ({accepted}). True scale 1.0 "
        + ("<strong>WITHIN</strong>" if 1.0 in [round(x, 2) for x in accepted]
           else "<strong>OUTSIDE</strong>")
        + " accepted region."
    )

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 3 sprint 7 — ABC-SMC stub</title>
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

<h1>Phase 3 sprint 7 — ABC-SMC stub</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  One-parameter forward-model inference on
  <code>transcript_init_prob_scale</code>, with the sprint-6 noise floor
  as the ABC acceptance threshold ε.
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

<h2>Inference setup</h2>
<ul>
  <li><strong>Parameter:</strong> <code>transcript_init_prob_scale</code>
      (multiplies Poisson rate in TranscriptInitiation's poisson mode).</li>
  <li><strong>True value:</strong> 1.0 (the unperturbed sampler).</li>
  <li><strong>Observed data:</strong> cross-replicate mean of <code>total</code>
      log-likelihood timeseries from the sprint-4 N=8 ensemble at scale=1.0.</li>
  <li><strong>Distance metric:</strong> SSE between proposed ensemble mean
      and observed mean over time.</li>
  <li><strong>Acceptance threshold ε:</strong> sprint-6's p95 of the
      within-null pairwise distance distribution = <code>{eps:.1f}</code>.</li>
</ul>

<h2>Sweep result</h2>
<img class="plot" src="{plot_uri}" alt="ABC acceptance curve + per-scale trajectories">

<table>
  <tr><th>proposed scale</th><th>SSE distance</th><th>distance / ε</th><th>verdict</th></tr>
  {rows_html}
</table>

<div class="verdict">
  <strong>Inference result.</strong> {posterior_summary}
  The acceptance threshold ε = {eps:.0f} (sprint-6 p95 of the within-null
  pairwise distance) is the parameter-distinguishability floor at the
  current ensemble size.
</div>

<h2>Pipeline status</h2>
<p>
  Phase 3 sprints 1–7 are end-to-end working: per-process Poisson
  likelihoods emitted, aggregated by LikelihoodCollector, persisted via
  XArrayEmitter, loaded back as <code>xarray.Dataset(replicate × time ×
  observable)</code>, with a calibrated noise-floor threshold and now a
  demonstrated forward-model ABC acceptance loop on a tunable parameter.
  The remaining Phase-3 work is the SMC particle propagation (sequential
  refinement of the posterior across decreasing-ε rounds), which builds
  directly on the per-scale ensembles persisted by this script.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.7,0.85,1.0,1.15,1.3",
                    help="Comma-separated list of proposed scales.")
    ap.add_argument("--n-per-scale", type=int, default=4)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-03/abc_smc_stub.html"))
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",")]

    # Step 0 — noise floor + ε from sprint-4's N=8.
    if not NOISE_FLOOR_RUNS.is_dir():
        print(f"ERROR: noise-floor reference ensemble at "
              f"{NOISE_FLOOR_RUNS} missing. Run "
              f"phase3_likelihood_xarray_ensemble.py first.")
        sys.exit(1)
    print(f"Loading sprint-4 noise-floor ensemble from {NOISE_FLOOR_RUNS}...")
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))
    nf = compute_noise_floor_max(ref_ds)
    eps = nf["p95"]
    print(f"  noise floor: median={nf['median']:.1f}, "
          f"p05={nf['p05']:.1f}, p95={nf['p95']:.1f} (= ε)")

    # "Observed" trajectory: cross-replicate mean at scale=1.0.
    observed_mean = ref_ds["total"].mean(dim="replicate").values
    t_observed = ref_ds["time"].values

    # Step 1 — run forward sims at each proposed scale, persist.
    print("\nRunning forward sims per proposed scale...")
    for s in scales:
        run_scale_ensemble(s, args.n_per_scale, args.duration, args.chunk)

    # Step 2 — load + distance for each scale.
    print("\nComputing per-scale distances to observed:")
    results = []
    for s in scales:
        out_dir = ABC_OUT_ROOT / f"scale_{s:.3f}"
        ds = _drop_trailing_nan(load_ensemble_at(out_dir, args.n_per_scale))
        mean_traj = ds["total"].mean(dim="replicate").values
        t_s = ds["time"].values
        # Align by intersecting time domain.
        common_t = np.intersect1d(t_s, t_observed)
        obs_aligned = np.interp(common_t, t_observed, observed_mean)
        mean_aligned = np.interp(common_t, t_s, mean_traj)
        d = float(np.sum((obs_aligned - mean_aligned) ** 2))
        accept = d < eps
        verdict = "ACCEPT" if accept else "REJECT"
        print(f"  scale={s:.3f}: SSE={d:.1f}  d/ε={d / eps:.2f}  {verdict}")
        results.append({
            "scale": s, "distance": d, "accept": accept,
            "mean_traj": mean_traj, "t": t_s,
            "observed_mean": observed_mean, "t_observed": t_observed,
        })

    plot_uri = make_figure(results, nf, eps)
    write_html(args.out, results, nf, eps, plot_uri)
    print(f"\nDone. Wrote {args.out}")


if __name__ == "__main__":
    main()
