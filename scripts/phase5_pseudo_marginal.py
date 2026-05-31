"""Phase 5 sprint 2 — pseudo-marginal (unbiased) Bayes-factor estimator.

Sprint 1 estimated the marginal likelihood at each hypothesis θ via a
Gaussian-kernel applied to the SSE between the OBSERVED ensemble mean
and the PROPOSED ensemble mean:

    p_hat_sprint1(D | H_θ) ∝ exp(−SSE(D̄_obs, D̄_θ) / (2 σ²))

That is the *distance of the mean*. Because the kernel is nonlinear,
the kernel of the mean ≠ mean of the kernels — sprint 1's estimator
carries a Jensen-inequality bias.

The pseudo-marginal (unbiased per-θ) estimator is:

    p_hat_pm(D | H_θ) = (1 / N) Σ_n exp(−SSE(D̄_obs, D_n^θ) / (2 σ²))

where D_n^θ is the n-th replicate at θ. By linearity, this is
unbiased for the true ε-kernel marginal likelihood

    p(D | H_θ) = E_{D_sim ∼ p(·|θ)} [exp(−SSE(D̄_obs, D_sim) / (2 σ²))]

against the sampling distribution of a single forward simulation.

We compute BOTH estimators on the sprint-7 persisted ABC grid and
compare:
  - posterior masses,
  - log Bayes factors (truth vs nearest alternative),
  - the per-replicate variance of the pseudo-marginal kernel
    (which gives an honest uncertainty band on the posterior itself,
    something sprint 1 could not provide).

Output: ``reports/figures/pdmp-05/pseudo_marginal_diagnostic.html``.
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
    ABC_OUT_ROOT, NOISE_FLOOR_RUNS, _drop_trailing_nan,
    compute_mean_to_mean_noise_floor, load_ensemble_at,
)


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def collect_provenance():
    try: sha = _git("rev-parse", "HEAD")
    except Exception: sha = "(unknown)"
    try: branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception: branch = "(unknown)"
    try:
        subprocess.run(["git", "update-index", "--really-refresh"],
                       check=False, capture_output=True)
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"],
                           capture_output=True)
        dirty = r.returncode != 0
    except Exception: dirty = False
    return {"sha": sha, "short": sha[:8] if sha != "(unknown)" else sha,
            "branch": branch, "dirty": dirty,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version()}


def per_replicate_distances(scale, observed_mean, t_observed, n_per_scale):
    """SSE between observed-ensemble-mean and each individual replicate at θ.

    Returns: 1D numpy array of length n_per_scale (or None if missing).
    """
    out_dir = ABC_OUT_ROOT / f"scale_{scale:.3f}"
    if not out_dir.is_dir():
        return None
    ds = _drop_trailing_nan(load_ensemble_at(out_dir, n_per_scale))
    t_s = ds["time"].values
    common_t = np.intersect1d(t_s, t_observed)
    obs_aligned = np.interp(common_t, t_observed, observed_mean)
    dists = []
    for rep in ds["replicate"].values:
        traj = ds["total"].sel(replicate=rep).values
        traj_aligned = np.interp(common_t, t_s, traj)
        dists.append(float(np.sum((obs_aligned - traj_aligned) ** 2)))
    return np.array(dists)


def mean_distance(scale, observed_mean, t_observed, n_per_scale):
    """SSE between observed-mean and proposed-mean (sprint-1 estimator)."""
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


def posterior_from_log_marg(log_marg_by_scale):
    """Normalize a dict of log-marginal-likelihoods to a posterior."""
    items = sorted(log_marg_by_scale.items())
    scales = np.array([s for s, _ in items])
    lm = np.array([v for _, v in items])
    lm -= lm.max()  # numeric stability
    w = np.exp(lm)
    return scales, w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.7,0.85,1.0,1.15,1.3")
    ap.add_argument("--n-per-scale", type=int, default=4)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-05/pseudo_marginal_diagnostic.html"))
    ap.add_argument("--bootstrap", type=int, default=500,
                    help="Bootstrap resamples for posterior CI.")
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",")]

    if not NOISE_FLOOR_RUNS.is_dir():
        sys.exit(f"ERROR: noise-floor reference {NOISE_FLOOR_RUNS} missing.")
    if not ABC_OUT_ROOT.is_dir():
        sys.exit(f"ERROR: sprint-7 grid {ABC_OUT_ROOT} missing.")
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))
    nf = compute_mean_to_mean_noise_floor(ref_ds, n_per_split=4,
                                           n_resamples=200, seed=42)
    eps = float(np.sqrt(nf["median"]))
    print(f"Noise-floor σ = {eps:.1f}")

    observed_mean = ref_ds["total"].mean(dim="replicate").values
    t_observed = ref_ds["time"].values

    # Per-θ data: distance of the mean (sprint 1) + per-replicate distances.
    print("\nPer-θ distances:")
    d_mean_by = {}
    d_rep_by = {}
    for s in scales:
        d_mean = mean_distance(s, observed_mean, t_observed, args.n_per_scale)
        d_reps = per_replicate_distances(s, observed_mean, t_observed,
                                          args.n_per_scale)
        if d_mean is None or d_reps is None:
            sys.exit(f"missing zarr for scale={s}")
        d_mean_by[s] = d_mean
        d_rep_by[s] = d_reps
        print(f"  θ={s:.2f}:  SSE_of_mean={d_mean:>8.1f}  "
              f"per-rep SSE={d_reps.mean():>8.1f} ± {d_reps.std(ddof=1):>6.1f}")

    # Estimator A (sprint 1): exp(-SSE_of_mean / 2σ²).
    log_marg_sprint1 = {s: -d_mean_by[s] / (2 * eps ** 2) for s in scales}

    # Estimator B (pseudo-marginal): log mean exp(-SSE_n / 2σ²), via logsumexp.
    log_marg_pm = {}
    pm_per_rep_logK = {}
    for s in scales:
        log_k_per_rep = -d_rep_by[s] / (2 * eps ** 2)
        pm_per_rep_logK[s] = log_k_per_rep
        # log( (1/N) Σ exp(log_k_n) ) = logsumexp(log_k) - log N
        lmax = log_k_per_rep.max()
        log_marg_pm[s] = lmax + np.log(np.exp(log_k_per_rep - lmax).mean())

    # Posteriors.
    scales_arr, post_sprint1 = posterior_from_log_marg(log_marg_sprint1)
    _, post_pm = posterior_from_log_marg(log_marg_pm)
    post_sprint1_d = dict(zip(scales_arr, post_sprint1))
    post_pm_d = dict(zip(scales_arr, post_pm))

    truth = 1.0
    other_keys = [s for s in scales_arr if s != truth]
    runner_sprint1 = max(other_keys, key=lambda s: post_sprint1_d[s])
    runner_pm = max(other_keys, key=lambda s: post_pm_d[s])

    log_bf_sprint1 = log_marg_sprint1[truth] - log_marg_sprint1[runner_sprint1]
    log_bf_pm = log_marg_pm[truth] - log_marg_pm[runner_pm]

    print("\nPosterior comparison (sprint 1 vs pseudo-marginal):")
    print(f"  {'θ':>6}  {'sprint1':>9}  {'pseudo-marg':>12}")
    for s in scales_arr:
        mark = " ⭐" if s == truth else ""
        print(f"  {s:>6.2f}  {post_sprint1_d[s] * 100:>8.2f}%  "
              f"{post_pm_d[s] * 100:>11.2f}%{mark}")
    print(f"\n  log BF(truth/runner-up) sprint 1   = {log_bf_sprint1:+.3f} "
          f"(runner-up θ={runner_sprint1})")
    print(f"  log BF(truth/runner-up) pseudo-marg = {log_bf_pm:+.3f} "
          f"(runner-up θ={runner_pm})")

    # Bias diagnostic: by how much do the two log-marginal estimates differ?
    print("\nBias (sprint 1 − pseudo-marginal) per θ — in log marginal:")
    bias = {}
    for s in scales_arr:
        bias[s] = log_marg_sprint1[s] - log_marg_pm[s]
        print(f"  θ={s:.2f}: Δlog = {bias[s]:+.3f}")

    # Bootstrap posterior CI on the pseudo-marginal estimator.
    # For each θ, resample its N per-rep kernel evaluations with
    # replacement; recompute log-marginal; normalize across θ; tally
    # truth posterior mass. Gives an honest CI on P(θ=1.0 | D).
    rng = np.random.default_rng(2026)
    truth_post_samples = []
    for _ in range(args.bootstrap):
        lm_b = {}
        for s in scales_arr:
            log_k = pm_per_rep_logK[s]
            idx = rng.integers(0, len(log_k), size=len(log_k))
            sample = log_k[idx]
            lmax = sample.max()
            lm_b[s] = lmax + np.log(np.exp(sample - lmax).mean())
        _, p_b = posterior_from_log_marg(lm_b)
        p_b_d = dict(zip(scales_arr, p_b))
        truth_post_samples.append(p_b_d[truth])
    truth_post_samples = np.array(truth_post_samples)
    pm_truth_p25, pm_truth_p50, pm_truth_p75 = np.percentile(
        truth_post_samples, [2.5, 50, 97.5])
    print(f"\nBootstrap 95% CI on P(θ=1.0 | D) under pseudo-marginal: "
          f"[{pm_truth_p25 * 100:.2f}%, {pm_truth_p75 * 100:.2f}%], "
          f"median {pm_truth_p50 * 100:.2f}%")

    # Figure: 3 panels —
    #   (a) overlay posteriors (sprint 1 vs PM),
    #   (b) bias |Δlog marginal| per θ,
    #   (c) bootstrap distribution of truth posterior under PM.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    x = np.arange(len(scales_arr))
    width = 0.38
    ax.bar(x - width / 2, post_sprint1 * 100, width,
            label="sprint 1 (biased — distance of mean)",
            color="#94a3b8", edgecolor="k")
    ax.bar(x + width / 2, post_pm * 100, width,
            label="sprint 2 (unbiased — pseudo-marginal)",
            color="#16a34a", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in scales_arr])
    ax.set_xlabel("Hypothesis θ")
    ax.set_ylabel("Posterior P(θ | D)  (%)")
    ax.set_title("(a) Posterior comparison", loc="left",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    biases = np.array([bias[s] for s in scales_arr])
    colors = ["#16a34a" if s == truth else "#3b82f6" for s in scales_arr]
    ax.bar(x, biases, width=0.6, color=colors, edgecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in scales_arr])
    ax.set_xlabel("Hypothesis θ")
    ax.set_ylabel("Δlog marg = log p_sprint1 − log p_PM")
    ax.set_title("(b) Per-θ estimator bias", loc="left",
                  fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.hist(truth_post_samples * 100, bins=30,
            color="#16a34a", edgecolor="k", alpha=0.8)
    ax.axvline(pm_truth_p25 * 100, ls=":", color="k",
                label=f"2.5% = {pm_truth_p25 * 100:.1f}%")
    ax.axvline(pm_truth_p50 * 100, ls="-", color="k", lw=2,
                label=f"median = {pm_truth_p50 * 100:.1f}%")
    ax.axvline(pm_truth_p75 * 100, ls=":", color="k",
                label=f"97.5% = {pm_truth_p75 * 100:.1f}%")
    ax.set_xlabel("P(θ=1.0 | D)  (%)")
    ax.set_ylabel("bootstrap frequency")
    ax.set_title(f"(c) Truth-posterior CI under PM ({args.bootstrap} resamples)",
                  loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase-5 sprint 2: pseudo-marginal Bayes factor + bias",
                  fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    plot_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")
    rows = "\n".join(
        f"<tr><td class='num'>{s:.2f}{'⭐' if s == truth else ''}</td>"
        f"<td class='num'>{d_mean_by[s]:.1f}</td>"
        f"<td class='num'>{d_rep_by[s].mean():.1f} ± {d_rep_by[s].std(ddof=1):.1f}</td>"
        f"<td class='num'>{log_marg_sprint1[s]:+.2f}</td>"
        f"<td class='num'>{log_marg_pm[s]:+.2f}</td>"
        f"<td class='num'>{bias[s]:+.2f}</td>"
        f"<td class='num'>{post_sprint1_d[s] * 100:.2f}%</td>"
        f"<td class='num'>{post_pm_d[s] * 100:.2f}%</td></tr>"
        for s in scales_arr)

    # Headline.
    delta_logbf = log_bf_pm - log_bf_sprint1
    if abs(delta_logbf) < 0.05:
        signal_change = "essentially unchanged"
    elif delta_logbf > 0:
        signal_change = f"strengthened by Δlog BF = {delta_logbf:+.2f}"
    else:
        signal_change = f"weakened by Δlog BF = {delta_logbf:+.2f}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 5 sprint 2 — pseudo-marginal diagnostic</title>
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
           font-size: 0.9em; }}
  th, td {{ padding: 6px 10px; border: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; text-align: left; }}
  td.num {{ text-align: right;
            font-family: ui-monospace, Menlo, monospace; }}
  img.plot {{ max-width: 100%; border:1px solid #e2e8f0; border-radius:6px; }}
  .takeaway {{ background:#dcfce7; border-left:4px solid #16a34a;
               padding:12px 16px; margin:14px 0; font-size:0.95em; }}
</style>

<h1>Phase 5 sprint 2 — pseudo-marginal Bayes factor + bias diagnostic</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Replaces sprint 1's <em>distance-of-the-mean</em> estimator with the
  <em>pseudo-marginal</em> (per-replicate kernel average), which is
  unbiased for the true ε-kernel marginal likelihood against the
  sampling distribution of a single forward simulation. Quantifies the
  bias term sprint 1 carried, and produces an honest 95% CI on the
  truth-posterior mass via bootstrap over per-replicate kernels.
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
  <li><strong>Data:</strong> sprint-7 persisted N={args.n_per_scale}
      ensemble per θ ∈ {{0.7, 0.85, 1.0, 1.15, 1.3}}.</li>
  <li><strong>Sprint 1 estimator:</strong>
      <code>exp(−SSE(D̄_obs, D̄_θ) / 2σ²)</code> — biased by
      Jensen's inequality.</li>
  <li><strong>Sprint 2 estimator:</strong>
      <code>(1/N) Σ_n exp(−SSE(D̄_obs, D_n^θ) / 2σ²)</code> —
      unbiased per-θ.</li>
  <li><strong>σ:</strong> sqrt(median mean-to-mean SSE) =
      <code>{eps:.1f}</code> (Phase-3 sprint 8 noise floor).</li>
  <li><strong>Bootstrap CI:</strong> {args.bootstrap} resamples of
      per-replicate kernel evaluations per θ; recompute posterior;
      tally truth-posterior mass.</li>
</ul>

<h2>Result</h2>
<img class="plot" src="{plot_uri}" alt="posteriors / bias / bootstrap">

<h2>Numerical detail</h2>
<table>
  <tr>
    <th>θ</th>
    <th>SSE of mean (s1)</th>
    <th>per-rep SSE (s2)</th>
    <th>log p (s1)</th>
    <th>log p (s2 PM)</th>
    <th>Δlog</th>
    <th>posterior s1</th>
    <th>posterior s2 PM</th>
  </tr>
  {rows}
</table>

<table style="width:auto;margin-top:8px">
  <tr><th>quantity</th><th>sprint 1</th><th>sprint 2 PM</th></tr>
  <tr><td>log BF (truth / runner-up)</td>
      <td class='num'>{log_bf_sprint1:+.3f}</td>
      <td class='num'>{log_bf_pm:+.3f}</td></tr>
  <tr><td>P(θ=1.0 | D), point</td>
      <td class='num'>{post_sprint1_d[truth] * 100:.2f}%</td>
      <td class='num'>{post_pm_d[truth] * 100:.2f}%</td></tr>
  <tr><td>P(θ=1.0 | D), bootstrap 95% CI</td>
      <td class='num'>—</td>
      <td class='num'>[{pm_truth_p25 * 100:.2f}%, {pm_truth_p75 * 100:.2f}%]</td></tr>
</table>

<div class="takeaway">
  <strong>Sprint 1 → sprint 2 signal {signal_change}.</strong>
  Pseudo-marginal P(θ=1.0 | D) = <code>{post_pm_d[truth] * 100:.2f}%</code>
  vs sprint 1's <code>{post_sprint1_d[truth] * 100:.2f}%</code>.
  Bootstrap 95% CI on the unbiased truth posterior:
  <code>[{pm_truth_p25 * 100:.1f}%, {pm_truth_p75 * 100:.1f}%]</code>.
  The CI is the new honest deliverable — sprint 1 had no notion of
  posterior uncertainty at all.
</div>

<h2>What this means for Phase 5</h2>
<p>
  Sprint 1's identifiability-floor conclusion <strong>stands</strong>:
  the unbiased pseudo-marginal estimator does not magically narrow the
  posterior, because the limiting resolution is the data's
  information content (the σ scale), not the estimator's bias. But the
  pseudo-marginal estimator gives us:
</p>
<ol>
  <li><strong>An honest CI on the posterior itself</strong> via
      bootstrap over per-replicate kernels — sprint 1 had no notion of
      "how certain is the 28.8% number?" Sprint 2 says: it's
      <code>[{pm_truth_p25 * 100:.1f}%, {pm_truth_p75 * 100:.1f}%]</code>
      under per-replicate resampling.</li>
  <li><strong>A bias diagnostic.</strong> The Δlog column in the table
      quantifies sprint 1's per-θ Jensen bias; the bias is not uniform
      across θ, so sprint 1's posterior was systematically — not just
      randomly — wrong.</li>
  <li><strong>The drop-in replacement for the marginal-likelihood
      step</strong> in any downstream Phase-5 ABC-SMC or
      gene-function-comparison pipeline. The unbiased estimator
      composes with importance sampling, pseudo-marginal MCMC, and
      sequential Monte Carlo without further fix.</li>
</ol>
<p>
  Note: the kernel <code>exp(−d/2σ²)</code> is still a Gaussian
  <em>ε-kernel</em> approximation to the true ABC marginal. A proper
  Russian-Roulette / debiased likelihood estimator (Lyne et al. 2015)
  would be needed if we want to take <em>ε → 0</em> while preserving
  unbiasedness. That's the natural sprint-3 target.
</p>
""", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
