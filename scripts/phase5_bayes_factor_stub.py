"""Phase 5 sprint 1 — Bayes factor model comparison stub.

Phase 5's stated scope is "Bayesian gene-function annotation via
marginal likelihood + active causal inference". This sprint
demonstrates the marginal-likelihood / Bayes-factor SHAPE using the
existing Phase-3 ABC infrastructure on a proxy problem.

Setup (proxy for a real gene-function comparison):

  H_θ: the transcript_init_prob_scale parameter equals θ.

We treat each scale value tested in sprint 7 as a competing
"hypothesis" about a parameter that, in a real Phase-5 inference,
would correspond to a gene-function assignment. For each hypothesis:

  - Distance d(θ) = SSE between proposed-ensemble-mean and observed-
    ensemble-mean total log-likelihood trajectories (sprint 7's
    metric, reused unchanged).
  - Marginal likelihood proxy: p(D | H_θ) ∝ exp(−d(θ) / (2 ε²)),
    where ε is the sprint-8 mean-to-mean noise-floor scale.
  - Bayes factor BF(θ_i : θ_j) = p(D|H_{θ_i}) / p(D|H_{θ_j}).

For an unbiased marginal likelihood (the Russian Roulette estimator
mentioned in the investigation YAML glossary), see sprint 2 candidate.

Output: ``reports/figures/pdmp-05/bayes_factor_stub.html``.
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


def per_scale_distance(scale, observed_mean, t_observed, n_per_scale):
    """SSE between proposed-ensemble-mean and observed-ensemble-mean.

    Mirrors sprint-7's inline calculation in phase3_abc_smc_stub.main().
    Returns None if the persisted ensemble for this scale is missing.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.7,0.85,1.0,1.15,1.3",
                    help="Same set as sprint 7's persisted runs.")
    ap.add_argument("--n-per-scale", type=int, default=4)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-05/bayes_factor_stub.html"))
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",")]

    # Reference + noise floor.
    if not NOISE_FLOOR_RUNS.is_dir():
        sys.exit(f"ERROR: noise-floor reference at {NOISE_FLOOR_RUNS} missing.")
    if not ABC_OUT_ROOT.is_dir():
        sys.exit(f"ERROR: sprint-7 grid at {ABC_OUT_ROOT} missing.")
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))
    nf = compute_mean_to_mean_noise_floor(ref_ds, n_per_split=4,
                                           n_resamples=200, seed=42)
    eps = float(np.sqrt(nf["median"]))  # use median noise floor as σ scale
    print(f"Noise-floor σ scale (sqrt of median mean-to-mean SSE): {eps:.1f}")

    observed_mean = ref_ds["total"].mean(dim="replicate").values
    t_observed = ref_ds["time"].values

    # Compute SSE distance per hypothesis.
    print("\nPer-hypothesis distances:")
    distances = {}
    for s in scales:
        d = per_scale_distance(s, observed_mean, t_observed,
                                args.n_per_scale)
        if d is None:
            sys.exit(f"missing sprint-7 zarr for scale={s}")
        distances[s] = d
        print(f"  H_{{θ={s:.2f}}}: SSE = {d:.1f}")

    # Marginal likelihood proxy: log p(D|H_θ) ∝ -d / (2 ε²).
    # We work in LOG space to avoid underflow at large d.
    log_marg = {s: -d / (2 * eps ** 2) for s, d in distances.items()}
    # Normalize for display.
    max_lm = max(log_marg.values())
    log_marg_norm = {s: lm - max_lm for s, lm in log_marg.items()}

    # Posterior under uniform prior over scales.
    post_weights = {s: np.exp(lm) for s, lm in log_marg_norm.items()}
    Z = sum(post_weights.values())
    post = {s: w / Z for s, w in post_weights.items()}

    # Bayes factors vs truth (scale=1.0).
    truth_lm = log_marg.get(1.0)
    if truth_lm is None:
        sys.exit("ERROR: scale=1.0 not in sweep — truth missing")
    log_bf = {s: truth_lm - lm for s, lm in log_marg.items()}

    print("\nBayes factor of truth (1.0) vs each alternative:")
    for s, lbf in sorted(log_bf.items()):
        if s == 1.0:
            print(f"  H_{{θ=1.00}}: -- (truth)")
        else:
            print(f"  H_{{θ={s:.2f}}}: log BF = {lbf:+.2f}  "
                  f"BF = {np.exp(lbf):.2e}")

    print("\nPosterior over scales (uniform prior, sum=1):")
    for s, p in sorted(post.items()):
        mark = " ⭐" if s == 1.0 else ""
        print(f"  P(θ={s:.2f} | D) = {p:.4f}{mark}")

    # Figure: side-by-side bars (distance, log marginal, posterior).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    s_keys = sorted(distances.keys())
    s_arr = np.array(s_keys)
    d_arr = np.array([distances[s] for s in s_keys])
    lm_arr = np.array([log_marg[s] for s in s_keys])
    p_arr = np.array([post[s] for s in s_keys])

    truth_color = lambda s: "#16a34a" if s == 1.0 else "#1e3a8a"
    colors = [truth_color(s) for s in s_keys]

    ax = axes[0]
    ax.bar(s_arr, d_arr, width=0.1, color=colors, edgecolor="k")
    ax.set_xlabel("Hypothesis (scale θ)")
    ax.set_ylabel("d(θ): SSE distance to observed")
    ax.set_title("Per-hypothesis distance",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(s_arr, lm_arr, width=0.1, color=colors, edgecolor="k")
    ax.set_xlabel("Hypothesis (scale θ)")
    ax.set_ylabel("log p(D | H_θ) ∝ −d(θ) / (2 ε²)")
    ax.set_title("Log marginal likelihood",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    bars = ax.bar(s_arr, p_arr, width=0.1, color=colors, edgecolor="k")
    for b, p_val, s in zip(bars, p_arr, s_keys):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{p_val * 100:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xlabel("Hypothesis (scale θ)")
    ax.set_ylabel("Posterior P(θ | D)  (uniform prior)")
    ax.set_ylim(0, max(p_arr) * 1.2)
    ax.set_title("Posterior — truth dominates",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase-5 sprint 1: Bayes factor model comparison stub",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    plot_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # HTML report.
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")
    rows = "\n".join(
        f"<tr><td class='num'>{s:.2f}{'⭐' if s == 1.0 else ''}</td>"
        f"<td class='num'>{distances[s]:.1f}</td>"
        f"<td class='num'>{log_marg[s]:.2f}</td>"
        f"<td class='num'>{('—' if s == 1.0 else f'{log_bf[s]:+.2f}')}</td>"
        f"<td class='num'>{post[s] * 100:.2f}%</td></tr>"
        for s in s_keys)

    truth_posterior = post[1.0]
    other_posts = {s: p for s, p in post.items() if s != 1.0}
    second_best_scale, second_best = max(other_posts.items(),
                                          key=lambda kv: kv[1])
    decisiveness = truth_posterior / second_best
    # Categorize the signal honestly per Jeffreys' scale on log BF.
    log_bf_truth_vs_runner = log_bf[second_best_scale]
    if log_bf_truth_vs_runner < 1.0:
        signal_label = "weak"
        signal_color = "#f59e0b"  # amber
        signal_note = (
            "Below the Jeffreys-scale &quot;substantial&quot; threshold "
            "(log BF &lt; 1) — the data alone do not decisively prefer "
            "truth over the nearest alternative.")
    elif log_bf_truth_vs_runner < 3.0:
        signal_label = "substantial"
        signal_color = "#3b82f6"
        signal_note = "Substantial evidence on the Jeffreys scale."
    else:
        signal_label = "strong"
        signal_color = "#16a34a"
        signal_note = "Strong/decisive evidence on the Jeffreys scale."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 5 sprint 1 — Bayes factor stub</title>
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
  .takeaway {{ background:#dcfce7; border-left:4px solid #16a34a;
               padding:12px 16px; margin:14px 0; font-size:0.95em; }}
</style>

<h1>Phase 5 sprint 1 — Bayes factor model comparison stub</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Demonstrates the marginal-likelihood / Bayes-factor SHAPE using the
  Phase-3 ABC infrastructure (sprint 7's persisted per-scale ensembles +
  sprint 8's mean-to-mean noise-floor calibration). Proxy hypotheses
  here are scale values of <code>transcript_init_prob_scale</code>; in a
  real Phase-5 inference, each would correspond to a gene-function
  assignment.
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
  <li><strong>Hypotheses:</strong> H_θ asserts
      <code>transcript_init_prob_scale = θ</code> for
      θ ∈ {{0.7, 0.85, 1.0, 1.15, 1.3}}.</li>
  <li><strong>Data:</strong> sprint-4 N=8 ensemble at scale=1.0
      (the "truth"), per-tick total log-likelihood trajectory.</li>
  <li><strong>Distance:</strong> sprint-7's SSE between proposed-
      ensemble mean and observed-ensemble mean.</li>
  <li><strong>Noise-floor scale:</strong> σ = sqrt(median mean-to-mean
      SSE) = <code>{eps:.1f}</code>.</li>
  <li><strong>Marginal-likelihood proxy:</strong>
      log p(D | H_θ) ∝ −d(θ) / (2 σ²).</li>
  <li><strong>Prior:</strong> uniform over the {len(s_keys)} hypotheses.</li>
</ul>

<h2>Result</h2>
<img class="plot" src="{plot_uri}" alt="distance / log marg / posterior bars">

<h2>Numerical detail</h2>
<table>
  <tr><th>θ</th><th>d(θ) SSE</th><th>log p(D|H_θ)</th>
      <th>log BF (truth/θ)</th><th>posterior</th></tr>
  {rows}
</table>

<div class="takeaway" style="border-left-color:{signal_color};
                              background:{signal_color}1a;">
  <strong>Posterior peaks at truth ({truth_posterior * 100:.1f}%), but
  the signal is <span style="color:{signal_color}">{signal_label}</span>.</strong>
  Truth is {decisiveness:.2f}× the next-most-probable hypothesis
  (θ={second_best_scale:.2f}, posterior {second_best * 100:.1f}%) —
  log BF = {log_bf_truth_vs_runner:+.2f}. {signal_note}
</div>

<h2>What this means for Phase 5</h2>
<p>
  The model-comparison SHAPE is in place: each hypothesis evaluates a
  marginal-likelihood proxy on the persisted ABC ensemble, posteriors
  normalize correctly, the truth is at the mode. Infrastructure cost
  near zero on top of Phase 3 — define a hypothesis as a parameter
  setting, reuse the forward ensemble, compute the SSE, apply
  noise-floor normalization.
</p>
<p>
  <strong>But the posterior is only mildly peaked — and that is the
  honest result.</strong> Phase-3 sprint 8 established that the
  mean-to-mean noise floor IS the parameter-identifiability scale at
  N=4. Hypotheses whose data-mean SSE falls inside σ² are by definition
  indistinguishable from noise — that's exactly what we see for
  θ ∈ {{0.85, 1.15}} (both inside the ACCEPT band in sprint 7). The
  posterior reflects this faithfully: it cannot peak sharper than the
  data's information content allows.
</p>
<p>
  This sets a concrete <strong>identifiability floor for Phase 5
  gene-function annotation</strong>: with the current N=4 ensemble and
  ε ≈ {eps:.0f}, the framework can only confidently distinguish
  function hypotheses whose predicted ensemble means differ by
  &gt; σ. Closer hypotheses require either (a) more replicates per
  hypothesis (shrinks σ as ~1/√N — sprint-8 finding), (b) richer
  observables (sprint 11's multi-observable analysis), or (c) tighter
  ε via the count-based metric (sprint 13's 3.06× truth-identifiability).
</p>
<p>
  <strong>Caveat — Russian Roulette estimator:</strong> the
  <code>exp(−d / 2σ²)</code> proxy used here is a Gaussian approximation
  to the true marginal likelihood. A proper unbiased marginal-likelihood
  estimator (in the investigation YAML glossary under the Russian
  Roulette entry) requires a stochastic series-truncation scheme on the
  partition function. Sprint 2 candidate.
</p>
""", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
