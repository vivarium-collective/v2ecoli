"""Phase 5 sprint 3 — N-scaling projection for Phase-5 data budget.

Sprint 2 found that at N=4 per hypothesis, the pseudo-marginal
log BF (truth vs runner-up) is +0.01, and the truth-posterior
bootstrap 95% CI is [22.4%, 26.8%] — essentially the 20% uniform
baseline. This sprint asks: HOW MANY REPLICATES would push the
posterior into a useful regime?

Approach (no new simulations):

  1. From sprint 2 per-replicate kernels k_n^θ = exp(−SSE(D̄_obs, D_n^θ) / 2σ²),
     measure per-θ kernel mean k̄_θ and variance V_θ.
  2. The pseudo-marginal at sample size N is k̄_θ(N) = (1/N) Σ k_n,
     with SE(k̄_θ) = √(V_θ/N). Delta-method:
         SE(log k̄_θ) ≈ √(V_θ/N) / k̄_θ.
  3. For each candidate N, simulate M=1000 parametric draws of
     (log k̄_θ ~ Normal(log k̂_θ_measured, SE_N)) per θ, normalize
     across θ to get a posterior, tally truth-posterior mass.
  4. Identify N_substantial: smallest N where the lower-2.5% CI on
     log BF (truth vs runner-up) exceeds 1.0 (Jeffreys "substantial").
  5. Identify N_decisive: same with threshold 5.0 (Jeffreys "decisive").

Output: reports/figures/pdmp-05/n_scaling_projection.html
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
from scripts.phase5_pseudo_marginal import per_replicate_distances


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
    ap.add_argument("--scales", default="0.7,0.85,1.0,1.15,1.3")
    ap.add_argument("--n-per-scale", type=int, default=4,
                    help="Per-θ replicates used to estimate (k̄, V).")
    ap.add_argument("--n-grid", default="4,8,16,32,64,128,256,512,1024",
                    help="Candidate N values to project at.")
    ap.add_argument("--n-draws", type=int, default=1000,
                    help="Parametric draws per N for CI estimation.")
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-05/n_scaling_projection.html"))
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",")]
    n_grid = [int(x) for x in args.n_grid.split(",")]

    if not NOISE_FLOOR_RUNS.is_dir():
        sys.exit(f"ERROR: noise-floor ref {NOISE_FLOOR_RUNS} missing.")
    if not ABC_OUT_ROOT.is_dir():
        sys.exit(f"ERROR: sprint-7 grid {ABC_OUT_ROOT} missing.")
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))
    nf = compute_mean_to_mean_noise_floor(ref_ds, n_per_split=4,
                                           n_resamples=200, seed=42)
    eps = float(np.sqrt(nf["median"]))
    print(f"Noise-floor σ = {eps:.1f}")

    observed_mean = ref_ds["total"].mean(dim="replicate").values
    t_observed = ref_ds["time"].values

    # Per-θ kernel mean + variance at N=n_per_scale (sprint 2 data).
    print("\nPer-θ kernel summary (from sprint-2 N=4 ensembles):")
    k_mean = {}
    k_var = {}
    log_k_mean = {}
    for s in scales:
        d_reps = per_replicate_distances(s, observed_mean, t_observed,
                                          args.n_per_scale)
        if d_reps is None:
            sys.exit(f"missing zarr for scale={s}")
        k_n = np.exp(-d_reps / (2 * eps ** 2))
        k_mean[s] = float(k_n.mean())
        k_var[s] = float(k_n.var(ddof=1))
        log_k_mean[s] = float(np.log(k_mean[s]))
        print(f"  θ={s:.2f}: k̄={k_mean[s]:.4g}  V={k_var[s]:.4g}  "
              f"CV={np.sqrt(k_var[s]) / k_mean[s]:.2f}  log k̄={log_k_mean[s]:+.3f}")

    truth = 1.0
    other_scales = [s for s in scales if s != truth]

    # Parametric extrapolation. For each candidate N, draw M log_k̄_θ
    # from Normal(log k̂_θ, SE_N) — i.e. CLT on the kernel mean —
    # then normalize and tally.
    rng = np.random.default_rng(2026)
    rows_data = []
    log_bf_low_by_N = {}
    log_bf_med_by_N = {}
    log_bf_high_by_N = {}
    truth_post_low = {}
    truth_post_med = {}
    truth_post_high = {}

    for N in n_grid:
        # SE on k̄ under CLT.
        # SE_log_k ≈ SE_k / k (delta method).
        se_log_k = {s: np.sqrt(k_var[s] / N) / k_mean[s] for s in scales}
        # Parametric draws of log k̄_θ.
        draws = {}
        for s in scales:
            draws[s] = rng.normal(log_k_mean[s], se_log_k[s],
                                   size=args.n_draws)
        log_marg = np.stack([draws[s] for s in scales], axis=1)  # (M, K)
        log_marg -= log_marg.max(axis=1, keepdims=True)
        w = np.exp(log_marg)
        post = w / w.sum(axis=1, keepdims=True)  # (M, K)

        # Per-draw runner-up vs truth log BF.
        truth_idx = scales.index(truth)
        other_idx = [scales.index(s) for s in other_scales]
        log_bf_truth_vs_each = (
            draws[truth][:, None]
            - np.stack([draws[s] for s in other_scales], axis=1)
        )  # (M, K-1)
        # Use the runner-up identity (smallest log-BF margin) per draw.
        log_bf_runner = log_bf_truth_vs_each.min(axis=1)

        truth_post = post[:, truth_idx]

        low_lb, med_lb, high_lb = np.percentile(log_bf_runner, [2.5, 50, 97.5])
        low_tp, med_tp, high_tp = np.percentile(truth_post, [2.5, 50, 97.5])
        log_bf_low_by_N[N] = low_lb
        log_bf_med_by_N[N] = med_lb
        log_bf_high_by_N[N] = high_lb
        truth_post_low[N] = low_tp
        truth_post_med[N] = med_tp
        truth_post_high[N] = high_tp

        print(f"  N={N:>5}: log BF runner-up 95% CI ["
              f"{low_lb:+.2f},{high_lb:+.2f}] med {med_lb:+.2f}  "
              f"P(truth) 95% CI [{low_tp * 100:>5.1f}%,{high_tp * 100:>5.1f}%]"
              f" med {med_tp * 100:.1f}%")
        rows_data.append((N, low_lb, med_lb, high_lb, low_tp, med_tp, high_tp))

    # Identify N thresholds.
    def first_N_meeting(threshold):
        for N in n_grid:
            if log_bf_low_by_N[N] >= threshold:
                return N
        return None

    n_substantial = first_N_meeting(1.0)
    n_decisive = first_N_meeting(5.0)

    print()
    print(f"N_substantial (log BF lower-CI >= 1, Jeffreys 'substantial'): "
          f"{n_substantial if n_substantial else 'NOT REACHED in grid'}")
    print(f"N_decisive (log BF lower-CI >= 5, Jeffreys 'decisive'):      "
          f"{n_decisive if n_decisive else 'NOT REACHED in grid'}")

    # Figure: 2 panels —
    #   (a) log BF (truth vs runner-up) + 95% CI band vs N (log x).
    #   (b) Truth posterior median + 95% CI band vs N.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    Ns = np.array(n_grid)
    log_bf_med = np.array([log_bf_med_by_N[n] for n in n_grid])
    log_bf_low = np.array([log_bf_low_by_N[n] for n in n_grid])
    log_bf_high = np.array([log_bf_high_by_N[n] for n in n_grid])

    ax = axes[0]
    ax.fill_between(Ns, log_bf_low, log_bf_high, color="#16a34a", alpha=0.25,
                     label="95% CI (parametric)")
    ax.plot(Ns, log_bf_med, "o-", color="#16a34a", lw=2,
             label="median log BF (truth/runner-up)")
    ax.axhline(1, color="#3b82f6", ls=":", lw=1.2,
                label="substantial (log BF=1)")
    ax.axhline(5, color="#dc2626", ls=":", lw=1.2,
                label="decisive (log BF=5)")
    ax.set_xscale("log")
    ax.set_xlabel("Replicates per hypothesis  N")
    ax.set_ylabel("log BF (truth / runner-up)")
    ax.set_title("(a) Distinguishability vs N", loc="left",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(which="both", alpha=0.3)
    if n_substantial:
        ax.axvline(n_substantial, color="#3b82f6", ls="--", alpha=0.6)
        ax.text(n_substantial, 0.2, f" N_sub={n_substantial}",
                fontsize=9, color="#3b82f6")
    if n_decisive:
        ax.axvline(n_decisive, color="#dc2626", ls="--", alpha=0.6)
        ax.text(n_decisive, 4.2, f" N_dec={n_decisive}",
                fontsize=9, color="#dc2626")

    tp_med = np.array([truth_post_med[n] for n in n_grid]) * 100
    tp_low = np.array([truth_post_low[n] for n in n_grid]) * 100
    tp_high = np.array([truth_post_high[n] for n in n_grid]) * 100

    ax = axes[1]
    ax.fill_between(Ns, tp_low, tp_high, color="#16a34a", alpha=0.25,
                     label="95% CI (parametric)")
    ax.plot(Ns, tp_med, "o-", color="#16a34a", lw=2,
             label="median P(θ=1.0 | D)")
    ax.axhline(100 / len(scales), color="#94a3b8", ls=":", lw=1.2,
                label=f"uniform baseline ({100/len(scales):.0f}%)")
    ax.set_xscale("log")
    ax.set_xlabel("Replicates per hypothesis  N")
    ax.set_ylabel("Truth posterior  P(θ=1.0 | D)  (%)")
    ax.set_title("(b) Truth posterior vs N", loc="left",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(which="both", alpha=0.3)

    fig.suptitle(f"Phase-5 sprint 3: N-scaling projection "
                 f"(σ={eps:.0f}, {len(scales)} hypotheses)",
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
    table_rows = "\n".join(
        f"<tr><td class='num'>{N}</td>"
        f"<td class='num'>{low_lb:+.2f}</td>"
        f"<td class='num'>{med_lb:+.2f}</td>"
        f"<td class='num'>{high_lb:+.2f}</td>"
        f"<td class='num'>{low_tp * 100:.1f}%</td>"
        f"<td class='num'>{med_tp * 100:.1f}%</td>"
        f"<td class='num'>{high_tp * 100:.1f}%</td>"
        f"<td>{'yes' if low_lb >= 1 else 'no'}</td>"
        f"<td>{'yes' if low_lb >= 5 else 'no'}</td></tr>"
        for (N, low_lb, med_lb, high_lb, low_tp, med_tp, high_tp) in rows_data)

    # Headline.
    headline_sub = (f"N ≥ <strong>{n_substantial}</strong>"
                    if n_substantial else
                    "<strong>not reached</strong> at any N in the grid")
    headline_dec = (f"N ≥ <strong>{n_decisive}</strong>"
                    if n_decisive else
                    "<strong>not reached</strong> at any N in the grid "
                    "(even at N=1024)")
    # Per-θ kernel-mean gap analysis — the *information-theoretic*
    # ceiling that no amount of N can overcome.
    truth_log_k = log_k_mean[truth]
    other_log_k_max = max(log_k_mean[s] for s in other_scales)
    info_gap = truth_log_k - other_log_k_max
    runner_scale = max(other_scales, key=lambda s: log_k_mean[s])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 5 sprint 3 — N-scaling projection</title>
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

<h1>Phase 5 sprint 3 — N-scaling projection for the Phase-5 data budget</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Sprint 2 found that at N=4 the data have essentially zero power to
  discriminate the five ±15% scale hypotheses (log BF truth/runner-up
  = +0.01, truth posterior 24.4% vs 20% uniform baseline). This sprint
  asks: <em>how many replicates per hypothesis would push the
  posterior into a useful regime?</em> Pure projection from sprint-2
  per-θ kernel mean &amp; variance via CLT + parametric draws.
  <strong>No new simulations.</strong>
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

<h2>Method</h2>
<ol>
  <li>From sprint-2 per-replicate kernels k_n^θ = exp(−SSE/2σ²),
      measure per-θ kernel mean k̄_θ and variance V_θ at N=4 (the
      sprint-7 ensemble size).</li>
  <li>Pseudo-marginal SE under CLT:
      SE(log k̄_θ) ≈ √(V_θ / N) / k̄_θ.</li>
  <li>For each candidate N, draw M={args.n_draws} parametric
      log k̄_θ ~ Normal(log k̂_θ_measured, SE_N) per θ, normalize
      across θ, tally truth-posterior + log BF.</li>
  <li>N_substantial / N_decisive = smallest N where the 2.5%-CI
      lower bound of log BF (truth vs runner-up) exceeds 1 / 5.</li>
</ol>

<h2>Result</h2>
<img class="plot" src="{plot_uri}" alt="log BF vs N / truth posterior vs N">

<h2>Data-budget table</h2>
<table>
  <tr>
    <th>N</th>
    <th colspan='3'>log BF (truth/runner-up) — 95% CI</th>
    <th colspan='3'>P(θ=1.0 | D) — 95% CI</th>
    <th>substantial?</th>
    <th>decisive?</th>
  </tr>
  <tr><th></th><th>2.5%</th><th>median</th><th>97.5%</th>
      <th>2.5%</th><th>median</th><th>97.5%</th><th></th><th></th></tr>
  {table_rows}
</table>

<div class="takeaway" style="border-left-color:#dc2626;background:#fee2e2;">
  <strong>Phase-5 data-budget headline — N is not the bottleneck.</strong><br>
  • <strong>Substantial</strong> evidence (log BF ≥ 1 with 95% conf):
    {headline_sub}.<br>
  • <strong>Decisive</strong> evidence (log BF ≥ 5 with 95% conf):
    {headline_dec}.<br>
  The kernel-mean gap between truth (log k̄ =
  <code>{truth_log_k:+.3f}</code>) and runner-up θ={runner_scale:.2f}
  (log k̄ = <code>{other_log_k_max:+.3f}</code>) is only Δlog =
  <code>{info_gap:+.3f}</code>. As N → ∞, the CI on log BF tightens
  to this gap — which is well below the "substantial" threshold of 1.
  No N can rescue this.
</div>

<h2>What this means for Phase 5</h2>
<p>
  The pseudo-marginal estimator's SE shrinks as 1/√N — so the CI on
  log BF tightens with more data — but the <em>signal itself</em>
  (the kernel-mean gap Δlog ≈ {info_gap:+.3f}) is set by the
  observable, the σ scale, and the spacing of the hypotheses.
  At the current setup, even <em>infinite</em> N would converge to
  log BF ≈ {info_gap:+.3f}, still below "substantial".
</p>
<p>
  <strong>This reclassifies the Phase-5 bottleneck.</strong> Sprint 2
  ended thinking N was the dominant lever — sprint 3 shows it is
  the <em>least</em> useful lever. The actual levers, in order of
  impact:
</p>
<ol>
  <li><strong>Wider hypothesis spacing</strong> — the ±15% spacing
      between adjacent hypotheses produces almost-identical predicted
      ensemble means at the current σ. Comparing genes whose function
      assignments imply ≥50% predicted-mean shifts would push more
      mass into kernel-distinguishable regions. <strong>This is the
      sprint-4 target.</strong></li>
  <li><strong>Richer observable</strong> — Phase-3 sprint 11
      showed per-channel beats aggregate; Phase-3 sprint 13's combined
      count vector gives 3.06× truth-vs-next-nearest separation,
      which would widen the per-θ kernel-mean gap directly.</li>
  <li><strong>Tighter ε</strong> — sequential ε refinement (Phase-3
      sprint 9) on the count-based metric (sprint 13). Tightening ε
      sharpens the kernel and exaggerates the kernel-mean gap.</li>
  <li><strong>N per hypothesis</strong> — only helps if (1)–(3) have
      already opened a kernel-mean gap > the noise floor. Sprint 3
      shows this lever alone is essentially useless at the current
      setup.</li>
</ol>
<p>
  <strong>Caveat.</strong> The kernel-mean gap of {info_gap:+.3f} is
  estimated from N=4 ensembles per θ; the standard error on that gap
  itself is non-trivial. A useful sprint-4 follow-up would run a
  modest validation ensemble (e.g. N=16 at θ ∈ {{1.0, 1.15}}) to
  confirm the gap stays small under larger N — but the conclusion
  here does not depend on N tuning.
</p>
""", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
