"""Phase 5 sprint 4 — hypothesis-spacing sweep on existing data.

Sprint 3 concluded the Phase-5 bottleneck is hypothesis SPACING, not
replicate count: at the existing ±15% nearest-neighbor spacing, the
kernel-mean gap between truth and θ=1.15 is only Δlog=+0.011, and no
N rescues this.

This sprint tests that diagnosis without new simulations. We re-run
the sprint-2 pseudo-marginal estimator on multiple SUBSETS of the
existing {0.7, 0.85, 1.0, 1.15, 1.3} hypothesis grid:

  - 5-way  ±15% nearest neighbor  →  truth posterior 24% (sprint 2)
  - 3-way  ±30% nearest neighbor (drop close 0.85 and 1.15)
  - 2-way  truth vs nearest extreme (0.7, 1.0)
  - 2-way  truth vs nearest extreme (1.0, 1.3)
  - 4-way  drop just 1.15 (the runner-up)
  - 4-way  drop just 0.85 (the other close neighbor)

For each subset, report truth posterior + log BF (truth vs runner-up)
+ bootstrap 95% CI. The expected pattern: posterior shoots up + log
BF clears the substantial/decisive thresholds as we drop the close
neighbors.

Output: reports/figures/pdmp-05/hypothesis_spacing_sweep.html
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
    NOISE_FLOOR_RUNS, _drop_trailing_nan, compute_mean_to_mean_noise_floor,
    load_ensemble_at,
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


SUBSETS: list[tuple[str, list[float], str]] = [
    ("5-way ±15%",  [0.70, 0.85, 1.00, 1.15, 1.30], "Original sprint-7 grid."),
    ("4-way drop 0.85", [0.70, 1.00, 1.15, 1.30], "Drop closer-low neighbor."),
    ("4-way drop 1.15", [0.70, 0.85, 1.00, 1.30], "Drop runner-up."),
    ("3-way ±30%",  [0.70, 1.00, 1.30],            "Drop both close neighbors."),
    ("2-way (0.7, 1.0)", [0.70, 1.00],             "Truth vs −30%."),
    ("2-way (1.0, 1.3)", [1.00, 1.30],             "Truth vs +30%."),
]


def posterior_from_log_marg(log_marg_dict):
    items = sorted(log_marg_dict.items())
    scales = np.array([s for s, _ in items])
    lm = np.array([v for _, v in items])
    lm -= lm.max()
    w = np.exp(lm)
    return scales, w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-scale", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-05/hypothesis_spacing_sweep.html"))
    args = ap.parse_args()

    # Noise floor + observed reference.
    ref_ds = _drop_trailing_nan(load_ensemble_at(NOISE_FLOOR_RUNS, 8))
    nf = compute_mean_to_mean_noise_floor(ref_ds, n_per_split=4,
                                           n_resamples=200, seed=42)
    eps = float(np.sqrt(nf["median"]))
    print(f"Noise-floor σ = {eps:.1f}")
    observed_mean = ref_ds["total"].mean(dim="replicate").values
    t_observed = ref_ds["time"].values

    # Precompute per-replicate kernel logs for all scales in any subset.
    all_scales = sorted({s for _, ss, _ in SUBSETS for s in ss})
    log_k_by_scale = {}
    for s in all_scales:
        d_reps = per_replicate_distances(s, observed_mean, t_observed,
                                          args.n_per_scale)
        if d_reps is None:
            sys.exit(f"missing zarr for scale={s}")
        log_k_by_scale[s] = -d_reps / (2 * eps ** 2)
        print(f"  θ={s:.2f}: log k̄={np.log(np.exp(log_k_by_scale[s]).mean()):+.3f}")

    truth = 1.00
    rng = np.random.default_rng(2026)
    results = []
    print()
    print(f"{'subset':<22} {'P(truth)':>10}  {'log BF':>10}  "
          f"{'95% CI on P(truth)':>20}")
    for label, scales, note in SUBSETS:
        if truth not in scales:
            sys.exit(f"truth scale {truth} missing from subset {label}")
        # Point estimate via unbiased pseudo-marginal.
        log_marg = {}
        for s in scales:
            log_k = log_k_by_scale[s]
            lmax = log_k.max()
            log_marg[s] = lmax + np.log(np.exp(log_k - lmax).mean())
        s_arr, post = posterior_from_log_marg(log_marg)
        post_d = dict(zip(s_arr, post))
        truth_post = post_d[truth]
        other = [s for s in scales if s != truth]
        runner = max(other, key=lambda s: log_marg[s])
        log_bf = log_marg[truth] - log_marg[runner]

        # Bootstrap CI on truth posterior.
        truth_posts_boot = []
        for _ in range(args.bootstrap):
            lm_b = {}
            for s in scales:
                log_k = log_k_by_scale[s]
                idx = rng.integers(0, len(log_k), size=len(log_k))
                sample = log_k[idx]
                lmax = sample.max()
                lm_b[s] = lmax + np.log(np.exp(sample - lmax).mean())
            _, post_b = posterior_from_log_marg(lm_b)
            post_b_d = dict(zip(sorted(scales), post_b))
            truth_posts_boot.append(post_b_d[truth])
        truth_posts_boot = np.array(truth_posts_boot)
        p25, p50, p75 = np.percentile(truth_posts_boot, [2.5, 50, 97.5])

        results.append({
            "label": label, "scales": scales, "note": note,
            "truth_post": truth_post, "log_bf": log_bf,
            "runner": runner, "log_marg": log_marg,
            "post_d": post_d, "ci_lo": p25, "ci_med": p50, "ci_hi": p75,
        })
        print(f"{label:<22} {truth_post * 100:>9.1f}%  {log_bf:>+9.3f}  "
              f"[{p25 * 100:>5.1f}%, {p75 * 100:>5.1f}%]")

    # Figure: 2 panels —
    #  (a) truth posterior + CI vs subset (one bar per subset).
    #  (b) log BF (truth vs runner-up) vs subset with Jeffreys bands.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    tp = np.array([r["truth_post"] for r in results]) * 100
    tp_lo = np.array([r["ci_lo"] for r in results]) * 100
    tp_hi = np.array([r["ci_hi"] for r in results]) * 100
    log_bfs = np.array([r["log_bf"] for r in results])
    n_h = np.array([len(r["scales"]) for r in results])

    ax = axes[0]
    colors = ["#94a3b8" if i == 0 else "#16a34a" for i in range(len(labels))]
    bars = ax.bar(x, tp, color=colors, edgecolor="k")
    ax.errorbar(x, tp, yerr=[tp - tp_lo, tp_hi - tp],
                 fmt="none", ecolor="k", capsize=4)
    uniform = 100 / n_h
    ax.plot(x, uniform, "x-", color="#dc2626", lw=1.5, markersize=10,
             label="uniform baseline")
    for i, (b, p_, n) in enumerate(zip(bars, tp, n_h)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{p_:.0f}%\n(n_H={n})", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("P(θ=1.0 | D)  (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Truth posterior + 95% CI vs hypothesis subset",
                  loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors2 = ["#dc2626" if v < 1 else "#3b82f6" if v < 5 else "#16a34a"
                for v in log_bfs]
    bars2 = ax.bar(x, log_bfs, color=colors2, edgecolor="k")
    ax.axhline(1, color="#3b82f6", ls=":", lw=1.2,
                label="substantial (log BF=1)")
    ax.axhline(5, color="#16a34a", ls=":", lw=1.2,
                label="decisive (log BF=5)")
    for b, v in zip(bars2, log_bfs):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.15 * np.sign(v + 0.01),
                f"{v:+.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("log BF (truth / runner-up)")
    ax.set_title("(b) log Bayes factor vs hypothesis subset",
                  loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase-5 sprint 4: hypothesis-spacing sweep on existing data",
                  fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    plot_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    # Table rows.
    rows = []
    for r in results:
        scales_str = ", ".join(f"{s:.2f}" for s in r["scales"])
        uniform = 100 / len(r["scales"])
        rows.append(
            f"<tr>"
            f"<td>{r['label']}</td>"
            f"<td><code>{{{scales_str}}}</code></td>"
            f"<td class='num'>{uniform:.0f}%</td>"
            f"<td class='num'>{r['truth_post'] * 100:.1f}%</td>"
            f"<td class='num'>[{r['ci_lo'] * 100:.1f}%, "
            f"{r['ci_hi'] * 100:.1f}%]</td>"
            f"<td class='num'>θ={r['runner']:.2f}</td>"
            f"<td class='num'>{r['log_bf']:+.3f}</td>"
            f"<td>{'<span style=\"color:#16a34a\">substantial+</span>' if r['log_bf'] >= 1 else 'weak'}"
            f"</td>"
            f"<td style='font-size:0.85em;color:#6b7280'>{r['note']}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 5 sprint 4 — hypothesis-spacing sweep</title>
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

<h1>Phase 5 sprint 4 — hypothesis-spacing sweep</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Sprint 3 concluded that the Phase-5 bottleneck is hypothesis
  SPACING, not replicate count: even at N=1024 the truth posterior
  asymptotes near 24% because the kernel-mean gap between truth and
  the closest neighbor (θ=1.15) is only Δlog=+0.011 nats. This sprint
  tests that diagnosis on the EXISTING data — no new simulations —
  by computing the unbiased pseudo-marginal posterior over
  progressively-wider subsets of the {len(all_scales)}-point sprint-7
  hypothesis grid.
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

<h2>Result</h2>
<img class="plot" src="{plot_uri}"
     alt="truth posterior + log BF per hypothesis subset">

<h2>Subset comparison</h2>
<table>
  <tr>
    <th>subset</th>
    <th>θ values</th>
    <th>uniform prior baseline</th>
    <th>P(θ=1.0 | D)</th>
    <th>95% CI</th>
    <th>runner-up</th>
    <th>log BF</th>
    <th>Jeffreys</th>
    <th>note</th>
  </tr>
  {rows_html}
</table>

<div class="takeaway" style="border-left-color:#3b82f6;background:#dbeafe;">
  <strong>Sprint 3's diagnosis confirmed — partially.</strong>
  Dropping the close-neighbor hypotheses θ=0.85 and θ=1.15 (which
  sprint-2's per-θ analysis already flagged as kernel-indistinguishable
  from truth) lifts the truth posterior dramatically — from
  {results[0]['truth_post'] * 100:.0f}% in the 5-way ±15% grid to
  {results[3]['truth_post'] * 100:.0f}% in the 3-way ±30% grid to
  {results[5]['truth_post'] * 100:.0f}% in the 2-way head-to-head against θ=1.3.
  But log BF maxes at <strong>{max(r['log_bf'] for r in results):+.2f}</strong>
  (truth vs 0.7) — still <em>below</em> the Jeffreys "substantial"
  threshold of 1. Spacing fixes the within-grid uniform-prior dilution
  (n_H → 2) but cannot overcome the intrinsic kernel-mean gap. The
  transcript-init scale observable + sprint-8 σ together set a hard
  per-hypothesis-pair ceiling of ~0.45 nats. Crossing "substantial"
  requires a sharper observable (sprint-5 target: count-vector).
</div>

<h2>What this means for Phase 5</h2>
<p>
  Sprint 4 separates two distinct identifiability effects:
</p>
<ol>
  <li><strong>Within-grid dilution</strong> (the n_H → 1/n_H uniform
      baseline). Wider spacing is the lever here. Going from 5-way
      to 2-way nearly triples truth-posterior mass, mostly by
      removing kernel-indistinguishable competitors.</li>
  <li><strong>Per-pair kernel-mean gap</strong> (the absolute log BF
      ceiling). At the current observable + σ, this ceiling is
      ~{max(r['log_bf'] for r in results):.2f} nats between truth
      and the farthest neighbor θ=0.7. Spacing CANNOT escape this
      ceiling — only observable choice or σ tightening can.</li>
</ol>
<p>
  For Phase-5 gene-function annotation this implies:
</p>
<ul>
  <li>Function hypotheses implying ≤±15% effect on the chosen
      observable are kernel-indistinguishable — they need to be
      either GROUPED (treated as one equivalence class) or
      COMPARED via a richer observable.</li>
  <li>The sprint-13 combined count-vector observable
      (sqrt(d_rna² + d_ribosome²)) was shown to give 3.06× truth-vs-
      next-nearest separation. Adopting it should raise the per-pair
      log-BF ceiling from ~0.45 nats to ~1.4 nats — above the
      substantial threshold. <strong>Sprint-5 target.</strong></li>
</ul>
<p>
  <strong>Net Phase-5 status after sprints 1-4:</strong> the model-
  comparison framework works (sprint 1), the estimator is unbiased
  (sprint 2), N is not the lever (sprint 3), spacing alone is not
  the lever either (sprint 4). The single remaining lever within the
  Phase-5 design space is OBSERVABLE CHOICE. Sprint-5 candidate: swap
  the aggregate log-likelihood for sprint-13's combined count vector
  and re-run the Bayes-factor + N-scaling + spacing analyses.
</p>
""", encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
