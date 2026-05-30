"""Generate the Phase-3 closeout report for the v2ecoli-pdmp investigation.

Mirrors ``scripts/phase2_progress_report.py``. Renders
``investigations/v2ecoli-pdmp/reports/pdmp-03-inference.html`` with a sprint
timeline, the headline ABC-SMC convergence result, and links to the
key figures emitted by sprints 5–9.
"""
from __future__ import annotations

import datetime as dt
import platform
import socket
import subprocess
from pathlib import Path

SPRINTS = [
    (1, "TranscriptInitiation log-likelihood emission", "✅",
     "Per-tick log P(observed | rates) emitted to listeners.rnap_data."
     " Discrete mode emits 0.0 sentinel; poisson mode emits the Σ_i "
     "log P(k_i | λ_i) under the rates the sampler used. Surfaces the "
     "pbg pruning quirk (task #14): scalar listener fields with no "
     "downstream consumer are dropped from the merged state."),
    (2, "PolypeptideInitiation likelihood + LikelihoodCollector", "✅",
     "Mirror of sprint 1 on the translation side. New "
     "LikelihoodCollector step reads both per-process likelihoods "
     "(pinning them in the merged state) and writes "
     "listeners.likelihood.{transcript_init, polypeptide_init, total}. "
     "Translation contributes ~88% of total magnitude — many more "
     "events per tick than transcription."),
    (3, "Likelihood persistence + readback (SQLite, then dropped)", "❌→✅",
     "First persistence attempt used SQLiteEmitter (wrong choice for "
     "PDMP per investigation YAML). Replaced by sprint 4."),
    (4, "XArrayEmitter ensemble (replaces sprint 3)", "✅",
     "N=8 PDMP+poisson sims, each writing per-tick observables to "
     ".pbg/runs/pdmp-03-likelihood/seed_NN/store.zarr/ via "
     "XArrayEmitter. DataTree readback into "
     "xarray.Dataset(replicate × time × observable) — the canonical "
     "inference shape. Σ_t cross-replicate σ on total = 488 fg, "
     "matching sprint 2's in-memory ensemble within MC noise. "
     "Documented trailing-buffer NaN quirk (buf_size ≥ 3 minimum)."),
    (5, "Per-tick ensemble figure", "✅",
     "3-panel matplotlib figure: per-tick log-likelihood for "
     "transcript_init, polypeptide_init, total. Faded individual "
     "replicate traces + bold mean line + ±σ band. HTML report with "
     "provenance banner, Σ_t summary table, and trailing-NaN warning."),
    (6, "Intra-ensemble inference noise floor", "✅",
     "Pairwise per-replicate SSE distance distribution on `total`. "
     "median = 158,294, p95 = 207,432. Initial ε estimate for "
     "ABC-SMC acceptance — later corrected by sprint 8."),
    (7, "ABC-SMC stub on transcript_init_prob_scale", "⚠️→✅",
     "Added tunable parameter (multiplies Poisson rate). 5 scales × "
     "N=4 sims. Distance metric IS sensitive (truth=1.0 is 7× closer "
     "than extremes) but sprint-6 ε was over-generous; all 5 scales "
     "accepted. Honest finding led directly to sprint 8 correction."),
    (8, "Mean-to-mean noise floor — corrected ε", "✅",
     "Subsamples sprint-4's N=8 into 4+4 splits, computes mean-to-mean "
     "SSE distribution. ε = p95 = 58,883 (3.5× tighter than sprint-6, "
     "matching theoretical √8 ≈ 2.83×). Sprint 7 re-verdict: scale=0.7 "
     "correctly REJECTED; posterior tightens to {0.85, 1.0, 1.15, 1.3}. "
     "Added per-seed caching for fast analysis iteration."),
    (9, "Sequential ε tightening — SMC refinement", "✅",
     "Iterates ε ∈ {p95, p75, p50, p25, p05} of the sprint-8 null. "
     "Posterior monotonically concentrates: p95 → {0.85, 1.0, 1.15, "
     "1.3}; p50 → {1.0, 1.15}; p25 and p05 → {1.0}. Truth retained at "
     "every ε level. Acceptance staircase visualization."),
]

ACCEPTANCE_TABLE = [
    ("p95", 53037.3, "{0.85, 1.0, 1.15, 1.3}", "4/5"),
    ("p75", 44440.3, "{0.85, 1.0, 1.15}", "3/5"),
    ("p50", 38968.2, "{1.0, 1.15}", "2/5"),
    ("p25", 33748.5, "**{1.0}**", "1/5"),
    ("p05", 29868.1, "**{1.0}**", "1/5"),
]

PER_SCALE_DISTANCES = [
    (0.70, 68884.7, "REJECT at all ε ≤ p95"),
    (0.85, 42419.4, "REJECT at p50 and below"),
    (1.00,  9793.1, "ACCEPT at every ε — the truth"),
    (1.15, 34532.5, "REJECT at p25 and below"),
    (1.30, 47459.0, "REJECT at p75 and below"),
]


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


def write_html(out_path: Path) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    sprint_rows = "\n".join(
        f"<tr><td>{n}</td><td>{title}</td><td>{verdict}</td>"
        f"<td>{detail}</td></tr>"
        for n, title, verdict, detail in SPRINTS
    )

    posterior_rows = "\n".join(
        f"<tr><td>{pct}</td><td class='num'>{eps:.1f}</td>"
        f"<td>{posterior}</td><td class='num'>{count}</td></tr>"
        for pct, eps, posterior, count in ACCEPTANCE_TABLE
    )

    scale_rows = "\n".join(
        f"<tr><td class='num'>{scale:.2f}</td>"
        f"<td class='num'>{sse:,.1f}</td><td>{note}</td></tr>"
        for scale, sse, note in PER_SCALE_DISTANCES
    )

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 3 inference — v2ecoli-pdmp closeout</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1500px; margin: 24px auto; padding: 0 18px;
         line-height: 1.55; }}
  h1 {{ margin: 0 0 6px 0; color:#1e3a8a; }}
  h2 {{ margin-top: 28px; border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px; color:#1e3a8a; }}
  h3 {{ margin-top: 18px; color:#334155; }}
  .meta {{ color: #6b7280; font-size: 0.9em; }}
  .provenance {{ background:#f8fafc; border:1px solid #e2e8f0;
                 border-radius:8px; padding:10px 14px; margin:14px 0 22px;
                 font-size:0.85em; }}
  .provenance dt {{ display:inline-block; min-width:110px; color:#475569;
                    font-weight:600; }}
  .provenance dd {{ display:inline; margin:0;
                    font-family: ui-monospace, Menlo, monospace; }}
  .provenance .row {{ margin: 1px 0; }}
  table {{ border-collapse: collapse; margin: 12px 0; width: 100%;
           font-size: 0.92em; }}
  th, td {{ padding: 6px 12px; border: 1px solid #e5e7eb;
            text-align: left; vertical-align: top; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  td.num {{ text-align: right;
            font-family: ui-monospace, Menlo, monospace; }}
  .takeaway {{ background:#dcfce7; border-left:4px solid #16a34a;
               padding:14px 18px; margin:18px 0; }}
  .takeaway h3 {{ margin-top: 0; color:#15803d; }}
  .next {{ background:#eff6ff; border-left:4px solid #3b82f6;
           padding:14px 18px; margin:18px 0; }}
  .next h3 {{ margin-top: 0; color:#1e40af; }}
  code {{ background: rgba(0,0,0,0.04); padding: 1px 6px;
          border-radius: 3px; font-size: 0.9em; }}
</style>

<h1>Phase 3 — Inference Infrastructure (closeout)</h1>
<p class="meta">
  Investigation <code>v2ecoli-pdmp</code> · Phase 3 of 6
  · PR <a href="https://github.com/vivarium-collective/v2ecoli/pull/100">#100</a>
  · {len(SPRINTS)} sprints landed
</p>

<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{prov["generated"]}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="https://github.com/vivarium-collective/v2ecoli/commit/{prov['sha']}"
       style="color:#0369a1;text-decoration:none">{prov['short']}</a>
       &nbsp;<code>{prov['sha']}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{prov["branch"]}</dd></div>
  <div class="row"><dt>host</dt><dd>{prov["host"]} &nbsp;
    <span style="color:#94a3b8">Python {prov["python"]}</span></dd></div>
</div>

<h2>Where Phase 3 stands</h2>
<p>
  Phase 3's stated scope is "observation likelihoods + likelihood
  accumulator + Vivarium observe/intervene effects; ABC-SMC baseline".
  Sprints 1–9 deliver the full mechanical loop and validate it on
  synthetic data:
</p>
<ul>
  <li><strong>Likelihoods</strong> per Phase-2 jump process (sprints 1, 2).</li>
  <li><strong>Aggregate collector</strong> step writing
      <code>listeners.likelihood.total</code> (sprint 2).</li>
  <li><strong>Persistence</strong> via XArrayEmitter to per-replicate
      zarr stores (sprint 4, after sprint 3's SQLite misstep).</li>
  <li><strong>Ensemble-shaped readback</strong> as
      <code>xarray.Dataset(replicate × time × observable)</code> —
      the canonical inference data shape (sprint 4).</li>
  <li><strong>Reviewable figure</strong> with provenance banner (sprint 5).</li>
  <li><strong>Calibrated noise floor</strong> — pairwise (sprint 6) and
      mean-to-mean (sprint 8) — setting the ABC acceptance ε empirically.</li>
  <li><strong>One-parameter forward-model ABC stub</strong> on
      <code>transcript_init_prob_scale</code>, distinguishing truth
      from perturbations (sprint 7).</li>
  <li><strong>SMC sequential refinement</strong> showing posterior
      concentration around the truth as ε tightens (sprint 9).</li>
</ul>

<h2>Sprint timeline</h2>
<table>
  <tr><th>#</th><th>sprint</th><th>verdict</th><th>finding</th></tr>
  {sprint_rows}
</table>

<h2>Headline result — posterior concentration on synthetic data</h2>
<p>
  Sprint 7 added <code>transcript_init_prob_scale</code> to
  <code>TranscriptInitiation</code> (multiplies the per-promoter
  Poisson rate). 5 proposed scales × N=4 forward sims each. The
  "observed" trajectory is the sprint-4 N=8 scale=1.0 ensemble mean.
  Distance metric: SSE between proposed-ensemble-mean and
  observed-ensemble-mean per-tick <code>total</code> log-likelihood.
</p>

<h3>Per-scale forward-model distances</h3>
<table>
  <tr><th>proposed scale</th><th>SSE</th><th>SMC behavior</th></tr>
  {scale_rows}
</table>

<p>
  The truth (scale = 1.0) sits 4× below the next-closest proposal —
  the forward model is meaningfully sensitive to the parameter.
</p>

<h3>Sequential ε refinement</h3>
<p>
  Sprint 8 calibrated ε from the mean-to-mean null distribution
  (4+4 random splits of the sprint-4 reference, 200 resamples).
  Sprint 9 iterates ε through {{p95, p75, p50, p25, p05}}:
</p>
<table>
  <tr><th>ε percentile</th><th>ε value</th>
      <th>accepted posterior</th><th>accept</th></tr>
  {posterior_rows}
</table>

<div class="takeaway">
  <h3>Closeout — SMC posterior collapses to the truth</h3>
  <p>
    At p25 and p05 the posterior contains only <strong>scale = 1.0</strong>,
    the synthetic truth. Each ε step monotonically eliminates the most
    extreme remaining proposal — classic SMC refinement. The
    Phase-2 closeout finding (cell_mass is the wrong observable; count-
    level listeners carry the jump-process signal) is now operationally
    actionable: a real ABC-SMC driver can read the persisted per-tick
    <code>likelihood.total</code> directly from disk and invert
    transcript-side parameters with no further infrastructure changes.
  </p>
</div>

<div class="next">
  <h3>Remaining ABC-SMC nice-to-haves (deferrable)</h3>
  <ol>
    <li>Proposal-perturbation kernel (Gaussian random walk around the
        accepted region) for real SMC propagation between rounds.</li>
    <li>Importance reweighting between rounds.</li>
    <li>Multi-parameter inference — extend the one-parameter
        infrastructure to (rate_scale, elongation_rate, ...) tuples.</li>
    <li>Bonus: fix the XArrayEmitter trailing-NaN at the source
        (loosen <code>buf_size ≥ 3</code> minimum).</li>
  </ol>
  <p>
    All four build directly on the persisted per-scale zarrs and the
    pipeline plumbing already in place; none block declaring Phase 3
    mechanically complete.
  </p>
</div>

<h2>Evidence — individual figures</h2>
<ul>
  <li><a href="../../../reports/figures/pdmp-03/likelihood_ensemble.html">
      Sprint 5+6 ensemble figure + intra-ensemble pairwise noise floor</a></li>
  <li><a href="../../../reports/figures/pdmp-03/abc_smc_stub.html">
      Sprint 7+8 ABC stub + mean-to-mean ε correction</a></li>
  <li><a href="../../../reports/figures/pdmp-03/abc_smc_sequential.html">
      Sprint 9 SMC acceptance staircase</a></li>
  <li><a href="pdmp-02-jump-processes.html">
      Phase-2 closeout report — why count-level observables matter</a></li>
</ul>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = Path("investigations/v2ecoli-pdmp/reports/pdmp-03-inference.html")
    write_html(out)
