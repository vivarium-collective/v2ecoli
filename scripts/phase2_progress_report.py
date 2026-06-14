"""Generate the Phase-2 closeout report for the v2ecoli-pdmp investigation.

Renders ``investigations/v2ecoli-pdmp/reports/pdmp-02-jump-processes.html``
with a sprint timeline, a 3-panel summary plot comparing PDMP-vs-baseline
ensemble statistics across the controller-knob sprints (6 / 7 / 9), and
links to the individual ensemble HTML artifacts as evidence.

The 3 panels surface the structural finding:
  1. σ at cm[-1]: PDMP stays sub-1 fg across all three knobs; baseline
     stays ~8–9 fg. The σ gap doesn't close.
  2. W₂(PDMP, baseline): GROWS monotonically as the controller loosens
     (sprint 6 → 7 → 9). Loosening makes things worse, not better.
  3. Mean offset μ(PDMP) − μ(baseline): the PDMP mean DROPS by 43 fg
     under sparse injection, surfacing the homeostat-collapses-to-target
     mechanism.
"""
from __future__ import annotations

import base64
import io
import subprocess
import socket
import platform
import sys
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sprint ensemble numbers — read out of the archived ensemble_validation
# HTMLs by parsing the summary table. Hardcoded here so the report stays
# reproducible even after the source HTML files churn.
SPRINTS = {
    "sprint 6\n(tau=1)": {
        "knob": "default",
        "discrete": dict(base_mu=1456.99, base_sd=7.97,
                         pdmp_mu=1458.85, pdmp_sd=0.42, w2_cm=7.79),
        "poisson":  dict(base_mu=1462.62, base_sd=9.17,
                         pdmp_mu=1458.56, pdmp_sd=0.45, w2_cm=9.73),
        "report":   "../../reports/figures/pdmp-02/ensemble_validation.html",
    },
    "sprint 7\n(tau=60 EMA)": {
        "knob": "feedback_tau_s=60",
        "discrete": dict(base_mu=1456.99, base_sd=7.97,
                         pdmp_mu=1449.67, pdmp_sd=0.59, w2_cm=10.42),
        "poisson":  dict(base_mu=1462.62, base_sd=9.17,
                         pdmp_mu=1449.93, pdmp_sd=0.59, w2_cm=15.35),
        "report":   "../../reports/figures/pdmp-02/ensemble_validation_tau60.html",
    },
    "sprint 9\n(period=60)": {
        "knob": "feedback_period_ticks=60",
        "discrete": dict(base_mu=1456.99, base_sd=7.97,
                         pdmp_mu=1413.95, pdmp_sd=0.08, w2_cm=43.76),
        "poisson":  dict(base_mu=1462.62, base_sd=9.17,
                         pdmp_mu=1414.00, pdmp_sd=0.11, w2_cm=49.46),
        "report":   "../../reports/figures/pdmp-02/ensemble_validation_period60.html",
    },
}

SPRINT_NARRATIVE = [
    (1, "TranscriptInitiation Poisson-per-promoter", "✅",
     "Added `pdmp_initiation_mode='poisson'`: per-promoter "
     "Poisson(n_target · p_i) sampling with resource cap pinned to "
     "actual inactive RNAP pool. Unit-equivalent to multinomial in mean."),
    (2, "Composite wiring + initiation-mode comparison viz", "✅",
     "Threaded transcript_initiation_mode through baseline + PDMP "
     "composites. Sanity-check viz: 4-panel comparison of per-tick "
     "initiation counts, cumulative, histogram, per-TU scatter."),
    (3, "PolypeptideInitiation Poisson-per-protein", "✅",
     "Same per-protein Poisson-tau-leap pattern as sprint 1. "
     "Resource cap = min(30S, 50S) inactive_ribosome_count."),
    (4, "Trajectory-shape divergence analysis", "✅",
     "Per-tick sampling across baseline kFBA + PDMP+cm. "
     "Result: peak |Δcm| = 150 fg at t=600 at the rnap_data listener "
     "(not at the cell_mass listener — the variance lives at counts)."),
    (5, "Data-driven water rate + Poisson-mode divergence", "✅",
     "Moved hard-coded WATER_RATE_PER_S out of source; sampler "
     "now walks WATER[c] in the kFBA precursor JSON. Driver reads "
     "the rate from data, falls back to the constant if absent."),
    (6, "N=8 W₂ ensemble validation (baseline)", "⚠️ Poisson INCREASED W₂",
     "First full ensemble (PDMP+cm vs baseline kFBA, N=8 ea, 600 s). "
     "Honest finding: poisson mode WIDENED W₂ vs discrete because "
     "PDMP+cm σ stays at 0.42 fg (controller homeostat damps) while "
     "baseline σ widens to 9.17 fg. Surfaces the structural question "
     "we spent the next three sprints answering."),
    (7, "Smoothed `consumption_matched` (EMA tau=60)", "❌ EMA on wrong term",
     "Added `feedback_tau_s`: EMA-smooth the inferred other_delta. "
     "Result: PDMP σ went 0.42 → 0.59 fg (basically no change); "
     "W₂ INCREASED to 10.42 / 15.35 fg. EMA only smooths the "
     "correction; target_delta = rate·tick_s still hard-pins."),
    (8, "Open-loop `measured_kfba` retest", "❌ ATP starvation",
     "Re-tested whether Poisson sampling rescues the pre-Phase-2 "
     "open-loop failure mode. It does not: ATP goes negative at "
     "t≈240 s regardless of sampler. Poisson cuts variance, not mean."),
    (9, "Sparse-injection `consumption_matched` (period=60)", "❌ catchup homeostasis",
     "Added `feedback_period_ticks`: controller acts every Nth tick, "
     "compensating cumulative consumption with a period-scaled target. "
     "PDMP mean dropped 43 fg (less total production); PDMP σ collapsed "
     "to 0.08–0.11 fg (catchup at each boundary perfectly resynchronizes "
     "all seeds). W₂ exploded to 43.76 / 49.46 fg."),
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


def make_summary_plot():
    labels = list(SPRINTS.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: σ at cm[-1] — PDMP vs baseline, both modes
    ax = axes[0]
    pdmp_sd_d = [SPRINTS[s]["discrete"]["pdmp_sd"] for s in labels]
    pdmp_sd_p = [SPRINTS[s]["poisson"]["pdmp_sd"]  for s in labels]
    base_sd_d = [SPRINTS[s]["discrete"]["base_sd"] for s in labels]
    base_sd_p = [SPRINTS[s]["poisson"]["base_sd"]  for s in labels]
    ax.bar(x - 1.5*width/2, base_sd_d, width/2, color="#94a3b8",
           label="baseline (discrete)")
    ax.bar(x - 0.5*width/2, base_sd_p, width/2, color="#475569",
           label="baseline (poisson)")
    ax.bar(x + 0.5*width/2, pdmp_sd_d, width/2, color="#60a5fa",
           label="PDMP+cm (discrete)")
    ax.bar(x + 1.5*width/2, pdmp_sd_p, width/2, color="#1e3a8a",
           label="PDMP+cm (poisson)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("σ at cm[-1] (fg)")
    ax.set_title("σ at cm[-1] — PDMP stays sub-1 fg across all knobs")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: W₂ at cm[-1]
    ax = axes[1]
    w2_d = [SPRINTS[s]["discrete"]["w2_cm"] for s in labels]
    w2_p = [SPRINTS[s]["poisson"]["w2_cm"]  for s in labels]
    ax.bar(x - width/2, w2_d, width, color="#7c3aed", label="discrete sampler")
    ax.bar(x + width/2, w2_p, width, color="#c084fc", label="poisson sampler")
    for i, (vd, vp) in enumerate(zip(w2_d, w2_p)):
        ax.text(i - width/2, vd + 0.6, f"{vd:.1f}", ha="center", fontsize=8)
        ax.text(i + width/2, vp + 0.6, f"{vp:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("W₂(PDMP, baseline) at cm[-1] (fg)")
    ax.set_title("W₂ — GROWS monotonically as the controller loosens")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: mean offset μ(PDMP) − μ(baseline)
    ax = axes[2]
    off_d = [SPRINTS[s]["discrete"]["pdmp_mu"] -
             SPRINTS[s]["discrete"]["base_mu"] for s in labels]
    off_p = [SPRINTS[s]["poisson"]["pdmp_mu"] -
             SPRINTS[s]["poisson"]["base_mu"]  for s in labels]
    ax.bar(x - width/2, off_d, width, color="#0891b2",
           label="discrete sampler")
    ax.bar(x + width/2, off_p, width, color="#67e8f9",
           label="poisson sampler")
    for i, (vd, vp) in enumerate(zip(off_d, off_p)):
        ax.text(i - width/2, vd - 1.8, f"{vd:+.1f}", ha="center", fontsize=8)
        ax.text(i + width/2, vp - 1.8, f"{vp:+.1f}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("μ(PDMP) − μ(baseline) at cm[-1] (fg)")
    ax.set_title("Mean offset — sparse injection produces LESS mass")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Phase-2 sprint comparison: PDMP+cm vs baseline kFBA, "
        "N=8 each, 600 s",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path) -> None:
    prov = collect_provenance()
    plot_uri = make_summary_plot()

    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    # Sprint narrative table.
    sprint_rows = "\n".join(
        f"<tr><td>{n}</td><td>{title}</td><td>{verdict}</td>"
        f"<td>{detail}</td></tr>"
        for n, title, verdict, detail in SPRINT_NARRATIVE
    )

    # Ensemble report links.
    ensemble_links = "\n".join(
        f'<li><a href="{cfg["report"]}">{label.replace(chr(10), " ")}</a> '
        f'— knob: <code>{cfg["knob"]}</code></li>'
        for label, cfg in SPRINTS.items()
    )

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 2 progress — v2ecoli-pdmp investigation</title>
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
  .takeaway {{ background:#fef3c7; border-left:4px solid #f59e0b;
               padding:14px 18px; margin:18px 0; }}
  .takeaway h3 {{ margin-top: 0; color:#92400e; }}
  .next {{ background:#dcfce7; border-left:4px solid #16a34a;
           padding:14px 18px; margin:18px 0; }}
  .next h3 {{ margin-top: 0; color:#15803d; }}
  img.plot {{ max-width: 100%; border:1px solid #e2e8f0; border-radius:6px;
              margin: 6px 0 14px; }}
  ul.evidence {{ font-size: 0.92em; }}
  code {{ background: rgba(0,0,0,0.04); padding: 1px 6px;
          border-radius: 3px; font-size: 0.9em; }}
</style>

<h1>Phase 2 — Jump processes (in-progress closeout)</h1>
<p class="meta">
  Investigation <code>v2ecoli-pdmp</code> · Phase 2 of 6
  · PR <a href="https://github.com/vivarium-collective/v2ecoli/pull/100">#100</a>
  · {len(SPRINT_NARRATIVE)} sprints landed
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

<h2>Where Phase 2 stands</h2>
<p>
  Phase 2's structural goal — replace discrete-time multinomial
  initiation with continuous-time per-promoter / per-protein Poisson
  tau-leap — <strong>is mechanically working</strong>. Sprints 1–5
  landed the sampler changes, composite wiring, sampler-comparison
  visualizations, and a data-driven water-rate cleanup. Sprint 4's
  trajectory-shape analysis confirmed the Poisson signal exists at the
  <code>rnap_data</code> listener with a 150 fg peak |Δcm|.
</p>
<p>
  Sprints 6–9 then asked a quantitative question: <em>does the
  Phase-2 Poisson sampler also close the ensemble W₂ gap to the
  kFBA baseline at the cell_mass listener?</em> The answer, after
  three controller-tuning attempts, is <strong>no — and the reason is
  structural, not algorithmic</strong>. The headline plot below
  surfaces it.
</p>

<h2>Sprint timeline</h2>
<table>
  <tr><th>#</th><th>sprint</th><th>verdict</th><th>finding</th></tr>
  {sprint_rows}
</table>

<h2>Headline comparison — sprints 6 / 7 / 9</h2>
<p>
  Three controller knobs were tested:
  default (every-tick compensation), EMA-smoothed correction
  (<code>feedback_tau_s=60</code>), and sparse injection
  (<code>feedback_period_ticks=60</code>). All three ran the same N=8 × 600 s
  ensemble protocol with both <code>discrete</code> (legacy multinomial)
  and <code>poisson</code> (Phase-2 tau-leap) samplers.
</p>
<img class="plot" src="{plot_uri}"
     alt="3-panel sprint comparison: σ, W₂, mean offset">

<h3>Detailed numbers</h3>
<table>
  <tr><th rowspan="2">sprint</th><th rowspan="2">knob</th>
      <th colspan="3">discrete sampler</th>
      <th colspan="3">poisson sampler</th></tr>
  <tr><th>baseline μ±σ</th><th>PDMP μ±σ</th><th>W₂(cm)</th>
      <th>baseline μ±σ</th><th>PDMP μ±σ</th><th>W₂(cm)</th></tr>
""" + "\n".join(
        '<tr><td>' + label.replace('\n', ' ') + '</td>'
        f'<td><code>{cfg["knob"]}</code></td>'
        f'<td class="num">{cfg["discrete"]["base_mu"]:.2f} ± '
        f'{cfg["discrete"]["base_sd"]:.2f}</td>'
        f'<td class="num">{cfg["discrete"]["pdmp_mu"]:.2f} ± '
        f'{cfg["discrete"]["pdmp_sd"]:.2f}</td>'
        f'<td class="num">{cfg["discrete"]["w2_cm"]:.2f}</td>'
        f'<td class="num">{cfg["poisson"]["base_mu"]:.2f} ± '
        f'{cfg["poisson"]["base_sd"]:.2f}</td>'
        f'<td class="num">{cfg["poisson"]["pdmp_mu"]:.2f} ± '
        f'{cfg["poisson"]["pdmp_sd"]:.2f}</td>'
        f'<td class="num">{cfg["poisson"]["w2_cm"]:.2f}</td></tr>'
        for label, cfg in SPRINTS.items()
    ) + f"""
</table>

<div class="takeaway">
  <h3>Structural takeaway</h3>
  <p>
    The <code>consumption_matched</code> ref-growth driver is <strong>by
    construction a homeostat</strong>: it compensates the OTHER processes'
    per-tick consumption to keep the bulk precursor pools at the kFBA-
    measured steady state. Any version of it that fully closes the
    consumption loop — even with EMA smoothing or sparse decimation —
    enforces a per-period production budget on the PDMP ensemble.
    That budget collapses per-tick variance at the bulk-pool level and
    therefore at the cell_mass listener. The σ across all three knobs
    stays below 1 fg on the PDMP side while baseline σ holds at ~8–9 fg.
  </p>
  <p>
    <strong>Cell_mass is the wrong observable for Phase-2 variance.</strong>
    The Poisson signal exists — sprint 4 found peak |Δcm| = 150 fg at
    <code>rnap_data</code>, which is upstream of the homeostat. It just
    doesn't reach the mass listener.
  </p>
</div>

<div class="next">
  <h3>Implication for Phase 3</h3>
  <p>
    Anchor likelihood inference on count-level observables
    (<code>rnap_data</code>, <code>ribosome_data</code>,
    <code>monomer_counts</code>) rather than aggregate cell_mass. Those
    listeners observe the jump-process variance directly, before the
    metabolic homeostat washes it out.
  </p>
  <p>
    Phase 2's mechanical deliverables (sprints 1–5) are <strong>done</strong>
    and ready to gate into Phase 3 — the count-listener observation
    likelihoods are the natural next sprint.
  </p>
</div>

<h2>Evidence — individual ensemble reports</h2>
<ul class="evidence">
  {ensemble_links}
  <li><a href="../../reports/figures/pdmp-02/trajectory_divergence.html">
      sprint-4 trajectory-shape divergence</a>
      — peak |Δcm| = 150 fg at rnap_data listener</li>
  <li><a href="../../reports/figures/pdmp-02/initiation_modes_comparison.html">
      sprint-2 initiation-modes comparison</a>
      — discrete vs poisson per-promoter event counts</li>
</ul>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = Path("investigations/v2ecoli-pdmp/reports/"
               "pdmp-02-jump-processes.html")
    write_html(out)
