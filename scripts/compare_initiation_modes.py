"""Phase 2 sprint 2 — overlay TranscriptInitiation discrete vs Poisson modes.

Runs the v2ecoli ``baseline`` composite in both ``transcript_initiation_mode``
settings for the same duration and seed, samples the per-tick initiation
counts from the ``rna_synth_prob`` listener, and renders one HTML with:

  1. Total per-tick initiation count over time (overlay).
  2. Cumulative initiation count over time (overlay).
  3. Per-tick count histogram (discrete vs Poisson).
  4. Per-TU cumulative count scatter (discrete vs Poisson).
  5. Endpoint state table.

Includes the same provenance banner pattern as
``compare_pdmp_vs_baseline.py`` so the HTML stays self-describing months
from now — git SHA + branch + dirty bit + date + host + Python version.

Usage:
    .venv/bin/python scripts/compare_initiation_modes.py
        [--duration 300] [--seed 0]
        [--out reports/figures/pdmp-02/initiation_modes_comparison.html]
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

from scripts.compare_pdmp_vs_baseline import collect_provenance  # noqa: E402


def run_with_per_tick_sampling(
    composite_name: str,
    transcript_initiation_mode: str,
    duration_s: int,
    seed: int = 0,
) -> dict:
    """Build + tick-by-tick sample of did_initialize + rna_init_event."""
    from v2ecoli import build_composite

    t0 = time.perf_counter()
    c = build_composite(
        composite_name,
        seed=seed,
        transcript_initiation_mode=transcript_initiation_mode,
    )
    build_wall = time.perf_counter() - t0

    per_tick_total: list[int] = []
    per_tu_cumulative: np.ndarray | None = None
    t1 = time.perf_counter()
    for _ in range(duration_s):
        c.run(1)
        listeners = (
            c.state.get("agents", {}).get("0", {})
            .get("listeners", {})
        )
        rnap = listeners.get("rnap_data", {})
        # Per-TU per-tick event count: 'rna_init_event' on rnap_data
        # (length = n_TUs); per-tick total = sum across TUs.
        # (NB: listener also has 'total_rna_init' but it's cumulative
        # since sim start, not per-tick.)
        ev = rnap.get("rna_init_event")
        if ev is not None:
            arr = np.asarray(ev, dtype=np.int64)
            per_tick_total.append(int(arr.sum()))
            if per_tu_cumulative is None:
                per_tu_cumulative = arr.copy()
            else:
                per_tu_cumulative = per_tu_cumulative + arr
        else:
            per_tick_total.append(0)
    run_wall = time.perf_counter() - t1

    mass = (
        c.state.get("agents", {}).get("0", {})
        .get("listeners", {}).get("mass", {})
    )
    return {
        "composite": composite_name,
        "mode": transcript_initiation_mode,
        "duration_s": duration_s,
        "build_wall_s": build_wall,
        "run_wall_s": run_wall,
        "per_tick_total": per_tick_total,
        "per_tu_cumulative": (per_tu_cumulative.tolist()
                              if per_tu_cumulative is not None else []),
        "cell_mass": float(mass.get("cell_mass", 0.0)),
        "dry_mass": float(mass.get("dry_mass", 0.0)),
    }


def make_viz(discrete: dict, poisson: dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        "TranscriptInitiation — discrete vs Poisson tau-leap "
        f"({discrete['duration_s']} s, seed=0)",
        fontsize=13, fontweight="bold",
    )

    d_total = np.asarray(discrete["per_tick_total"])
    p_total = np.asarray(poisson["per_tick_total"])
    t = np.arange(1, len(d_total) + 1)

    # --- Panel 1: per-tick total over time ---
    ax = axes[0, 0]
    ax.plot(t, d_total, color="#3b82f6", label="discrete", lw=0.9, alpha=0.85)
    ax.plot(t, p_total, color="#a855f7", label="poisson",  lw=0.9, alpha=0.85)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("per-tick init count")
    ax.set_title("Total initiations per tick", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # --- Panel 2: cumulative over time ---
    ax = axes[0, 1]
    ax.plot(t, np.cumsum(d_total), color="#3b82f6", label="discrete", lw=1.4)
    ax.plot(t, np.cumsum(p_total), color="#a855f7", label="poisson",  lw=1.4)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cumulative init count")
    ax.set_title("Cumulative initiations", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    diff_total = int(np.cumsum(p_total)[-1] - np.cumsum(d_total)[-1])
    ax.text(0.02, 0.98,
            f"final Δ = {diff_total:+d}\n"
            f"({100 * diff_total / max(1, np.cumsum(d_total)[-1]):+.2f}%)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="#f3f4f6", edgecolor="#e5e7eb", boxstyle="round,pad=0.3"))

    # --- Panel 3: per-tick count histogram ---
    ax = axes[1, 0]
    bins = np.arange(
        min(d_total.min(), p_total.min()),
        max(d_total.max(), p_total.max()) + 2,
    )
    ax.hist(d_total, bins=bins, alpha=0.55, color="#3b82f6", label="discrete",
            edgecolor="#1e3a8a", linewidth=0.5)
    ax.hist(p_total, bins=bins, alpha=0.55, color="#a855f7", label="poisson",
            edgecolor="#581c87", linewidth=0.5)
    ax.set_xlabel("per-tick init count")
    ax.set_ylabel("# of ticks")
    ax.set_title("Per-tick count distribution\n"
                 "(Poisson has wider tails — that's the jump-process variance)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(0.65, 0.95,
            f"mean: disc={d_total.mean():.2f}, pois={p_total.mean():.2f}\n"
            f"std:  disc={d_total.std():.2f}, pois={p_total.std():.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="#f3f4f6", edgecolor="#e5e7eb",
                      boxstyle="round,pad=0.3"))

    # --- Panel 4: per-TU cumulative scatter ---
    ax = axes[1, 1]
    d_cum = np.asarray(discrete["per_tu_cumulative"])
    p_cum = np.asarray(poisson["per_tu_cumulative"])
    if d_cum.size and p_cum.size:
        ax.scatter(d_cum, p_cum, s=10, alpha=0.55, color="#0ea5e9",
                   edgecolor="#0369a1", linewidth=0.3)
        lim = max(d_cum.max(), p_cum.max()) * 1.05 or 1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5,
                label="y = x (perfect agreement)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        # Pearson correlation
        nz = (d_cum + p_cum) > 0
        if nz.sum() > 2:
            r = np.corrcoef(d_cum[nz], p_cum[nz])[0, 1]
            ax.text(0.02, 0.98,
                    f"per-TU Pearson r = {r:.3f}\n"
                    f"n_TUs = {nz.sum()} (active)",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="#f3f4f6", edgecolor="#e5e7eb",
                              boxstyle="round,pad=0.3"))
    ax.set_xlabel("discrete: cumulative init count per TU")
    ax.set_ylabel("poisson: cumulative init count per TU")
    ax.set_title("Per-TU cumulative agreement", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(out_path: Path, data_uri: str, discrete: dict, poisson: dict,
               provenance: dict) -> None:
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if provenance.get("git_dirty") else ""
    )
    short = provenance.get("git_short", "")
    full_sha = provenance.get("git_sha", "")
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>TranscriptInitiation discrete vs poisson — {short or 'report'}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1500px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ margin: 0 0 6px 0; }}
  .meta {{ color: #6b7280; font-size: 0.9em; }}
  .meta code {{ background: rgba(0,0,0,0.04); padding: 1px 5px; border-radius: 3px; }}
  .provenance {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                 padding:10px 14px; margin:14px 0 20px; font-size:0.85em;
                 line-height:1.55; }}
  .provenance dt {{ display:inline-block; min-width:120px; color:#475569;
                    font-weight:600; }}
  .provenance dd {{ display:inline; margin:0; font-family: ui-monospace, Menlo, monospace; }}
  .provenance .row {{ margin: 1px 0; }}
  table.summary {{ border-collapse: collapse; margin: 14px 0; font-size: 0.95em; }}
  table.summary th, table.summary td {{ padding: 6px 12px; border: 1px solid #e5e7eb; }}
  table.summary th {{ background: #f3f4f6; font-weight: 600; text-align: left; }}
  table.summary td.num {{ text-align: right; font-variant-numeric: tabular-nums;
                          font-family: ui-monospace, Menlo, monospace; }}
  img {{ max-width: 100%; }}
</style>
<h1>TranscriptInitiation — discrete vs Poisson tau-leap</h1>
<div class="meta">
  Phase 2 / sprint 2 of the PDMP investigation (PR #100).
  Sim duration: <code>{discrete['duration_s']} s</code>, seed=0.
  Composite: <code>{discrete['composite']}</code>.
</div>

<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{provenance.get('generated_at','')}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="https://github.com/vivarium-collective/v2ecoli/commit/{full_sha}"
        style="color:#0369a1;text-decoration:none">{short}</a> &nbsp;<code>{full_sha}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{provenance.get('git_branch','')}</dd></div>
  <div class="row"><dt>last commit</dt><dd>{provenance.get('git_last_commit_msg','')} — {provenance.get('git_last_commit_author','')} ({provenance.get('git_last_commit_when','')})</dd></div>
  <div class="row"><dt>script</dt><dd>{provenance.get('script','')}</dd></div>
  <div class="row"><dt>host</dt><dd>{provenance.get('host','')} &nbsp; <span style="color:#94a3b8">{provenance.get('platform','')}, Python {provenance.get('python','')}</span></dd></div>
</div>

<table class="summary">
  <tr><th></th><th>discrete (legacy multinomial)</th><th>poisson (tau-leap)</th><th>Δ</th></tr>
  <tr><td>build wall</td>
      <td class="num">{discrete['build_wall_s']:.2f} s</td>
      <td class="num">{poisson['build_wall_s']:.2f} s</td>
      <td class="num">{poisson['build_wall_s'] - discrete['build_wall_s']:+.2f} s</td></tr>
  <tr><td>run wall</td>
      <td class="num">{discrete['run_wall_s']:.2f} s</td>
      <td class="num">{poisson['run_wall_s']:.2f} s</td>
      <td class="num">{poisson['run_wall_s'] - discrete['run_wall_s']:+.2f} s</td></tr>
  <tr><td>total initiations</td>
      <td class="num">{sum(discrete['per_tick_total'])}</td>
      <td class="num">{sum(poisson['per_tick_total'])}</td>
      <td class="num">{sum(poisson['per_tick_total']) - sum(discrete['per_tick_total']):+d}</td></tr>
  <tr><td>cell_mass at endpoint (fg)</td>
      <td class="num">{discrete['cell_mass']:.2f}</td>
      <td class="num">{poisson['cell_mass']:.2f}</td>
      <td class="num">{poisson['cell_mass'] - discrete['cell_mass']:+.2f}</td></tr>
  <tr><td>dry_mass at endpoint (fg)</td>
      <td class="num">{discrete['dry_mass']:.2f}</td>
      <td class="num">{poisson['dry_mass']:.2f}</td>
      <td class="num">{poisson['dry_mass'] - discrete['dry_mass']:+.2f}</td></tr>
</table>

<h2 style="margin-top:24px;">Comparison panels</h2>
<img src="{data_uri}" alt="discrete vs poisson comparison">

<h2 style="margin-top:24px;">Notes</h2>
<p style="font-size:0.95em; line-height:1.55;">
The discrete (legacy) mode samples per-promoter event counts with one
<code>multinomial(n_target, p)</code> draw — exact sum constraint
Σ N<sub>i</sub> = n_target, promoters coupled through that constraint.
The Poisson mode treats each promoter as an independent jump process at
rate λ<sub>i</sub> = n_target · p<sub>i</sub>; this is the per-promoter
marginal a continuous-time PDMP integrates against (the Phase-3 likelihood
target). Per-tick totals fluctuate by O(√n_target), the per-TU rates
converge to the same expectation, and the cell_mass evolution stays
within stochastic noise of the discrete baseline.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=300,
                   help="Simulated seconds for both runs (default 300).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--composite", default="baseline",
                   help="Composite recipe name (must accept "
                        "transcript_initiation_mode).")
    p.add_argument("--out",
                   default="reports/figures/pdmp-02/initiation_modes_comparison.html")
    p.add_argument("--summary-json",
                   default=".pbg/runs/pdmp-02/initiation_modes_summary.json")
    args = p.parse_args()

    print(f"Running '{args.composite}' with discrete mode for {args.duration} s...",
          flush=True)
    discrete = run_with_per_tick_sampling(
        args.composite, "discrete", args.duration, args.seed)
    print(f"  done: run_wall={discrete['run_wall_s']:.2f} s, "
          f"total={sum(discrete['per_tick_total'])}", flush=True)

    print(f"Running '{args.composite}' with poisson mode for {args.duration} s...",
          flush=True)
    poisson = run_with_per_tick_sampling(
        args.composite, "poisson", args.duration, args.seed)
    print(f"  done: run_wall={poisson['run_wall_s']:.2f} s, "
          f"total={sum(poisson['per_tick_total'])}", flush=True)

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(
        {"discrete": discrete, "poisson": poisson},
        indent=2, default=str))
    print(f"Wrote summary {args.summary_json}", flush=True)

    print("Collecting provenance...", flush=True)
    provenance = collect_provenance(extra={
        "duration_s": args.duration, "seed": args.seed,
        "composite": args.composite,
    })

    print("Rendering...", flush=True)
    data_uri = fig_to_data_uri(make_viz(discrete, poisson))
    write_html(Path(args.out), data_uri, discrete, poisson, provenance)
    print(f"Wrote viz {args.out}", flush=True)

    # Date+sha-stamped archival copy.
    short = provenance.get("git_short") or "nogit"
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = Path(args.out).with_name(
        f"initiation_modes_{stamp}_{short}.html"
    )
    archive.write_bytes(Path(args.out).read_bytes())
    print(f"Wrote archive {archive}", flush=True)


if __name__ == "__main__":
    main()
