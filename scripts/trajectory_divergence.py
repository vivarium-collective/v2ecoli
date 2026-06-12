"""PDMP+consumption_matched vs baseline kFBA — trajectory-shape divergence.

Phase 1 closed with endpoint mass within Phase-0's ±σ band, but
``cm_max_abs_z = 1594`` over the trajectory: somewhere mid-run, the
PDMP composite diverged hugely from the reference's mean-at-that-time
and then came back by 600 s. This script characterises *where* in time
that divergence peaks and *which submass* drives it.

Runs the v2ecoli ``baseline`` (kFBA) composite and the PDMP composite
(``with_ref_growth=True, ref_growth_flux_source='consumption_matched'``)
for the same duration + seed, sampling every tick. Renders 5 panels:

  1. cell_mass(t) overlaid (baseline blue, PDMP purple)
  2. dry_mass(t) overlaid
  3. Δcm(t) = PDMP − baseline, with the timepoint of max|Δcm| marked
  4. Per-submass divergence over time (one curve per submass)
  5. Submass-composition bar at the moment of max divergence

Optional: ``--pdmp-transcript-mode {discrete,poisson}`` to also see
how the Phase-2 jump-process toggle changes the shape.

Usage:
    .venv/bin/python scripts/trajectory_divergence.py
        [--duration 600] [--seed 0]
        [--pdmp-transcript-mode discrete]
        [--pdmp-polypeptide-mode discrete]
        [--out reports/figures/pdmp-02/trajectory_divergence.html]
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
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

# Submasses to trace across the run.
SUBMASS_FIELDS = (
    "cell_mass",
    "dry_mass",
    "water_mass",
    "protein_mass",
    "rna_mass",
    "dna_mass",
    "smallMolecule_mass",
    "rRna_mass",
    "mRna_mass",
    "tRna_mass",
)


def run_with_per_tick_mass(
    composite_name: str,
    duration_s: int,
    seed: int = 0,
    **build_kwargs,
) -> dict:
    from v2ecoli import build_composite

    t0 = time.perf_counter()
    c = build_composite(composite_name, seed=seed, **build_kwargs)
    build_wall = time.perf_counter() - t0

    series: dict[str, list[float]] = {f: [] for f in SUBMASS_FIELDS}
    times: list[int] = []

    t1 = time.perf_counter()
    for tick in range(1, duration_s + 1):
        c.run(1)
        mass = (
            c.state.get("agents", {}).get("0", {})
            .get("listeners", {}).get("mass", {})
        )
        times.append(tick)
        for f in SUBMASS_FIELDS:
            series[f].append(float(mass.get(f, 0.0)))
    run_wall = time.perf_counter() - t1

    return {
        "composite": composite_name,
        "build_kwargs": build_kwargs,
        "duration_s": duration_s,
        "build_wall_s": build_wall,
        "run_wall_s": run_wall,
        "times": times,
        "series": series,
    }


def make_viz(baseline: dict, pdmp: dict):
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)
    fig.suptitle(
        f"PDMP+consumption_matched − baseline kFBA — trajectory-shape divergence "
        f"({baseline['duration_s']} s, seed=0)",
        fontsize=13, fontweight="bold",
    )

    t = np.asarray(baseline["times"])
    cm_b = np.asarray(baseline["series"]["cell_mass"])
    cm_p = np.asarray(pdmp["series"]["cell_mass"])
    dm_b = np.asarray(baseline["series"]["dry_mass"])
    dm_p = np.asarray(pdmp["series"]["dry_mass"])
    dcm = cm_p - cm_b
    ddm = dm_p - dm_b
    t_peak = int(t[np.argmax(np.abs(dcm))])

    # --- Panel 1: cell_mass overlay ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, cm_b, color="#3b82f6", lw=1.4, label="baseline (kFBA)")
    ax.plot(t, cm_p, color="#a855f7", lw=1.4, label="PDMP+cm")
    ax.axvline(t_peak, color="#dc2626", ls="--", lw=0.9, alpha=0.6,
               label=f"max|Δcm| @ t={t_peak} s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cell_mass (fg)")
    ax.set_title("cell_mass(t)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)

    # --- Panel 2: dry_mass overlay ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, dm_b, color="#3b82f6", lw=1.4, label="baseline (kFBA)")
    ax.plot(t, dm_p, color="#a855f7", lw=1.4, label="PDMP+cm")
    ax.axvline(t_peak, color="#dc2626", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("dry_mass (fg)")
    ax.set_title("dry_mass(t)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)

    # --- Panel 3: Δcm(t) divergence ---
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(t, dcm, 0, where=(dcm > 0), color="#a855f7", alpha=0.3,
                    label="PDMP heavier")
    ax.fill_between(t, dcm, 0, where=(dcm <= 0), color="#3b82f6", alpha=0.3,
                    label="baseline heavier")
    ax.plot(t, dcm, color="#1f2937", lw=1.0)
    ax.axhline(0, color="#475569", lw=0.6)
    ax.axvline(t_peak, color="#dc2626", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δ cell_mass (fg)")
    ax.set_title("Cell-mass divergence: PDMP − baseline")
    ax.text(0.02, 0.98,
            f"endpoint Δ = {dcm[-1]:+.1f} fg\n"
            f"peak |Δ| = {np.abs(dcm).max():.1f} fg @ t={t_peak} s",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="#f3f4f6", edgecolor="#e5e7eb",
                      boxstyle="round,pad=0.3"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)

    # --- Panel 4: per-submass divergence over time ---
    ax = fig.add_subplot(gs[1, 1])
    palette = {
        "protein_mass":     "#10b981",
        "rna_mass":         "#0ea5e9",
        "dna_mass":         "#f59e0b",
        "smallMolecule_mass":"#ef4444",
        "water_mass":       "#94a3b8",
        "rRna_mass":        "#06b6d4",
        "mRna_mass":        "#7c3aed",
        "tRna_mass":        "#db2777",
    }
    for field in ("protein_mass", "rna_mass", "dna_mass",
                  "smallMolecule_mass", "water_mass"):
        if field not in baseline["series"]:
            continue
        delta = (np.asarray(pdmp["series"][field])
                 - np.asarray(baseline["series"][field]))
        ax.plot(t, delta, color=palette.get(field, "#000"), lw=1.1,
                label=field.replace("_mass", ""))
    ax.axhline(0, color="#475569", lw=0.6)
    ax.axvline(t_peak, color="#dc2626", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Δ mass (fg)")
    ax.set_title("Per-submass divergence: PDMP − baseline")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, linestyle=":", alpha=0.4)

    # --- Panel 5: bar chart at t_peak ---
    ax = fig.add_subplot(gs[2, :])
    peak_idx = int(np.argmax(np.abs(dcm)))
    bar_fields = [
        "protein_mass", "rna_mass", "dna_mass",
        "smallMolecule_mass", "water_mass",
        "rRna_mass", "mRna_mass", "tRna_mass",
    ]
    deltas = [
        float(np.asarray(pdmp["series"][f])[peak_idx]
              - np.asarray(baseline["series"][f])[peak_idx])
        if f in baseline["series"] else 0.0
        for f in bar_fields
    ]
    colors = ["#ef4444" if d > 0 else "#3b82f6" for d in deltas]
    bars = ax.bar(
        [f.replace("_mass", "") for f in bar_fields],
        deltas, color=colors, edgecolor="white", linewidth=1,
    )
    for b, d in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_y() + b.get_height() + (1.0 if d >= 0 else -1.0),
                f"{d:+.1f}",
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=9, color="#1f2937")
    ax.axhline(0, color="#475569", lw=0.6)
    ax.set_ylabel("Δ submass (fg)")
    ax.set_title(f"Submass divergence at t = {t_peak} s (peak |Δcm|)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    return fig, t_peak, dcm, ddm


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(out_path: Path, data_uri: str, baseline: dict, pdmp: dict,
               t_peak: int, dcm: np.ndarray, ddm: np.ndarray,
               provenance: dict) -> None:
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if provenance.get("git_dirty") else ""
    )
    short = provenance.get("git_short", "")
    full_sha = provenance.get("git_sha", "")
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Trajectory-shape divergence — {short or 'report'}</title>
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
<h1>PDMP+cm − baseline kFBA: trajectory-shape divergence</h1>
<div class="meta">
  Phase 2 follow-up to the PR-72 endpoint-vs-shape note. Sim duration:
  <code>{baseline['duration_s']} s</code>, seed=0. Per-tick sampling.
  Composites: <code>{baseline['composite']}</code> /
  <code>{pdmp['composite']}</code> with
  <code>{json.dumps(pdmp['build_kwargs'])}</code>.
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
  <tr><th></th><th>baseline (kFBA)</th><th>PDMP+cm</th><th>Δ</th></tr>
  <tr><td>run wall</td>
      <td class="num">{baseline['run_wall_s']:.2f} s</td>
      <td class="num">{pdmp['run_wall_s']:.2f} s</td>
      <td class="num">{pdmp['run_wall_s'] - baseline['run_wall_s']:+.2f} s</td></tr>
  <tr><td>cell_mass at t=0</td>
      <td class="num">{baseline['series']['cell_mass'][0]:.2f} fg</td>
      <td class="num">{pdmp['series']['cell_mass'][0]:.2f} fg</td>
      <td class="num">{pdmp['series']['cell_mass'][0] - baseline['series']['cell_mass'][0]:+.2f} fg</td></tr>
  <tr><td>cell_mass at t={baseline['duration_s']}</td>
      <td class="num">{baseline['series']['cell_mass'][-1]:.2f} fg</td>
      <td class="num">{pdmp['series']['cell_mass'][-1]:.2f} fg</td>
      <td class="num">{dcm[-1]:+.2f} fg</td></tr>
  <tr><td>peak |Δcm|</td>
      <td colspan=2 class="num">{np.abs(dcm).max():.2f} fg @ t={t_peak} s</td>
      <td class="num"></td></tr>
  <tr><td>peak |Δdm|</td>
      <td colspan=2 class="num">{np.abs(ddm).max():.2f} fg</td>
      <td class="num"></td></tr>
</table>

<h2 style="margin-top:24px;">Divergence panels</h2>
<img src="{data_uri}" alt="trajectory divergence">

<h2 style="margin-top:24px;">Reading the figure</h2>
<p style="font-size:0.95em; line-height:1.55;">
The top row is the trajectories themselves — same shape qualitatively,
quantitatively offset. The middle row pinpoints <em>when</em> the
PDMP-vs-baseline mismatch peaks and <em>what</em> submass leads. The
bottom bar chart freezes the moment of maximum cell-mass divergence and
decomposes it: which pools are pulling PDMP heavier (red bars) vs lighter
(blue bars) than baseline. The endpoint match from PR&nbsp;#72 reads here
as a near-zero net cell_mass at t={baseline['duration_s']} s but a non-
trivial trajectory shape — exactly the "mid-run divergence" note.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pdmp-composite", default="millard_pdmp_baseline")
    p.add_argument("--pdmp-transcript-mode", default="discrete",
                   choices=["discrete", "poisson"])
    p.add_argument("--pdmp-polypeptide-mode", default="discrete",
                   choices=["discrete", "poisson"])
    p.add_argument("--out",
                   default="reports/figures/pdmp-02/trajectory_divergence.html")
    args = p.parse_args()

    print(f"Running baseline for {args.duration} s...", flush=True)
    baseline = run_with_per_tick_mass(
        "baseline", args.duration, args.seed,
        transcript_initiation_mode=args.pdmp_transcript_mode,
        polypeptide_initiation_mode=args.pdmp_polypeptide_mode,
    )
    print(f"  done: run_wall={baseline['run_wall_s']:.2f} s", flush=True)

    print(f"Running PDMP+consumption_matched for {args.duration} s...",
          flush=True)
    pdmp = run_with_per_tick_mass(
        args.pdmp_composite, args.duration, args.seed,
        with_ref_growth=True,
        ref_growth_flux_source="consumption_matched",
        transcript_initiation_mode=args.pdmp_transcript_mode,
        polypeptide_initiation_mode=args.pdmp_polypeptide_mode,
    )
    print(f"  done: run_wall={pdmp['run_wall_s']:.2f} s", flush=True)

    provenance = collect_provenance(extra={
        "duration_s": args.duration, "seed": args.seed,
        "pdmp_transcript_mode": args.pdmp_transcript_mode,
        "pdmp_polypeptide_mode": args.pdmp_polypeptide_mode,
    })

    fig, t_peak, dcm, ddm = make_viz(baseline, pdmp)
    data_uri = fig_to_data_uri(fig)
    write_html(Path(args.out), data_uri, baseline, pdmp,
               t_peak, dcm, ddm, provenance)
    print(f"Wrote viz {args.out}", flush=True)

    short = provenance.get("git_short") or "nogit"
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = Path(args.out).with_name(
        f"trajectory_divergence_{stamp}_{short}.html"
    )
    archive.write_bytes(Path(args.out).read_bytes())
    print(f"Wrote archive {archive}", flush=True)


if __name__ == "__main__":
    main()
