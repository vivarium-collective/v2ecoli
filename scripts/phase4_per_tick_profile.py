"""Phase 4 sprint 1 — real per-tick compute profile.

Replaces ``reports/figures/pdmp-04/per_step_compute_decomposition.html``
(previously a PLANNED scaffold) with actual cProfile measurements
of one composite tick of millard_pdmp_baseline + ref_growth_driver +
poisson initiation modes — the Phase-3 production composite that
Phase 4 needs to compile down.

Output: a single bar chart attributing per-tick exclusive time to:

  - Each v2ecoli Process under v2ecoli/processes/
  - Each v2ecoli Step under v2ecoli/steps/
  - The Millard ODE + LQR controller
  - pbg / process_bigraph framework overhead
  - numpy / scipy / lower-level libs
  - everything else

Identifies the dominant per-tick cost — the highest-priority
compilation target for Phase 4.
"""
from __future__ import annotations

import argparse
import base64
import cProfile
import datetime as dt
import io
import os
import platform
import pstats
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


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
    return {"sha": sha, "short": sha[:8] if sha != "(unknown)" else sha,
            "branch": branch, "dirty": dirty,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version()}


def categorize(filename: str) -> str:
    """Bucket a profile entry by source file path."""
    if filename in ("~", "<built-in>", "<frozen importlib._bootstrap>"):
        return "Python internals"
    if "/v2ecoli/processes/" in filename:
        return "Process: " + Path(filename).stem
    if "/v2ecoli/steps/millard" in filename or "millard" in filename.lower():
        return "Millard ODE / LQR"
    if "/v2ecoli/steps/" in filename:
        return "Step: " + Path(filename).stem
    if "/v2ecoli/library/" in filename:
        return "v2ecoli library"
    if "/v2ecoli/composites/" in filename:
        return "Composite scaffold"
    if "/process_bigraph/" in filename:
        return "pbg framework"
    if "/wholecell/" in filename:
        return "wholecell util"
    if "/numpy/" in filename or "/scipy/" in filename:
        return "numpy / scipy"
    if "/basico/" in filename or "/COPASI/" in filename:
        return "Millard ODE / LQR"
    if "site-packages" in filename:
        return "other site-packages"
    return "other"


# Stable ordering + colors for plot consistency.
CATEGORY_ORDER = [
    "Process: transcript_initiation",
    "Process: transcript_elongation",
    "Process: polypeptide_initiation",
    "Process: polypeptide_elongation",
    "Process: protein_degradation",
    "Process: rna_degradation",
    "Process: rna_maturation",
    "Process: chromosome_replication",
    "Process: chromosome_structure",
    "Process: complexation",
    "Process: equilibrium",
    "Process: two_component_system",
    "Process: tf_binding",
    "Process: tf_unbinding",
    "Millard ODE / LQR",
    "Step: ref_growth_driver",
    "Step: likelihood_collector",
    "Step: partition",
    "Step: rnap_data",
    "Step: ribosome_data",
    "Step: mass_listener",
    "v2ecoli library",
    "Composite scaffold",
    "pbg framework",
    "wholecell util",
    "numpy / scipy",
    "other site-packages",
    "Python internals",
    "other",
]


def profile_one_tick(duration: int = 1, warmup: int = 3):
    from v2ecoli import build_composite

    c = build_composite(
        "millard_pdmp_baseline",
        seed=0,
        with_ref_growth=True,
        ref_growth_flux_source="consumption_matched",
        transcript_initiation_mode="poisson",
        polypeptide_initiation_mode="poisson",
    )
    print(f"Warming up {warmup} ticks...", flush=True)
    c.run(warmup)

    print(f"Profiling {duration} tick(s)...", flush=True)
    pr = cProfile.Profile()
    pr.enable()
    c.run(duration)
    pr.disable()

    stats = pstats.Stats(pr)
    # tt = own time (exclusive); ct = cumulative time.
    bucket: dict[str, float] = defaultdict(float)
    bucket_calls: dict[str, int] = defaultdict(int)
    total_exclusive = 0.0
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, funcname = func
        cat = categorize(filename)
        bucket[cat] += tt
        bucket_calls[cat] += nc
        total_exclusive += tt
    return bucket, bucket_calls, total_exclusive, duration


def make_figure(bucket, bucket_calls, total, duration) -> str:
    # Sort by time descending, keep top ~14 + collapse the rest.
    items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
    TOP_N = 14
    top = items[:TOP_N]
    rest = items[TOP_N:]
    if rest:
        top.append(("(other smaller categories)",
                    sum(t for _, t in rest)))

    labels = [name for name, _ in top]
    times = np.array([t for _, t in top])
    pct = 100.0 * times / total

    fig, axes = plt.subplots(1, 2, figsize=(15, 8),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel 1: horizontal bar chart, top categories
    ax = axes[0]
    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(labels)))
    bars = ax.barh(y_pos, times * 1000, color=colors, edgecolor="black",
                   linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f"Per-tick own time (ms) — {duration} tick measured, "
                  f"total {total * 1000:.0f} ms")
    for i, (b, p) in enumerate(zip(bars, pct)):
        ax.text(b.get_width() + max(times) * 1000 * 0.005,
                b.get_y() + b.get_height() / 2,
                f"{p:.1f}%", ha="left", va="center", fontsize=9)
    ax.set_title("Per-tick compute decomposition by source",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: cumulative-fraction line
    ax = axes[1]
    cumfrac = np.cumsum(pct) / 100
    ax.plot(np.arange(1, len(labels) + 1), cumfrac, marker="o", lw=2,
            color="#1e3a8a")
    ax.axhline(0.8, color="#dc2626", lw=1.5, ls="--",
               label="80% threshold")
    ax.axhline(0.9, color="#f59e0b", lw=1.5, ls=":",
               label="90% threshold")
    ax.set_xlabel("# of categories included (sorted hi → lo)")
    ax.set_ylabel("cumulative fraction of per-tick time")
    ax.set_ylim(0, 1.05)
    ax.set_title("Compilation target prioritization",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Phase-4 sprint 1: real per-tick compute profile "
        f"(PDMP+poisson, total {total * 1000:.0f} ms / tick)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, bucket, bucket_calls, total, duration,
               plot_uri: str) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
    rows = []
    cum = 0.0
    for name, t in items:
        cum += t
        pct = 100.0 * t / total
        cumpct = 100.0 * cum / total
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td class='num'>{t * 1000:.2f}</td>"
            f"<td class='num'>{pct:.2f}%</td>"
            f"<td class='num'>{bucket_calls.get(name, 0):,}</td>"
            f"<td class='num'>{cumpct:.2f}%</td></tr>")
    table_rows = "\n".join(rows)

    # Identify top compilation target.
    top_cat, top_t = items[0]
    second_cat, second_t = items[1] if len(items) > 1 else (None, 0)
    target_msg = (
        f"<strong>{top_cat}</strong> accounts for "
        f"<strong>{100.0 * top_t / total:.1f}%</strong> "
        f"of per-tick own-time. The natural Phase-4 compilation target."
    )

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 4 sprint 1 — per-tick compute profile</title>
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

<h1>Phase 4 sprint 1 — real per-tick compute profile</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  cProfile measurement on one tick of the millard_pdmp_baseline
  composite with poisson initiation modes + consumption_matched
  ref-growth driver — the Phase-3 production stack Phase 4 needs to
  compile down. Replaces the previous planned scaffold figure.
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

<h2>Decomposition</h2>
<img class="plot" src="{plot_uri}" alt="per-tick bar chart + cumulative fraction">

<div class="takeaway">
  <strong>Top compilation target.</strong> {target_msg}
</div>

<h2>Full breakdown</h2>
<p>
  Total measured per-tick own-time: <code>{total * 1000:.1f} ms</code>
  ({duration} tick(s) profiled with cProfile; numbers reflect own/
  exclusive time, not cumulative).
</p>
<table>
  <tr><th>category</th>
      <th>own time (ms)</th>
      <th>% of tick</th>
      <th>call count</th>
      <th>cumulative %</th></tr>
  {table_rows}
</table>

<h2>Phase 4 implications</h2>
<ul>
  <li>The Phase-4 deliverable is "≥10³ parallel single-cell
      trajectories with near-zero marshalling overhead." A 600× speedup
      at the dominant category above is the natural first attack.</li>
  <li>If the breakdown is dominated by ONE Process (e.g. a
      polymerize-style hot loop), the compilation target is well-
      defined: lift that one Process into Catalyst.jl / a JIT-compiled
      kernel.</li>
  <li>If the breakdown is dominated by FRAMEWORK overhead (pbg
      marshalling, dict packing/unpacking), the answer is the
      column-centric runtime — replace nested-dict state with
      pre-allocated arrays indexed by (trajectory, time).</li>
  <li>If the breakdown is fragmented across many Processes, parallel
      throughput (multiple trajectories in lockstep) is the cleaner win
      than per-process compilation — Catalyst.jl excels at
      vector-state SSA-like systems.</li>
</ul>
<p>
  The Phase 4 plan can now be informed by the actual hot path instead
  of the planned scaffold projection.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-04/per_step_compute_decomposition.html"))
    args = ap.parse_args()

    bucket, bucket_calls, total, duration = profile_one_tick(
        args.duration, args.warmup)

    print(f"\nTotal own-time: {total * 1000:.1f} ms over {duration} tick(s)")
    print("Top categories:")
    for cat, t in sorted(bucket.items(), key=lambda x: -x[1])[:12]:
        print(f"  {cat:<48} {t * 1000:8.2f} ms  "
              f"({100.0 * t / total:5.2f}%)")

    plot_uri = make_figure(bucket, bucket_calls, total, duration)
    write_html(args.out, bucket, bucket_calls, total, duration, plot_uri)


if __name__ == "__main__":
    main()
