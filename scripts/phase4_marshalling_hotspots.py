"""Phase 4 sprint 2 — identify per-tick marshalling hotspot functions.

Sprint 1's category profile showed 76% of per-tick compute lives in
"framework overhead" buckets (Python internals + numpy + v2ecoli
library + other site-packages). This sprint drills in:

  - Top functions by CUMULATIVE time (where the wall actually goes).
  - Top functions by CALL COUNT (which operations are hammered per
    tick — the column-centric runtime should aim to eliminate these
    from the per-tick hot path).

Identifies the specific functions Phase 4 should target for
elimination via the column-centric refactor.
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


def short_label(filename: str, funcname: str, max_len: int = 60) -> str:
    """Compress a (filename, funcname) tuple into a readable bar label."""
    if filename in ("~", "<built-in>"):
        loc = "built-in"
    else:
        parts = Path(filename).parts
        # Find a recognizable anchor: parent dir + filename
        if "site-packages" in parts:
            i = parts.index("site-packages")
            loc = "/".join(parts[i + 1: i + 4])
        elif "v2ecoli" in parts:
            i = parts.index("v2ecoli")
            loc = "/".join(parts[i:])
        else:
            loc = "/".join(parts[-3:])
    label = f"{loc}::{funcname}"
    if len(label) > max_len:
        label = "…" + label[-(max_len - 1):]
    return label


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
    rows = []  # (filename, lineno, funcname, primitive_calls, total_calls, own_time, cumulative_time)
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, funcname = func
        rows.append((filename, lineno, funcname, cc, nc, tt, ct))
    return rows, duration


def make_figure(top_by_cum, top_by_calls, total_own_time) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    # Panel 1: top by cumulative time
    ax = axes[0]
    labels = [short_label(r[0], r[2]) for r in top_by_cum]
    times = np.array([r[6] for r in top_by_cum])  # cumulative time
    pct = 100.0 * times / total_own_time
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(labels)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, times * 1000, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative time (ms) — includes time in callees")
    for b, p in zip(bars, pct):
        ax.text(b.get_width() + max(times) * 1000 * 0.005,
                b.get_y() + b.get_height() / 2,
                f"{p:.1f}% of own-time", ha="left", va="center", fontsize=8)
    ax.set_title("Top 20 functions by CUMULATIVE time",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: top by call count
    ax = axes[1]
    labels = [short_label(r[0], r[2]) for r in top_by_calls]
    calls = np.array([r[4] for r in top_by_calls])  # total calls
    own = np.array([r[5] for r in top_by_calls])  # own time
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(labels)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, calls, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Call count per tick (log scale)")
    ax.set_xscale("log")
    for b, c, t in zip(bars, calls, own):
        ax.text(b.get_width() * 1.05,
                b.get_y() + b.get_height() / 2,
                f"{c:,} calls, {t * 1000:.2f} ms own",
                ha="left", va="center", fontsize=8)
    ax.set_title("Top 20 functions by CALL COUNT — the marshalling hotspots",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Phase-4 sprint 2: function-level marshalling hotspots in one tick",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def write_html(out_path: Path, rows, total_own, duration, plot_uri) -> None:
    prov = collect_provenance()
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if prov["dirty"] else "")

    by_cum = sorted(rows, key=lambda r: r[6], reverse=True)[:20]
    by_calls = sorted(rows, key=lambda r: r[4], reverse=True)[:20]
    by_own = sorted(rows, key=lambda r: r[5], reverse=True)[:20]

    def render_row(r, rank):
        filename, lineno, funcname, cc, nc, tt, ct = r
        return (
            f"<tr><td class='num'>{rank}</td>"
            f"<td><code>{short_label(filename, funcname, 80)}</code></td>"
            f"<td class='num'>{nc:,}</td>"
            f"<td class='num'>{tt * 1000:.2f}</td>"
            f"<td class='num'>{ct * 1000:.2f}</td>"
            f"<td class='num'>{100.0 * tt / total_own:.1f}%</td></tr>")

    cum_rows = "\n".join(render_row(r, i + 1)
                          for i, r in enumerate(by_cum))
    calls_rows = "\n".join(render_row(r, i + 1)
                            for i, r in enumerate(by_calls))
    own_rows = "\n".join(render_row(r, i + 1)
                          for i, r in enumerate(by_own))

    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>Phase 4 sprint 2 — marshalling hotspot functions</title>
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
  th, td {{ padding: 5px 10px; border: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; text-align: left; }}
  td.num {{ text-align: right;
            font-family: ui-monospace, Menlo, monospace; }}
  code {{ background:#f1f5f9; padding:1px 4px; border-radius:3px;
          font-size:0.85em; }}
  img.plot {{ max-width: 100%; border:1px solid #e2e8f0; border-radius:6px; }}
</style>

<h1>Phase 4 sprint 2 — function-level marshalling hotspots</h1>
<p style='color:#6b7280;font-size:0.9em;'>
  Sprint 1's category profile showed 76% of per-tick compute lives in
  framework overhead. This sprint drills in: which specific functions
  dominate by cumulative time and which are hammered by call count.
  Total profiled per-tick own time: <code>{total_own * 1000:.1f} ms</code>
  across {duration} tick(s).
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

<h2>Visualization</h2>
<img class="plot" src="{plot_uri}" alt="top functions by cumulative time + by call count">

<h2>Top 20 by CUMULATIVE time</h2>
<table>
  <tr><th>#</th><th>function</th><th>calls</th>
      <th>own ms</th><th>cum ms</th><th>% own</th></tr>
  {cum_rows}
</table>

<h2>Top 20 by CALL COUNT</h2>
<p>
  The functions called millions of times per tick are the
  column-centric refactor's primary elimination targets. Each call
  inside a per-trajectory inner loop becomes one vector op across
  10³ trajectories under the column-centric runtime.
</p>
<table>
  <tr><th>#</th><th>function</th><th>calls</th>
      <th>own ms</th><th>cum ms</th><th>% own</th></tr>
  {calls_rows}
</table>

<h2>Top 20 by OWN time</h2>
<p>
  The exclusive-time leaderboard. These are the functions whose
  inner code (not callees) actually consumes the wall clock. Worth
  inspecting for any pure-Python interpretation overhead or dict
  manipulation that a JIT or column-centric refactor would eliminate.
</p>
<table>
  <tr><th>#</th><th>function</th><th>calls</th>
      <th>own ms</th><th>cum ms</th><th>% own</th></tr>
  {own_rows}
</table>

<h2>Phase 4 implication</h2>
<p>
  The Phase-4 column-centric runtime should specifically eliminate the
  per-tick instances of the top "call count" functions from the inner
  loop — those are pure marshalling overhead that vectorize cleanly
  across trajectories. Functions in the cumulative-time leaderboard
  that aren't in the call-count list are the real per-tick work
  (numeric kernels, sampler draws); those benefit from per-Process
  compilation more than column-centric refactor.
</p>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument(
        "--out", type=Path,
        default=Path("reports/figures/pdmp-04/marshalling_hotspots.html"))
    args = ap.parse_args()

    rows, duration = profile_one_tick(args.duration, args.warmup)
    total_own = sum(r[5] for r in rows)

    print(f"\nTotal own-time: {total_own * 1000:.1f} ms over {duration} tick(s)")

    by_cum = sorted(rows, key=lambda r: r[6], reverse=True)[:args.top_n]
    by_calls = sorted(rows, key=lambda r: r[4], reverse=True)[:args.top_n]

    print(f"\nTop {args.top_n} by CUMULATIVE time:")
    for r in by_cum[:10]:
        print(f"  {short_label(r[0], r[2]):<60} "
              f"{r[6] * 1000:7.2f} ms cum  {r[5] * 1000:6.2f} ms own  "
              f"{r[4]:>8,} calls")

    print(f"\nTop {args.top_n} by CALL COUNT:")
    for r in by_calls[:10]:
        print(f"  {short_label(r[0], r[2]):<60} "
              f"{r[4]:>8,} calls  {r[5] * 1000:7.2f} ms own  "
              f"{r[6] * 1000:6.2f} ms cum")

    plot_uri = make_figure(by_cum, by_calls, total_own)
    write_html(args.out, rows, total_own, duration, plot_uri)


if __name__ == "__main__":
    main()
