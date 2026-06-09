#!/usr/bin/env python3
"""Render a self-contained HTML performance report from perf_results.json.

v2ecoli vs vEcoli on the same multiseed/multigen job. Pure stdlib; inline CSS
+ inline-SVG bar charts so the report is a single portable file (no server,
no JS, survives being emailed). Tolerant of partial results — renders whatever
engines have run so far.

Usage:  python scripts/perf_report.py [--open]
"""
from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "perf"
RESULTS = OUT_DIR / "perf_results.json"
HTML = OUT_DIR / "perf_compare.html"

V2_COLOR, VE_COLOR = "#2563eb", "#dc2626"


def _bars(rows, *, unit, width=460, label_w=120, fmt="{:.1f}"):
    """rows: [(label, value, color)] → horizontal-bar SVG. Skips None values."""
    rows = [(l, v, c) for l, v, c in rows if v is not None]
    if not rows:
        return '<p style="color:#94a3b8;font-style:italic">no data yet</p>'
    vmax = max(v for _, v, _ in rows) or 1.0
    bar_w = width - label_w - 70
    rh, gap = 26, 10
    h = len(rows) * (rh + gap) + 8
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" '
           f'style="width:100%;max-width:{width}px;height:auto">']
    for i, (label, v, color) in enumerate(rows):
        y = i * (rh + gap) + 4
        w = max(2, (v / vmax) * bar_w)
        out.append(
            f'<text x="{label_w-8}" y="{y+rh/2+4:.0f}" font-size="12" fill="#334155" '
            f'text-anchor="end">{escape(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{rh}" rx="3" fill="{color}"/>'
            f'<text x="{label_w+w+6:.1f}" y="{y+rh/2+4:.0f}" font-size="12" '
            f'font-weight="600" fill="#0f172a">{fmt.format(v)} {unit}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _fmt_ram(mb):
    if mb is None:
        return "n/a"
    return f"{mb/1024:.0f} GB" if mb >= 10240 else f"{mb:.0f} MB"


def _kpi(label, value, sub=""):
    sub = f'<div class="kpi-sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="kpi"><div class="kpi-val">{escape(str(value))}</div>'
            f'<div class="kpi-label">{escape(label)}</div>{sub}</div>')


def _engine_summary(e: dict | None) -> dict:
    """Normalize an engine block into headline numbers."""
    if not e:
        return {}
    wall = e.get("wall_s")
    rss = e.get("maxrss_mb")
    cells = [c for c in (e.get("per_cell") or []) if c.get("wall_s") is not None]
    per_cell = round(sum(c["wall_s"] for c in cells) / len(cells), 1) if cells else None
    # Peak RAM: a single-process engine's /usr/bin/time maxrss IS its peak.
    # For a multi-task Nextflow engine the launcher maxrss EXCLUDES the sim
    # subprocesses, so it is NOT representative — only the per-task peak_rss
    # is (and that is unavailable on macOS-local without a container). Report
    # None there rather than the misleading launcher number.
    is_multitask = bool(e.get("tasks"))
    task_peak = [t.get("peak_rss_mb") for t in (e.get("tasks") or []) if t.get("peak_rss_mb")]
    if task_peak:
        peak = max(task_peak)
    elif is_multitask:
        peak = None                      # per-task RAM not captured
    else:
        peak = rss
    return {"wall": wall, "rss": rss, "peak_rss": peak, "per_cell": per_cell,
            "n_cells": len(cells), "rc": e.get("returncode"), "multitask": is_multitask,
            "rss_note": "launcher only" if is_multitask else "",
            "model": e.get("execution_model", "")}


def render(data: dict) -> str:
    spec = data.get("spec", {})
    prov = data.get("provenance", {})
    v2 = _engine_summary(data.get("v2ecoli"))
    ve = _engine_summary(data.get("vEcoli"))
    cells = spec.get("cells", "—")

    # caveat-text values
    ve_rss = ve.get("rss") if ve.get("rss") is not None else "—"
    v2_rss_gb = (f"{v2['peak_rss']/1024:.0f} GB peak RSS"
                 if v2.get("peak_rss") and v2["peak_rss"] >= 10240
                 else (f"{v2['peak_rss']:.0f} MB peak RSS" if v2.get("peak_rss")
                       else "its measured peak RSS"))
    max_steps = spec.get("max_steps", "—")
    gens = spec.get("generations", "—")
    v2_mode = spec.get("v2_mode", "seq")
    v2_is_ray = v2_mode == "ray"
    caveat_heading = ("Catch-up result — now an apples-to-apples comparison" if v2_is_ray
                      else "Why the totals aren't apples-to-apples")

    # Execution-model caveat — adapts to v2ecoli seq vs ray.
    if v2_is_ray:
        exec_note = (
            f"<b>Both engines now run seeds concurrently</b> — v2ecoli via the process-bigraph "
            f"Ray protocol (<i>{escape(v2.get('model',''))}</i>, one worker process per seed), "
            f"vEcoli via Nextflow (<i>{escape(ve.get('model',''))}</i>). Total wall ≈ the critical "
            f"path (slowest seed) for both, so the totals are now an apples-to-apples throughput "
            f"comparison rather than sequential-vs-parallel.")
        ram_note = (
            f"<b>Peak RAM is now real and bounded for both per-cell paths.</b> Each v2ecoli Ray "
            f"worker is its own process and reports its own peak via <code>resource.getrusage</code> "
            f"— <b>{v2_rss_gb}</b> per seed after the fix (internal full-state emitter disabled via "
            f"<code>set_null_emitter_override</code> + the <code>chromosome_history</code> leak "
            f"removed), down from ~57 GB whole-process before. vEcoli per-task RAM is still "
            f"<i>n/a</i> on macOS-local (Nextflow needs a container engine to emit it).")
    else:
        exec_note = (
            f"<b>Execution model differs.</b> v2ecoli — <i>{escape(v2.get('model',''))}</i>: seeds "
            f"run one after another in a single GIL-bound Python process, so total wall ≈ sum of "
            f"per-cell costs. vEcoli — <i>{escape(ve.get('model',''))}</i>: Nextflow fans the "
            f"{cells} sim tasks across processes/cores, so total wall ≈ critical path, not the sum. "
            f"<b>Per-cell wall</b> is the fairer single-engine compute number; <b>total wall</b> "
            f"reflects each engine's real end-to-end throughput on this host.")
        ram_note = (
            f"<b>Peak RAM is only directly measured for v2ecoli.</b> It runs as one process, so "
            f"<code>/usr/bin/time -l</code> captures its true peak. vEcoli's sim tasks are separate "
            f"Nextflow subprocesses, and macOS-local Nextflow (no container engine) does not emit "
            f"per-task RAM — so vEcoli per-cell RAM shows <i>n/a</i> (the launcher's {ve_rss} MB "
            f"excludes the sim children). v2ecoli's {v2_rss_gb} reflects <b>unbounded in-process "
            f"accumulation</b> across the whole sequential run (the internal full-state emitter "
            f"holds every step in memory) — disable it (<code>set_null_emitter_override</code>) and "
            f"run <code>--v2-mode ray</code> to bound it.")

    # headline speedup (total wall)
    speed = ""
    if v2.get("wall") and ve.get("wall"):
        if ve["wall"] < v2["wall"]:
            speed = f"vEcoli {v2['wall']/ve['wall']:.2f}× faster (total wall)"
        else:
            speed = f"v2ecoli {ve['wall']/v2['wall']:.2f}× faster (total wall)"

    def status_badge(s):
        if not s:
            return '<span class="badge pend">not run</span>'
        if s.get("rc") == 0:
            return '<span class="badge ok">completed</span>'
        return f'<span class="badge fail">exit {s.get("rc")}</span>'

    # charts
    wall_chart = _bars([("v2ecoli", v2.get("wall"), V2_COLOR),
                        ("vEcoli", ve.get("wall"), VE_COLOR)], unit="s", fmt="{:.0f}")
    rss_chart = _bars([("v2ecoli", v2.get("peak_rss"), V2_COLOR),
                       ("vEcoli", ve.get("peak_rss"), VE_COLOR)], unit="MB", fmt="{:.0f}")
    percell_chart = _bars([("v2ecoli", v2.get("per_cell"), V2_COLOR),
                           ("vEcoli", ve.get("per_cell"), VE_COLOR)], unit="s", fmt="{:.0f}")

    # per-cell / per-task tables
    def cell_table(e, eng):
        cells_ = (e or {}).get("per_cell") or []
        if not cells_:
            return '<p style="color:#94a3b8">no per-cell data</p>'
        head = "".join(f"<th>{escape(k)}</th>" for k in cells_[0].keys())
        body = ""
        for c in cells_:
            body += "<tr>" + "".join(
                f"<td>{escape(str(c.get(k)))}</td>" for k in cells_[0].keys()) + "</tr>"
        return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    # vEcoli full task breakdown (all stages)
    task_rows = ""
    for t in (data.get("vEcoli") or {}).get("tasks", []) or []:
        task_rows += ("<tr>" + "".join(f"<td>{escape(str(t.get(k)))}</td>" for k in
                      ("task", "status", "realtime_s", "cpu_pct", "peak_rss_mb")) + "</tr>")
    task_table = (f'<table class="data"><thead><tr><th>task</th><th>status</th>'
                  f'<th>realtime (s)</th><th>%cpu</th><th>peak RSS (MB)</th></tr></thead>'
                  f'<tbody>{task_rows}</tbody></table>') if task_rows else \
        '<p style="color:#94a3b8">no Nextflow trace parsed yet</p>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v2ecoli vs vEcoli — multiseed/multigen performance</title>
<style>
  :root {{ font-family: -apple-system, system-ui, sans-serif; }}
  body {{ margin: 0; background: #f8fafc; color: #0f172a; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .lede {{ color: #475569; margin: 0 0 24px; font-size: 14px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
           padding: 20px 22px; margin: 16px 0; }}
  h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: .04em;
        color: #64748b; margin: 0 0 14px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .kpi {{ background: #f1f5f9; border-radius: 8px; padding: 14px; text-align: center; }}
  .kpi-val {{ font-size: 22px; font-weight: 700; }}
  .kpi-label {{ font-size: 11px; color: #64748b; text-transform: uppercase;
                letter-spacing: .03em; margin-top: 2px; }}
  .kpi-sub {{ font-size: 11px; color: #94a3b8; margin-top: 4px; }}
  .verdict {{ font-size: 18px; font-weight: 700; padding: 14px 18px;
              background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px;
              color: #065f46; }}
  .verdict.pending {{ background:#fffbeb; border-color:#fde68a; color:#92400e; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }}
  .badge.ok {{ background:#dcfce7; color:#166534; }}
  .badge.pend {{ background:#fef9c3; color:#854d0e; }}
  .badge.fail {{ background:#fee2e2; color:#991b1b; }}
  table.data {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  table.data th {{ text-align: left; color: #64748b; border-bottom: 2px solid #e2e8f0;
                   padding: 6px 8px; }}
  table.data td {{ border-bottom: 1px solid #f1f5f9; padding: 6px 8px; }}
  .legend span {{ display:inline-block; width:11px; height:11px; border-radius:2px;
                  margin-right:5px; vertical-align:middle; }}
  .note {{ font-size: 12.5px; color:#475569; background:#f8fafc;
           border-left: 3px solid #cbd5e1; padding: 10px 14px; margin: 8px 0; }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .prov {{ font-size: 12px; color:#64748b; }}
</style></head>
<body><div class="wrap">
  <h1>v2ecoli vs vEcoli — multiseed/multigen performance</h1>
  <p class="lede">Same workload — <b>{spec.get('seeds','—')} seeds × {spec.get('generations','—')} generations</b>
     ({cells} cell-sims), condition: {escape(str(spec.get('condition','—')))} — run on both engines.
     v2ecoli's process-bigraph runner vs vEcoli's Nextflow DAG.</p>

  <div class="card">
    <div class="verdict {'' if speed else 'pending'}">{escape(speed) if speed else 'Run in progress — awaiting both engines.'}</div>
    <div class="kpis" style="margin-top:16px">
      {_kpi('v2ecoli total', f"{v2.get('wall')}s" if v2.get('wall') is not None else '—', status_badge(v2).replace('<span','').replace('</span>','') if False else '')}
      {_kpi('vEcoli total', f"{ve.get('wall')}s" if ve.get('wall') is not None else '—')}
      {_kpi('v2ecoli peak RAM', _fmt_ram(v2.get('peak_rss')), 'whole-process, real')}
      {_kpi('vEcoli peak RAM', _fmt_ram(ve.get('peak_rss')), 'per-task n/a (no container)')}
    </div>
    <p style="margin:12px 0 0">v2ecoli {status_badge(v2)} &nbsp; vEcoli {status_badge(ve)}</p>
  </div>

  <div class="card">
    <h2>Performance metrics</h2>
    <div class="legend" style="margin-bottom:10px;font-size:12px;color:#475569">
      <span style="background:{V2_COLOR}"></span>v2ecoli &nbsp;&nbsp;
      <span style="background:{VE_COLOR}"></span>vEcoli</div>
    <div class="grid2">
      <div><b style="font-size:13px">Total wall-clock</b>{wall_chart}</div>
      <div><b style="font-size:13px">Peak resident RAM</b>{rss_chart}</div>
    </div>
    <div style="margin-top:14px"><b style="font-size:13px">Mean per-cell wall (one seed×gen E. coli)</b>{percell_chart}</div>
  </div>

  <div class="card">
    <h2>{caveat_heading}</h2>
    <div class="note">{exec_note}</div>
    <div class="note">Both reuse a prebuilt ParCa <code>sim_data</code> (v2ecoli: <code>out/cache</code>;
      vEcoli: <code>out/kb/simData.cPickle</code> via <code>sim_data_path</code>) and vEcoli analyses
      are stripped, so this isolates <b>simulation</b> execution, not ParCa or plotting.</div>
    <div class="note">{ram_note}</div>
    <div class="note"><b>Sim span differs slightly.</b> v2ecoli ran to a fixed {max_steps}-step cap
      (reaching {gens} generations of one lineage); vEcoli ran each generation to natural division.
      The per-cell wall is therefore per-engine "one cell-generation," not an identical sim-time span.</div>
  </div>

  <div class="card">
    <h2>Per-cell breakdown</h2>
    <div class="grid2">
      <div><b style="font-size:13px">v2ecoli</b> (per seed×gen)<br>{cell_table(data.get('v2ecoli'),'v2')}</div>
      <div><b style="font-size:13px">vEcoli</b> (per sim task)<br>{cell_table(data.get('vEcoli'),'vecoli')}</div>
    </div>
  </div>

  <div class="card">
    <h2>vEcoli Nextflow task trace (all stages)</h2>
    {task_table}
  </div>

  <div class="card">
    <h2>Provenance &amp; method</h2>
    <p class="prov">v2ecoli @ <code>{escape(str(prov.get('v2ecoli_commit')))}</code> &nbsp;·&nbsp;
       vEcoli @ <code>{escape(str(prov.get('vecoli_commit')))}</code> &nbsp;·&nbsp;
       host <code>{escape(str(prov.get('host')))}</code></p>
    <p class="prov">v2ecoli driver: <code>scripts/run_phase0_multigen.py</code> under <code>/usr/bin/time -l</code>.
       vEcoli driver: <code>python -m runscripts.workflow</code>; per-task metrics from the Nextflow trace CSV.
       Regenerate: <code>python scripts/perf_compare.py --engine both</code> then
       <code>python scripts/perf_report.py</code>.</p>
  </div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    if not RESULTS.is_file():
        # Render an empty shell so the structure is visible before any run.
        data = {"spec": {"seeds": 2, "generations": 2, "cells": 4,
                         "condition": "baseline (minimal glucose)"}, "provenance": {}}
    else:
        data = json.loads(RESULTS.read_text())
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(render(data), encoding="utf-8")
    print(f"wrote {HTML}")
    if args.open:
        import subprocess
        subprocess.run(["open", str(HTML)])


if __name__ == "__main__":
    main()
