#!/usr/bin/env python
"""Generate pdmp-02 jump-process figures from the real run summary.

Source data: ``.pbg/runs/pdmp-02/initiation_modes_summary.json`` — a discrete
vs. poisson (per-promoter/per-protein tau-leap) comparison of the v2ecoli
baseline composite over a 300 s run. Each mode records:

  per_tick_total      300-tick time series of total initiation events / tick
  per_tu_cumulative   per-transcription-unit cumulative initiation counts (3277 TUs)
  cell_mass, dry_mass scalar endpoints (fg)
  build_wall_s, run_wall_s   wall-clock

The figure is rendered with matplotlib (Agg) and base64-embedded into a
self-contained HTML page matching the workspace's reports/figures/ convention
(720 px fixed-height card + caption). No fabricated values — every number is
read straight from the JSON.

Run from the workspace root:
  .venv/bin/python workspace/investigations/v2ecoli-pdmp/studies/pdmp-02-jump-processes/sims/make_charts.py
"""
from __future__ import annotations

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
SRC = os.path.join(WS_ROOT, ".pbg", "runs", "pdmp-02", "initiation_modes_summary.json")
OUT_DIR = os.path.join(WS_ROOT, "reports", "figures", "pdmp-02")

DISCRETE_C = "#475569"  # slate
POISSON_C = "#dc2626"   # red

_HTML = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:720px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:720px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
.imgbox{{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}}
.imgbox img{{max-width:100%;max-height:100%;object-fit:contain}}
p.caption{{font-size:0.82em;color:#475569;line-height:1.4}}
</style></head><body>
<div class="wrap">
  <h1>{title}</h1>
  <div class="imgbox"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
  <p class="caption">{caption}</p>
</div></body></html>
"""


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write(name: str, title: str, caption: str, fig) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(_HTML.format(title=title, caption=caption, b64=_png_b64(fig)))
    return path


def main() -> None:
    data = json.load(open(SRC))
    d, p = data["discrete"], data["poisson"]

    d_tick = np.asarray(d["per_tick_total"], float)
    p_tick = np.asarray(p["per_tick_total"], float)
    d_tu = np.asarray(d["per_tu_cumulative"], float)
    p_tu = np.asarray(p["per_tu_cumulative"], float)

    # --- jump_event_rates_real: per-tick initiation rate, discrete vs poisson ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    t = np.arange(len(d_tick))
    ax.plot(t, d_tick, color=DISCRETE_C, lw=1.1, label=f"discrete (mean {d_tick.mean():.1f}, sd {d_tick.std():.1f})")
    ax.plot(t, p_tick, color=POISSON_C, lw=0.9, alpha=0.8, label=f"poisson (mean {p_tick.mean():.1f}, sd {p_tick.std():.1f})")
    ax.set_xlabel("tick"); ax.set_ylabel("initiation events / tick")
    ax.set_title("Per-tick initiation rate")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)

    ax2 = axes[1]
    bins = np.arange(min(d_tick.min(), p_tick.min()) - 0.5, max(d_tick.max(), p_tick.max()) + 1.5)
    ax2.hist(d_tick, bins=bins, color=DISCRETE_C, alpha=0.6, label="discrete", orientation="horizontal")
    ax2.hist(p_tick, bins=bins, color=POISSON_C, alpha=0.5, label="poisson", orientation="horizontal")
    ax2.set_xlabel("ticks"); ax2.set_title("Rate distribution")
    ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)
    fig.tight_layout()

    dm = (p["cell_mass"] - d["cell_mass"]) / d["cell_mass"] * 100
    drym = (p["dry_mass"] - d["dry_mass"]) / d["dry_mass"] * 100
    cap = (
        f"Total transcription-initiation events per tick over a {d['duration_s']} s v2ecoli baseline run, "
        f"discrete vs. the per-promoter/per-protein Poisson tau-leap sampler. Both modes track the same rising "
        f"trend (initiation grows as the cell accumulates gene copies; discrete mean {d_tick.mean():.1f}, "
        f"sd {d_tick.std():.1f}), but the Poisson sampler adds substantial per-tick stochasticity on top "
        f"(mean {p_tick.mean():.1f}, sd {p_tick.std():.1f}, range {p_tick.min():.0f}-{p_tick.max():.0f}) — the "
        f"continuous-time jump signature. Yet aggregate endpoints are conserved: cell_mass "
        f"{d['cell_mass']:.1f}->{p['cell_mass']:.1f} fg (Δ{dm:+.3f}%), dry_mass {d['dry_mass']:.1f}->{p['dry_mass']:.1f} fg "
        f"(Δ{drym:+.3f}%) — the consumption-matched homeostat washes per-tick jump noise out at the mass level. "
        f"Source: .pbg/runs/pdmp-02/initiation_modes_summary.json."
    )
    out1 = _write("jump_event_rates_real.html", "Jump initiation rates — discrete vs Poisson (real run)", cap, fig)

    # --- jump_initiation_per_tu_distribution_real: per-TU cumulative count distribution ---
    fig2, ax = plt.subplots(figsize=(9, 4.6))
    tu_max = int(max(d_tu.max(), p_tu.max()))
    bins = np.arange(-0.5, tu_max + 1.5)
    ax.hist(d_tu, bins=bins, color=DISCRETE_C, alpha=0.55, label=f"discrete (mean {d_tu.mean():.2f}/TU)")
    ax.hist(p_tu, bins=bins, color=POISSON_C, alpha=0.45, label=f"poisson (mean {p_tu.mean():.2f}/TU)")
    ax.set_xlabel("cumulative initiations per transcription unit")
    ax.set_ylabel("number of TUs")
    ax.set_title(f"Per-TU initiation count distribution ({len(d_tu)} TUs)")
    ax.set_yscale("log")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.2)
    fig2.tight_layout()
    cap2 = (
        f"Distribution of cumulative transcription-initiation counts across all {len(d_tu)} transcription units, "
        f"discrete vs. Poisson tau-leap. Both modes produce the same per-TU mean "
        f"({d_tu.mean():.2f} vs {p_tu.mean():.2f} initiations/TU) and the same heavy-tailed shape (log y-axis), "
        f"confirming the Poisson sampler reproduces the discrete model's per-gene initiation statistics, not just "
        f"the aggregate rate. Source: .pbg/runs/pdmp-02/initiation_modes_summary.json."
    )
    out2 = _write("jump_initiation_per_tu_distribution_real.html",
                  "Per-TU initiation distribution — discrete vs Poisson (real run)", cap2, fig2)

    for o in (out1, out2):
        print("wrote", os.path.relpath(o, WS_ROOT))


if __name__ == "__main__":
    main()
