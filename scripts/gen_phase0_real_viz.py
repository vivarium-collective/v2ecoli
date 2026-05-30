"""Generate REAL Phase 0 viz from the N=8 × 600-step pilot ensemble.

Replaces 3 synthetic placeholders in pdmp-00:
  - rng_seeding_fix_proof: had synthetic 5% CV projection → real 0.54% CV
  - per_condition_growth_rate: had synthetic 3-condition curves → real single-condition
    endpoint, with disclaimer that 3-condition needs Phase 0's conditions-3 run
  - ensemble divergence per-seed bars (new, replacing nothing)

Source data: .pbg/runs/phase0-pilot/summary.json (8/8 seeds, 600 steps each).
Note: only endpoint state is persisted (composite.run() doesn't auto-emit
trajectories without explicit emitter wiring); the next ensemble will use
XArrayEmitter for full time-series.

Run from worktree root:
    python scripts/gen_phase0_real_viz.py
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = Path("reports/figures/pdmp-00")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:{pinned_h}px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:{pinned_h}px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
p.caption{{color:#475569;font-size:0.85em;line-height:1.4}}
.fig{{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.fig img{{max-width:100%;max-height:100%;width:auto;height:auto;display:block;object-fit:contain}}
.tag{{display:inline-block;background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:0.7em;margin-right:6px}}
.tag.diag{{background:#dbeafe;color:#1e40af}}
</style></head>
<body><div class="wrap">
  <h1>{title}</h1>
  <p><span class="tag">real-data</span><span class="tag diag">N=8 × 600 steps</span></p>
  <div class="fig"><img src='data:image/png;base64,{png_b64}' alt='{title}' /></div>
  <p class="caption">{caption}</p>
</div></body></html>"""


def _save(name: str, title: str, caption: str, pinned_h: int = 760):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("ascii")
    html = TEMPLATE.format(title=title, caption=caption, png_b64=png_b64, pinned_h=pinned_h)
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def load_summary() -> dict:
    s = json.loads(Path(".pbg/runs/phase0-pilot/summary.json").read_text())
    return s


def viz_rng_fix_proof_real(s: dict):
    """Replace the synthetic 5% projection with the actual 0.54% CV."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    per_seed = s["per_seed"]
    seeds = [r["seed"] for r in per_seed]
    atp = [r["ATP[c]_count"] for r in per_seed]
    mass = [r["dry_mass_fg"] for r in per_seed]
    atp_mean, atp_std = s["ATP[c]_count_stats"]["mean"], s["ATP[c]_count_stats"]["std"]
    atp_cv = s["ATP[c]_count_stats"]["cv_pct"]
    mass_mean, mass_std = s["dry_mass_fg_stats"]["mean"], s["dry_mass_fg_stats"]["std"]
    mass_cv = s["dry_mass_fg_stats"]["cv_pct"]

    # Panel 1: ATP count per seed
    ax = axes[0]
    bars = ax.bar(seeds, atp, color="#3b82f6", alpha=0.85, edgecolor="#1e3a8a", lw=1)
    ax.axhline(atp_mean, ls="--", color="#10b981", lw=1.5, label=f"mean = {atp_mean:.2e}")
    ax.fill_between([-0.5, 7.5], atp_mean - atp_std, atp_mean + atp_std,
                    color="#10b981", alpha=0.15, label=f"±1σ band (CV={atp_cv:.2f}%)")
    ax.set_xlim(-0.5, 7.5)
    ax.set_xlabel("Master seed")
    ax.set_ylabel("ATP[c] count at t=600 s")
    ax.set_title("ATP[c] endpoint per seed — N=8, 600 model-seconds")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: dry mass per seed
    ax = axes[1]
    ax.bar(seeds, mass, color="#a855f7", alpha=0.85, edgecolor="#581c87", lw=1)
    ax.axhline(mass_mean, ls="--", color="#10b981", lw=1.5, label=f"mean = {mass_mean:.2f} fg")
    ax.fill_between([-0.5, 7.5], mass_mean - mass_std, mass_mean + mass_std,
                    color="#10b981", alpha=0.15, label=f"±1σ band (CV={mass_cv:.2f}%)")
    ax.set_xlim(-0.5, 7.5)
    ax.set_xlabel("Master seed")
    ax.set_ylabel("Dry mass (fg) at t=600 s")
    ax.set_title("Dry mass endpoint per seed — N=8")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Per-process RNG seeding fix — REAL pilot data (replaces earlier 5% synthetic projection)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    _save("rng_seeding_fix_proof_REAL",
          "Per-process RNG fix — real N=8 pilot (CV ≈ 0.5%)",
          (
            f"Real measurement after 600 model-seconds (one cycle target) across N=8 master "
            f"seeds. ATP[c] CV across seeds = {atp_cv:.2f}%, dry mass CV = {mass_cv:.2f}%. "
            "The per-process RNG fix (crc32(process_name, master_seed) in baseline.py) DOES "
            "produce divergence — but smaller than the 5% I drew in the synthetic 'projected' viz. "
            "Why smaller: most baseline-process behavior at sub-doubling timescales is constraint-"
            "bound (FBA, mass conservation) and only weakly stochastic; the stochastic processes "
            "(transcription, etc.) drive divergence on a longer timescale. The 0.5% CV at 600s is "
            "real ensemble diversity and unblocks the gating Phase 0 reference ensemble."
          ),
          pinned_h=820)


def viz_phase0_single_condition_endpoint(s: dict):
    """REPLACES per_condition_growth_rate placeholder. Shows the single-condition
    endpoint state across 8 seeds. The 3-condition view will land when we run
    against the M9-acetate + M9-glucose+aa caches."""
    per_seed = s["per_seed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    obs = ["ATP[c]_count", "dry_mass_fg", "cell_mass_fg"]
    titles = ["ATP[c] count", "Dry mass (fg)", "Cell mass (fg)"]
    colors = ["#3b82f6", "#a855f7", "#ec4899"]
    for ax, key, title, c in zip(axes, obs, titles, colors):
        values = [r.get(key, np.nan) for r in per_seed]
        ax.scatter(range(len(values)), values, s=80, c=c, alpha=0.85, edgecolor="black", lw=0.5)
        m = np.nanmean(values)
        sd = np.nanstd(values)
        ax.axhline(m, color="#10b981", ls="--", lw=1.2, label=f"mean ± σ")
        ax.fill_between([-1, len(values)], m - sd, m + sd, color="#10b981", alpha=0.12)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_xlabel("Seed")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Phase 0 pilot — single-condition (M9-glucose) endpoint state, N=8 seeds × 600 steps",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    _save("phase0_pilot_single_condition_REAL",
          "Phase 0 pilot — single condition × N=8 endpoint",
          (
            "Real endpoint state from N=8 × 600-step ensemble on M9-glucose. The 3-condition "
            "reference (across M9-glucose, M9-acetate, M9-glucose+aa) requires building the "
            "per-condition ParCa caches first; this is the M9-glucose slice of that planned grid. "
            "Endpoint-only (no time series) because composite.run() didn't have XArrayEmitter "
            "wired in this pilot — next pass attaches the emitter to get full trajectories."
          ),
          pinned_h=720)


def viz_phase0_summary_table(s: dict):
    """Per-seed table view — diagnostic for any anomalies."""
    per_seed = s["per_seed"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    cols = ["seed", "wall_s", "ATP[c]", "WATER[c]", "dry_mass_fg", "cell_mass_fg"]
    rows = []
    for r in per_seed:
        rows.append([
            f"{r['seed']:02d}",
            f"{r['wall_seconds']:.1f}",
            f"{r['ATP[c]_count']:,}",
            f"{r['WATER[c]_count']:,}",
            f"{r['dry_mass_fg']:.3f}",
            f"{r['cell_mass_fg']:.3f}",
        ])
    # Mean row
    rows.append([""] * len(cols))
    rows.append([
        "mean",
        f"{np.mean([r['wall_seconds'] for r in per_seed]):.1f}",
        f"{int(np.mean([r['ATP[c]_count'] for r in per_seed])):,}",
        f"{int(np.mean([r['WATER[c]_count'] for r in per_seed])):,}",
        f"{np.mean([r['dry_mass_fg'] for r in per_seed]):.3f}",
        f"{np.mean([r['cell_mass_fg'] for r in per_seed]):.3f}",
    ])
    rows.append([
        "CV %",
        "—",
        f"{s['ATP[c]_count_stats']['cv_pct']:.2f}",
        "—",
        f"{s['dry_mass_fg_stats']['cv_pct']:.2f}",
        "—",
    ])
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for i in range(len(cols)):
        table.get_celld()[(0, i)].set_facecolor("#e0e7ff")
        table.get_celld()[(0, i)].set_text_props(weight="bold")
    # Highlight mean + CV rows
    for j in range(len(cols)):
        table.get_celld()[(len(per_seed) + 2, j)].set_facecolor("#fef3c7")
        table.get_celld()[(len(per_seed) + 3, j)].set_facecolor("#dcfce7")
    ax.set_title("Phase 0 pilot — full per-seed summary (N=8, 600 steps each)", pad=20)
    _save("phase0_pilot_summary_table",
          "Phase 0 pilot — per-seed summary table",
          (
            "Endpoint state across all 8 seeds of the 600-step pilot run. ATP and dry mass "
            "show 0.5% CV — consistent ensemble divergence, no single-seed outlier. WATER[c] "
            "(largest pool, ~3e10 molecules) is shown for scale. Wall time ~48s/seed; total "
            "6.5 min serial. This is the immediate baseline; the gating N=64 ensemble will be "
            "run via the same script with --n-seeds 64."
          ),
          pinned_h=820)


def main():
    s = load_summary()
    print(f"loaded pilot summary: N={s.get('n_seeds_successful')} × {s.get('n_steps')} steps, "
          f"ATP CV={s.get('ATP[c]_count_stats', {}).get('cv_pct'):.2f}%")
    viz_rng_fix_proof_real(s)
    viz_phase0_single_condition_endpoint(s)
    viz_phase0_summary_table(s)
    print("done.")


if __name__ == "__main__":
    main()
