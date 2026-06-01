"""Analyze multi-gen Phase 0 SQLite outputs → real viz for pdmp-02.

For each seed's run.db (one per replicate from run_phase0_multigen.py):
  1. Pull the history table (per-tick state JSON).
  2. Extract cell_mass(t) time-series.
  3. Detect divisions = local cell_mass drops > 30% (the mother→daughter split).
  4. For each division event: record mother_mass, daughter_mass, time-of-division.
  5. Aggregate across all replicates.

Produces:
  reports/figures/pdmp-02/multigen_cell_mass_trajectories.html
    Per-replicate cell_mass(t) with division events marked.
  reports/figures/pdmp-02/multigen_division_time_distribution.html
    Histogram of inter-division times (proxy for doubling time).
  reports/figures/pdmp-02/multigen_inheritance_mass_split.html
    Distribution of daughter_mass / mother_mass at division. Under binomial
    partition the ratio is concentrated around 0.5 with binomial-variance.

Run from worktree root:
    python scripts/gen_phase0_multigen_viz.py
"""
from __future__ import annotations
import base64
import io
import json
import os
import sqlite3
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = Path("reports/figures/pdmp-02")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})


def _save_html(name: str, title: str, caption: str, pinned_h: int = 760):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    html = f"""<!DOCTYPE html>
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
</style></head><body><div class="wrap">
<h1>{title}</h1>
<p><span class="tag">real-data</span><span class="tag">Phase 0 multi-gen</span></p>
<div class="fig"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
<p class="caption">{caption}</p>
</div></body></html>"""
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def load_db(db_path: Path) -> list[dict]:
    """Pull (step, global_time, state) rows from one seed's history."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT step, global_time, state FROM history ORDER BY step"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for step, t, state_json in rows:
        try:
            state = json.loads(state_json)
        except Exception:
            continue
        out.append({"step": step, "t": float(t) if t is not None else None, "state": state})
    return out


def extract_cell_mass(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Pull (time, cell_mass) from the rows. Cell_mass is at
    state.agents.<agent_id>.listeners.mass.cell_mass (the followed agent)."""
    ts, ms = [], []
    for r in rows:
        s = r["state"]
        ag = s.get("agents") or {}
        if not ag: continue
        # Take first agent (the followed one in single-lineage)
        aid = next(iter(ag))
        m = ag[aid].get("listeners", {}).get("mass", {}).get("cell_mass")
        if m is None or r["t"] is None: continue
        try:
            ts.append(float(r["t"])); ms.append(float(m))
        except (TypeError, ValueError):
            continue
    return ts, ms


def detect_divisions(ts: list[float], ms: list[float], threshold: float = 0.3) -> list[dict]:
    """Find division events = consecutive mass drop > threshold (fractional).
    Returns: list of {t_div, mother_mass, daughter_mass, ratio}."""
    events = []
    for i in range(1, len(ts)):
        if ms[i-1] <= 0: continue
        ratio = ms[i] / ms[i-1]
        if ratio < (1 - threshold):
            events.append({
                "t_div": ts[i],
                "mother_mass": ms[i-1],
                "daughter_mass": ms[i],
                "ratio": ratio,
            })
    return events


def viz_cell_mass_trajectories(per_seed: list[tuple[int, list[float], list[float], list[dict]]]):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = plt.cm.tab10.colors
    for (seed, ts, ms, divs), c in zip(per_seed, colors):
        ax.plot(ts, ms, color=c, lw=1.6, label=f"seed={seed:02d} ({len(divs)} div)")
        for d in divs:
            ax.scatter([d["t_div"]], [d["daughter_mass"]], color=c, s=60,
                       marker="v", edgecolor="black", lw=0.8, zorder=5)
    ax.set_xlabel("Time (model seconds)")
    ax.set_ylabel("Cell mass (fg)")
    ax.set_title(f"Phase 0 multi-gen — cell mass(t) with division events (▼)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    _save_html("multigen_cell_mass_trajectories",
               "Phase 0 multi-gen — cell mass(t) with division events",
               (
                 "Real cell-mass trajectories from N seeds × multi-generation runs. "
                 "Division events detected as fractional mass drops > 30% (mother → daughter); "
                 "marked with ▼. Inter-division spacing is the empirical doubling time "
                 "(should match the M9-glucose published ~44 min = ~2640 simulated seconds)."
               ),
               pinned_h=720)


def viz_division_time_distribution(all_divs: list[dict], per_seed_divisions: dict[int, list[dict]]):
    fig, ax = plt.subplots(figsize=(11, 5))
    # Per-seed inter-division intervals
    intervals = []
    for divs in per_seed_divisions.values():
        if len(divs) < 2: continue
        sorted_t = sorted(d["t_div"] for d in divs)
        for a, b in zip(sorted_t[:-1], sorted_t[1:]):
            intervals.append(b - a)
    # First-division times (from t=0)
    first_div_times = [
        min(d["t_div"] for d in divs)
        for divs in per_seed_divisions.values() if divs
    ]
    if first_div_times:
        ax.hist(first_div_times, bins=12, color="#3b82f6", alpha=0.85,
                edgecolor="black", label=f"First division time (N={len(first_div_times)})")
    if intervals:
        ax.hist(intervals, bins=12, color="#10b981", alpha=0.65,
                edgecolor="black", label=f"Inter-division interval (N={len(intervals)})")
    ax.axvline(2640, color="red", ls="--", lw=1.5, label="M9-glucose τ ≈ 2640 s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Phase 0 multi-gen — division time distribution")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save_html("multigen_division_time_distribution",
               "Phase 0 multi-gen — division time distribution",
               (
                 "Blue = first-division time (from t=0); green = inter-division intervals "
                 "(time between successive divisions). Both should center near the M9-glucose "
                 "doubling time ~2640 s (red line). Spread reflects per-replicate variation "
                 "in division-trigger conditions."
               ),
               pinned_h=720)


def viz_inheritance_mass_split(all_divs: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ratios = np.array([d["ratio"] for d in all_divs])
    if len(ratios):
        axes[0].hist(ratios, bins=20, color="#a855f7", alpha=0.85,
                     edgecolor="black", label=f"N={len(ratios)} divisions")
        axes[0].axvline(0.5, color="green", ls="--", lw=1.5, label="binomial mean = 0.5")
        axes[0].set_xlabel("daughter_mass / mother_mass")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Inheritance: daughter/mother mass ratio")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        # Stats panel
        mean_r = np.mean(ratios); std_r = np.std(ratios)
        axes[1].axis("off")
        stats = (
            f"N divisions observed:  {len(ratios)}\n\n"
            f"Mean daughter/mother:  {mean_r:.4f}\n"
            f"  (perfect binomial would give 0.5)\n\n"
            f"Std of ratio:  {std_r:.4f}\n"
            f"  CV = {std_r/max(mean_r, 1e-9)*100:.2f}%\n\n"
            f"Test contract: variance of daughter1−daughter2\n"
            f"under binomial(mother, 0.5) is mother/4.\n"
            f"Per-division observed:\n"
        )
        for i, d in enumerate(all_divs[:5]):
            stats += f"  div{i+1}: mother={d['mother_mass']:.1f} fg, daughter={d['daughter_mass']:.1f}\n"
        if len(all_divs) > 5:
            stats += f"  ... and {len(all_divs) - 5} more\n"
        axes[1].text(0.01, 0.98, stats, va="top", ha="left", fontsize=10,
                     family="monospace")
    else:
        axes[0].text(0.5, 0.5, "No divisions detected in this dataset",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    fig.suptitle("Phase 0 multi-gen — inheritance distribution (binomial-partition test)",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    _save_html("multigen_inheritance_mass_split",
               "Phase 0 multi-gen — inheritance distribution",
               (
                 "Distribution of (daughter_mass / mother_mass) across all division events. "
                 "Under perfect binomial(mother, 0.5) cell-mass partition the ratio should "
                 "center on 0.5 with binomial-derived variance. Right panel: per-division "
                 "stats including the variance test the pdmp-02 multi-gen-inheritance-binomial "
                 "primary test will evaluate against."
               ),
               pinned_h=720)


def main():
    root = Path(".pbg/runs/phase0-multigen")
    if not root.is_dir():
        sys.exit(f"missing {root}; run scripts/run_phase0_multigen.py first")
    db_files = sorted(root.glob("seed_*/run.db"))
    if not db_files:
        sys.exit(f"no run.db files under {root}")

    per_seed = []
    all_divs = []
    per_seed_divisions = {}
    for dbp in db_files:
        seed = int(dbp.parent.name.split("_")[-1])
        rows = load_db(dbp)
        if not rows:
            print(f"  seed_{seed:02d}: empty db, skip")
            continue
        ts, ms = extract_cell_mass(rows)
        divs = detect_divisions(ts, ms)
        print(f"  seed_{seed:02d}: {len(ts)} timepoints, {len(divs)} divisions")
        per_seed.append((seed, ts, ms, divs))
        all_divs.extend(divs)
        per_seed_divisions[seed] = divs

    print(f"\nTotal: {len(per_seed)} replicates × {len(all_divs)} division events")
    print()
    viz_cell_mass_trajectories(per_seed)
    viz_division_time_distribution(all_divs, per_seed_divisions)
    viz_inheritance_mass_split(all_divs)


if __name__ == "__main__":
    main()
