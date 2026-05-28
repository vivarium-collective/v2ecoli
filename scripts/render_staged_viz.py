"""Staged visualization renderer.

A "staged" viz declares (1) what it WILL show + (2) where its data lives.
The renderer:
  - If the data exists on disk → render the real chart
  - If the data is missing → render an explicit SKELETON: the chart's axes
    + a clear "Awaiting data from <run>" overlay banner. NEVER synthesises
    data to dress up an empty viz as real.

The user complaint this fixes: 15-20 placeholder viz across pdmp-* are
matplotlib-drawn from np.random with caption tags like "synth:projection".
Readers can't distinguish "this is a methodology demo" from "this is what
the run produced" — and tiny tag chips don't help. The skeleton state
shows the chart shape (axes, gridlines, expected band placeholder) with
a clear NOT-YET-RUN overlay instead.

Viz specs live in scripts/viz_specs/<study>.yaml. Each spec has a list of
viz; per-viz fields:
  name             slug for output filename
  title            chart title
  caption          description (real-data + pending share the caption)
  study            owning study slug
  data:
    source_glob    list of file patterns the chart needs (must all resolve)
    loader         "json-trajectory" | "json-endpoint-summary" | "csv-pandas"
                   determines how the renderer parses what it finds
    keys           which keys/columns to extract
  plot:
    kind           ensemble-band | endpoint-bars | line-curves | heatmap | diagram
    x_label, y_label
    overlays       optional list of horizontal lines, bands, anchors
  awaits:
    run_name       the planned_runs[] entry whose completion fills this viz
    fallback       skeleton | scaffold (drawn when data missing)

Outputs go to reports/figures/<study>/<name>.html, same pinned-height
template as other generators so the iframe auto-sizer works.

Run from worktree root:
    python scripts/render_staged_viz.py [--study pdmp-02] [--dry-run]
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:{pinned_h}px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:{pinned_h}px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
p.caption{{color:#475569;font-size:0.85em;line-height:1.4}}
.fig{{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.fig img{{max-width:100%;max-height:100%;width:auto;height:auto;display:block;object-fit:contain}}
.state{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.7em;margin-right:6px;font-weight:600}}
.state.real{{background:#d1fae5;color:#065f46}}
.state.skeleton{{background:#fef3c7;color:#92400e}}
.state.scaffold{{background:#fef3c7;color:#92400e}}
.tag{{display:inline-block;background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:4px;font-size:0.7em;margin-right:6px}}
</style></head>
<body><div class="wrap">
  <h1>{title}</h1>
  <p><span class="state {state}">{state_label}</span><span class="tag">{tag}</span></p>
  <div class="fig"><img src='data:image/png;base64,{png_b64}' alt='{title}' /></div>
  <p class="caption">{caption}</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_trajectory_dir(source_glob: list[str], keys: list[str]) -> dict | None:
    """Load all matching trajectory.json files and stack by key.

    source_glob like [".pbg/runs/phase0-traj/seed_*/trajectory.json"].
    Returns {key: array (n_replicates, n_t), time: array (n_t,)} or None
    if zero files match.

    Robust to early-terminated replicates (seed crashed mid-run, returned a
    short trajectory): pad short series with NaN up to the longest, so the
    array is regular and downstream nanmean/nanstd handle the missing tail.
    Records the per-replicate-completion vector in out["completed_steps"].
    """
    files = []
    for pattern in source_glob:
        files.extend(sorted(glob(pattern)))
    if not files:
        return None
    series_lists: dict[str, list[list[float]]] = {k: [] for k in keys}
    completed: list[int] = []
    time_arr = None
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        # Pick the longest time vector seen as the canonical timeline.
        if "time" in d:
            t = np.array(d["time"], dtype=float)
            if time_arr is None or len(t) > len(time_arr):
                time_arr = t
        for k in keys:
            if k in d:
                series_lists[k].append([
                    (v if v is not None else np.nan) for v in d[k]
                ])
        completed.append(len(d.get("time", [])))
    if time_arr is None or not any(series_lists[k] for k in keys):
        return None
    n_t = len(time_arr)
    # Pad short series with NaN to length n_t so the array is regular.
    padded: dict[str, np.ndarray] = {}
    for k in keys:
        if not series_lists[k]:
            continue
        rows = []
        for s in series_lists[k]:
            if len(s) < n_t:
                s = list(s) + [np.nan] * (n_t - len(s))
            elif len(s) > n_t:
                s = s[:n_t]
            rows.append(s)
        padded[k] = np.array(rows, dtype=float)
    out: dict = {"time": time_arr, "n_replicates": len(files),
                 "completed_steps": np.array(completed)}
    out.update(padded)
    return out


def load_endpoint_summary(source_glob: list[str], keys: list[str]) -> dict | None:
    files = []
    for pattern in source_glob:
        files.extend(sorted(glob(pattern)))
    if not files:
        return None
    rows: list[dict] = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        # Two shapes: per-seed summary (has 'seed') or ensemble summary (has 'per_seed')
        if "per_seed" in d:
            rows.extend(d["per_seed"])
        elif "seed" in d:
            rows.append(d)
    if not rows:
        return None
    out: dict = {"seeds": [r.get("seed") for r in rows], "n": len(rows)}
    for k in keys:
        out[k] = np.array([r.get(k, np.nan) for r in rows], dtype=float)
    return out


def load_csv(source_glob: list[str], keys: list[str]) -> dict | None:
    import pandas as pd
    files = []
    for pattern in source_glob:
        files.extend(sorted(glob(pattern)))
    if not files:
        return None
    df = pd.read_csv(files[0], index_col=0)
    out: dict = {"time": df.index.values}
    for k in keys:
        if k in df.columns:
            out[k] = df[k].values
    return out


LOADERS = {
    "json-trajectory": load_trajectory_dir,
    "json-endpoint-summary": load_endpoint_summary,
    "csv-pandas": load_csv,
}


# ---------------------------------------------------------------------------
# Renderers — one per plot.kind. Each takes (data | None, spec) and returns
# (state, png_b64). When data is None, render the skeleton.
# ---------------------------------------------------------------------------

def render_ensemble_band(data: dict | None, spec: dict) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    keys = spec["data"]["keys"]
    main_key = keys[0]
    plot = spec.get("plot", {})

    if data is not None and main_key in data:
        arr = data[main_key]
        t = data["time"]
        for i in range(arr.shape[0]):
            ax.plot(t, arr[i], color="#3b82f6", alpha=0.10, lw=0.6)
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax.plot(t, mean, color="#1e3a8a", lw=2.5,
                label=f"ensemble mean (N={arr.shape[0]})")
        ax.fill_between(t, mean - sd, mean + sd, color="#3b82f6", alpha=0.20, label="±1 SD")
        ax.legend(loc="best")
        state = "real"
    else:
        # Skeleton: empty axes with explanatory overlay.
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.55,
                f"Awaiting data from {spec['awaits']['run_name']}",
                ha="center", va="center", fontsize=15, color="#92400e",
                bbox=dict(boxstyle="round,pad=0.7", fc="#fef3c7", ec="#92400e", lw=1.5))
        ax.text(0.5, 0.35,
                f"Will plot: {main_key}",
                ha="center", va="center", fontsize=10, color="#475569", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        state = "skeleton"

    ax.set_xlabel(plot.get("x_label", ""))
    ax.set_ylabel(plot.get("y_label", ""))
    ax.set_title(spec.get("title", "")); ax.grid(True, alpha=0.3)
    return state, _to_b64()


def render_endpoint_bars(data: dict | None, spec: dict) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = spec["data"]["keys"]
    plot = spec.get("plot", {})

    if data is not None and keys[0] in data:
        values = data[keys[0]]
        seeds = data.get("seeds") or list(range(len(values)))
        ax.bar(seeds, values, color="#3b82f6", alpha=0.85, edgecolor="#1e3a8a", lw=1)
        m = np.nanmean(values); sd = np.nanstd(values)
        ax.axhline(m, ls="--", color="#10b981", lw=1.5,
                   label=f"mean = {m:.2e}, CV = {100*sd/m:.2f}%" if m else "mean")
        ax.legend()
        state = "real"
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.55,
                f"Awaiting data from {spec['awaits']['run_name']}",
                ha="center", va="center", fontsize=15, color="#92400e",
                bbox=dict(boxstyle="round,pad=0.7", fc="#fef3c7", ec="#92400e", lw=1.5))
        ax.text(0.5, 0.35, f"Will plot: {keys[0]} per seed",
                ha="center", va="center", fontsize=10, color="#475569", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        state = "skeleton"

    ax.set_xlabel(plot.get("x_label", "")); ax.set_ylabel(plot.get("y_label", ""))
    ax.set_title(spec.get("title", ""))
    return state, _to_b64()


def render_cv_growth(data: dict | None, spec: dict) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    keys = spec["data"]["keys"]
    plot = spec.get("plot", {})

    if data is not None and all(k in data for k in keys):
        t = data["time"]
        for k in keys:
            arr = data[k]
            mean = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cv_pct = np.where(np.abs(mean) > 0, sd / np.abs(mean) * 100, np.nan)
            ax.plot(t, cv_pct, label=k.split(".")[-1], lw=2)
        ax.legend()
        state = "real"
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.55,
                f"Awaiting data from {spec['awaits']['run_name']}",
                ha="center", va="center", fontsize=15, color="#92400e",
                bbox=dict(boxstyle="round,pad=0.7", fc="#fef3c7", ec="#92400e", lw=1.5))
        ax.text(0.5, 0.35, "Will plot: cross-seed CV(t) per observable",
                ha="center", va="center", fontsize=10, color="#475569", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        state = "skeleton"

    ax.set_xlabel(plot.get("x_label", "")); ax.set_ylabel(plot.get("y_label", ""))
    ax.set_title(spec.get("title", "")); ax.grid(True, alpha=0.3)
    return state, _to_b64()


def render_skeleton_only(data: dict | None, spec: dict) -> tuple[str, str]:
    """For viz where the chart is fundamentally a methodology DIAGRAM, not
    data-bound — but we still want the same skeleton-vs-real machinery so
    the reader sees an explicit "this is a methodology stub, not a result"
    label until the matching run lands."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot = spec.get("plot", {})
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if data is None:
        ax.text(0.5, 0.55,
                f"Awaiting data from {spec['awaits']['run_name']}",
                ha="center", va="center", fontsize=15, color="#92400e",
                bbox=dict(boxstyle="round,pad=0.7", fc="#fef3c7", ec="#92400e", lw=1.5))
        ax.text(0.5, 0.35, plot.get("skeleton_note", "Empty until run lands."),
                ha="center", va="center", fontsize=10, color="#475569", style="italic")
        state = "skeleton"
    else:
        ax.text(0.5, 0.5, "(diagram-only viz; nothing to render)",
                ha="center", va="center", fontsize=12, color="#475569")
        state = "real"
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(spec.get("title", ""))
    return state, _to_b64()


PLOT_RENDERERS = {
    "ensemble-band": render_ensemble_band,
    "endpoint-bars": render_endpoint_bars,
    "cv-growth": render_cv_growth,
    "skeleton-only": render_skeleton_only,
}


def _to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_one(spec: dict, fig_root: Path, dry_run: bool) -> tuple[str, str]:
    """Render one viz spec. Returns (state, output_path)."""
    study = spec["study"]
    data = None
    if "data" in spec:
        loader = LOADERS.get(spec["data"].get("loader"))
        if loader:
            data = loader(
                source_glob=spec["data"]["source_glob"],
                keys=spec["data"]["keys"],
            )
    renderer = PLOT_RENDERERS.get(spec["plot"]["kind"], render_skeleton_only)
    state, png_b64 = renderer(data, spec)

    state_label = {
        "real": "real-data",
        "skeleton": f"skeleton — awaiting {spec.get('awaits', {}).get('run_name', '<run>')}",
        "scaffold": "scaffold",
    }.get(state, state)
    tag = spec.get("tag", "")
    if not tag:
        tag = "N=" + str(data.get("n_replicates") or data.get("n", "?")) if data else "design-stage"

    html = HTML_TEMPLATE.format(
        title=spec["title"],
        caption=spec.get("caption", ""),
        tag=tag,
        state=state,
        state_label=state_label,
        png_b64=png_b64,
        pinned_h=spec.get("pinned_h", 760),
    )
    out = fig_root / study / f"{spec['name']}.html"
    if dry_run:
        return state, str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return state, str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default=None, help="Only render viz for this study slug (e.g. pdmp-02)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    fig_root = Path("reports/figures")
    specs_dir = Path("scripts/viz_specs")
    if not specs_dir.is_dir():
        sys.exit(f"missing viz specs dir: {specs_dir}")

    n_real, n_skel = 0, 0
    for spec_file in sorted(specs_dir.glob("*.yaml")):
        bundle = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
        for spec in bundle.get("viz", []) or []:
            if args.study and spec.get("study") != args.study:
                continue
            state, out_path = render_one(spec, fig_root, args.dry_run)
            marker = "[real]    " if state == "real" else "[SKELETON]"
            print(f"  {marker} {out_path}")
            if state == "real":
                n_real += 1
            else:
                n_skel += 1
    print()
    print(f"done: {n_real} real-data, {n_skel} skeleton")


if __name__ == "__main__":
    main()
