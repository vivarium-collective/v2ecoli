"""Side-by-side comparison: PDMP+consumption_matched vs the v2ecoli baseline.

Runs both composites for ``--duration`` s (default 600) with the same seed,
captures per-process invoke timing via ``Composite.timing_summary()`` and the
full mass-listener submass breakdown at the endpoint. Writes a 3-panel HTML:

  1. Wall-time comparison — overall + top-N processes side-by-side bars.
  2. Mass-fraction comparison — stacked bar showing the dry-mass component
     split (protein, RNA, DNA, small-molecule, …) for each composite.
  3. Endpoint state table — cm, dm, water, submasses, with absolute values
     and (composite − baseline) deltas.

Usage:
    .venv/bin/python scripts/compare_pdmp_vs_baseline.py
        [--duration 600] [--seed 0]
        [--baseline-composite baseline]
        [--pdmp-composite millard_pdmp_baseline]
        [--out reports/figures/pdmp-01/pdmp_vs_baseline.html]
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


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    """True iff there are uncommitted edits to *tracked* files.

    ``git status --porcelain`` reports untracked files too — but newly-
    written report HTML, ``.venv`` symlinks, scratch logs, etc. show up
    there even on an otherwise pristine checkout. Use ``git diff --quiet``
    against HEAD (covers both staged and unstaged), which ignores
    untracked entirely.
    """
    # Refresh the index so stat-only diffs (mtime updates without content
    # changes — common after a fresh checkout) don't false-trigger.
    try:
        subprocess.run(
            ["git", "update-index", "--refresh"],
            cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass
    try:
        rc = subprocess.call(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return rc != 0
    except Exception:
        return False


def collect_provenance(extra: dict | None = None) -> dict:
    """Collect identifying metadata about this run for the report header.

    Captured: git SHA + short + branch + dirty bit + date, host OS + Python,
    plus the path to the script that generated the report. Helpful when
    the HTML lives on disk for months and the reviewer needs to know
    exactly which code produced it.
    """
    sha = _git("rev-parse", "HEAD")
    short = sha[:8] if sha else ""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git_dirty()
    last_msg = _git("log", "-1", "--format=%s")
    last_author = _git("log", "-1", "--format=%an")
    last_when = _git("log", "-1", "--format=%ai")
    prov = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": sha,
        "git_short": short,
        "git_branch": branch,
        "git_dirty": dirty,
        "git_last_commit_msg": last_msg,
        "git_last_commit_author": last_author,
        "git_last_commit_when": last_when,
        "host": platform.node(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
    }
    if extra:
        prov.update(extra)
    return prov


# Mass-listener fields that compose dry_mass. Names match
# v2ecoli/steps/listeners/mass_listener.py:submass_listener_indices.
DRY_MASS_COMPONENTS = (
    "protein_mass",
    "rna_mass",
    "dna_mass",
    "smallMolecule_mass",
    # rRna/mRna/tRna are subsets of rna_mass; omit from the stacked bar to
    # avoid double-counting.
)

# Extra fields we want in the endpoint state table.
ENDPOINT_FIELDS = (
    "cell_mass", "dry_mass", "water_mass", "volume",
    "protein_mass", "rna_mass", "dna_mass", "smallMolecule_mass",
    "rRna_mass", "mRna_mass", "tRna_mass",
    "membrane_mass", "inner_membrane_mass", "outer_membrane_mass",
    "periplasm_mass", "cytosol_mass", "extracellular_mass",
)


def run_with_timing(composite_name: str, duration_s: int, seed: int = 0,
                    sample_every_s: int = 60,
                    **build_kwargs) -> dict:
    """Build → run in chunks → snapshot timing + endpoint state + timeseries."""
    from v2ecoli import build_composite
    from scripts.render_dnaa00_chromosome_viz import extract_snapshot

    t0 = time.perf_counter()
    composite = build_composite(composite_name, seed=seed, **build_kwargs)
    build_wall = time.perf_counter() - t0

    samples: list[dict] = []
    sim_time = 0
    t1 = time.perf_counter()
    while sim_time < duration_s:
        chunk = min(sample_every_s, duration_s - sim_time)
        composite.run(chunk)
        sim_time += chunk
        mass = (composite.state.get("agents") or {}).get("0", {}).get(
            "listeners", {}).get("mass", {})
        snap = {
            "t": sim_time,
            **{k: float(mass.get(k, 0.0)) for k in ENDPOINT_FIELDS if k in mass},
        }
        chrom = extract_snapshot(composite.state, sim_time)
        if chrom is not None:
            snap["n_chromosomes"] = chrom["n_chromosomes"]
            snap["n_domains"] = chrom["n_domains"]
            snap["n_forks"] = len(chrom["fork_coords"])
            snap["n_rnap"] = chrom["n_rnap"]
            snap["fork_coords"] = chrom["fork_coords"]
            snap["fork_domains"] = chrom["fork_domains"]
            snap["rnap_coords"] = chrom["rnap_coords"]
            snap["rnap_domains"] = chrom["rnap_domains"]
            snap["domain_children"] = chrom["domain_children"]
        samples.append(snap)
    run_wall = time.perf_counter() - t1

    ts = composite.timing_summary()
    per_process = {
        " / ".join(str(p) for p in path): seconds
        for path, seconds in (ts.per_process or {}).items()
    }
    timing = {
        "build_wall_s": build_wall,
        "run_wall_s": run_wall,
        "total_s": float(getattr(ts, "total", 0.0)),
        "process_time_s": float(getattr(ts, "process_time", 0.0)),
        "framework_time_s": float(getattr(ts, "framework_time", 0.0)),
        "fraction_in_processes": float(ts.fraction_in_processes() if callable(getattr(ts, "fraction_in_processes", None)) else getattr(ts, "fraction_in_processes", 0.0)),
        "per_process": per_process,
    }

    mass = composite.state.get("agents", {}).get("0", {}).get(
        "listeners", {}).get("mass", {})
    endpoint = {k: float(mass.get(k, 0.0)) for k in ENDPOINT_FIELDS if k in mass}

    return {
        "composite": composite_name,
        "duration_s": duration_s,
        "timing": timing,
        "endpoint": endpoint,
        "samples": samples,
        "state": composite.state,  # for bigraph
    }


def _process_short_name(path: str) -> str:
    """Strip 'agents/0/' prefix and 'ecoli-' prefix for readability."""
    short = path.replace("agents / 0 / ", "")
    if short.startswith("ecoli-"):
        short = short[len("ecoli-"):]
    return short


def _top_processes(per_process: dict, n: int = 12) -> list[tuple[str, float]]:
    items = sorted(per_process.items(), key=lambda kv: kv[1], reverse=True)
    return [(_process_short_name(k), v) for k, v in items[:n]]


def make_viz(baseline: dict, pdmp: dict, top_n: int = 12):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"PDMP+consumption_matched vs baseline (kFBA)  "
        f"— {baseline['duration_s']} s sim, seed=0",
        fontsize=14, fontweight="bold",
    )

    # --- Panel 1: wall-time bars (top-N processes) ---
    ax = axes[0]
    base_top = _top_processes(baseline["timing"]["per_process"], n=top_n)
    pdmp_top = _top_processes(pdmp["timing"]["per_process"], n=top_n)
    # Union of process names so both bars stay aligned.
    names = []
    for k, _ in base_top + pdmp_top:
        if k not in names:
            names.append(k)
    names = names[:top_n + 4]
    base_seconds = [baseline["timing"]["per_process"].get(
        "agents / 0 / ecoli-" + n,
        baseline["timing"]["per_process"].get("agents / 0 / " + n, 0.0)
    ) for n in names]
    pdmp_seconds = [pdmp["timing"]["per_process"].get(
        "agents / 0 / ecoli-" + n,
        pdmp["timing"]["per_process"].get("agents / 0 / " + n, 0.0)
    ) for n in names]

    y = np.arange(len(names))
    h = 0.4
    ax.barh(y - h / 2, base_seconds, h, label=f"baseline ({baseline['timing']['run_wall_s']:.1f}s wall)",
            color="#3b82f6")
    ax.barh(y + h / 2, pdmp_seconds, h, label=f"PDMP+cm ({pdmp['timing']['run_wall_s']:.1f}s wall)",
            color="#a855f7")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cumulative invoke time (s)")
    ax.set_title("Top process timings", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # --- Panel 2: mass-fraction stacked bar ---
    ax = axes[1]
    labels = ["baseline\n(kFBA)", "PDMP+cm"]
    component_data = {c: [] for c in DRY_MASS_COMPONENTS}
    component_data["water_mass"] = []
    component_data["other"] = []  # cell - water - sum(components)
    for run in (baseline, run := pdmp):
        ep = run["endpoint"]
        cm = ep.get("cell_mass", 0.0)
        water = ep.get("water_mass", 0.0)
        component_data["water_mass"].append(water)
        comp_sum = 0.0
        for c in DRY_MASS_COMPONENTS:
            v = ep.get(c, 0.0)
            component_data[c].append(v)
            comp_sum += v
        component_data["other"].append(max(0.0, cm - water - comp_sum))

    component_data = {**{k: v for k, v in component_data.items()}}
    colors = ["#9ca3af", "#3b82f6", "#10b981", "#f59e0b", "#a78bfa", "#ef4444"]
    component_order = ["water_mass"] + list(DRY_MASS_COMPONENTS) + ["other"]
    bottom = np.zeros(len(labels))
    for i, comp in enumerate(component_order):
        vals = component_data[comp]
        bars = ax.bar(labels, vals, bottom=bottom, color=colors[i % len(colors)],
                      label=comp.replace("_mass", ""), edgecolor="white", linewidth=1)
        # Annotate non-trivial slices.
        for j, b in enumerate(bars):
            if vals[j] > 30:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f"{vals[j]:.0f}",
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom = bottom + np.asarray(vals)
    ax.set_ylabel("Mass (fg)")
    ax.set_title("Cell-mass composition at t=600 s", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # --- Panel 3: endpoint table ---
    ax = axes[2]
    ax.axis("off")
    rows = []
    base_ep = baseline["endpoint"]
    pdmp_ep = pdmp["endpoint"]
    all_keys = [k for k in ENDPOINT_FIELDS if k in base_ep or k in pdmp_ep]
    for k in all_keys:
        bv = base_ep.get(k, float("nan"))
        pv = pdmp_ep.get(k, float("nan"))
        delta = pv - bv
        rows.append([k.replace("_mass", ""),
                     f"{bv:.2f}", f"{pv:.2f}",
                     f"{delta:+.2f}"])
    tbl = ax.table(
        cellText=rows,
        colLabels=["field", "baseline", "PDMP+cm", "Δ"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Endpoint state (fg)", fontsize=12, pad=8)

    plt.tight_layout()
    return fig


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_timeseries_fig(baseline: dict, pdmp: dict):
    """Time-series comparison — totals + replication state."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Time-series — totals + replication state",
                 fontsize=13, fontweight="bold")

    panels = [
        ("cell_mass", "cell_mass (fg)", axes[0, 0]),
        ("dry_mass",  "dry_mass (fg)",  axes[0, 1]),
        ("water_mass","water_mass (fg)",axes[1, 0]),
        ("n_forks",   "active replisomes (forks)", axes[1, 1]),
    ]
    for field, ylabel, ax in panels:
        for run, color, label in [
            (baseline, "#3b82f6", "baseline (kFBA)"),
            (pdmp,     "#a855f7", "PDMP+cm"),
        ]:
            ts_t = [s["t"] for s in run["samples"]]
            ts_v = [s.get(field, 0.0) for s in run["samples"]]
            ax.plot(ts_t, ts_v, "-o", ms=4, color=color, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("t (s)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return fig


# Submass groupings for the multi-panel time-series grid.
SUBMASS_GROUPS = [
    # (group title, [(field, display_label)])
    ("Macromolecules", [
        ("protein_mass", "protein"),
        ("rna_mass", "RNA (total)"),
        ("dna_mass", "DNA"),
        ("smallMolecule_mass", "small molecules"),
    ]),
    ("RNA breakdown", [
        ("rRna_mass", "rRNA"),
        ("mRna_mass", "mRNA"),
        ("tRna_mass", "tRNA"),
    ]),
    ("Membranes / compartments", [
        ("inner_membrane_mass", "inner membrane"),
        ("outer_membrane_mass", "outer membrane"),
        ("membrane_mass", "membrane (total)"),
        ("periplasm_mass", "periplasm"),
        ("cytosol_mass", "cytosol"),
        ("extracellular_mass", "extracellular"),
    ]),
]


def make_submass_grid_fig(baseline: dict, pdmp: dict):
    """Per-submass time-series grid (one subplot per submass field, two
    curves each: baseline + PDMP)."""
    # Count total fields across groups.
    fields = [
        (group_title, field, label)
        for group_title, items in SUBMASS_GROUPS
        for field, label in items
    ]
    n = len(fields)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows),
                             sharex=True)
    fig.suptitle("Submass time-series — baseline vs PDMP+consumption_matched",
                 fontsize=14, fontweight="bold", y=1.0)

    axes_flat = axes.ravel()
    for ax, (group_title, field, label) in zip(axes_flat, fields):
        any_data = False
        for run, color, run_label in [
            (baseline, "#3b82f6", "baseline (kFBA)"),
            (pdmp,     "#a855f7", "PDMP+cm"),
        ]:
            ts_t = [s["t"] for s in run["samples"]]
            ts_v = [s.get(field, 0.0) for s in run["samples"]]
            if any(v != 0.0 for v in ts_v):
                any_data = True
            ax.plot(ts_t, ts_v, "-o", ms=3, color=color, label=run_label)
        ax.set_title(f"{label}", fontsize=10, fontweight="bold")
        ax.set_ylabel("fg", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        if not any_data:
            ax.text(0.5, 0.5, "(zero throughout)", ha="center", va="center",
                    color="#94a3b8", fontsize=9, transform=ax.transAxes)
        # Top-left subplot gets the legend.
        if ax is axes_flat[0]:
            ax.legend(fontsize=8, loc="best")
        # Group header annotation on the leftmost column.
        ax_idx = list(axes_flat).index(ax)
        if ax_idx % ncols == 0:
            ax.set_ylabel(f"{label}\n(fg)", fontsize=8)
    # Bottom row x-labels.
    for ax in axes_flat[-ncols:]:
        ax.set_xlabel("t (s)", fontsize=9)
    # Hide unused panels.
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# (bigraph_viz2 embedding removed — kept for future use if needed.)


def make_chromosome_fig_uri(samples: list[dict], title: str = "") -> str:
    """Render the chromosome-state diagram (oriC, replisomes, forks) using
    v2ecoli's :func:`_plot_chromosome_timeline` — same figure the dnaa-02 and
    dnaa-06 studies use to show the chromosome state evolving over the
    cell cycle. Returns a data URI for embedding."""
    from v2ecoli.visualizations.workflow import _plot_chromosome_timeline

    # _plot_chromosome_timeline expects 'time' key, list of fork_coords,
    # fork_domains, n_rnap, etc. Translate from our 't'-keyed samples.
    converted = []
    for s in samples:
        converted.append({
            "time": s.get("t", 0.0),
            "n_chromosomes": s.get("n_chromosomes", 0),
            "fork_coords": s.get("fork_coords", []),
            "fork_domains": s.get("fork_domains", []),
            "rnap_coords": s.get("rnap_coords", []),
            "rnap_domains": s.get("rnap_domains", []),
            "n_rnap": s.get("n_rnap", 0),
            "domain_children": s.get("domain_children", {}),
        })
    b64 = _plot_chromosome_timeline(converted, title=title, annotate_events=True)
    return "data:image/png;base64," + b64


def write_html(
    out_path: Path,
    summary_data_uri: str,
    timeseries_data_uri: str,
    submass_grid_uri: str,
    chromosome_base_uri: str | None,
    chromosome_pdmp_uri: str | None,
    baseline: dict,
    pdmp: dict,
    provenance: dict,
) -> None:
    bt = baseline["timing"]
    pt = pdmp["timing"]
    speedup = (bt["run_wall_s"] / pt["run_wall_s"]) if pt["run_wall_s"] > 0 else float("nan")
    dirty_badge = (
        '<span style="color:#dc2626;font-weight:600">  · DIRTY TREE</span>'
        if provenance.get("git_dirty") else ""
    )
    short = provenance.get("git_short", "")
    branch = provenance.get("git_branch", "")
    gen_at = provenance.get("generated_at", "")
    full_sha = provenance.get("git_sha", "")
    last_msg = provenance.get("git_last_commit_msg", "")
    last_author = provenance.get("git_last_commit_author", "")
    last_when = provenance.get("git_last_commit_when", "")
    host = provenance.get("host", "")
    pyver = provenance.get("python", "")
    platf = provenance.get("platform", "")
    script_rel = provenance.get("script", "")
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>PDMP+consumption_matched vs baseline — {short or 'report'}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         color: #1f2937; max-width: 1400px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ margin: 0 0 6px 0; }}
  .meta {{ color: #6b7280; font-size: 0.9em; margin-bottom: 14px; }}
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
<h1>PDMP + consumption_matched vs baseline (kFBA)</h1>
<div class="meta">
  Sim duration: <code>{baseline['duration_s']} s</code>, seed=0, M9-glucose.
  Composite: baseline = <code>{baseline['composite']}</code>,
  PDMP = <code>{pdmp['composite']}</code> with
  <code>with_ref_growth=True, ref_growth_flux_source='consumption_matched'</code>.
</div>

<div class="provenance">
  <div class="row"><dt>generated</dt><dd>{gen_at}</dd></div>
  <div class="row"><dt>git commit</dt>
    <dd><a href="https://github.com/vivarium-collective/v2ecoli/commit/{full_sha}"
        style="color:#0369a1;text-decoration:none">{short}</a> &nbsp;<code>{full_sha}</code>{dirty_badge}</dd></div>
  <div class="row"><dt>git branch</dt><dd>{branch}</dd></div>
  <div class="row"><dt>last commit</dt><dd>{last_msg} — {last_author} ({last_when})</dd></div>
  <div class="row"><dt>script</dt><dd>{script_rel}</dd></div>
  <div class="row"><dt>host</dt><dd>{host} &nbsp; <span style="color:#94a3b8">{platf}, Python {pyver}</span></dd></div>
</div>

<table class="summary">
  <tr><th></th><th>baseline (kFBA)</th><th>PDMP+cm</th><th>Δ</th></tr>
  <tr><td>build wall</td>
      <td class="num">{bt['build_wall_s']:.2f} s</td>
      <td class="num">{pt['build_wall_s']:.2f} s</td>
      <td class="num">{pt['build_wall_s'] - bt['build_wall_s']:+.2f} s</td></tr>
  <tr><td>run wall</td>
      <td class="num">{bt['run_wall_s']:.2f} s</td>
      <td class="num">{pt['run_wall_s']:.2f} s</td>
      <td class="num">{pt['run_wall_s'] - bt['run_wall_s']:+.2f} s ({speedup:.2f}×)</td></tr>
  <tr><td>process time</td>
      <td class="num">{bt['process_time_s']:.2f} s</td>
      <td class="num">{pt['process_time_s']:.2f} s</td>
      <td class="num">—</td></tr>
  <tr><td>framework time</td>
      <td class="num">{bt['framework_time_s']:.2f} s</td>
      <td class="num">{pt['framework_time_s']:.2f} s</td>
      <td class="num">—</td></tr>
  <tr><td>cell_mass (fg)</td>
      <td class="num">{baseline['endpoint'].get('cell_mass', 0):.2f}</td>
      <td class="num">{pdmp['endpoint'].get('cell_mass', 0):.2f}</td>
      <td class="num">{pdmp['endpoint'].get('cell_mass', 0) - baseline['endpoint'].get('cell_mass', 0):+.2f}</td></tr>
  <tr><td>dry_mass (fg)</td>
      <td class="num">{baseline['endpoint'].get('dry_mass', 0):.2f}</td>
      <td class="num">{pdmp['endpoint'].get('dry_mass', 0):.2f}</td>
      <td class="num">{pdmp['endpoint'].get('dry_mass', 0) - baseline['endpoint'].get('dry_mass', 0):+.2f}</td></tr>
</table>

<h2 style="margin-top:24px;">Summary panels</h2>
<img src="{summary_data_uri}" alt="comparison panels">

<h2 style="margin-top:24px;">Time-series</h2>
<img src="{timeseries_data_uri}" alt="time-series comparison">

<h2 style="margin-top:24px;">Submass time-series — all components</h2>
<img src="{submass_grid_uri}" alt="submass time-series grid">

<h2 style="margin-top:24px;">Chromosome / replication state</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
"""
    if chromosome_base_uri:
        html += (f'<div><h3>baseline (kFBA)</h3>'
                 f'<img src="{chromosome_base_uri}" alt="chromosome (baseline)"></div>\n')
    else:
        html += '<div><h3>baseline (kFBA)</h3><em>render failed</em></div>\n'
    if chromosome_pdmp_uri:
        html += (f'<div><h3>PDMP+cm</h3>'
                 f'<img src="{chromosome_pdmp_uri}" alt="chromosome (PDMP)"></div>\n')
    else:
        html += '<div><h3>PDMP+cm</h3><em>render failed</em></div>\n'
    html += "</div>\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=600,
                   help="Simulated seconds for both runs (default 600).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-composite", default="baseline")
    p.add_argument("--pdmp-composite", default="millard_pdmp_baseline")
    p.add_argument("--out",
                   default="reports/figures/pdmp-01/pdmp_vs_baseline.html")
    p.add_argument("--summary-json",
                   default=".pbg/runs/pdmp-vs-baseline/summary.json")
    args = p.parse_args()

    print(f"Running baseline ({args.baseline_composite}) for {args.duration}s...",
          flush=True)
    baseline = run_with_timing(
        args.baseline_composite, args.duration, args.seed)
    print(f"  done: run_wall={baseline['timing']['run_wall_s']:.2f}s  "
          f"cm={baseline['endpoint'].get('cell_mass', float('nan')):.2f}  "
          f"dm={baseline['endpoint'].get('dry_mass', float('nan')):.2f}",
          flush=True)

    print(f"Running PDMP ({args.pdmp_composite}) + consumption_matched "
          f"for {args.duration}s...", flush=True)
    pdmp = run_with_timing(
        args.pdmp_composite, args.duration, args.seed,
        with_ref_growth=True,
        ref_growth_flux_source="consumption_matched",
    )
    print(f"  done: run_wall={pdmp['timing']['run_wall_s']:.2f}s  "
          f"cm={pdmp['endpoint'].get('cell_mass', float('nan')):.2f}  "
          f"dm={pdmp['endpoint'].get('dry_mass', float('nan')):.2f}",
          flush=True)

    # Summary JSON (omit non-serializable state).
    def _strip(run):
        return {k: v for k, v in run.items() if k != "state"}
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(
        {"baseline": _strip(baseline), "pdmp": _strip(pdmp)},
        indent=2, default=str))
    print(f"Wrote summary {args.summary_json}", flush=True)

    print("Rendering summary panels...", flush=True)
    summary_uri = fig_to_data_uri(make_viz(baseline, pdmp))
    print("Rendering time-series...", flush=True)
    timeseries_uri = fig_to_data_uri(make_timeseries_fig(baseline, pdmp))
    print("Rendering submass grid...", flush=True)
    submass_grid_uri = fig_to_data_uri(make_submass_grid_fig(baseline, pdmp))
    print("Rendering chromosome figures (oriC + replisomes + forks)...", flush=True)
    chrom_base_uri = make_chromosome_fig_uri(
        baseline["samples"], title="baseline (kFBA) — chromosome state")
    chrom_pdmp_uri = make_chromosome_fig_uri(
        pdmp["samples"], title="PDMP + consumption_matched — chromosome state")
    print("Collecting provenance...", flush=True)
    provenance = collect_provenance(extra={
        "duration_s": args.duration,
        "seed": args.seed,
        "baseline_composite": args.baseline_composite,
        "pdmp_composite": args.pdmp_composite,
    })

    write_html(
        Path(args.out),
        summary_uri, timeseries_uri, submass_grid_uri,
        chrom_base_uri, chrom_pdmp_uri,
        baseline, pdmp, provenance,
    )

    # Also write a date+commit-stamped archival copy so each run is
    # individually addressable as evidence on PR #72.
    short = provenance.get("git_short") or "nogit"
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = Path(args.out).with_name(
        f"pdmp_vs_baseline_{stamp}_{short}.html"
    )
    archive.write_bytes(Path(args.out).read_bytes())
    print(f"Wrote archive copy {archive}", flush=True)
    print(f"Wrote viz {args.out}", flush=True)


if __name__ == "__main__":
    main()
