"""V1V2Visualization — three-way comparison: vEcoli 1.0 vs 2.0 vs v2ecoli.

Migrated rendering from reports/v1_v2_report.py. The Step takes three
trajectories and produces a side-by-side comparison HTML with overlaid
matplotlib plots. The wrapper at reports/v1_v2_report.py handles the
three-subprocess orchestration.
"""

from __future__ import annotations

import base64
import io
import warnings
from html import escape
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_document

# ---------------------------------------------------------------------------
# Engine display metadata (key, display label, hex color, line style)
# ---------------------------------------------------------------------------
_ENGINES = [
    ("vecoli_v1",         "vEcoli 1.0 (vivarium)",     "#3b82f6", "--"),
    ("vecoli_composite",  "vEcoli 2.0 (composite)",    "#8b5cf6", "-."),
    ("v2ecoli",           "v2ecoli (pure PBG)",         "#ef4444", "-"),
]

# Map the three input port names to the engine keys above.
_PORT_TO_ENGINE = {
    "history_v1":      "vecoli_v1",
    "history_v2":      "vecoli_composite",
    "history_v2ecoli": "v2ecoli",
}

_TD = 'style="padding:6px 12px;border:1px solid #e2e8f0;"'
_TH = 'style="padding:8px 12px;text-align:left;border:1px solid #e2e8f0;background:#f1f5f9;"'


# ---------------------------------------------------------------------------
# Plot helpers (migrated from reports/v1_v2_report.py)
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_comparison(datasets: dict, metric: str, ylabel: str, title: str) -> str:
    """Single metric overlay plot for all available engines. Returns base64 PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, color, ls in _ENGINES:
        snaps = datasets.get(key, {}).get("snapshots", [])
        if not snaps:
            continue
        t = [s["time"] / 60 for s in snaps]
        y = [s.get(metric, 0) for s in snaps]
        ax.plot(t, y, color=color, linestyle=ls, label=label, linewidth=1.5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _plot_side_by_side(datasets: dict, metrics: list, ylabel: str, title: str) -> str:
    """Side-by-side panels for each engine showing multiple metrics. Returns base64 PNG."""
    active = [(k, l, c, ls) for k, l, c, ls in _ENGINES if datasets.get(k, {}).get("snapshots")]
    n = max(len(active), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=13)

    for i, (key, label, _, _) in enumerate(active):
        ax = axes[0][i]
        snaps = datasets[key]["snapshots"]
        t = [s["time"] / 60 for s in snaps]
        for mkey, name, color in metrics:
            y = [s.get(mkey, 0) for s in snaps]
            ax.plot(t, y, color=color, label=name, linewidth=1.2)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(len(active), n):
        axes[0][i].text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[0][i].transAxes)

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_mass_components(datasets: dict) -> str:
    """Side-by-side mass component plots for each engine. Returns base64 PNG."""
    components = [
        ("protein_mass",       "Protein",    "#22c55e"),
        ("dna_mass",           "DNA",        "#8b5cf6"),
        ("rRna_mass",          "rRNA",       "#3b82f6"),
        ("tRna_mass",          "tRNA",       "#06b6d4"),
        ("mRna_mass",          "mRNA",       "#f97316"),
        ("smallMolecule_mass", "Small mol",  "#f59e0b"),
    ]
    return _plot_side_by_side(datasets, components, "Mass (fg)", "Mass Components")


# ---------------------------------------------------------------------------
# HTML rendering (migrated and adapted from generate_report() in legacy script)
# ---------------------------------------------------------------------------

def _snap_at(snaps: list, t: float) -> dict:
    """Find snapshot closest to time t."""
    best: dict = {}
    for s in snaps:
        if abs(s["time"] - t) < abs(best.get("time", 1e9) - t):
            best = s
    return best


def _render_comparison_html(datasets: dict, duration: float, title: str) -> str:
    """Generate three-way comparison HTML body.

    Migrated from reports/v1_v2_report.generate_report().  The ``datasets``
    dict maps engine keys (``vecoli_v1``, ``vecoli_composite``, ``v2ecoli``) to
    dicts with at least a ``"snapshots"`` list.  Each snapshot is a flat dict
    with numeric fields (``time``, ``dry_mass``, ``cell_mass``, etc.).
    """
    intervals = [t for t in [0, 60, 300, 600, 1200, 1800, 2400] if t <= duration]

    # Generate plots only when there is at least some data.
    plots: dict[str, str] = {}
    has_data = any(datasets.get(k, {}).get("snapshots") for k, _, _, _ in _ENGINES)
    if has_data:
        plots["dry_mass"]   = _plot_comparison(datasets, "dry_mass",   "Dry Mass (fg)",       "Dry Mass Over Time")
        plots["cell_mass"]  = _plot_comparison(datasets, "cell_mass",  "Cell Mass (fg)",       "Cell Mass (wet)")
        plots["growth_rate"] = _plot_comparison(datasets, "instantaneous_growth_rate",
                                                "Growth Rate (1/s)", "Instantaneous Growth Rate")
        plots["volume"]      = _plot_comparison(datasets, "volume",     "Volume (fL)",         "Cell Volume")
        plots["chromosomes"] = _plot_comparison(datasets, "n_chromosomes", "Chromosomes",      "Chromosome Count")
        plots["forks"]       = _plot_comparison(datasets, "n_forks",    "Forks",               "Replication Forks")
        plots["mass_components"] = _plot_mass_components(datasets)

        rna_metrics = [
            ("rRna_mass", "rRNA", "#3b82f6"),
            ("tRna_mass", "tRNA", "#06b6d4"),
            ("mRna_mass", "mRNA", "#f97316"),
        ]
        plots["rna_breakdown"] = _plot_side_by_side(datasets, rna_metrics, "Mass (fg)", "RNA Mass Breakdown")

        struct_metrics = [
            ("protein_mass",       "Protein",        "#22c55e"),
            ("dna_mass",           "DNA",             "#8b5cf6"),
            ("smallMolecule_mass", "Small molecules", "#f59e0b"),
        ]
        plots["structural"] = _plot_side_by_side(datasets, struct_metrics, "Mass (fg)", "Structural Components")

    # --- Dry-mass comparison table ---
    mass_table_rows = ""
    for t in intervals:
        row = f'<tr><td style="font-weight:bold">{t/60:.0f} min</td>'
        for key, _, _, _ in _ENGINES:
            snaps = datasets.get(key, {}).get("snapshots", [])
            st = _snap_at(snaps, t) if snaps else {}
            dm = st.get("dry_mass", 0)
            row += f"<td>{dm:.1f}</td>" if dm > 0 else "<td>—</td>"
        row += "</tr>"
        mass_table_rows += row

    # --- Performance cards ---
    perf_cards = ""
    for key, label, color, _ in _ENGINES:
        d = datasets.get(key, {})
        speed = d.get("speed", 0)
        wall  = d.get("wall_time", 0)
        sim   = d.get("sim_time", 0)
        if speed > 0:
            perf_cards += (
                f'<div class="perf-card">'
                f'<div class="label">{escape(label)}</div>'
                f'<div class="value" style="color:{color}">{speed:.1f}x</div>'
                f'<div class="label">{wall:.0f}s wall for {sim:.0f}s sim</div>'
                f"</div>\n"
            )
        else:
            perf_cards += (
                f'<div class="perf-card">'
                f'<div class="label">{escape(label)}</div>'
                f'<div class="value" style="color:#ccc">N/A</div>'
                f'<div class="label">not available</div>'
                f"</div>\n"
            )

    # --- Overview table ---
    overview_rows = ""
    metrics_list = [
        ("Engine",       lambda d: d.get("engine", "N/A")),
        ("Load time",    lambda d: f"{d.get('load_time',0):.2f}s"             if d.get("load_time") else "N/A"),
        ("Sim duration", lambda d: f"{d.get('sim_time',0):.0f}s ({d.get('sim_time',0)/60:.1f} min)" if d.get("sim_time") else "N/A"),
        ("Wall time",    lambda d: f"{d.get('wall_time',0):.1f}s"             if d.get("wall_time") else "N/A"),
        ("Speed",        lambda d: f"{d.get('speed',0):.1f}x"                 if d.get("speed")     else "N/A"),
    ]
    for name, fn in metrics_list:
        row = f"<tr><td>{escape(name)}</td>"
        for key, _, _, _ in _ENGINES:
            d = datasets.get(key, {})
            row += f"<td>{escape(str(fn(d)))}</td>"
        row += "</tr>"
        overview_rows += row

    for name, metric in [
        ("Final dry mass", "dry_mass"),
        ("Chromosomes",    "n_chromosomes"),
        ("Forks",          "n_forks"),
    ]:
        row = f"<tr><td>{escape(name)}</td>"
        for key, _, _, _ in _ENGINES:
            snaps = datasets.get(key, {}).get("snapshots", [])
            val = snaps[-1].get(metric, 0) if snaps else 0
            fmt = f"{val:.1f} fg" if "mass" in name.lower() else str(int(val))
            row += f"<td>{escape(fmt)}</td>"
        row += "</tr>"
        overview_rows += row

    # --- Plot img tags ---
    def _img(key: str, alt: str) -> str:
        if not plots.get(key):
            return ""
        return f'<div class="plot"><img src="data:image/png;base64,{plots[key]}" alt="{escape(alt)}"></div>\n'

    html = (
        """<style>
body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }
h1 { color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }
h2 { color: #1e40af; margin-top: 2em; }
h3 { color: #334155; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { padding: 6px 12px; border: 1px solid #e2e8f0; text-align: center; }
th { background: #f1f5f9; font-weight: 600; }
.plot { margin: 1em 0; text-align: center; }
.plot img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }
.perf { display: flex; gap: 2em; justify-content: center; margin: 1em 0; }
.perf-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em 2em; text-align: center; }
.perf-card .value { font-size: 2em; font-weight: bold; }
.perf-card .label { color: #64748b; font-size: 0.9em; }
</style>
"""
        f"<h1>{escape(title)}</h1>\n"
        "<p>Three-way comparison of vEcoli 1.0 (vivarium engine), vEcoli 2.0 (composite migration), "
        "and v2ecoli (pure process-bigraph). All use the same ParCa parameters, initial state, "
        "and biological process logic.</p>\n"
        "<h2>Performance</h2>\n"
        f'<div class="perf">\n{perf_cards}</div>\n'
        "<table>\n"
        "<tr><th>Metric</th><th>vEcoli 1.0</th><th>vEcoli 2.0</th><th>v2ecoli</th></tr>\n"
        f"{overview_rows}\n"
        "</table>\n"
        "<h2>Growth Comparison</h2>\n"
    )

    for key, alt in [
        ("dry_mass",    "Dry Mass"),
        ("cell_mass",   "Cell Mass"),
        ("growth_rate", "Growth Rate"),
        ("volume",      "Volume"),
    ]:
        html += _img(key, alt)

    html += "<h2>Chromosome Replication</h2>\n"
    for key, alt in [("chromosomes", "Chromosomes"), ("forks", "Replication Forks")]:
        html += _img(key, alt)

    html += "<h2>Mass Components (side-by-side)</h2>\n"
    html += _img("mass_components", "Mass Components")
    html += _img("rna_breakdown",   "RNA Breakdown")
    html += _img("structural",      "Structural Components")

    html += (
        "<h2>Dry Mass Comparison Table (fg)</h2>\n"
        "<table>\n"
        "<tr><th>Time</th><th>vEcoli 1.0</th><th>vEcoli 2.0</th><th>v2ecoli</th></tr>\n"
        f"{mass_table_rows}\n"
        "</table>\n"
        "<footer style='margin-top:2em;color:#64748b;font-size:0.85em;'>"
        "Generated by V1V2Visualization &middot; v2ecoli (pure process-bigraph)"
        "</footer>\n"
    )

    return html


# ---------------------------------------------------------------------------
# Visualization Step
# ---------------------------------------------------------------------------

class V1V2Visualization(Visualization):
    """Render three-way comparison from three trajectory inputs.

    Inputs
    ------
    history_v1:      trajectory rows from vEcoli 1.0 (vivarium engine).
    history_v2:      trajectory rows from vEcoli 2.0 (composite migration).
    history_v2ecoli: trajectory rows from v2ecoli (pure process-bigraph).
    metadata:        run metadata (seed, duration_sec, etc.).

    Each history is a list of flat snapshot dicts.  Recognised snapshot keys:
    ``time``, ``dry_mass``, ``cell_mass``, ``protein_mass``, ``rna_mass``,
    ``rRna_mass``, ``tRna_mass``, ``mRna_mass``, ``dna_mass``,
    ``smallMolecule_mass``, ``water_mass``, ``volume``,
    ``instantaneous_growth_rate``, ``n_chromosomes``, ``n_forks``.
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history_v1":      "list[map[any]]",
            "history_v2":      "list[map[any]]",
            "history_v2ecoli": "list[map[any]]",
            "metadata":        "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        h_v1  = state.get("history_v1")      or []
        h_v2  = state.get("history_v2")       or []
        h_v2e = state.get("history_v2ecoli")  or []
        meta  = state.get("metadata")          or {}
        title = self.config.get("title")       or "v1 vs v2 vs v2ecoli"

        # Build the datasets dict expected by the rendering helpers.
        duration = float(meta.get("duration_sec") or 0)
        if not duration and (h_v1 or h_v2 or h_v2e):
            # Derive from the longest trajectory if not explicitly provided.
            for hist in (h_v1, h_v2, h_v2e):
                if hist:
                    t = hist[-1].get("time", 0)
                    if t > duration:
                        duration = float(t)

        datasets: dict[str, Any] = {
            "vecoli_v1":        {"snapshots": h_v1,  **_meta_from_rows(h_v1,  meta)},
            "vecoli_composite": {"snapshots": h_v2,  **_meta_from_rows(h_v2,  meta)},
            "v2ecoli":          {"snapshots": h_v2e, **_meta_from_rows(h_v2e, meta)},
        }

        body = _render_comparison_html(datasets, duration, title)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_from_rows(rows: list, meta: dict) -> dict:
    """Extract performance metadata from trajectory rows or the metadata dict."""
    wall_time = float(meta.get("wall_time", 0) or 0)
    sim_time  = float(meta.get("sim_time",  0) or 0)
    load_time = float(meta.get("load_time", 0) or 0)
    speed     = (sim_time / wall_time) if wall_time > 0 and sim_time > 0 else 0.0
    return {
        "wall_time": wall_time,
        "sim_time":  sim_time,
        "load_time": load_time,
        "speed":     speed,
        "engine":    meta.get("engine", ""),
    }
