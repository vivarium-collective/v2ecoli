#!/usr/bin/env python
"""Figures for the dnaa n×K in-sim parameter sweep (dnaa-7).

Turns the 20 multigen runs (n∈{1,2,4,6,8} × K∈{15,20,30,40}, mechanistic oriC-low
trigger, seed 1) into the report figures Rashmi asked for:

  1. metric heatmap        — n×K grid, cell = composite (cyclic + steady-DnaA-band)
                             score, annotated with CYC/BAND pass glyphs + mean DnaA.
  2. trajectory browser    — interactive 4-panel lineage trajectories with a dropdown
                             to pick any (n,K): number of oriC, total DnaA (band
                             [300,800]), DnaA-ATP fraction (band [0.2,0.5]), oriC-low
                             occupancy. THESE are the actual-simulation trajectories
                             the analytic dnaa-5 switch/K curves lacked.
  3. static per-condition  — matplotlib 4-panel SVG+PNG for a curated subset
                             (reference n4/K30 + the four n/K extremes) for the
                             report charts dir, with generation-boundary marks.
  4. metric CSV            — per-condition + per-generation tables for the Data tab.

Usage:
  make_dnaa_sweep_figures.py --out-root out --tag-glob 'nk_n*_K*' \
      --study-dir workspace/studies/dnaa-7-parameter-sweep
"""
from __future__ import annotations
import argparse, csv, glob, os, re, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.dnaa_sweep_analysis import (
    load_trajectory, compute_metric, DNAA_BAND, ATP_BAND, RD)
from scripts.pbg_plot_style import PALETTE

PANELS = [
    ("number_of_oric", "oriC count", None, PALETTE["purple"]),
    ("total_dnaa", "total DnaA (counts)", DNAA_BAND, PALETTE["blue"]),
    ("atp_fraction", "DnaA-ATP fraction", ATP_BAND, PALETTE["amber"]),
    ("oric_low_occ", "oriC-low occupancy", None, PALETTE["green"]),
]


def _parse_nk(tag):
    m = re.search(r"n(\d+)_K(\d+)", tag)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def collect(out_root, tag_glob):
    """Return {(n,K): {'tag','df','bounds','metric'}} for every run dir found."""
    dirs = sorted(glob.glob(os.path.join(out_root, tag_glob)))
    data = {}
    for d in dirs:
        tag = os.path.basename(d)
        n, K = _parse_nk(tag)
        if n is None:
            continue
        df, bounds = load_trajectory(d)
        if df is None:
            print(f"  skip {tag}: no data yet")
            continue
        data[(n, K)] = {"tag": tag, "df": df, "bounds": bounds,
                        "metric": compute_metric(d)}
        print(f"  loaded {tag}: {df.height} steps, {len(bounds)+1} gens")
    return data


# --------------------------------------------------------------------------- #
def fig_heatmap(data, out_html, out_png=None):
    ns = sorted({n for n, _ in data})
    Ks = sorted({K for _, K in data})
    Z = np.full((len(ns), len(Ks)), np.nan)
    text = [["" for _ in Ks] for _ in ns]
    for (n, K), d in data.items():
        m = d["metric"]
        i, j = ns.index(n), Ks.index(K)
        Z[i, j] = m.get("composite", np.nan)
        cyc = "✓" if m.get("cyclic_ok") else "✗"
        band = "✓" if m.get("band_ok") else "✗"
        text[i][j] = (f"<b>{m.get('composite', 0):.2f}</b><br>"
                      f"CYC {cyc}  BAND {band}<br>DnaA {m.get('dnaa_mean', 0):.0f}")
    fig = go.Figure(go.Heatmap(
        z=Z, x=[str(k) for k in Ks], y=[str(n) for n in ns],
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(title="composite<br>(cyclic +<br>steady band)"),
        hovertemplate="n=%{y}, K=%{x} nM<br>composite=%{z:.2f}<extra></extra>"))
    fig.update_layout(
        title=dict(text="<b>oriC-low cooperativity sweep — cyclic-replication + steady-DnaA-band score</b>"
                        "<br><span style='font-size:12px;color:#475569'>"
                        "in-sim, mechanistic oriC-low trigger, 8 generations, seed 1 · "
                        "CYC = once-per-cycle (oriC 1↔2, one fire/gen) · BAND = total DnaA in [300,800]"
                        "</span>",
                   x=0.5, xanchor="center", font=dict(size=16), yref="container", y=0.96, yanchor="top"),
        xaxis=dict(title="K — oriC-low half-saturation (nM)", type="category"),
        yaxis=dict(title="n — Hill cooperativity", type="category"),
        template="plotly_white", height=460, width=760,
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(t=92, b=60, l=70, r=90))
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    if out_png:
        try:
            fig.write_image(out_png, scale=2)
        except Exception as e:
            print(f"  (heatmap png skipped: {e})")
    print(f"  wrote {out_html}")
    return fig


def fig_heatmap_mpl(data, out_svg, out_png):
    """Static matplotlib heatmap for the charts strip (no kaleido dependency)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted({n for n, _ in data})
    Ks = sorted({K for _, K in data})
    Z = np.full((len(ns), len(Ks)), np.nan)
    for (n, K), d in data.items():
        Z[ns.index(n), Ks.index(K)] = d["metric"].get("composite", np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels(Ks)
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
    ax.set_xlabel("K — oriC-low half-saturation (nM)")
    ax.set_ylabel("n — Hill cooperativity")
    ax.set_title("oriC-low cooperativity sweep — cyclic-replication + steady-DnaA-band score\n"
                 "in-sim, mechanistic trigger, 8 generations, seed 1", fontsize=10.5)
    for (n, K), d in data.items():
        m = d["metric"]; i, j = ns.index(n), Ks.index(K)
        cyc = "✓" if m.get("cyclic_ok") else "✗"
        band = "✓" if m.get("band_ok") else "✗"
        val = m.get("composite", 0)
        ax.text(j, i, f"{val:.2f}\nCYC {cyc} BAND {band}\nDnaA {m.get('dnaa_mean',0):.0f}",
                ha="center", va="center", fontsize=7.5,
                color="white" if val < 0.7 else "black")
    fig.colorbar(im, ax=ax, label="composite (cyclic + steady band)")
    fig.tight_layout()
    fig.savefig(out_svg); fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_svg} + {out_png}")


def fig_trajectory_browser(data, out_html):
    keys = sorted(data.keys())
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[p[1] for p in PANELS])
    # one set of 4 traces per condition; only the reference visible initially
    ref = (4, 30) if (4, 30) in data else keys[0]
    trace_cond = []   # condition key per trace, for the dropdown visibility mask
    for k in keys:
        df = data[k]["df"]
        # stride to keep the self-contained embed light (20 conditions × 4 traces);
        # ~1200 points/trace preserves the sawtooth + fire spikes at report scale.
        stride = max(1, df.height // 1200)
        df = df.gather_every(stride)
        x = df["t_min"].to_numpy()
        vis = (k == ref)
        for row, (col, _lab, _band, color) in enumerate(PANELS, start=1):
            y = df[col].to_numpy()
            shape = "hv" if col == "number_of_oric" else "linear"
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", line=dict(color=color, width=1.8, shape=shape),
                name=f"n{k[0]} K{k[1]}", legendgroup=str(k), showlegend=False,
                visible=vis, hovertemplate="%{y:.3g} @ %{x:.0f} min<extra></extra>"),
                row=row, col=1)
            trace_cond.append(k)
    # static band shapes (always shown)
    for row, (_col, _lab, band, _color) in enumerate(PANELS, start=1):
        if band:
            fig.add_hrect(y0=band[0], y1=band[1], line_width=0,
                          fillcolor="rgba(100,116,139,0.10)", row=row, col=1)
    fig.update_yaxes(title_text="count", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="counts", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="fraction", row=3, col=1, range=[0, 1])
    fig.update_yaxes(title_text="occupancy", row=4, col=1, range=[0, 1])
    fig.update_xaxes(title_text="lineage time (min)", row=4, col=1)
    # dropdown: pick condition -> toggle its 4 traces visible
    buttons = []
    for k in keys:
        mask = [tc == k for tc in trace_cond]
        m = data[k]["metric"]
        lab = f"n={k[0]}  K={k[1]}  " + ("✓ cyclic+band" if m.get("pass") else
              ("cyclic" if m.get("cyclic_ok") else "—"))
        buttons.append(dict(label=lab, method="update",
                            args=[{"visible": mask},
                                  {"title.text": _traj_title(k, m)}]))
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=1.0, xanchor="right", y=1.12, yanchor="top",
                          bgcolor="white", bordercolor="#cbd5e1")],
        title=dict(text=_traj_title(ref, data[ref]["metric"]), x=0.5, xanchor="center",
                   font=dict(size=16), yref="container", y=0.97, yanchor="top"),
        template="plotly_white", height=880, width=900,
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        margin=dict(t=120, b=55, l=70, r=40), hovermode="x unified", showlegend=False)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"  wrote {out_html}")
    return fig


def _traj_title(k, m):
    tag = "once-per-cycle + steady DnaA band" if m.get("pass") else \
          ("cyclic (DnaA drifts)" if m.get("cyclic_ok") else "not once-per-cycle")
    return (f"<b>Lineage trajectories — n={k[0]}, K={k[1]} nM</b>"
            f"<br><span style='font-size:12px;color:#475569'>{tag} · "
            f"mean DnaA {m.get('dnaa_mean', 0):.0f} (drift {m.get('dnaa_drift', 0):.2f}) · "
            f"ATP fraction {m.get('atp_frac', 0):.2f}</span>")


def fig_static_panels(data, keys, out_svg, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, len(keys), figsize=(3.4 * len(keys), 9.2),
                             sharex="col", squeeze=False)
    for c, k in enumerate(keys):
        d = data.get(k)
        for r, (col, lab, band, color) in enumerate(PANELS):
            ax = axes[r][c]
            if d is None:
                ax.text(0.5, 0.5, "no run", ha="center", va="center", transform=ax.transAxes)
                continue
            df = d["df"]; x = df["t_min"].to_numpy(); y = df[col].to_numpy()
            if band:
                ax.axhspan(band[0], band[1], color="0.5", alpha=0.10, lw=0)
            step = col == "number_of_oric"
            (ax.step if step else ax.plot)(x, y, color=color, lw=1.4,
                                           **({"where": "post"} if step else {}))
            for b in d["bounds"]:
                ax.axvline(b, color=PALETTE["muted"], lw=0.7, ls=":", alpha=0.5)
            ax.grid(False)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                m = d["metric"]
                flag = "✓cyc+band" if m.get("pass") else ("cyc" if m.get("cyclic_ok") else "—")
                ax.set_title(f"n={k[0]}, K={k[1]}\n{flag}", fontsize=10)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9)
            if col in ("atp_fraction", "oric_low_occ"):
                ax.set_ylim(0, 1)
            if col in ("number_of_oric", "total_dnaa"):
                ax.set_ylim(bottom=0)
            if r == 3:
                ax.set_xlabel("lineage time (min)", fontsize=9)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_svg); fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_svg} + {out_png}")


def write_csvs(data, cond_csv, gen_csv):
    with open(cond_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "K_nM", "composite", "cyclic_ok", "band_ok", "pass",
                    "mean_total_dnaa", "dnaa_drift", "dnaa_cv", "atp_fraction",
                    "atp_in_band", "n_gens"])
        for (n, K) in sorted(data):
            m = data[(n, K)]["metric"]
            w.writerow([n, K, m.get("composite"), m.get("cyclic_ok"), m.get("band_ok"),
                        m.get("pass"), m.get("dnaa_mean"), m.get("dnaa_drift"),
                        m.get("dnaa_cv"), m.get("atp_frac"), m.get("atp_ok"), m.get("n_gens")])
    with open(gen_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "K_nM", "generation", "fire_events", "max_oric", "mean_total_dnaa",
                    "mean_atp_fraction"])
        for (n, K) in sorted(data):
            for g in data[(n, K)]["metric"].get("per_gen", []):
                w.writerow([n, K, g["gen"], g["events"], g["max_oric"],
                            round(g["mean_dnaa"], 1), round(g["mean_atpfr"], 3)])
    print(f"  wrote {cond_csv} + {gen_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--tag-glob", default="nk_n*_K*")
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--report-figdir", default=None,
                    help="where interactive HTML embeds go (default reports/figures/<slug>)")
    args = ap.parse_args()
    slug = os.path.basename(args.study_dir.rstrip("/"))
    charts = os.path.join(args.study_dir, "charts")
    # interactive HTML embeds live under reports/figures/<slug>/ so the report
    # auto-discovers them (pbg interactive-viz embed convention).
    figdir = args.report_figdir or os.path.join("reports", "figures", slug)
    ana = os.path.join(args.study_dir, "analyses")
    for d in (charts, figdir, ana):
        os.makedirs(d, exist_ok=True)
    print("collecting runs...")
    data = collect(args.out_root, args.tag_glob)
    if not data:
        raise SystemExit("no runs loaded")
    print(f"loaded {len(data)} conditions")
    fig_heatmap(data, os.path.join(figdir, "nk_metric_heatmap.html"),
                os.path.join(charts, "nk_metric_heatmap.png"))
    fig_heatmap_mpl(data, os.path.join(charts, "nk_metric_heatmap.svg"),
                    os.path.join(charts, "nk_metric_heatmap.png"))
    fig_trajectory_browser(data, os.path.join(figdir, "nk_trajectory_browser.html"))
    # curated static subset: reference + 4 extremes (present ones only)
    curated = [k for k in [(4, 30), (1, 30), (8, 30), (4, 15), (4, 40)] if k in data]
    if curated:
        fig_static_panels(data, curated,
                          os.path.join(charts, "nk_trajectories_curated.svg"),
                          os.path.join(charts, "nk_trajectories_curated.png"),
                          "n×K lineage trajectories (in-sim, mechanistic trigger, seed 1)")
    write_csvs(data, os.path.join(ana, "nk_sweep_metrics.csv"),
               os.path.join(ana, "nk_sweep_per_generation.csv"))


if __name__ == "__main__":
    main()
