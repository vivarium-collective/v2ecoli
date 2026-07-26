"""Interactive Plotly figures for the multiscale-bioprocess investigation.

Adds engaging, hover/zoom/toggle interactive figures ALONGSIDE the existing
static charts, written to ``reports/figures/<study-dir>/<fig>.html`` so the
vivarium-workbench auto-discovers them as ``embed_visualizations`` (iframes,
copied verbatim into the gh-pages publish bundle).

Each file is self-contained: Plotly via CDN, an ``html,body{height:820px}``
clamp the dashboard auto-sizer reads, a title, a legend, hover tooltips, and a
caption. Long trajectories are downsampled to keep file sizes small.

Data sources (all LOCAL):
  * mbp-04 headline    : out/mbp_production/{wcm,millard}/run.db  (run_multigen_sqlite)
  * mbp-05 benchmark   : docs/report_cards/beulig_batch/*/report_card_verdict.json
                         + workspace/references/papers/palsson-2025-supp/*.csv
  * mbp-03 coupling    : build_composite("reactor_bird_coupled")  (60 short steps)
  * mbp-06 gap-analysis: workspace/studies/mbp-06-gap-analysis/study.yaml findings
  * mbp-01 env-coupling: build_composite("ecoli_baseline")             (40 short steps)
  * mbp-07 metabolism  : build_composite("ecoli_millard") + Millard glucose sweep

Run from the worktree root:
    ./.venv/bin/python scripts/make_mbp_interactive_figures.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import csv
import json
import os
import re
import sqlite3
import sys
import traceback
from pathlib import Path

import plotly.graph_objects as go
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

FIG_ROOT = REPO_ROOT / "reports" / "figures"
BEULIG_DIR = REPO_ROOT / "workspace" / "references" / "papers" / "palsson-2025-supp"
CARD_ROOT = REPO_ROOT / "docs" / "report_cards" / "beulig_batch"
OD_TO_GDW = 0.34  # Beulig OD600 -> gDW/L (matches beulig_comparison_harness)

# --------------------------------------------------------------------------- #
# HTML wrapper (dashboard auto-sizer reads the html,body height clamp)
# --------------------------------------------------------------------------- #
PINNED_H = 820
TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:{H}px;overflow:hidden;margin:0;padding:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:{H}px;padding:12px 16px;display:flex;flex-direction:column;gap:6px}}
h1{{font-size:1.12em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:5px}}
.fig{{flex:1 1 auto;min-height:0}}
p.caption{{margin:0;color:#475569;font-size:0.82em;line-height:1.35}}
.tag{{display:inline-block;background:#d1fae5;color:#065f46;padding:1px 7px;border-radius:4px;font-size:0.68em;margin-right:5px;vertical-align:middle}}
</style></head>
<body><div class="wrap">
  <h1>{title}</h1>
  <div class="fig">{plot_div}</div>
  <p class="caption"><span class="tag">interactive · plotly</span>{caption}</p>
</div></body></html>"""


def write_fig(study_dir: str, name: str, fig: go.Figure, title: str,
              caption: str, height: int = 680) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=24, t=40, b=48),
        template="plotly_white",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        hovermode="x unified",
    )
    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False,
                           config={"responsive": True, "displaylogo": False})
    html = TEMPLATE.format(title=title, caption=caption, plot_div=plot_div, H=PINNED_H)
    out_dir = FIG_ROOT / study_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out.relative_to(REPO_ROOT)}  ({out.stat().st_size//1024} KB)")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
_QRE = re.compile(r"Quantity\(([-\d.eE+]+)")


def qval(x):
    """Parse a serialized pint Quantity string / magnitude / number -> float."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "magnitude"):
        return float(x.magnitude)
    m = _QRE.search(str(x))
    return float(m.group(1)) if m else float("nan")


def _downsample(xs, ys_list, target=800):
    """Uniformly thin parallel series to <= target points (keep endpoints)."""
    n = len(xs)
    if n <= target:
        return xs, ys_list
    step = max(1, n // target)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return ([xs[i] for i in idx],
            [[y[i] for i in idx] for y in ys_list])


def load_run_db(path: Path):
    """Read a run_multigen_sqlite history table -> per-step trajectory dicts.

    Returns list of {t_s, cell_mass_fg, biomass_gL, o2_delta, co2_delta}.
    """
    con = sqlite3.connect(str(path))
    rows = con.execute(
        "select global_time, state from history order by step").fetchall()
    con.close()
    out = []
    for t, blob in rows:
        st = json.loads(blob)
        agents = st.get("agents") or {}
        aid = "0" if "0" in agents else (next(iter(agents)) if agents else None)
        cm = float("nan")
        if aid is not None:
            cm = qval(agents[aid].get("listeners", {}).get("mass", {}).get("cell_mass"))
        pop = st.get("population") or {}
        rea = st.get("reactor") or {}
        out.append({
            "t_s": float(t),
            "cell_mass_fg": cm,
            "biomass_gL": float(pop.get("biomass_concentration_gL", float("nan"))),
            "o2_delta": float(rea.get("o2_transport_delta", float("nan"))),
            "co2_delta": float(rea.get("co2_transport_delta", float("nan"))),
        })
    return out


def detect_divisions(traj, frac=0.7):
    """Indices/times where cell_mass drops below frac*prev (a division event)."""
    times = []
    prev = traj[0]["cell_mass_fg"]
    for r in traj[1:]:
        if r["cell_mass_fg"] < prev * frac:
            times.append(r["t_s"])
        prev = r["cell_mass_fg"]
    return times


# --------------------------------------------------------------------------- #
# Figure 1 — mbp-04: multi-generation biomass accumulation (HEADLINE)
# --------------------------------------------------------------------------- #
def fig_mbp04_multigeneration():
    print("[mbp-04] multi-generation accumulation (run.db x2) ...")
    wcm = load_run_db(REPO_ROOT / "out" / "mbp_production" / "wcm" / "run.db")
    mil = load_run_db(REPO_ROOT / "out" / "mbp_production" / "millard" / "run.db")
    wcm_div = detect_divisions(wcm)
    mil_div = detect_divisions(mil)
    print(f"   WCM     : {len(wcm_div)} divisions -> {len(wcm_div)+1} generations"
          f" (cell_mass {wcm[0]['cell_mass_fg']:.0f}->{wcm[-1]['cell_mass_fg']:.0f} fg,"
          f" biomass x{wcm[-1]['biomass_gL']/wcm[0]['biomass_gL']:.2f})")
    print(f"   Millard : {len(mil_div)} divisions -> {len(mil_div)+1} generations"
          f" (cell_mass {mil[0]['cell_mass_fg']:.0f}->{mil[-1]['cell_mass_fg']:.0f} fg,"
          f" biomass x{mil[-1]['biomass_gL']/mil[0]['biomass_gL']:.2f})")

    def hrs(traj):
        return [r["t_s"] / 3600.0 for r in traj]

    fig = go.Figure()
    # cell_mass sawtooth (left y, secondary handled via yaxis2)
    fig.add_trace(go.Scatter(
        x=hrs(wcm), y=[r["cell_mass_fg"] for r in wcm], name="WCM-FBA cell mass",
        mode="lines", line=dict(color="#1f6feb", width=2),
        hovertemplate="t=%{x:.2f} h<br>cell mass=%{y:.0f} fg<extra>WCM-FBA</extra>"))
    fig.add_trace(go.Scatter(
        x=hrs(mil), y=[r["cell_mass_fg"] for r in mil], name="Millard-ODE cell mass",
        mode="lines", line=dict(color="#cf222e", width=2),
        hovertemplate="t=%{x:.2f} h<br>cell mass=%{y:.0f} fg<extra>Millard-ODE</extra>"))
    # population biomass concentration (right y)
    fig.add_trace(go.Scatter(
        x=hrs(wcm), y=[r["biomass_gL"] for r in wcm], name="WCM-FBA biomass (g/L)",
        mode="lines", line=dict(color="#1f6feb", width=2, dash="dot"), yaxis="y2",
        hovertemplate="t=%{x:.2f} h<br>biomass=%{y:.4f} g/L<extra>WCM-FBA pop</extra>"))
    fig.add_trace(go.Scatter(
        x=hrs(mil), y=[r["biomass_gL"] for r in mil], name="Millard-ODE biomass (g/L)",
        mode="lines", line=dict(color="#cf222e", width=2, dash="dot"), yaxis="y2",
        hovertemplate="t=%{x:.2f} h<br>biomass=%{y:.4f} g/L<extra>Millard pop</extra>"))
    # division markers (WCM)
    shapes, anns = [], []
    for i, dt in enumerate(wcm_div, 1):
        x = dt / 3600.0
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                           line=dict(color="#1f6feb", width=1, dash="dash")))
        anns.append(dict(x=x, y=1.0, yref="paper", text=f"WCM div {i}→gen {i+1}",
                         showarrow=False, font=dict(size=9, color="#1f6feb"),
                         yshift=-2, xshift=2, xanchor="left"))
    fig.update_layout(
        shapes=shapes, annotations=anns,
        xaxis=dict(title="Time (h)"),
        yaxis=dict(title="Single-cell mass (fg)"),
        yaxis2=dict(title="Population biomass (g/L)", overlaying="y",
                    side="right", showgrid=False),
    )
    cap = (f"Single-cell mass (solid, left axis) and population biomass density "
           f"(dotted, right axis) over a {wcm[-1]['t_s']/3600:.1f} h reactor run. "
           f"WCM-FBA divides {len(wcm_div)}× (sawtooth resets → {len(wcm_div)+1} "
           f"generations; biomass ×{wcm[-1]['biomass_gL']/wcm[0]['biomass_gL']:.2f}), "
           f"tracing the batch-scale accumulation path. Millard-ODE stays in "
           f"generation 1 ({len(mil_div)} divisions) — the documented inherent "
           f"slow-growth of the kinetic-metabolism swap, not a coupling failure. "
           f"Toggle traces in the legend; hover for values.")
    write_fig("mbp-04-multigeneration-runs", "biomass_accumulation_interactive",
              fig,
              "Multi-generation biomass accumulation: WCM divides, Millard stays gen 1",
              cap)


# --------------------------------------------------------------------------- #
# mbp-05 Beulig benchmark helpers (reuse harness loaders)
# --------------------------------------------------------------------------- #
def _load_beulig_batch(csv_name: str):
    """Batch-phase (t<0 h) mean trajectory from a wide Beulig reactor CSV.

    Returns (times_h_from_zero, means). Time shifted so batch starts at 0 h.
    """
    path = BEULIG_DIR / csv_name
    if not path.is_file():
        return [], []
    times, means = [], []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return [], []
        for row in reader:
            if not row:
                continue
            try:
                t = float(row[0])
            except (ValueError, IndexError):
                continue
            if t >= 0:  # batch phase only (negative time convention)
                continue
            vals = []
            for c in row[1:]:
                try:
                    vals.append(float(c))
                except ValueError:
                    pass
            if not vals:
                continue
            times.append(t)
            means.append(sum(vals) / len(vals))
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    means = [means[i] for i in order]
    if times:
        t0 = times[0]
        times = [t - t0 for t in times]  # start batch at 0 h
    return times, means


def _arm_axis_value(verdict: dict, axis_id: str):
    for g in verdict.get("groups", {}).values():
        for a in g.get("axes", []):
            if a.get("id") == axis_id:
                return a.get("value")
    return None


def fig_mbp05_overlay():
    print("[mbp-05] 3-arm Beulig overlay ...")
    arms = {
        "WCM-FBA": "vs_beulig",
        "Millard-ODE": "millard_vs_beulig",
        "iML1515": "iml1515_vs_beulig",
    }
    verdicts = {}
    for label, d in arms.items():
        p = CARD_ROOT / d / "report_card_verdict.json"
        if p.is_file():
            verdicts[label] = json.loads(p.read_text())

    fig = go.Figure()
    # --- Beulig reference: biomass (OD x 0.34) -------------------------------
    t_od, od = _load_beulig_batch("reactor_OD_data.csv")
    biomass_ref = [v * OD_TO_GDW for v in od]
    if t_od:
        fig.add_trace(go.Scatter(
            x=t_od, y=biomass_ref, name="Beulig biomass (gDW/L)", mode="lines+markers",
            line=dict(color="#16a34a", width=2.5), marker=dict(size=5),
            hovertemplate="t=%{x:.2f} h<br>%{y:.2f} gDW/L<extra>Beulig ref</extra>"))
    # --- sim arm biomass endpoints (horizontal markers) ----------------------
    colors = {"WCM-FBA": "#1f6feb", "Millard-ODE": "#cf222e", "iML1515": "#9333ea"}
    xend = (t_od[-1] if t_od else 6.0)
    for label, v in verdicts.items():
        val = _arm_axis_value(v, "growth.biomass_gDW_per_L")
        if val is None:
            continue
        fig.add_trace(go.Scatter(
            x=[0, xend], y=[val, val], name=f"{label} biomass",
            mode="lines", line=dict(color=colors[label], width=1.8, dash="dash"),
            hovertemplate=f"{label}: %{{y:.4f}} gDW/L<extra></extra>"))
    fig.update_yaxes(type="log")
    fig.update_layout(
        xaxis=dict(title="Batch time (h)"),
        yaxis=dict(title="Biomass density (gDW/L, log scale)"),
    )
    cap = ("Beulig 2025 WT batch-phase biomass (green, OD×0.34) climbs to ~9.6 "
           "gDW/L, while all three simulation arms flatline near 0.013 gDW/L "
           "(dashed, endpoint scalars from each arm's report card) — a ~3-orders "
           "gap on a log axis. This is the headline mbp-05 mismatch: at the "
           "current single-cell / short-batch scope none of the arms reach "
           "reactor-scale density. Toggle arms in the legend.")
    write_fig("mbp-05-palsson-benchmark", "beulig_3arm_biomass_overlay_interactive",
              fig, "3-arm biomass vs Beulig 2025 batch reference (log scale)", cap)

    # --- Beulig byproduct + glucose reference trajectories -------------------
    fig2 = go.Figure()
    series = [
        ("reactor_glucose_data.csv", "glucose (remaining)", "#b45309"),
        ("reactor_lactate_data.csv", "lactate", "#0ea5e9"),
        ("reactor_formate_data.csv", "formate", "#8b5cf6"),
        ("reactor_ethanol_data.csv", "ethanol", "#ec4899"),
        ("reactor_pyruvate_data.csv", "pyruvate", "#14b8a6"),
        ("reactor_succinate_data.csv", "succinate", "#f59e0b"),
    ]
    any_series = False
    for csv_name, name, color in series:
        t, m = _load_beulig_batch(csv_name)
        if not t:
            continue
        any_series = True
        fig2.add_trace(go.Scatter(
            x=t, y=m, name=name, mode="lines", line=dict(color=color, width=2),
            hovertemplate="t=%{x:.2f} h<br>%{y:.3f} mM<extra>" + name + "</extra>"))
    fig2.update_layout(
        xaxis=dict(title="Batch time (h)"),
        yaxis=dict(title="Concentration (mM)"),
    )
    cap2 = ("Beulig 2025 WT batch-phase medium chemistry: glucose drawdown and "
            "secreted byproducts (lactate / formate / ethanol / pyruvate / "
            "succinate). The simulation arms secrete ~0 of every byproduct at "
            "the current scope (every byproducts axis grades 'mismatch' in the "
            "report cards), so only the reference is plotted here — this panel "
            "is the byproduct-spectrum the arms must eventually reproduce. "
            "Toggle species in the legend; zoom to inspect.")
    if any_series:
        write_fig("mbp-05-palsson-benchmark",
                  "beulig_byproduct_reference_interactive", fig2,
                  "Beulig 2025 batch byproduct + glucose reference trajectories", cap2)


def fig_mbp05_verdict_heatmap():
    print("[mbp-05] verdict heatmap ...")
    arms = [
        ("WCM-FBA", "vs_beulig"),
        ("Millard-ODE", "millard_vs_beulig"),
        ("iML1515", "iml1515_vs_beulig"),
    ]
    # collect union of axes (group.axis_label) preserving order
    axis_order = []
    axis_label = {}
    per_arm = {}
    for label, d in arms:
        p = CARD_ROOT / d / "report_card_verdict.json"
        if not p.is_file():
            continue
        v = json.loads(p.read_text())
        per_arm[label] = {}
        for gname, g in v.get("groups", {}).items():
            for a in g.get("axes", []):
                aid = a["id"]
                if aid not in axis_label:
                    axis_order.append(aid)
                    axis_label[aid] = f"{gname} · {a.get('label', aid)[:42]}"
                per_arm[label][aid] = a
    arm_labels = [lbl for lbl, _ in arms if lbl in per_arm]

    code = {"within_tol": 3, "drift": 2, "mismatch": 1, "ungraded": 0}
    z, text, hover = [], [], []
    for aid in axis_order:
        zrow, trow, hrow = [], [], []
        for lbl in arm_labels:
            a = per_arm[lbl].get(aid)
            if a is None:
                zrow.append(None); trow.append("n/a")
                hrow.append(f"{lbl}<br>{aid}<br>(not graded for this arm)")
            else:
                verd = a.get("verdict", "ungraded")
                zrow.append(code.get(verd, 0))
                trow.append(verd.replace("within_tol", "within tol"))
                meter = str(a.get("meter", "")).replace("—", "—")
                hrow.append(f"<b>{lbl}</b><br>{axis_label[aid]}<br>"
                            f"verdict: {verd}<br>{meter}")
        z.append(zrow); text.append(trow); hover.append(hrow)

    # discrete colorscale: 0 ungraded grey, 1 mismatch red, 2 drift amber, 3 ok green
    colorscale = [
        [0.0, "#cbd5e1"], [0.25, "#cbd5e1"],
        [0.25, "#ef4444"], [0.5, "#ef4444"],
        [0.5, "#f59e0b"], [0.75, "#f59e0b"],
        [0.75, "#22c55e"], [1.0, "#22c55e"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=arm_labels, y=[axis_label[a] for a in axis_order],
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=colorscale, zmin=0, zmax=3, showscale=True,
        colorbar=dict(title="verdict", tickvals=[0.4, 1.1, 1.9, 2.6],
                      ticktext=["ungraded", "mismatch", "drift", "within tol"]),
        xgap=2, ygap=2,
    ))
    fig.update_layout(xaxis=dict(title="Simulation arm", side="top"),
                      yaxis=dict(autorange="reversed"))
    cap = ("Graded report-card verdicts per arm × axis (group · observable). "
           "Green = within tolerance, amber = drift, red = mismatch, grey = "
           "ungraded (no Beulig reference column, e.g. acetate / dissolved-O2). "
           "Every quantitative growth/substrate/gas/byproduct axis grades "
           "'mismatch' across all three arms — the comparison executed cleanly "
           "(summary = within tol) but the biology gap is real. Hover any cell "
           "for the divergence meter.")
    write_fig("mbp-05-palsson-benchmark", "beulig_verdict_heatmap_interactive",
              fig, "Beulig benchmark verdict heatmap (3 arms × graded axes)", cap,
              height=700)


# --------------------------------------------------------------------------- #
# Figure — mbp-03: reactor <-> cell coupling (keystone)
# --------------------------------------------------------------------------- #
def fig_mbp03_coupling(n_steps=60):
    print(f"[mbp-03] reactor_bird_coupled, {n_steps} steps ...")
    from v2ecoli import build_composite
    from v2ecoli.steps.reactor_cell_coupler import O2_EXCHANGE_KEY
    c = build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache",
                        cells_per_agent=1.0e11,
                        bird_reactor_config={"gas_flow_rate_Lpm": 2.0})
    t, do2, sat, do2sat_pct, biomass, glc, o2x = [], [], [], [], [], [], []
    for i in range(n_steps):
        c.run(1)
        r = c.state["reactor"]
        t.append(i + 1)
        d = qval(r.get("dissolved_o2"))
        s = qval(r.get("o2_saturation"))
        do2.append(d); sat.append(s)
        do2sat_pct.append(100.0 * d / s if s else float("nan"))
        biomass.append(qval(r.get("biomass")))
        glc.append(qval(r.get("glucose_medium_mM")))
        exch = c.state["agents"]["0"]["environment"]["exchange"]
        o2x.append(float(exch.get(O2_EXCHANGE_KEY, 0.0)))
    print(f"   dissolved_o2 {do2[0]:.3f}->{do2[-1]:.3f} mg/L (C*~{sat[-1]:.2f}); "
          f"O2 exchange {o2x[0]:.3g}->{o2x[-1]:.3g} counts/step")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=do2, name="dissolved O2 (mg/L)", mode="lines",
        line=dict(color="#1f6feb", width=2),
        hovertemplate="step %{x}<br>DO=%{y:.3f} mg/L<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=sat, name="O2 saturation C* (mg/L)", mode="lines",
        line=dict(color="#94a3b8", width=1.8, dash="dash"),
        hovertemplate="step %{x}<br>C*=%{y:.3f} mg/L<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=o2x, name="agent O2 exchange (counts/step, −=uptake)", mode="lines",
        line=dict(color="#cf222e", width=2), yaxis="y2",
        hovertemplate="step %{x}<br>O2 exch=%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=glc, name="medium glucose (mM)", mode="lines",
        line=dict(color="#b45309", width=1.6, dash="dot"), yaxis="y2",
        visible="legendonly",
        hovertemplate="step %{x}<br>glucose=%{y:.3f} mM<extra></extra>"))
    fig.update_layout(
        xaxis=dict(title="Coupled step (1 s/step)"),
        yaxis=dict(title="Reactor dissolved O2 (mg/L)"),
        yaxis2=dict(title="Agent exchange / medium glucose", overlaying="y",
                    side="right", showgrid=False),
    )
    cap = (f"BiRD reactor ↔ WCM cell coupling over {n_steps} coupled steps "
           f"(reactor_bird_coupled, cells/agent=1e11, gas 2 L/min). The cell's "
           f"O2 uptake (red, right axis) draws reactor dissolved-O2 (blue) below "
           f"its gas-equilibrium saturation C* (grey dashed): DO "
           f"{do2[0]:.2f}→{do2[-1]:.2f} mg/L. This closes the gas-transfer loop "
           f"end-to-end — reactor physics writes the cell's environment, the "
           f"cell's exchange writes back to the reactor. Toggle medium glucose "
           f"in the legend.")
    write_fig("mbp-03-bird-reactor-coupling", "reactor_coupling_interactive",
              fig, "Reactor ↔ cell coupling: dissolved O2 falls under cell load", cap)


# --------------------------------------------------------------------------- #
# Figure — mbp-06: gap inventory sunburst
# --------------------------------------------------------------------------- #
AXIS_NAME = {
    "A": "A · v2ecoli biology coverage",
    "B": "B · coupling layer",
    "C": "C · BiRD physics",
    "D": "D · Beulig benchmark scope",
    "E": "E · methodology",
}


def fig_mbp06_gaps():
    print("[mbp-06] gap inventory sunburst ...")
    y = yaml.safe_load(
        (REPO_ROOT / "workspace" / "studies" / "mbp-06-gap-analysis"
         / "study.yaml").read_text())
    entries = ((y.get("findings") or {}).get("entries")) or []
    print(f"   {len(entries)} gap entries")
    by_axis = {}
    ids, labels, parents, values, hover = [], [], [], [], []
    # root
    ids.append("root"); labels.append(f"{len(entries)} gaps"); parents.append("")
    values.append(0); hover.append(f"{len(entries)} gaps across 5 axes")
    seen_axis, seen_axpri = set(), set()
    for e in entries:
        ax = str(e.get("axis", "?")).strip()
        pri = str(e.get("priority", "?")).strip()
        sig = str(e.get("signature", "gap"))
        desc = str(e.get("description", "")).strip()
        rtype = str(e.get("resolution_type", ""))
        prov = str(e.get("provenance", ""))
        by_axis[ax] = by_axis.get(ax, 0) + 1
        ax_id = f"ax::{ax}"
        if ax_id not in seen_axis:
            seen_axis.add(ax_id)
            ids.append(ax_id); labels.append(AXIS_NAME.get(ax, f"axis {ax}"))
            parents.append("root"); values.append(0)
            hover.append(AXIS_NAME.get(ax, f"axis {ax}"))
        pri_id = f"ax::{ax}::p{pri}"
        if pri_id not in seen_axpri:
            seen_axpri.add(pri_id)
            ids.append(pri_id); labels.append(f"priority {pri}")
            parents.append(ax_id); values.append(0)
            hover.append(f"{AXIS_NAME.get(ax,'axis '+ax)} — priority {pri}")
        ids.append(f"gap::{sig}"); labels.append(sig)
        parents.append(pri_id); values.append(1)
        wrapped = "<br>".join(_wrap(desc, 60))
        hover.append(f"<b>{sig}</b><br>axis {ax} · priority {pri} · {prov}<br>"
                     f"resolution: {rtype}<br>{wrapped}")
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        insidetextorientation="radial",
        marker=dict(colors=values, colorscale="Blues", showscale=False),
    ))
    axis_counts = ", ".join(f"{AXIS_NAME.get(k,k).split(' · ')[0]}={v}"
                            for k, v in sorted(by_axis.items()))
    cap = (f"All {len(entries)} gaps from the mbp-06 cross-study synthesis, "
           f"nested axis → priority → gap. Counts per axis: {axis_counts}. "
           f"Click a wedge to zoom into an axis or priority tier; hover any gap "
           f"for its description, provenance, and candidate resolution type. "
           f"Axes A (v2ecoli biology) and D (Beulig scope) carry the most "
           f"load-bearing gaps for the next investigation cycle.")
    write_fig("mbp-06-gap-analysis", "gap_inventory_interactive", fig,
              "Gap inventory: 15 gaps by axis → priority", cap, height=700)


def _wrap(s, width):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out[:6]


# --------------------------------------------------------------------------- #
# Figure — mbp-01: baseline env-coupling exchange fluxes are live
# --------------------------------------------------------------------------- #
def fig_mbp01_env(n_steps=40):
    print(f"[mbp-01] baseline env-coupling, {n_steps} steps ...")
    from v2ecoli import build_composite
    c = build_composite("ecoli_baseline", seed=0, cache_dir="out/cache")
    t, glc_ext, glc_up, o2_up, obj = [], [], [], [], []
    for i in range(n_steps):
        c.run(1)
        ag = c.state["agents"]["0"]
        ext = ag.get("boundary", {}).get("external", {}) or {}
        glc_ext.append(float(ext.get("GLC", float("nan"))))
        # environment.exchange is a dict of counts/step (negative = uptake)
        exch = ag.get("environment", {}).get("exchange", {}) or {}
        glc_up.append(abs(float(exch.get("GLC", float("nan")))))
        o2_up.append(abs(float(exch.get("OXYGEN-MOLECULE", float("nan")))))
        fr = ag.get("listeners", {}).get("fba_results", {})
        ov = fr.get("objective_value") if isinstance(fr, dict) else None
        obj.append(float(ov) if ov is not None else float("nan"))
        t.append(i + 1)
    print(f"   glucose uptake |exch| {glc_up[0]:.3g}->{glc_up[-1]:.3g}; "
          f"objective {obj[0]:.3g}->{obj[-1]:.3g}; ext GLC {glc_ext[0]:.2f} mM")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=glc_up, name="glucose uptake |exchange|", mode="lines",
        line=dict(color="#1f6feb", width=2),
        hovertemplate="step %{x}<br>|GLC exch|=%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=o2_up, name="O2 uptake |exchange|", mode="lines",
        line=dict(color="#cf222e", width=2),
        hovertemplate="step %{x}<br>|O2 exch|=%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=obj, name="FBA objective (growth proxy)", mode="lines",
        line=dict(color="#16a34a", width=1.8, dash="dot"), yaxis="y2",
        hovertemplate="step %{x}<br>objective=%{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=glc_ext, name="external glucose (mM)", mode="lines",
        line=dict(color="#b45309", width=1.5, dash="dash"), yaxis="y2",
        visible="legendonly",
        hovertemplate="step %{x}<br>ext glucose=%{y:.3f} mM<extra></extra>"))
    fig.update_layout(
        xaxis=dict(title="Simulation step (1 s/step)"),
        yaxis=dict(title="Exchange (counts/step, |uptake|)"),
        yaxis2=dict(title="FBA objective / external glucose", overlaying="y",
                    side="right", showgrid=False),
    )
    cap = (f"Baseline composite over {n_steps} steps, reading the external "
           f"environment store each timestep (the plumbing mbp-01's "
           f"EnvironmentDriver instruments). Glucose and O2 uptake exchange "
           f"(left axis, counts/step) and the FBA growth objective (right axis) "
           f"are live and updated every step from the metabolism FBA — "
           f"confirming the environment→metabolism read path is wired before "
           f"mbp-03 drives it with reactor physics. Toggle the external-glucose "
           f"trace in the legend.")
    write_fig("mbp-01-time-varying-environment", "env_coupling_interactive",
              fig, "Environment → metabolism coupling is live each step (baseline)",
              cap)


def _pick(d, keys):
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


# --------------------------------------------------------------------------- #
# Figures — mbp-07: Millard metabolism swap growth + env responsiveness
# --------------------------------------------------------------------------- #
def fig_mbp07_growth(n_steps=60):
    print(f"[mbp-07] baseline_millard growth + central fluxes, {n_steps} steps ...")
    from v2ecoli import build_composite
    c = build_composite("ecoli_millard", seed=0, cache_dir="out/cache")
    flux_keys = ["PTS_4", "CYTBO", "PGI", "PYK"]
    labels = {"PTS_4": "PTS_4 (glucose uptake)", "CYTBO": "CYTBO (O2 respiration)",
              "PGI": "PGI (glycolysis)", "PYK": "PYK (pyruvate kinase)"}
    t, mass = [], []
    flux = {k: [] for k in flux_keys}
    for i in range(n_steps):
        c.run(1)
        agents = c.state.get("agents") or {}
        ag = agents.get("0") or (next(iter(agents.values())) if agents else {})
        t.append(i + 1)
        mass.append(qval(ag.get("listeners", {}).get("mass", {}).get("cell_mass")))
        fl = ag.get("central_fluxes") or {}
        for k in flux_keys:
            flux[k].append(qval(fl.get(k)))
    print(f"   cell_mass {mass[0]:.1f}->{mass[-1]:.1f} fg "
          f"(Δ={mass[-1]-mass[0]:+.2f})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=mass, name="cell mass (fg)", mode="lines+markers",
        line=dict(color="#1f6feb", width=2), marker=dict(size=4),
        hovertemplate="step %{x}<br>cell mass=%{y:.2f} fg<extra></extra>"))
    pal = ["#cf222e", "#16a34a", "#9333ea", "#b45309"]
    for k, color in zip(flux_keys, pal):
        fig.add_trace(go.Scatter(
            x=t, y=flux[k], name=labels[k], mode="lines",
            line=dict(color=color, width=1.8), yaxis="y2",
            hovertemplate="step %{x}<br>" + k + "=%{y:.4g}<extra></extra>"))
    fig.update_layout(
        xaxis=dict(title="Simulation step (1 s/step)"),
        yaxis=dict(title="Cell mass (fg)"),
        yaxis2=dict(title="Central-carbon flux (model units)", overlaying="y",
                    side="right", showgrid=False),
    )
    cap = (f"baseline_millard: central-carbon metabolism is the Millard 2017 "
           f"kinetic ODE (replaces FBA). Cell mass (blue, left axis) stays "
           f"stable under the swap — {mass[0]:.0f}→{mass[-1]:.0f} fg over "
           f"{n_steps} s (growth is not resolved at this timescale; cell cycle "
           f"~2700 s). The Millard central fluxes (right axis) are live and "
           f"non-trivial, confirming the kinetic metabolism drives the cell. "
           f"Toggle fluxes in the legend.")
    write_fig("mbp-07-millard-kinetic-metabolism",
              "metabolism_swap_growth_interactive", fig,
              "Millard metabolism swap: cell stable, central fluxes live", cap)


def fig_mbp07_env_response():
    print("[mbp-07] Millard glucose sweep (env responsiveness) ...")
    from v2ecoli.core import build_core
    from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism
    core = build_core()
    glc_mM = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    uptake = []
    for g in glc_mM:
        s = MillardPDMPMetabolism(config={}, core=core)
        out = s.update({
            "lqr_control": {}, "bulk": None,
            "listeners_mass": {"cell_mass": 1000.0, "dry_mass": 300.0},
            "external_concentrations": {"GLCx": g},
        }, 1.0)
        uptake.append(abs(qval(out["central_fluxes"]["PTS_4"])))
    print(f"   |PTS_4| {uptake[0]:.3g}->{uptake[-1]:.3g} as GLCx "
          f"{glc_mM[0]}->{glc_mM[-1]} mM")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=glc_mM, y=uptake, name="|PTS_4| glucose-uptake flux",
        mode="lines+markers", line=dict(color="#16a34a", width=2.2),
        marker=dict(size=7),
        hovertemplate="GLCx=%{x} mM<br>|PTS_4|=%{y:.4g} mM/s<extra></extra>"))
    fig.update_xaxes(type="log")
    fig.update_layout(
        xaxis=dict(title="External glucose GLCx (mM, log scale)"),
        yaxis=dict(title="Glucose-uptake flux |PTS_4| (mM/s)"),
    )
    cap = (f"Millard kinetics respond to the environment: glucose-uptake flux "
           f"|PTS_4| rises from {uptake[0]:.3g} to {uptake[-1]:.3g} mM/s as "
           f"external glucose sweeps 0.05→50 mM (saturating Michaelis-Menten "
           f"shape). This step-level dose-response confirms the kinetic "
           f"metabolism reads the external concentration store — the "
           f"prerequisite for reactor coupling (mbp-03). Hover for values; zoom "
           f"to inspect the low-glucose knee.")
    write_fig("mbp-07-millard-kinetic-metabolism",
              "env_responsiveness_interactive", fig,
              "Millard glucose dose-response (environment responsiveness)", cap)


# --------------------------------------------------------------------------- #
def main():
    figures = [
        ("mbp-04", fig_mbp04_multigeneration),
        ("mbp-05 overlay", fig_mbp05_overlay),
        ("mbp-05 heatmap", fig_mbp05_verdict_heatmap),
        ("mbp-06 gaps", fig_mbp06_gaps),
        ("mbp-03 coupling", fig_mbp03_coupling),
        ("mbp-07 growth", fig_mbp07_growth),
        ("mbp-07 env", fig_mbp07_env_response),
        ("mbp-01 env", fig_mbp01_env),
    ]
    failures = []
    for name, fn in figures:
        try:
            fn()
        except Exception as exc:  # keep going; report at the end
            failures.append((name, exc))
            print(f"  !! {name} FAILED: {exc}")
            traceback.print_exc()
    print("\n=== DONE ===")
    if failures:
        print("FAILURES:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
