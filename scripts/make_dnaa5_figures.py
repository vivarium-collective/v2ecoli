"""Interactive Plotly figures demonstrating the IMPACT of Study 5 — Cooperative oriC Switch.

Writes self-contained HTML embeds into reports/figures/dnaa-5-cooperativity/ (auto-discovered
by the dashboard study card + report). Run series come from scripts/extract_dnaa5.py
(/tmp/dnaa5_data.json); the switch-curve figure is analytic (the Hill model itself).

Usage: .venv/bin/python scripts/make_dnaa5_figures.py [--data /tmp/dnaa5_data.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = "reports/figures/dnaa-5-cooperativity"
K_NM = 30.0  # half-saturation free [DnaA-ATP]
# palette
GREY, GREEN, RED, BLUE, AMBER = "#94a3b8", "#16a34a", "#dc2626", "#2563eb", "#d97706"
BAND_GREEN = "rgba(34,197,94,0.10)"
BAND_BLUE = "rgba(37,99,235,0.08)"


def _wrap(s, width=78):
    """Wrap a subtitle into <br>-separated lines so it doesn't overflow/clip the iframe."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def _layout(fig, title, subtitle, h=480):
    sub = _wrap(subtitle)
    nlines = sub.count("<br>") + 1
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#475569'>{sub}</span>",
                   x=0.5, xanchor="center", font=dict(size=17)),
        template="plotly_white", height=h + 18 * max(0, nlines - 1),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(t=64 + 16 * nlines, b=55, l=70, r=40), hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="#e2e8f0", borderwidth=1),
    )
    return fig


def _write(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".html")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True,
                   div_id=name, config={"displayModeBar": False, "responsive": True})
    print(f"  wrote {path}")


def fig_switch_curves():
    """Analytic: the cooperative Hill switch sharpening from Langmuir to all-or-none."""
    A = np.linspace(0, 4 * K_NM, 400)
    fig = go.Figure()
    ns = [(1, GREY, "n=1 — Langmuir (gradual)", "dash"),
          (2, "#60a5fa", "n=2", "solid"),
          (4, GREEN, "n=4 — operating point (sharp switch)", "solid"),
          (6, RED, "n=6 — too sharp (over-depletes pool)", "solid"),
          (8, "#7c3aed", "n=8", "dot")]
    for n, c, lab, dash in ns:
        theta = A ** n / (K_NM ** n + A ** n)
        fig.add_trace(go.Scatter(x=A, y=theta, name=lab, mode="lines",
                                 line=dict(color=c, width=4 if n == 4 else 2.2, dash=dash),
                                 hovertemplate="free DnaA-ATP %{x:.0f} nM<br>occupancy %{y:.2f}<extra></extra>"))
    fig.add_vline(x=K_NM, line=dict(color="#64748b", width=1, dash="dot"))
    fig.add_annotation(x=K_NM, y=1.04, text="K = 30 nM (half-saturation)", showarrow=False,
                       font=dict(size=11, color="#64748b"))
    fig.add_annotation(x=K_NM * 0.45, y=0.30, text="gradual fill", showarrow=False,
                       font=dict(size=12, color=GREY))
    fig.add_annotation(x=K_NM * 1.7, y=0.93, text="all-or-none switch", showarrow=False,
                       font=dict(size=12, color=GREEN))
    fig.update_xaxes(title="free DnaA-ATP concentration (nM)")
    fig.update_yaxes(title="oriC-low occupancy θ", range=[-0.03, 1.10])
    _layout(fig, "Cooperativity sharpens the oriC-low switch",
            "θ = Aⁿ / (Kⁿ + Aⁿ), K = 30 nM — Langmuir (n=1) fills gradually; the cooperative filament (n≥4) snaps all-or-none")
    _write(fig, "dnaa5_switch_curve_interactive")


def fig_response_sharpness():
    """Analytic: occupancy contrast below/at/above threshold — the switch sharpens with n.
    Matches the study's unit-verified values (n=6: 0.015/0.5/0.985; n=1: 0.33/0.5/0.67)."""
    ns = [1, 2, 4, 6, 8]
    def theta(n, frac):  # frac * K
        A = frac * K_NM
        return A ** n / (K_NM ** n + A ** n)
    below = [theta(n, 0.5) for n in ns]
    atK = [theta(n, 1.0) for n in ns]
    above = [theta(n, 2.0) for n in ns]
    dyn = [a - b for a, b in zip(above, below)]
    xlab = [("n=1 (Langmuir)" if n == 1 else ("n=4 (operating)" if n == 4 else f"n={n}")) for n in ns]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("occupancy below / at / above threshold",
                                        "on/off contrast  θ(2K) − θ(½K)"))
    fig.add_trace(go.Bar(name="½K (below)", x=xlab, y=below, marker_color=GREY,
                         hovertemplate="%{x}<br>θ(½K) %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(name="K (threshold)", x=xlab, y=atK, marker_color="#cbd5e1",
                         hovertemplate="%{x}<br>θ(K) %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(name="2K (above)", x=xlab, y=above, marker_color=GREEN,
                         hovertemplate="%{x}<br>θ(2K) %{y:.3f}<extra></extra>"), row=1, col=1)
    bar_cols = [GREY if n == 1 else (GREEN if n == 4 else "#60a5fa") for n in ns]
    fig.add_trace(go.Bar(x=xlab, y=dyn, marker_color=bar_cols, showlegend=False,
                         text=[f"{v:.2f}" for v in dyn], textposition="outside",
                         hovertemplate="%{x}<br>contrast %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title="oriC-low occupancy θ", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title="contrast (0=flat, 1=ideal switch)", range=[0, 1.05], row=1, col=2)
    fig.update_layout(barmode="group")
    _layout(fig, "Cooperativity widens the on/off contrast of the switch",
            "a tiny rise in free DnaA-ATP (½K→2K) flips occupancy from ~0 to ~1 at n≥4 — the Langmuir (n=1) only spans 0.33→0.67", h=470)
    _write(fig, "dnaa5_response_sharpness")


def fig_hill_vs_kd():
    """Analytic: the Hill-in-[DnaA-ATP] switch and the classical varying-K_d (Adair)
    switch are the SAME curve — n=4 (Hill) coincides with a per-neighbour cooperativity
    factor coop~1.9 (each bound DnaA lowers the next K_d ~1.9x). Shows n=4's conclusion
    is formulation-independent."""
    import sys, numpy as np
    sys.path.insert(0, ".")
    from v2ecoli.steps.dnaa_box_binding import _oric_low_occupancy_kd as kd
    A = np.linspace(0, 4 * K_NM, 400)
    r = A / K_NM
    hill4 = r ** 4 / (1 + r ** 4)
    lang = r / (1 + r)
    # best-fit cooperativity factor to the Hill n=4 curve
    coops = np.linspace(1.4, 2.4, 41)
    def kdcurve(c): return np.array([kd(a, K_NM, 8, c) for a in A])
    best = min(coops, key=lambda c: float(np.mean((kdcurve(c) - hill4) ** 2)))
    kdbest = kdcurve(best)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=A, y=lang, name="Langmuir (no cooperativity)", mode="lines",
                             line=dict(color=GREY, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=A, y=hill4, name="Hill in [DnaA-ATP] (n=4)", mode="lines",
                             line=dict(color=GREEN, width=4)))
    fig.add_trace(go.Scatter(x=A, y=kdbest, name=f"varying-K_d / Adair (8 sites, coop≈{best:.1f})",
                             mode="lines", line=dict(color="#ea580c", width=2.4, dash="dot")))
    fig.add_vline(x=K_NM, line=dict(color="#64748b", width=1, dash="dot"))
    fig.update_xaxes(title="free DnaA-ATP concentration (nM)")
    fig.update_yaxes(title="oriC-low occupancy θ", range=[-0.03, 1.08])
    _layout(fig, "Two formulations, one switch — Hill (n=4) ≈ classical varying-K_d",
            "the phenomenological Hill switch (n=4) and the mechanistic varying-dissociation-constant "
            f"(Adair, 8 sites, each neighbour lowering K_d ~{best:.1f}×) give the SAME occupancy curve — "
            "so n=4's sharp-switch conclusion is formulation-independent")
    _write(fig, "dnaa5_hill_vs_kd")


def fig_saturation_per_gen():
    """Test 'oriC saturates roughly once per generation' (Haochen): per-generation count of
    oriC-low saturation episodes (occ>=0.9) + max occupancy. Real values from the n=4 run."""
    gens = list(range(1, 10))
    sats = [11, 93, 95, 0, 0, 0, 0, 0, 0]
    maxocc = [1.00, 1.00, 1.00, 0.25, 0.25, 0.12, 0.25, 0.50, 0.12]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=gens, y=sats, name="saturation episodes (occ≥0.9)", marker_color="#ea580c",
                         text=sats, textposition="outside",
                         hovertemplate="gen %{x}<br>%{y} saturations<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=gens, y=maxocc, name="max oriC-low occupancy", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=7),
                             hovertemplate="gen %{x}<br>max occ %{y:.2f}<extra></extra>"), secondary_y=True)
    fig.add_hline(y=1, line=dict(color="#16a34a", width=1, dash="dot"),
                  annotation_text="once-per-generation target", annotation_position="top left")
    fig.update_xaxes(title="generation")
    fig.update_yaxes(title_text="saturation episodes per generation", secondary_y=False)
    fig.update_yaxes(title_text="max oriC-low occupancy", range=[0, 1.1], secondary_y=True)
    _layout(fig, "Test: does oriC saturate once per generation? — NOT under the mass trigger",
            "the cooperative switch over-fires in gens 1-3 (11-95 saturation episodes, far more than once) "
            "then STOPS saturating in gens 4-9 (max occupancy 0.12-0.50, never reaches full) as free DnaA-ATP "
            "drifts down. Clean once-per-cycle firing needs the mechanistic DnaA-ATP/oriC trigger phase-locked "
            "to initiation (out of scope for this report) plus a sustained DnaA-ATP level — a real open item.", h=480)
    _write(fig, "dnaa5_saturation_per_gen")


def fig_parent_daughter(domain_path="/tmp/dnaa5_domain_occ.json"):
    """Parent vs daughter oriC-low occupancy — the two oriC copies after replication.
    Answers Haochen's 'two trajectories' question: after the chromosome replicates the
    oriC-low sites double (8->16) and split into two domains; each copy fills its own
    low-affinity sites. The aggregate occupancy mixed them — here they're marked separately."""
    import json as _json
    if not os.path.exists(domain_path):
        print(f"  (skip parent/daughter: no {domain_path})"); return
    dd = _json.load(open(domain_path))
    t = dd["t_min"]; cA = dd["copyA"]; cB = dd["copyB"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=cA, name="daughter oriC copy 1", mode="lines",
                             line=dict(color=BLUE, width=1.4), connectgaps=False,
                             hovertemplate="%{x:.0f} min<br>occ %{y:.2f}<extra>daughter 1</extra>"))
    fig.add_trace(go.Scatter(x=t, y=cB, name="daughter oriC copy 2", mode="lines",
                             line=dict(color="#ea580c", width=1.4), connectgaps=False,
                             hovertemplate="%{x:.0f} min<br>occ %{y:.2f}<extra>daughter 2</extra>"))
    fig.update_xaxes(title="lineage time (min)")
    fig.update_yaxes(title="oriC-low occupancy (per chromosome domain)", range=[-0.03, 1.08])
    _layout(fig, "The two daughter oriC copies, tracked separately",
            "at replication initiation the origin fires and the chromosome forms TWO sister (daughter) "
            "copies, so the oriC-low sites double 8→16 into two domains; each daughter fills its own "
            "sites cooperatively. Before replication (oriC=1) only copy 1 is present; copy 2 (orange) "
            "appears when oriC=2. Both daughters are now tracked separately — the 'second trajectory' "
            "in the aggregate view is the daughter chromosome's oriC, not a mixed/stale artifact.")
    _write(fig, "dnaa5_parent_vs_daughter")


def fig_kd_sweep():
    """In-sim varying-K_d sweep: the Adair formulation over-depletes the DnaA-ATP fraction
    at every cooperativity tested (1.3-1.9), while Hill n=4 holds it in band. Real g3+ means
    from the 8-gen runs (dnaa5_kd_coop1p{3,5,7} + 1p9 crashed-at-gen-4 + Hill n=4 reference)."""
    coops = [1.3, 1.5, 1.7, 1.9]
    atpfr = [0.081, 0.086, 0.089, 0.071]   # kd-mode g3+ means (seed 1)
    crashed = [False, False, False, True]
    fig = go.Figure()
    fig.add_hrect(y0=0.2, y1=0.5, fillcolor=BAND_GREEN, line_width=0,
                  annotation_text="DnaA-ATP band [0.2,0.5]", annotation_position="top left")
    cols = [RED if c else "#ea580c" for c in crashed]
    fig.add_trace(go.Scatter(x=coops, y=atpfr, mode="lines+markers+text",
                             text=[f"{v:.2f}" + (" ✗crash" if c else "") for v, c in zip(atpfr, crashed)],
                             textposition="top center", line=dict(color="#ea580c", width=2),
                             marker=dict(size=11, color=cols), name="varying-K_d (Adair)"))
    fig.add_hline(y=0.276, line=dict(color=GREEN, width=2.5, dash="dash"),
                  annotation_text="Hill n=4 = 0.276 (in band)", annotation_position="bottom right")
    fig.update_xaxes(title="varying-K_d cooperativity factor (coop)")
    fig.update_yaxes(title="DnaA-ATP fraction (gen3+ mean)", range=[0, 0.6])
    _layout(fig, "Varying-K_d over-depletes DnaA-ATP at every cooperativity",
            "across the kd-mode sweep (coop 1.3-1.9) the DnaA-ATP fraction collapses to ~0.08 — well below "
            "the [0.2,0.5] band — and at coop=1.9 the cell crashes; the Hill n=4 form holds 0.276 in band. "
            "Static-curve equivalence does NOT survive the dynamic DnaA-ATP balance.", h=470)
    _write(fig, "dnaa5_kd_sweep")


def fig_langmuir_vs_switch(d):
    """Occupancy time-trace: Langmuir half-fills; the cooperative switch reaches FULL."""
    lang, coop = d.get("langmuir_n1"), d.get("coop_n4")
    if not (coop and coop.get("occ")):
        return
    fig = go.Figure()
    if lang and lang.get("occ"):
        fig.add_trace(go.Scatter(x=lang["t_min"], y=lang["occ"], name=f"Langmuir n=1 (max {lang['occ_max']:.2f})",
                                 mode="lines", line=dict(color=GREY, width=1.6),
                                 hovertemplate="%{x:.0f} min<br>occ %{y:.2f}<extra>Langmuir</extra>"))
    if coop:
        fig.add_trace(go.Scatter(x=coop["t_min"], y=coop["occ"], name=f"Cooperative n=4 (max {coop['occ_max']:.2f})",
                                 mode="lines", line=dict(color=GREEN, width=1.8),
                                 hovertemplate="%{x:.0f} min<br>occ %{y:.2f}<extra>Cooperative n=4</extra>"))
    fig.add_hline(y=0.5, line=dict(color=GREY, width=1, dash="dot"))
    fig.add_hline(y=1.0, line=dict(color=GREEN, width=1, dash="dot"))
    fig.update_xaxes(title="lineage time (min)")
    fig.update_yaxes(title="oriC-low occupancy", range=[-0.03, 1.10])
    _layout(fig, "Under the cell cycle: Langmuir stays half-full; the switch reaches FULL",
            "oriC-low occupancy over an 8-generation succinate lineage (dnaa-4 reference config, seed 1)")
    _write(fig, "dnaa5_langmuir_vs_switch")


def fig_sweet_spot(d):
    """Why n=4: full switch with homeostasis preserved; n=6 over-depletes the pool."""
    pts = [("n=1\nLangmuir", d.get("langmuir_n1"), GREY),
           ("n=4\noperating", d.get("coop_n4"), GREEN),
           ("n=6\ntoo strong", d.get("coop_n6"), RED)]
    pts = [(lab, v, c) for lab, v, c in pts if v and v.get("occ_max") is not None]
    if not pts:
        return
    labs = [p[0] for p in pts]
    occ = [p[1]["occ_max"] for p in pts]
    pool = [p[1]["pool_g3p_mean"] for p in pts]
    cols = [p[2] for p in pts]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "switch completeness", "DnaA homeostasis"))
    fig.add_trace(go.Bar(x=labs, y=occ, marker_color=cols, text=[f"{v:.2f}" for v in occ],
                         textposition="outside", showlegend=False,
                         hovertemplate="%{x}<br>max occupancy %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=labs, y=pool, marker_color=cols, text=[f"{v:.0f}" for v in pool],
                         textposition="outside", showlegend=False,
                         hovertemplate="%{x}<br>steady pool %{y:.0f}<extra></extra>"), row=1, col=2)
    # band shading for the pool subplot
    fig.add_hrect(y0=300, y1=800, fillcolor=BAND_GREEN, line_width=0, row=1, col=2)
    fig.add_hline(y=300, line=dict(color=GREEN, width=1, dash="dash"), row=1, col=2)
    fig.add_hline(y=800, line=dict(color=GREEN, width=1, dash="dash"), row=1, col=2)
    fig.update_yaxes(title="max oriC-low occupancy", range=[0, 1.18], row=1, col=1)
    fig.update_yaxes(title="steady DnaA pool (molecules)", range=[0, 900], row=1, col=2)
    _layout(fig, "Why n = 4: a full switch with homeostasis preserved",
            "n=4 reaches full occupancy AND keeps the DnaA pool inside the [300,800] band; n=6 over-depletes it (pool dips below 300)", h=470)
    _write(fig, "dnaa5_sweet_spot")


def fig_homeostasis(d):
    """Cooperativity preserves the dnaa-3/4 bands: pool [300,800] + DnaA-ATP fraction [0.2,0.5]."""
    coop = d.get("coop_n4")
    if not (coop and coop.get("pool")):
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("total DnaA pool", "DnaA-ATP fraction"))
    fig.add_hrect(y0=300, y1=800, fillcolor=BAND_GREEN, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=coop["t_min"], y=coop["pool"], name="DnaA pool", line=dict(color=BLUE, width=1.6),
                             hovertemplate="%{x:.0f} min<br>%{y:.0f}<extra>pool</extra>"), row=1, col=1)
    fig.add_hrect(y0=0.2, y1=0.5, fillcolor=BAND_GREEN, line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=coop["t_min"], y=coop["atpfr"], name="DnaA-ATP fraction", line=dict(color=AMBER, width=1.4),
                             hovertemplate="%{x:.0f} min<br>%{y:.2f}<extra>ATP frac</extra>"), row=2, col=1)
    fig.update_yaxes(title="molecules", range=[0, 900], row=1, col=1)
    fig.update_yaxes(title="fraction", range=[0, 1.0], row=2, col=1)
    fig.update_xaxes(title="lineage time (min)", row=2, col=1)
    _layout(fig, "Cooperativity preserves DnaA homeostasis (n=4)",
            f"the dnaa-3/4 bands hold: pool in [300,800] (g3+ mean {coop['pool_g3p_mean']:.0f}), DnaA-ATP fraction in [0.2,0.5] (g3+ mean {coop['atpfr_g3p_mean']:.2f})", h=560)
    _write(fig, "dnaa5_homeostasis_preserved")


def fig_multiseed(d):
    """Seed-robust: the switch + homeostasis hold across seeds 0/1/2."""
    seeds = [("seed 0", d.get("coop_n4_s0"), "#0ea5e9"),
             ("seed 1", d.get("coop_n4"), GREEN),
             ("seed 2", d.get("coop_n4_s2"), "#a855f7")]
    seeds = [(lab, v, c) for lab, v, c in seeds if v]
    if not seeds:
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("oriC-low occupancy", "DnaA pool (g3+ mean)"))
    for lab, v, c in seeds:
        fig.add_trace(go.Scatter(x=v["t_min"], y=v["occ"], name=lab, line=dict(color=c, width=1.3),
                                 legendgroup=lab, hovertemplate="%{x:.0f} min<br>occ %{y:.2f}<extra>"+lab+"</extra>"), row=1, col=1)
    fig.add_hrect(y0=300, y1=800, fillcolor=BAND_GREEN, line_width=0, row=1, col=2)
    fig.add_trace(go.Bar(x=[s[0] for s in seeds], y=[s[1]["pool_g3p_mean"] for s in seeds],
                         marker_color=[s[2] for s in seeds], showlegend=False,
                         text=[f"{s[1]['pool_g3p_mean']:.0f}" for s in seeds], textposition="outside",
                         hovertemplate="%{x}<br>pool %{y:.0f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title="occupancy", range=[-0.03, 1.10], row=1, col=1)
    fig.update_yaxes(title="molecules", range=[0, 900], row=1, col=2)
    fig.update_xaxes(title="lineage time (min)", row=1, col=1)
    _layout(fig, "Seed-robust: the cooperative switch holds across seeds 0/1/2",
            "occupancy reaches full and the DnaA pool stays in [300,800] on all three seeds (homeostasis robust; the in-band ATP fraction is seed-1-calibrated)", h=470)
    _write(fig, "dnaa5_multiseed_robustness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/dnaa5_data.json")
    args = ap.parse_args()
    d = json.load(open(args.data)) if os.path.exists(args.data) else {}
    print(f"loaded {len(d)} run series from {args.data}")
    fig_switch_curves()                # analytic, always
    fig_response_sharpness()           # analytic, always
    fig_hill_vs_kd()                   # analytic, always
    fig_kd_sweep()                     # in-sim sweep summary, always
    fig_parent_daughter()              # per-domain daughter-copy occupancy (if data present)
    fig_saturation_per_gen()           # the once-per-generation saturation test
    fig_langmuir_vs_switch(d)
    fig_sweet_spot(d)
    fig_homeostasis(d)
    fig_multiseed(d)
    print("done.")


if __name__ == "__main__":
    main()
