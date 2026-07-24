#!/usr/bin/env python
"""Interactive Plotly figures for the v2ecoli<->vEcoli comparison + rpoBC fix.

Reads the ORIGINAL (unfixed) and FIXED comparison verdicts under out/goldstd_<cond>/
and emits self-contained interactive HTML embeds into the workspace reports/figures/
tree, where the read-only dashboard auto-discovers them per study.

Figures:
  investigation-level (written under every study dir so each report shows them):
    fix-headline-heatmap   per-observable |median Δ%|, before-fix vs after-fix, all conds
    growth-before-after    growth-rate Δ% per condition, before vs after (signif. marked)
    rna-before-after       RNA-mass Δ% per condition, before vs after
    rnap-rootcause         rpoBC ppGpp-expression ratio v2/vEcoli, before vs after fix
    nutrient-gradient      the discovered growth divergence ordered by doubling time
  per-study:
    <cond>-observables     all 5 observables Δ% for THAT condition, before vs after
"""
from __future__ import annotations
import json
import os
import sys

import plotly.graph_objects as go
from plotly.offline import plot

V2 = "/Users/eranagmon/code/v2ecoli"
WS = "/Users/eranagmon/code/v2e-goldstd"
FIG_ROOT = f"{WS}/reports/figures"
CONDS = ["with_aa", "basal", "succinate", "no_oxygen", "acetate"]  # by doubling time
DOUBLING = {"with_aa": 25, "basal": 44, "succinate": 82, "no_oxygen": 100, "acetate": 136}
OBS = ["Growth rate", "RNA mass", "Cell mass", "Dry mass", "Protein mass"]
C_BEFORE, C_AFTER, C_VE = "#d1495b", "#2a9d8f", "#7f7f7f"
LAYOUT = dict(template="plotly_white", font=dict(size=13),
              margin=dict(l=60, r=30, t=60, b=50), height=430,
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def _verdict(cond, kind):
    """kind: 'report' (original) or 'report_fixed'. Returns {obs: (delta%, p, verdict)}."""
    p = f"{V2}/out/goldstd_{cond}/{kind}/verdict.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for ax in d.get("groups", {}).get("physiology", {}).get("axes", []):
        det = ax.get("detail") or {}
        if det.get("delta_rel") is not None:
            out[ax["label"]] = (det["delta_rel"] * 100, det.get("p"), ax["verdict"])
    return out


def _write(fig, study, name):
    d = f"{FIG_ROOT}/{study}"
    os.makedirs(d, exist_ok=True)
    html = plot(fig, output_type="div", include_plotlyjs="cdn", config={"displayModeBar": False})
    path = f"{d}/{name}.html"
    open(path, "w").write(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif}</style>"
        + html)
    return path


def collect():
    data = {}
    for c in CONDS:
        data[c] = {"before": _verdict(c, "report"), "after": _verdict(c, "report_fixed")}
    return data


def fig_headline(data, present):
    # heatmap: rows=conditions, cols=observables, value=|Δ%| after fix; annotate before->after
    z, text = [], []
    for c in present:
        row, trow = [], []
        for o in OBS:
            b = data[c]["before"][o][0] if data[c]["before"] else None
            a = data[c]["after"][o][0] if data[c]["after"] else None
            row.append(abs(a) if a is not None else None)
            trow.append(f"{b:+.1f}% → {a:+.1f}%" if (a is not None and b is not None) else "")
        z.append(row); text.append(trow)
    fig = go.Figure(go.Heatmap(
        z=z, x=OBS, y=[f"{c} ({DOUBLING[c]}m)" for c in present], text=text,
        texttemplate="%{text}", textfont={"size": 11},
        colorscale=[[0, "#2a9d8f"], [0.5, "#e9c46a"], [1, "#d1495b"]], zmin=0, zmax=25,
        colorbar=dict(title="|Δ%|<br>after fix")))
    fig.update_layout(title="v2ecoli vs vEcoli: |median Δ| before → after the rpoBC fix "
                            "(4-seed, gen-1)", **LAYOUT)
    return fig


def fig_before_after(data, present, obs, title):
    before = [data[c]["before"][obs][0] if data[c]["before"] else None for c in present]
    after = [data[c]["after"][obs][0] if data[c]["after"] else None for c in present]
    sig = ["★" if (data[c]["before"] and data[c]["before"][obs][1] is not None
                   and data[c]["before"][obs][1] < 0.05) else "" for c in present]
    x = [f"{c}<br>{DOUBLING[c]}m" for c in present]
    fig = go.Figure()
    fig.add_bar(name="before fix", x=x, y=before, marker_color=C_BEFORE,
                text=[f"{v:+.1f}%{s}" for v, s in zip(before, sig)], textposition="outside")
    fig.add_bar(name="after fix", x=x, y=after, marker_color=C_AFTER,
                text=[f"{v:+.1f}%" for v in after], textposition="outside")
    fig.add_hrect(y0=-5, y1=5, fillcolor="#2a9d8f", opacity=0.08, line_width=0)
    fig.update_layout(title=title + "  (★ = statistically significant before fix; "
                              "shaded = ±5% tolerance)",
                      yaxis_title="median Δ vs vEcoli (%)", barmode="group", **LAYOUT)
    return fig


def fig_rnap_rootcause(present):
    rc = json.load(open(f"{os.environ['SP']}/rootcause_data.json"))
    x = [f"{c}<br>{DOUBLING[c]}m" for c in present]
    fig = go.Figure()
    fig.add_bar(name="before fix", x=x, y=[rc["rpoBC_orig_ratio"][c] for c in present],
                marker_color=C_BEFORE, text=[f"{rc['rpoBC_orig_ratio'][c]:.2f}" for c in present],
                textposition="outside")
    fig.add_bar(name="after fix", x=x, y=[rc["rpoBC_fixed_ratio"][c] for c in present],
                marker_color=C_AFTER, text=[f"{rc['rpoBC_fixed_ratio'][c]:.2f}" for c in present],
                textposition="outside")
    fig.add_hline(y=1.0, line_dash="dash", line_color=C_VE, annotation_text="vEcoli (=1.0)")
    fig.update_layout(
        title="Root cause: rpoBC operon ppGpp-expression ratio v2ecoli / vEcoli"
              "<br><sub>the NNLS fit zeroed exp_free[rpoBC], inverting its nutrient "
              "response; the fix restores the fold-change-consistent split</sub>",
        yaxis_title="rpoBC expression ratio (v2 / vEcoli)", barmode="group", **LAYOUT)
    return fig


def fig_gradient(data, present):
    # original growth-rate divergence, ordered by doubling time (the discovery)
    order = sorted(present, key=lambda c: DOUBLING[c])
    y = [data[c]["before"][OBS[0]][0] if data[c]["before"] else None for c in order]
    col = ["#d1495b" if (data[c]["before"] and data[c]["before"][OBS[0]][1] is not None
                         and data[c]["before"][OBS[0]][1] < 0.05) else "#e9c46a" for c in order]
    fig = go.Figure(go.Bar(x=[f"{c}<br>{DOUBLING[c]}m" for c in order], y=y, marker_color=col,
                           text=[f"{v:+.0f}%" for v in y], textposition="outside"))
    fig.add_hline(y=0, line_color=C_VE)
    fig.update_layout(
        title="The discovered signal: growth-rate divergence scales with nutrient shift"
              "<br><sub>(before fix) v2 over-grows on poor carbon, under-grows on rich AA; "
              "clean only at the basal fit point. Red = significant.</sub>",
        yaxis_title="growth-rate Δ vs vEcoli (%)", **LAYOUT)
    return fig


def main():
    data = collect()
    present = [c for c in CONDS if data[c]["before"] and data[c]["after"]]
    print(f"conditions with both before+after: {present}")
    if not present:
        print("no fixed verdicts yet"); return 1

    # investigation-level figures — write into EVERY present study dir so each shows them
    inv_figs = {
        "01-fix-headline": fig_headline(data, present),
        "02-growth-before-after": fig_before_after(data, present, "Growth rate",
                                                    "Growth rate: before → after the rpoBC fix"),
        "03-rna-before-after": fig_before_after(data, present, "RNA mass",
                                                "RNA mass: before → after the rpoBC fix"),
        "04-rnap-rootcause": fig_rnap_rootcause(present),
        "05-nutrient-gradient": fig_gradient(data, present),
    }
    for c in present:
        for name, fig in inv_figs.items():
            _write(fig, c, name)
        # per-study: all observables before/after for this condition
        f = fig_before_after(data, [c], "Growth rate", "")  # placeholder replaced below
        pf = go.Figure()
        before = [data[c]["before"][o][0] for o in OBS]
        after = [data[c]["after"][o][0] for o in OBS]
        pf.add_bar(name="before fix", x=OBS, y=before, marker_color=C_BEFORE,
                   text=[f"{v:+.1f}%" for v in before], textposition="outside")
        pf.add_bar(name="after fix", x=OBS, y=after, marker_color=C_AFTER,
                   text=[f"{v:+.1f}%" for v in after], textposition="outside")
        pf.add_hrect(y0=-5, y1=5, fillcolor="#2a9d8f", opacity=0.08, line_width=0)
        pf.update_layout(title=f"{c}: all observables, before → after the rpoBC fix "
                               f"(doubling {DOUBLING[c]}m)",
                         yaxis_title="median Δ vs vEcoli (%)", barmode="group", **LAYOUT)
        _write(pf, c, f"00-{c}-observables")
    print(f"wrote figures for {len(present)} studies under {FIG_ROOT}/<study>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
