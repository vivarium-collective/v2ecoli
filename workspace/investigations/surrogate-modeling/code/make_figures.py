"""Build self-contained interactive Plotly figures from a surrogate run's
metrics.json. Writes one HTML report suitable for the dashboard / PR evidence.

Usage:
    .venv/bin/python make_figures.py --data <run-dir> --out <report.html>
"""
from __future__ import annotations

import argparse
import json
import os


def _fig_per_group_r2(metrics):
    import plotly.graph_objects as go

    groups, r2, dims, frac = [], [], [], []
    for g, m in metrics["per_group"].items():
        groups.append(g)
        r2.append(max(m["median_r2"], -1.0))  # clamp for readability
        dims.append(m["n_dims"])
        frac.append(m["frac_r2_above_0.5"])
    fig = go.Figure(go.Bar(
        x=groups, y=r2,
        text=[f"median R²={v:.3f}<br>{f*100:.0f}% cols R²>0.5<br>{d} dims"
              for v, f, d in zip(r2, frac, dims)],
        textposition="outside",
        marker_color=["#2ca02c" if v > 0.8 else "#ff7f0e" if v > 0 else "#d62728" for v in r2],
    ))
    fig.update_layout(
        title="One-step prediction fidelity per observable group (held-out trajectory)",
        yaxis_title="median per-column R² (clamped at −1)", xaxis_title="observable group",
        yaxis_range=[-1.1, 1.2], height=440, template="plotly_white",
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="green",
                  annotation_text="R²=0.95 target")
    return fig


def _fig_rollout(metrics):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    roll = metrics["rollout"]
    names = list(roll.keys())
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)
    for i, name in enumerate(names, start=1):
        actual = roll[name]["actual"]
        surr = roll[name]["surrogate"]
        steps = list(range(len(actual)))
        fig.add_trace(go.Scatter(x=steps, y=actual, name="baseline",
                                 line=dict(color="#1f77b4"), showlegend=(i == 1)), row=1, col=i)
        fig.add_trace(go.Scatter(x=steps, y=surr, name="surrogate",
                                 line=dict(color="#d62728", dash="dash"),
                                 showlegend=(i == 1)), row=1, col=i)
        fig.update_xaxes(title_text="step (s)", row=1, col=i)
    fig.update_layout(
        title=f"Autoregressive surrogate rollout vs. baseline ({metrics['rollout_steps']} steps)",
        height=420, template="plotly_white",
    )
    return fig


def _speed_html(metrics):
    s = metrics["speed"]
    return (
        f"<div style='font-family:sans-serif;margin:18px 0;padding:14px;"
        f"background:#f0f7ff;border-left:4px solid #1f77b4'>"
        f"<b>Performance:</b> surrogate <b>{s['surrogate_steps_per_sec']:,.0f}</b> steps/s "
        f"vs. simulator <b>{s['simulator_steps_per_sec']:.2f}</b> steps/s "
        f"&nbsp;→&nbsp; <b>{s['speedup']:,.0f}×</b> speedup "
        f"(over {metrics['observable_dims']:,} observables)."
        f"</div>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import plotly.io as pio

    metrics = json.load(open(os.path.join(args.data, "metrics.json")))

    f1 = _fig_per_group_r2(metrics)
    f2 = _fig_rollout(metrics)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>v2ecoli NN surrogate — fidelity & speed</title></head><body>",
        "<h1 style='font-family:sans-serif'>v2ecoli neural-network surrogate</h1>",
        "<p style='font-family:sans-serif;max-width:780px'>A pbg-torch residual-MLP "
        "surrogate trained on baseline rollouts, predicting the per-step delta of "
        f"{metrics['observable_dims']:,} observables across growth/mass, exchange "
        "fluxes, metabolic fluxes, protein and transcript abundance, and chromosome "
        "state. One whole trajectory was held out for evaluation.</p>",
        _speed_html(metrics),
        pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
        pio.to_html(f2, include_plotlyjs=False, full_html=False),
        "</body></html>",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(parts))
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
