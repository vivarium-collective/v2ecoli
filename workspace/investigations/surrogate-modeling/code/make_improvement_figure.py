"""Follow-up figure: does multi-step rollout-loss training improve the surrogate?

Reads metrics_improvement.json (8-fold CV: one-step NN vs rollout-loss NN vs
linear) and renders the rollout-error comparison with the honest verdict.

Usage:
    .venv/bin/python make_improvement_figure.py --data ../run_compact --out <html>
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import plotly.graph_objects as go
    import plotly.io as pio

    m = json.load(open(os.path.join(args.data, "metrics_improvement.json")))
    order = ["onestep", "rollout_loss", "linear"]
    labels = {"onestep": "neural net<br>(one-step loss)",
              "rollout_loss": "neural net<br>(rollout loss)",
              "linear": "linear<br>baseline"}
    colors = {"onestep": "#d62728", "rollout_loss": "#ff7f0e", "linear": "#2ca02c"}
    med = [m["models"][k]["rollout_mass_nrmse_median"] for k in order]
    inst = [m["models"][k]["unstable_folds"] for k in order]
    nf = m["models"][order[0]]["n_folds"]

    fig = go.Figure(go.Bar(
        x=[labels[k] for k in order], y=med, marker_color=[colors[k] for k in order],
        text=[f"{v:.3f}<br>{u}/{nf} unstable" for v, u in zip(med, inst)],
        textposition="outside"))
    fig.update_layout(
        title=f"Does rollout-loss training help? ({nf}-fold CV, cell_mass rollout)",
        yaxis=dict(title="median rollout nRMSE (log, lower=better)", type="log"),
        height=440, template="plotly_white", margin=dict(t=80))

    verdict = (
        "<div style='font-family:sans-serif;max-width:820px;margin:14px 0;padding:12px;"
        "background:#fffaf0;border-left:4px solid #ff7f0e'>"
        f"<b>Follow-up finding:</b> multi-step rollout-loss training genuinely helps — it "
        f"<b>eliminates the instability</b> ({m['models']['rollout_loss']['unstable_folds']}/{nf} "
        f"unstable vs {m['models']['onestep']['unstable_folds']}/{nf} for one-step) and improves "
        f"rollout accuracy ({m['models']['rollout_loss']['rollout_mass_nrmse_median']:.3f} vs "
        f"{m['models']['onestep']['rollout_mass_nrmse_median']:.3f}). But it still does <b>not beat "
        f"the linear baseline</b> ({m['models']['linear']['rollout_mass_nrmse_median']:.3f}). The "
        "right training technique narrows the gap; for a target this smooth, a linear model remains "
        "the honest choice. Rollout-loss is the technique to carry to harder, genuinely nonlinear "
        "targets — where a linear model would fail.</div>"
    )
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Improving the surrogate</title></head>",
        "<body style='max-width:900px;margin:0 auto'>",
        "<h1 style='font-family:sans-serif'>Can we do better? Rollout-loss training</h1>",
        "<p style='font-family:sans-serif;max-width:820px'>sm-01 diagnosed the neural net's "
        "failure as autoregressive instability (one-step training never penalizes compounding "
        "error). The principled fix: train the net on its own multi-step unrolled predictions.</p>",
        verdict,
        pio.to_html(fig, include_plotlyjs="cdn", full_html=False),
        "</body></html>",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(html))
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
