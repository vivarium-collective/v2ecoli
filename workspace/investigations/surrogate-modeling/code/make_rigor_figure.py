"""Rigor figure: surrogate vs. trivial baselines.

Reads the compact panel's k-fold metrics (metrics_rigor.json) and the broad
panel's per-observable skill (metrics_skill.json) and renders one interactive
report showing (a) cross-validated rollout error per model with a 'beats
baselines?' reading, (b) error-vs-horizon, and (c) the broad-panel
learnability (fraction of columns that beat persistence).

Usage:
    .venv/bin/python make_rigor_figure.py --compact ../run_compact --broad ../run --out <html>
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", required=True)
    ap.add_argument("--broad", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import plotly.graph_objects as go
    import plotly.io as pio

    rig = json.load(open(os.path.join(args.compact, "metrics_rigor.json")))
    skill = json.load(open(os.path.join(args.broad, "metrics_skill.json")))
    summ = rig["summary"]
    models = ["linear", "mean-delta", "surrogate", "persistence"]
    colors = {"surrogate": "#d62728", "linear": "#2ca02c",
              "mean-delta": "#1f77b4", "persistence": "#999999"}

    # (a) cross-validated rollout error per model (log scale; lower is better)
    med = [summ[m]["rollout_mass_nrmse_median"] for m in models]
    unstable = [summ[m]["rollout_folds_unstable"] for m in models]
    nf = summ[models[0]]["n_folds"]
    bar = go.Figure(go.Bar(
        x=models, y=med, marker_color=[colors[m] for m in models],
        text=[f"{v:.3f}<br>{u}/{nf} unstable" for v, u in zip(med, unstable)],
        textposition="outside",
    ))
    bar.update_layout(
        title=f"Cross-validated rollout error (cell_mass, {nf}-fold leave-one-trajectory-out)",
        yaxis=dict(title="median rollout nRMSE (log, lower=better)", type="log"),
        xaxis_title="model", height=440, template="plotly_white", margin=dict(t=80))
    bar.add_annotation(xref="paper", yref="paper", x=0.5, y=1.12, showarrow=False,
                       text="<b>The neural surrogate does NOT beat a linear model — and is the only one that destabilizes in rollout.</b>",
                       font=dict(size=12, color="#444"))

    # (b) error vs horizon (one fold)
    hz = rig["horizon"]
    steps = hz["steps"]
    line = go.Figure()
    for m in models:
        ys = hz["models"].get(m)
        if ys is None:
            continue
        line.add_trace(go.Scatter(x=steps, y=ys, name=m, line=dict(color=colors[m])))
    line.update_layout(
        title="Autoregressive rollout error vs. horizon (cell_mass, one held-out trajectory)",
        xaxis_title="rollout step (s)", yaxis=dict(title="absolute error (fg)"),
        height=420, template="plotly_white")

    # (c) broad-panel learnability: fraction of columns beating persistence
    pg = skill["per_group"]
    groups = list(pg.keys())
    fracs = [pg[g]["frac_beats_persistence"] for g in groups]
    learn = go.Figure(go.Bar(
        x=groups, y=fracs, marker_color="#d62728",
        text=[f"{f*100:.1f}%<br>n={pg[g]['n_varying']}" for g, f in zip(groups, fracs)],
        textposition="outside"))
    learn.update_layout(
        title=f"Broad panel: fraction of observables that beat persistence "
              f"(overall {skill['overall_frac_beats_persistence']*100:.1f}%)",
        yaxis=dict(title="fraction beating persistence", range=[0, max(0.1, max(fracs) * 1.3 or 0.1)],
                   tickformat=".0%"),
        xaxis_title="observable group", height=420, template="plotly_white")

    intro = ("<p style='font-family:sans-serif;max-width:820px'>The honest test: does the "
             "neural surrogate beat <i>trivial</i> baselines (persistence = next equals current; "
             "mean-delta = constant drift; linear = ridge regression on the per-step delta)? "
             "One-step R² is uninformative here — every model scores ~1.000 because the signal "
             "is smooth (a persistence artifact). The discriminating test is the multi-step "
             "<b>autoregressive rollout</b>, cross-validated leave-one-trajectory-out.</p>")
    verdict = ("<div style='font-family:sans-serif;max-width:820px;margin:14px 0;padding:12px;"
               "background:#fff5f5;border-left:4px solid #d62728'>"
               "<b>Finding:</b> for coarse growth/mass a <b>linear model is the best and most "
               "stable emulator</b> (rollout nRMSE ≈ 0.02, 0 unstable folds); the neural surrogate "
               "matches it only when it doesn't destabilize. For the broad 11.5k-observable panel, "
               "<b>0% of observables beat persistence</b> — they are not predictable from the "
               "observable view alone; they require the cell's hidden state. The convincing "
               "conclusion is a scoping one, not a capability claim.</div>")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Surrogate vs. baselines — rigor</title></head>",
        "<body style='max-width:900px;margin:0 auto'>",
        "<h1 style='font-family:sans-serif'>Does the surrogate beat trivial baselines?</h1>",
        intro, verdict,
        pio.to_html(bar, include_plotlyjs="cdn", full_html=False),
        pio.to_html(line, include_plotlyjs=False, full_html=False),
        pio.to_html(learn, include_plotlyjs=False, full_html=False),
        "</body></html>",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(parts))
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
