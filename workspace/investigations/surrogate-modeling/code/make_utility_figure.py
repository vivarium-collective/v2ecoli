"""Utility / break-even figure: cumulative cost vs. number of evaluations for a
cheap emulator (the linear model that actually works) vs. the full WCM.

Usage:
    .venv/bin/python make_utility_figure.py --data ../run_compact --out <html>
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

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    m = json.load(open(os.path.join(args.data, "metrics_utility.json")))
    be = m["breakeven_n_evaluations"]
    e_per = m["per_emulator_traj_seconds"]
    w_per = m["per_wcm_traj_seconds"]
    train = m["train_cost_seconds"]

    ns = np.unique(np.concatenate([
        np.logspace(0, np.log10(max(m["sweep_n"], 10)), 60),
        [be],
    ]))
    ns.sort()
    emu = train + ns * e_per
    wcm = ns * w_per

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=wcm / 3600.0, name="full WCM simulator",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=ns, y=emu / 3600.0, name="linear emulator (+ 8 training sims)",
                             line=dict(color="#2ca02c", width=3)))
    fig.add_vline(x=be, line_dash="dash", line_color="#d62728",
                  annotation_text=f"break-even ≈ {be:.0f} evals")
    fig.update_layout(
        title="Cost to run N coarse-growth evaluations — emulator vs. simulator",
        xaxis=dict(title="number of trajectory evaluations", type="log"),
        yaxis=dict(title="cumulative wall-clock (hours)", type="log"),
        height=460, template="plotly_white", legend=dict(x=0.02, y=0.98))

    headline = (
        "<div style='font-family:sans-serif;max-width:820px;margin:14px 0;padding:12px;"
        "background:#f0fff4;border-left:4px solid #2ca02c'>"
        f"The linear emulator swept <b>{m['sweep_n']:,}</b> {m['steps']}-step trajectories in "
        f"<b>{m['emulator_sweep_seconds']:.2f}s</b> — the same sweep on the WCM would take "
        f"<b>~{m['wcm_time_for_sweep_seconds']/3600:.0f} hours</b> "
        f"(~{m['per_traj_speedup']:,.0f}× per trajectory). It pays back its "
        f"{int(train/w_per)} training sims after only <b>~{be:.0f} evaluations</b>. "
        "The expensive part is generating training data, not the emulator — so a cheap, "
        "honest emulator of a smooth observable is worth building even when a neural net isn't.</div>"
    )
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Surrogate utility & break-even</title></head>",
        "<body style='max-width:900px;margin:0 auto'>",
        "<h1 style='font-family:sans-serif'>What a cheap emulator buys you</h1>",
        headline,
        pio.to_html(fig, include_plotlyjs="cdn", full_html=False),
        "</body></html>",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(html))
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
