"""sm-04 figure: do the emulators hold across a division discontinuity?

Reads metrics_through_division.json and renders two panels:
  (top)    one held-out lineage's true cell_mass across a division vs each
           emulator's autoregressive rollout — the rollout sails straight
           through the halving because a smooth emulator has no way to
           represent the discontinuity.
  (bottom) within-generation rollout nRMSE vs the teacher-forced one-step error
           AT the division tick, per model — the boundary error dwarfs the
           within-generation error for every emulator, trivial or neural.

Usage:
    .venv/bin/python make_through_division_figure.py --data <dir> --out <html>
"""
from __future__ import annotations

import argparse
import json
import os


LABELS = {
    "persistence": "persistence",
    "mean-delta": "mean-delta",
    "linear": "linear",
    "onestep_nn": "neural net<br>(one-step)",
    "rollout_nn": "neural net<br>(rollout-loss)",
}
COLORS = {
    "persistence": "#7f7f7f",
    "mean-delta": "#9467bd",
    "linear": "#2ca02c",
    "onestep_nn": "#d62728",
    "rollout_nn": "#ff7f0e",
}
ORDER = ["persistence", "mean-delta", "linear", "onestep_nn", "rollout_nn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    m = json.load(open(os.path.join(args.data, "metrics_through_division.json")))
    s = m["summary"]
    tr = m.get("trace")
    nf = m["n_folds"]
    drop = m["boundary_actual_drop"]["median"]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.58, 0.42], vertical_spacing=0.14,
        subplot_titles=(
            "One lineage across a division: true cell_mass vs emulator rollout",
            "Within-generation error vs error AT the division tick (per emulator)"))

    # --- top: rollout trace across division ---
    if tr is not None:
        x = list(range(len(tr["actual_mass"])))
        div_i = tr["div_index"]
        fig.add_trace(go.Scatter(
            x=x, y=tr["actual_mass"], name="true cell_mass", mode="lines",
            line=dict(color="black", width=3)), row=1, col=1)
        for k in ORDER:
            if k in tr["models"]:
                fig.add_trace(go.Scatter(
                    x=x, y=tr["models"][k], name=LABELS[k], mode="lines",
                    line=dict(color=COLORS[k], width=1.5, dash="dot")), row=1, col=1)
        fig.add_vline(x=div_i + 0.5, line=dict(color="crimson", width=1.5, dash="dash"),
                      row=1, col=1)
        fig.add_annotation(x=div_i + 0.5, y=max(tr["actual_mass"]),
                           text="division<br>(mass halves)", showarrow=True, arrowhead=2,
                           ax=40, ay=-30, font=dict(color="crimson", size=11), row=1, col=1)
        fig.update_xaxes(title_text="rollout step (s from birth)", row=1, col=1)
        fig.update_yaxes(title_text="cell_mass (fg)", row=1, col=1)

    # --- bottom: within-gen vs boundary one-step error bars (teacher-forced) ---
    wg = [s[k]["wg_onestep"]["median"] for k in ORDER]
    bd = [s[k]["boundary_onestep"]["median"] for k in ORDER]
    fig.add_trace(go.Bar(
        x=[LABELS[k] for k in ORDER], y=wg, name="within-generation one-step",
        marker_color="#4c78a8", text=[f"{v:.4f}" for v in wg], textposition="outside"),
        row=2, col=1)
    fig.add_trace(go.Bar(
        x=[LABELS[k] for k in ORDER], y=bd, name="division-tick one-step",
        marker_color="crimson", text=[f"{v:.1f}" for v in bd], textposition="outside"),
        row=2, col=1)
    fig.update_yaxes(title_text="cell_mass one-step error<br>(nRMSE, log)", type="log", row=2, col=1)

    fig.update_layout(
        title=f"sm-04 — do the emulators survive division? ({nf}-fold leave-one-lineage-out CV)",
        height=820, template="plotly_white", barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12), margin=dict(t=90))

    lin = s["linear"]
    verdict = (
        "<div style='font-family:sans-serif;max-width:860px;margin:14px 0;padding:12px;"
        "background:#fff5f5;border-left:4px solid crimson'>"
        f"<b>Finding (negative, and informative):</b> every emulator — trivial or neural — "
        f"emulates coarse growth <i>within</i> a generation (linear teacher-forced one-step "
        f"nRMSE {lin['wg_onestep']['median']:.4f}) but <b>none represents the division "
        f"discontinuity</b>. At the division tick the true cell_mass drops by "
        f"~{drop*100:.0f}% (a clean halving), yet each model, fed the true pre-division "
        f"state, predicts continued smooth growth — the linear model's one-step error at the "
        f"boundary is {lin['boundary_onestep']['median']:.1f} nRMSE, "
        f"~{lin['boundary_over_within_onestep_ratio']:.0f}× its within-generation error. The "
        "emulator does <b>not</b> follow the halving; it has no state variable that encodes "
        "'about to divide', so autoregressive rollout sails straight through and full-lineage "
        f"rollout error jumps ~{lin['full_rollout_nrmse']['median']/lin['wg_rollout_nrmse']['median']:.0f}× "
        f"(within-gen {lin['wg_rollout_nrmse']['median']:.2f} → through-division "
        f"{lin['full_rollout_nrmse']['median']:.2f}). This extends the investigation's scoping "
        "result to the cell-cycle boundary: the observable-view emulator is valid only within "
        "a generation. Crossing division needs an explicit division event/reset — the "
        "motivation for the latent encode–decode follow-up.</div>"
    )
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>sm-04 through division</title></head>",
        "<body style='max-width:920px;margin:0 auto'>",
        "<h1 style='font-family:sans-serif'>Rolling the emulator through division</h1>",
        "<p style='font-family:sans-serif;max-width:860px'>sm-00 halted every trajectory at "
        "division; every sm-0x conclusion was conditioned on pre-division single-cell "
        "rollouts. sm-04 follows one daughter lineage across the reset and asks whether the "
        "emulators that win within a generation still hold across the cell_mass halving.</p>",
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
