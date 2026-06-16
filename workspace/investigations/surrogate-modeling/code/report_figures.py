"""Comprehensive interactive report figures for the surrogate-modeling studies.

Recomputes everything it plots directly from a run directory's artifacts
(transitions.npz, layout.json, surrogate.pt, train_history.json, metrics.json),
so the figures demonstrate each claim from primary data rather than restating
summary numbers.

Two report kinds:

  --kind dataset   (sm-00): observable-panel composition + cell_mass / growth
                   trajectories across every sampled seed (coverage + finiteness).

  --kind surrogate (sm-01): per-group one-step R^2, a one-step parity scatter for
                   mass, the DISTRIBUTION of per-column R^2 (shows how many
                   observables are actually emulated), and an autoregressive
                   rollout of cell_mass + growth vs. baseline (tracks for the
                   compact panel; diverges to NaN for the broad panel) — plus the
                   measured speedup.

Usage:
    .venv/bin/python report_figures.py --kind dataset   --data ../run          --out <html>
    .venv/bin/python report_figures.py --kind surrogate --data ../run_compact  --out <html> --title "Compact panel"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout  # noqa: E402


def _load(data_dir):
    from pbg_torch import TransitionDataset
    ds = TransitionDataset.load(os.path.join(data_dir, "transitions.npz"))
    layout = PanelLayout.from_dict(json.load(open(os.path.join(data_dir, "layout.json"))))
    return ds, layout


def _speed_banner(metrics):
    s = metrics["speed"]
    return (
        "<div style='font-family:sans-serif;margin:18px 0;padding:14px;"
        "background:#f0f7ff;border-left:4px solid #1f77b4'>"
        f"<b>Performance:</b> surrogate <b>{s['surrogate_steps_per_sec']:,.0f}</b> steps/s "
        f"vs. simulator <b>{s['simulator_steps_per_sec']:.2f}</b> steps/s "
        f"&nbsp;→&nbsp; <b>{s['speedup']:,.0f}×</b> speedup "
        f"(over {metrics['observable_dims']:,} observables)."
        "</div>"
    )


def _wrap(parts, title):
    import plotly.io as pio
    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{title}</title></head>",
            "<body style='max-width:900px;margin:0 auto'>",
            f"<h1 style='font-family:sans-serif'>{title}</h1>"]
    first = True
    for p in parts:
        if isinstance(p, str):
            html.append(p)
        else:  # a plotly figure
            html.append(pio.to_html(p, include_plotlyjs="cdn" if first else False,
                                    full_html=False))
            first = False
    html.append("</body></html>")
    return "\n".join(html)


# ---------------------------------------------------------------- dataset report
def _dataset_report(data_dir, out, title):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ds, layout = _load(data_dir)
    n_traj = len(np.unique(ds.traj_id))

    # group composition
    groups = layout.groups
    dims = [layout.group_slices[g][1] - layout.group_slices[g][0] for g in groups]
    comp = go.Figure(go.Bar(x=groups, y=dims, text=dims, textposition="outside",
                            marker_color="#1f77b4"))
    comp.update_layout(title=f"Observable panel composition ({sum(dims):,} total)",
                       yaxis_title="dimensions", xaxis_title="group", height=380,
                       template="plotly_white", yaxis_type="log")

    # per-seed trajectories for cell_mass + growth
    mass_col = layout.labels.index(("mass", "cell_mass"))
    growth_col = layout.labels.index(("mass", "instantaneous_growth_rate"))
    traj = make_subplots(rows=1, cols=2,
                         subplot_titles=("cell_mass (fg)", "instantaneous_growth_rate (1/s)"))
    for t in np.unique(ds.traj_id):
        sel = ds.traj_id == t
        # obs sequence for this trajectory: X rows in order + final Y
        seq_mass = np.concatenate([ds.X[sel, mass_col], ds.Y[sel][-1:, mass_col]])
        seq_grow = np.concatenate([ds.X[sel, growth_col], ds.Y[sel][-1:, growth_col]])
        steps = np.arange(len(seq_mass))
        traj.add_trace(go.Scatter(x=steps, y=seq_mass, name=f"seed {t}",
                                  legendgroup=f"s{t}", showlegend=True,
                                  line=dict(width=1)), row=1, col=1)
        traj.add_trace(go.Scatter(x=steps, y=seq_grow, name=f"seed {t}",
                                  legendgroup=f"s{t}", showlegend=False,
                                  line=dict(width=1)), row=1, col=2)
    traj.update_xaxes(title_text="step (s)")
    traj.update_layout(title=f"Sampled trajectories across {n_traj} seeds "
                             f"({ds.n_transitions:,} transitions, all finite)",
                       height=440, template="plotly_white")

    finite = "yes" if np.isfinite(ds.X).all() and np.isfinite(ds.Y).all() else "NO"
    intro = ("<p style='font-family:sans-serif'>The broad observable panel "
             f"({sum(dims):,} dimensions across {len(groups)} groups) extracted from "
             f"{n_traj} baseline rollouts into {ds.n_transitions:,} transition pairs. "
             f"All values finite: <b>{finite}</b>.</p>")
    with open(out, "w") as fh:
        fh.write(_wrap([intro, comp, traj], title))
    print(f"  saved -> {out}")


# -------------------------------------------------------------- surrogate report
def _surrogate_report(data_dir, out, title):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pbg_torch import SurrogateNet

    ds, layout = _load(data_dir)
    metrics = json.load(open(os.path.join(data_dir, "metrics.json")))
    hist = json.load(open(os.path.join(data_dir, "train_history.json")))
    net = SurrogateNet.load(os.path.join(data_dir, "surrogate.pt"))
    holdout = hist["holdout_traj"]

    sel = ds.traj_id == holdout
    Xh, Yh = ds.X[sel], ds.Y[sel]
    pred = net.predict_next(Xh)

    # (1) per-group median R^2 bar
    groups, med, frac, dims = [], [], [], []
    for g, m in metrics["per_group"].items():
        if not np.isfinite(m["median_r2"]):
            continue
        groups.append(g)
        med.append(max(m["median_r2"], -1.0))
        frac.append(m["frac_r2_above_0.5"])
        dims.append(m["n_dims"])
    f_bar = go.Figure(go.Bar(
        x=groups, y=med,
        text=[f"med R²={v:.3f}<br>{f*100:.0f}% cols>0.5<br>{d} dims"
              for v, f, d in zip(med, frac, dims)],
        textposition="outside",
        marker_color=["#2ca02c" if v > 0.8 else "#ff7f0e" if v > 0 else "#d62728" for v in med]))
    f_bar.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="R²=0.95")
    f_bar.update_layout(title="One-step fidelity per group (median per-column R², held-out)",
                        yaxis_title="median per-column R² (clamped −1)", yaxis_range=[-1.1, 1.2],
                        height=420, template="plotly_white")

    # (2) parity scatter for mass (one-step, pooled across the 7 mass dims)
    ms, me = layout.group_slices["mass"]
    a = Yh[:, ms:me].ravel()
    p = pred[:, ms:me].ravel()
    lim = [float(min(a.min(), p.min())), float(max(a.max(), p.max()))]
    parity = go.Figure()
    parity.add_trace(go.Scatter(x=a, y=p, mode="markers",
                                marker=dict(size=4, opacity=0.4, color="#1f77b4"),
                                name="mass observables"))
    parity.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                line=dict(color="black", dash="dash"), name="y = x"))
    parity.update_layout(title="One-step parity: predicted vs. actual next mass observables (held-out)",
                         xaxis_title="actual", yaxis_title="predicted",
                         height=460, template="plotly_white")

    # (3) distribution of per-column R^2 across ALL varying observables
    ss_res = np.sum((Yh - pred) ** 2, axis=0)
    ss_tot = np.sum((Yh - Yh.mean(axis=0)) ** 2, axis=0)
    valid = ss_tot > 0
    r2_col = 1.0 - ss_res[valid] / ss_tot[valid]
    r2_clamped = np.clip(r2_col, -1.0, 1.0)
    frac_good = float(np.mean(r2_col > 0.5))
    hist_fig = go.Figure(go.Histogram(x=r2_clamped, nbinsx=40, marker_color="#1f77b4"))
    hist_fig.add_vline(x=0.5, line_dash="dash", line_color="green",
                       annotation_text=f"{frac_good*100:.1f}% of {valid.sum():,} cols > 0.5")
    hist_fig.update_layout(
        title="Distribution of per-column one-step R² across all varying observables",
        xaxis_title="per-column R² (clamped to −1)", yaxis_title="# observables",
        height=420, template="plotly_white")

    # (4) autoregressive rollout vs baseline (tracks=compact, diverges=broad)
    actual_seq = np.vstack([Xh[0][None, :], Yh])
    n_roll = min(60, actual_seq.shape[0] - 1)
    s = Xh[0].copy()
    roll = [s.copy()]
    diverged_at = None
    for i in range(n_roll):
        s = net.predict_next(s[None, :])[0]
        if diverged_at is None and not np.isfinite(s).all():
            diverged_at = i + 1
        roll.append(s.copy())
    roll = np.asarray(roll)
    mass_col = layout.labels.index(("mass", "cell_mass"))
    grow_col = layout.labels.index(("mass", "instantaneous_growth_rate"))
    roll_fig = make_subplots(rows=1, cols=2, subplot_titles=("cell_mass (fg)",
                                                             "instantaneous_growth_rate (1/s)"))
    for ci, col in enumerate([mass_col, grow_col], start=1):
        steps = np.arange(roll.shape[0])
        roll_fig.add_trace(go.Scatter(x=steps, y=actual_seq[: n_roll + 1, col],
                                      name="baseline", line=dict(color="#1f77b4"),
                                      showlegend=(ci == 1)), row=1, col=ci)
        rser = roll[:, col].astype(float)
        rser[~np.isfinite(rser)] = None
        roll_fig.add_trace(go.Scatter(x=steps, y=rser, name="surrogate",
                                      line=dict(color="#d62728", dash="dash"),
                                      showlegend=(ci == 1)), row=1, col=ci)
        roll_fig.update_xaxes(title_text="step (s)", row=1, col=ci)
    rtitle = f"Autoregressive rollout vs. baseline ({n_roll} steps)"
    if diverged_at:
        rtitle += f" — surrogate DIVERGED at step {diverged_at} (NaN)"
    roll_fig.update_layout(title=rtitle, height=440, template="plotly_white")

    intro = ("<p style='font-family:sans-serif'>Held-out trajectory "
             f"{holdout} ({Xh.shape[0]} transitions, {ds.X.shape[1]:,} observables). "
             "Parity + R² distribution are one-step; the rollout is fully "
             "autoregressive (the surrogate's own predictions feed back).</p>")
    parts = [intro, _speed_banner(metrics), f_bar, parity, hist_fig, roll_fig]
    with open(out, "w") as fh:
        fh.write(_wrap(parts, title))
    print(f"  saved -> {out}  (per-col R²>0.5: {frac_good*100:.1f}%"
          + (f", diverged@{diverged_at}" if diverged_at else ", rollout stable") + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["dataset", "surrogate"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.kind == "dataset":
        _dataset_report(args.data, args.out, args.title or "Surrogate training dataset")
    else:
        _surrogate_report(args.data, args.out, args.title or "Surrogate fidelity & speed")


if __name__ == "__main__":
    main()
