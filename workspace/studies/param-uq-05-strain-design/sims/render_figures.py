#!/usr/bin/env python
"""Render REAL figures for param-uq-05-strain-design from an ensemble result dir.

Reads results/<dir>/sobol.json + raw_records.json and writes:
  charts/sobol_by_strategy.png     — grouped total-order Sobol (exp/trl_eff/decoy)
                                      for GFP product level under each aggregation
                                      strategy (population, by-seed whiskers,
                                      by-generation if present).
  charts/pce_response_strain_design.png — order-2 PCE surrogate response of GFP
                                      product level vs exp (trl_eff at midpoint)
                                      and vs trl_eff (exp at midpoint), refit from
                                      the population-aggregated ensemble.

Usage:
    python render_figures.py --results <study>/results_smoke --charts <study>/charts \
        --label "SMOKE (n=14, 3 seeds, 1 gen, 200 steps)"
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("POLARS_MAX_THREADS", "1")

PARAMS = ["exp", "trl_eff", "inert_decoy"]
PCOLORS = {"exp": "#2E5EAA", "trl_eff": "#E07B39", "inert_decoy": "#9AA0A6"}
PLABEL = {"exp": "exp", "trl_eff": "trl_eff", "inert_decoy": "inert (decoy)"}


def _total(strat_entry, obs):
    if not strat_entry or "total_order" not in strat_entry:
        return None
    return strat_entry["total_order"].get(obs)


def fig_sobol(sobol, obs, out_path, label):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    groups = []       # (name, {param: value}, {param: (lo,hi) or None})
    s1 = _total(sobol.get("strategy_1_population"), obs)
    if s1:
        groups.append(("Strategy 1\npopulation", s1, None))
    # Strategy 2: by generation
    s2 = sobol.get("strategy_2_by_generation", {})
    for gk in sorted(s2):
        t = _total(s2[gk], obs)
        if t:
            groups.append((f"Strategy 2\n{gk}", t, None))
    # Strategy 3: by seed -> mean +/- range across seeds
    s3 = sobol.get("strategy_3_by_seed", {})
    seed_totals = {p: [] for p in PARAMS}
    for sk in sorted(s3):
        t = _total(s3[sk], obs)
        if t:
            for p in PARAMS:
                seed_totals[p].append(t.get(p, np.nan))
    if any(len(v) for v in seed_totals.values()):
        mean = {p: float(np.nanmean(seed_totals[p])) if seed_totals[p] else np.nan
                for p in PARAMS}
        err = {p: (float(np.nanmin(seed_totals[p])), float(np.nanmax(seed_totals[p])))
               if seed_totals[p] else None for p in PARAMS}
        n_seeds = max(len(v) for v in seed_totals.values())
        groups.append((f"Strategy 3\nby-seed (n={n_seeds})", mean, err))

    n_groups = len(groups)
    x = np.arange(n_groups)
    w = 0.26
    for pi, p in enumerate(PARAMS):
        vals = [g[1].get(p, np.nan) for g in groups]
        yerr = None
        errs = [g[2][p] if g[2] and g[2].get(p) else None for g in groups]
        if any(e is not None for e in errs):
            lo = [v - (e[0] if e else v) for v, e in zip(vals, errs)]
            hi = [(e[1] if e else v) - v for v, e in zip(vals, errs)]
            yerr = np.array([lo, hi])
            yerr = np.clip(yerr, 0, None)
        ax.bar(x + (pi - 1) * w, vals, w, label=PLABEL[p], color=PCOLORS[p],
               yerr=yerr, capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("Total-order Sobol index")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.05, color="#B00020", ls=":", lw=1, alpha=0.7)
    ax.text(n_groups - 0.5, 0.06, "adversarial null band (0.05)", ha="right",
            fontsize=7.5, color="#B00020")
    ax.set_title(f"REAL results — Strain-design Sobol (GFP product level)\n"
                 f"exp vs trl_eff vs inert decoy · {label}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def fig_pce_response(records, out_path, label, bounds):
    from pbg_uq.uqpc import (fit_pce_and_sobol, _physical_to_germ,
                             _predict_and_variance)
    # population aggregate per sample (train only), GFP product level
    by_sample = {}
    for r in records:
        if r["split"] != "train":
            continue
        by_sample.setdefault(r["sample"], []).append(r)
    Xs, Ys = [], []
    for si in sorted(by_sample):
        rows = by_sample[si]
        Xs.append([rows[0]["exp"], rows[0]["trl_eff"], rows[0]["inert_decoy"]])
        Ys.append(np.mean([rr["gfp_product_level"] for rr in rows]))
    X = np.array(Xs)
    Y = np.array(Ys).reshape(-1, 1)
    b = np.array(bounds)
    res = fit_pce_and_sobol(X, Y, b, polynomial_order=2, parameter_names=PARAMS)
    mid = b.mean(axis=1)

    def _pce_predict(Xg):
        # res.surrogate.predict is broken in this pbg_uq build (returns 0); use
        # the fitted PCRV + linregs directly (the same path fit uses internally).
        yg, _ = _predict_and_variance(res.pcrv, res.linregs,
                                      _physical_to_germ(Xg, b), 1)
        return np.asarray(yg).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # vs exp
    ge = np.linspace(b[0, 0], b[0, 1], 60)
    Xe = np.column_stack([ge, np.full_like(ge, mid[1]), np.full_like(ge, mid[2])])
    ye = _pce_predict(Xe)
    axes[0].plot(ge, ye, color=PCOLORS["exp"], lw=2.2)
    axes[0].scatter(X[:, 0], Y.ravel(), s=22, color="#333", alpha=0.55,
                    label="ensemble samples")
    axes[0].set_xlabel("exp (new-gene expression factor)")
    axes[0].set_ylabel("GFP product level (monomer count)")
    axes[0].set_title(f"vs exp (trl_eff={mid[1]:.2f})", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    # vs trl_eff
    gt = np.linspace(b[1, 0], b[1, 1], 60)
    Xt = np.column_stack([np.full_like(gt, mid[0]), gt, np.full_like(gt, mid[2])])
    yt = _pce_predict(Xt)
    axes[1].plot(gt, yt, color=PCOLORS["trl_eff"], lw=2.2)
    axes[1].scatter(X[:, 1], Y.ravel(), s=22, color="#333", alpha=0.55)
    axes[1].set_xlabel("trl_eff (new-gene translation efficiency)")
    axes[1].set_title(f"vs trl_eff (exp={mid[0]:.2f})", fontsize=10)
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"REAL results — order-2 PCE surrogate: GFP product level · {label}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--charts", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    sobol = json.load(open(os.path.join(args.results, "sobol.json")))
    records = json.load(open(os.path.join(args.results, "raw_records.json")))
    bounds = sobol.get("run_config", {}).get(
        "bounds", [[0.5, 5.0], [0.5, 2.0], [0.0, 1.0]])
    os.makedirs(args.charts, exist_ok=True)
    fig_sobol(sobol, "gfp_product_level",
              os.path.join(args.charts, "sobol_by_strategy.png"), args.label)
    fig_pce_response(records, os.path.join(args.charts, "pce_response_strain_design.png"),
                     args.label, bounds)


if __name__ == "__main__":
    main()
