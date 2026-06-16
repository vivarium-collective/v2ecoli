"""Evaluate a trained surrogate against held-out v2ecoli baseline behavior, and
profile its inference speed against the full simulator.

Produces ``metrics.json`` containing:
  - per_group: one-step prediction R2 / RMSE / nRMSE on the held-out trajectory,
    for each observable group (mass, exchange, base_flux, monomer, mrna, ...)
  - rollout: autoregressive surrogate vs. actual trajectory for the scalar
    observables (cell_mass, instantaneous_growth_rate), for the figures
  - speed: surrogate steps/sec, simulator steps/sec, and the speedup factor

Usage:
    .venv/bin/python evaluate_surrogate.py --data <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout  # noqa: E402

CACHE_DIR = "/Users/eranagmon/code/v2ecoli/out/cache"


def _group_metrics(actual, pred):
    """Per-column emulation metrics for a (possibly high-dimensional) group.

    Pooling raw R^2 across columns of vastly different magnitude is misleading
    (a few huge-variance columns dominate). We compute R^2 and nRMSE PER COLUMN
    on the absolute next-state, drop zero-variance columns, and summarize with
    the median and the fraction of columns predicted well (R^2 > 0.5).
    """
    actual = actual if actual.ndim > 1 else actual[:, None]
    pred = pred if pred.ndim > 1 else pred[:, None]
    ss_res = np.sum((actual - pred) ** 2, axis=0)
    mu = actual.mean(axis=0)
    ss_tot = np.sum((actual - mu) ** 2, axis=0)
    valid = ss_tot > 0
    r2_col = 1.0 - ss_res[valid] / ss_tot[valid]
    rmse_col = np.sqrt(np.mean((actual - pred) ** 2, axis=0))[valid]
    std_col = actual.std(axis=0)[valid]
    nrmse_col = rmse_col / std_col
    return {
        "median_r2": float(np.median(r2_col)) if r2_col.size else float("nan"),
        "frac_r2_above_0.5": float(np.mean(r2_col > 0.5)) if r2_col.size else float("nan"),
        "median_nrmse": float(np.median(nrmse_col)) if nrmse_col.size else float("nan"),
        "n_dims": int(actual.shape[1]),
        "n_varying_dims": int(valid.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout-steps", type=int, default=60)
    ap.add_argument("--profile-surrogate-steps", type=int, default=2000)
    ap.add_argument("--profile-sim-steps", type=int, default=40)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()

    from pbg_torch import TransitionDataset, SurrogateNet

    ds = TransitionDataset.load(os.path.join(args.data, "transitions.npz"))
    layout = PanelLayout.from_dict(
        json.load(open(os.path.join(args.data, "layout.json"))))
    net = SurrogateNet.load(os.path.join(args.data, "surrogate.pt"))
    hist = json.load(open(os.path.join(args.data, "train_history.json")))
    holdout = hist["holdout_traj"]

    sel = ds.traj_id == holdout
    Xh, Yh = ds.X[sel], ds.Y[sel]
    print(f"Held-out trajectory {holdout}: {Xh.shape[0]} transitions.")

    # --- one-step prediction metrics, per observable group ---
    pred_next = net.predict_next(Xh)        # absolute next-step prediction
    per_group = {}
    for group, (start, end) in layout.group_slices.items():
        per_group[group] = _group_metrics(Yh[:, start:end], pred_next[:, start:end])
        m = per_group[group]
        print(f"  {group:11s} dim={m['n_dims']:5d} varying={m['n_varying_dims']:5d}  "
              f"median_R2={m['median_r2']:+.3f}  frac(R2>0.5)={m['frac_r2_above_0.5']:.2f}  "
              f"median_nRMSE={m['median_nrmse']:.3f}")
    overall = _group_metrics(Yh, pred_next)
    print(f"  {'OVERALL':11s} dim={overall['n_dims']:5d} varying={overall['n_varying_dims']:5d}  "
          f"median_R2={overall['median_r2']:+.3f}  frac(R2>0.5)={overall['frac_r2_above_0.5']:.2f}")

    # --- autoregressive rollout of scalar observables ---
    # reconstruct the actual held-out obs sequence: obs[0]=Xh[0], obs[t]=Yh[t-1]
    actual_seq = np.vstack([Xh[0][None, :], Yh])
    n_roll = min(args.rollout_steps, actual_seq.shape[0] - 1)
    s = Xh[0].copy()
    roll = [s.copy()]
    for _ in range(n_roll):
        s = net.predict_next(s[None, :])[0]
        roll.append(s.copy())
    roll = np.asarray(roll)

    rollout = {}
    for name in ("cell_mass", "instantaneous_growth_rate"):
        col = layout.labels.index(("mass", name))
        rollout[name] = {
            "actual": actual_seq[: n_roll + 1, col].tolist(),
            "surrogate": roll[: n_roll + 1, col].tolist(),
        }

    # --- speed profiling ---
    x1 = Xh[:1].copy()
    t0 = time.time()
    for _ in range(args.profile_surrogate_steps):
        net.predict_next(x1)
    sur_sps = args.profile_surrogate_steps / (time.time() - t0)

    from v2ecoli import build_composite
    comp = build_composite("baseline", seed=0, cache_dir=args.cache_dir, emitter="null")
    comp.run(1)
    t0 = time.time()
    for _ in range(args.profile_sim_steps):
        comp.run(1)
    sim_sps = args.profile_sim_steps / (time.time() - t0)

    speed = {"surrogate_steps_per_sec": sur_sps, "simulator_steps_per_sec": sim_sps,
             "speedup": sur_sps / sim_sps}
    print(f"\nSpeed: surrogate {sur_sps:,.0f} steps/s vs simulator "
          f"{sim_sps:.2f} steps/s  ->  {speed['speedup']:,.0f}x")

    metrics = {
        "holdout_traj": int(holdout),
        "n_holdout_transitions": int(Xh.shape[0]),
        "observable_dims": int(ds.X.shape[1]),
        "per_group": per_group,
        "overall": overall,
        "rollout": rollout,
        "rollout_steps": int(n_roll),
        "speed": speed,
    }
    with open(os.path.join(args.data, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  saved -> {args.data}/metrics.json")


if __name__ == "__main__":
    main()
