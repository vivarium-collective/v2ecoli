"""Utility + break-even: what a cheap emulator buys you.

The rigor analysis showed a LINEAR model is the best, most stable emulator of
coarse growth — so the honest "so what" features the linear emulator, not the
neural net. We:

  1. Fit the linear emulator on the compact dataset and roll out mass+growth
     trajectories for a large sweep of initial conditions, timing it.
  2. Compare the amortized cost against the full WCM simulator: a sweep of N
     trajectories costs (training sims) + N·(emulator rollout) for the emulator
     vs N·(WCM rollout) for the simulator. Report the break-even N.

This makes "fast" meaningful: a cheap emulator pays for its training data once
you need more than a handful of evaluations.

Usage:
    .venv/bin/python utility_demo.py --data ../run_compact --out <dir>
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
from baselines import LinearPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sweep", type=int, default=5000, help="# initial conditions")
    ap.add_argument("--steps", type=int, default=250, help="rollout horizon per traj")
    ap.add_argument("--n-train-sims", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pbg_torch import TransitionDataset
    ds = TransitionDataset.load(os.path.join(args.data, "transitions.npz"))
    layout = PanelLayout.from_dict(json.load(open(os.path.join(args.data, "layout.json"))))
    mass_col = layout.labels.index(("mass", "cell_mass"))

    # Fit the linear emulator on all data.
    lin = LinearPredictor(ds.spec).fit(ds.X, ds.Y)

    # Sweep initial conditions: sample initial observable vectors spanning the
    # observed range (perturb real starting states so drivers/co-features stay
    # physically plausible).
    rng = np.random.default_rng(0)
    starts = ds.X[rng.integers(0, ds.X.shape[0], size=args.sweep)].copy()
    mass_lo, mass_hi = ds.X[:, mass_col].min(), ds.X[:, mass_col].max()
    starts[:, mass_col] = rng.uniform(mass_lo, mass_hi, size=args.sweep)

    # Time the full sweep (vectorized autoregressive rollout).
    t0 = time.time()
    s = starts.copy()
    final_mass = np.empty(args.sweep)
    traj0 = None
    for step in range(args.steps):
        s = lin.predict_next(s)
        # feed predicted targets back; drivers (none here) stay fixed
        starts[:, : ds.spec.n_targets] = s
        s = starts
        if traj0 is None:
            traj0 = []
        traj0.append(s[0, mass_col])
    final_mass[:] = s[:, mass_col]
    emulator_sweep_s = time.time() - t0
    per_emulator_traj_s = emulator_sweep_s / args.sweep

    # Measured simulator cost (from the broad run's profiling, falls back to a
    # conservative per-step estimate).
    sim_steps_per_sec = 14.0
    broad_metrics = os.path.join(os.path.dirname(args.data), "run", "metrics.json")
    if os.path.isfile(broad_metrics):
        sim_steps_per_sec = json.load(open(broad_metrics))["speed"]["simulator_steps_per_sec"]
    per_wcm_traj_s = args.steps / sim_steps_per_sec

    # Amortized cost: training sims (one WCM run each) + N emulator rollouts.
    train_cost_s = args.n_train_sims * per_wcm_traj_s
    def emulator_total(n):
        return train_cost_s + n * per_emulator_traj_s
    def wcm_total(n):
        return n * per_wcm_traj_s
    # break-even N: smallest N where emulator_total <= wcm_total
    # train + N*e <= N*w  ->  N >= train / (w - e)
    breakeven = train_cost_s / max(per_wcm_traj_s - per_emulator_traj_s, 1e-12)

    speedup = per_wcm_traj_s / per_emulator_traj_s
    out = {
        "sweep_n": args.sweep, "steps": args.steps,
        "emulator_sweep_seconds": emulator_sweep_s,
        "per_emulator_traj_seconds": per_emulator_traj_s,
        "per_wcm_traj_seconds": per_wcm_traj_s,
        "sim_steps_per_sec": sim_steps_per_sec,
        "train_cost_seconds": train_cost_s,
        "per_traj_speedup": speedup,
        "breakeven_n_evaluations": breakeven,
        "wcm_time_for_sweep_seconds": wcm_total(args.sweep),
        "cost_curve": {
            "n": [1, 10, 100, 1000, args.sweep],
            "emulator_s": [emulator_total(n) for n in [1, 10, 100, 1000, args.sweep]],
            "wcm_s": [wcm_total(n) for n in [1, 10, 100, 1000, args.sweep]],
        },
    }
    print(f"Linear emulator swept {args.sweep:,} {args.steps}-step trajectories in "
          f"{emulator_sweep_s:.2f}s ({per_emulator_traj_s*1e3:.3f} ms/traj).")
    print(f"  WCM per-traj: {per_wcm_traj_s:.1f}s  ->  per-traj speedup ~{speedup:,.0f}x")
    print(f"  Same sweep on the WCM: {wcm_total(args.sweep)/3600:.1f} h "
          f"(vs {emulator_sweep_s:.1f}s).")
    print(f"  Break-even: emulator pays back its {args.n_train_sims} training sims "
          f"after ~{breakeven:.1f} evaluations.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metrics_utility.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  saved -> {args.out}/metrics_utility.json")


if __name__ == "__main__":
    main()
