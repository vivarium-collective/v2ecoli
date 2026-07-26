"""Roll out an ensemble of v2ecoli baseline trajectories and build a pbg-torch
TransitionDataset over the broad observable panel.

Each trajectory steps the baseline composite one simulated second at a time,
extracting the flat observable vector before and after each step to form the
(features_t, targets_{t+1}) transition pairs the surrogate trains on. A
trajectory halts at division (the single-agent assumption breaks once daughters
appear).

Usage:
    .venv/bin/python sample_baseline.py --seeds 0 1 2 3 4 5 \
        --n-steps 250 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout, build_spec, DEFAULT_GROUPS  # noqa: E402

CACHE_DIR = "/Users/eranagmon/code/v2ecoli/out/cache"


def sample_trajectory(seed, n_steps, layout, cache_dir):
    """Step one baseline trajectory; return (X_list, Y_list) of obs vectors.

    layout is None on the first call: it is discovered from the first stepped
    state and returned so subsequent trajectories reuse the same ordering.
    """
    from v2ecoli import build_composite

    comp = build_composite("ecoli_baseline", seed=seed, cache_dir=cache_dir, emitter="null")
    comp.run(1)  # one step to populate listeners before discovery
    if layout is None:
        layout = PanelLayout.discover(comp.state)

    prev = layout.extract(comp.state)
    Xs, Ys = [], []
    for _ in range(n_steps):
        comp.run(1)
        cell = comp.state["agents"]
        if len(cell) != 1 or list(cell.values())[0].get("divide"):
            break  # division fired — stop before the agent key changes
        cur = layout.extract(comp.state)
        Xs.append(prev)
        Ys.append(cur)
        prev = cur
    return Xs, Ys, layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-steps", type=int, default=250)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from pbg_torch import TransitionDataset

    layout = None
    Xall, Yall, ids = [], [], []
    t0 = time.time()
    for traj_index, seed in enumerate(args.seeds):
        ts = time.time()
        # discover layout (first traj) with the requested groups
        if layout is None:
            from v2ecoli import build_composite
            comp = build_composite("ecoli_baseline", seed=seed, cache_dir=args.cache_dir, emitter="null")
            comp.run(1)
            layout = PanelLayout.discover(comp.state, groups=tuple(args.groups))
            del comp
        Xs, Ys, layout = sample_trajectory(seed, args.n_steps, layout, args.cache_dir)
        Xall.extend(Xs)
        Yall.extend(Ys)
        ids.extend([traj_index] * len(Xs))
        print(f"  seed {seed}: {len(Xs)} transitions ({time.time()-ts:.1f}s)", flush=True)

    X = np.asarray(Xall, dtype=np.float64)
    Y = np.asarray(Yall, dtype=np.float64)
    DT = np.ones(X.shape[0], dtype=np.float64)
    traj_id = np.asarray(ids, dtype=np.int64)

    spec = build_spec(layout)
    ds = TransitionDataset(X=X, Y=Y, DT=DT, traj_id=traj_id, spec=spec)
    ds.save(os.path.join(args.out, "transitions.npz"))
    with open(os.path.join(args.out, "layout.json"), "w") as fh:
        json.dump(layout.to_dict(), fh)

    print(f"\nDataset: {X.shape[0]} transitions x {X.shape[1]} observables "
          f"from {len(args.seeds)} trajectories ({time.time()-t0:.1f}s total)")
    print(f"  groups: " + ", ".join(
        f"{g}={layout.group_slices[g][1]-layout.group_slices[g][0]}" for g in layout.groups))
    print(f"  saved -> {args.out}/transitions.npz + layout.json")


if __name__ == "__main__":
    main()
