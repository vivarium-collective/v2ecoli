"""Train a pbg-torch surrogate on a baseline TransitionDataset.

One whole trajectory is held out as a true test set (excluded from training);
the remaining trajectories train the residual-MLP surrogate. The checkpoint is
self-contained (weights + spec + normalization stats).

Usage:
    .venv/bin/python train_surrogate.py --data <dir> --holdout-traj -1 \
        --hidden 256 256 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing transitions.npz")
    ap.add_argument("--holdout-traj", type=int, default=-1,
                    help="trajectory index to hold out for test (-1 = last)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pbg_torch import TransitionDataset, train_surrogate

    ds = TransitionDataset.load(os.path.join(args.data, "transitions.npz"))
    traj_ids = np.unique(ds.traj_id)
    holdout = traj_ids[args.holdout_traj]
    keep = ds.traj_id != holdout

    train_ds = TransitionDataset(
        X=ds.X[keep], Y=ds.Y[keep], DT=ds.DT[keep],
        traj_id=ds.traj_id[keep], spec=ds.spec,
    )
    print(f"Training on {train_ds.n_transitions} transitions "
          f"({len(traj_ids)-1} trajectories), holding out traj {int(holdout)} "
          f"({int((~keep).sum())} transitions) for test.")
    print(f"Observable dimensionality: {ds.X.shape[1]}")

    net, history = train_surrogate(
        train_ds, hidden=tuple(args.hidden), epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, seed=args.seed,
    )

    ckpt = os.path.join(args.data, "surrogate.pt")
    net.save(ckpt)
    with open(os.path.join(args.data, "train_history.json"), "w") as fh:
        json.dump({"history": history, "holdout_traj": int(holdout),
                   "hidden": args.hidden, "epochs": args.epochs}, fh)

    print(f"  final train_loss={history['train_loss'][-1]:.4g}, "
          f"val_loss={history['val_loss'][-1]:.4g}")
    print(f"  saved -> {ckpt}")


if __name__ == "__main__":
    main()
