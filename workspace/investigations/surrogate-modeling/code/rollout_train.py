"""Improvement experiment: multi-step ROLLOUT-LOSS training.

The diagnosis (sm-01): the one-step surrogate predicts fine but its
autoregressive rollout destabilizes (some folds explode) because one-step
training never penalizes compounding error. The fix is to train the net on its
own K-step unrolled predictions, so the loss sees and punishes drift.

This module trains a pbg_torch.SurrogateNet with a multi-step rollout loss
(differentiable through the unroll) for the COMPACT panel (no exogenous drivers
→ the target vector is the full feature vector, so the unroll is closed and
differentiable). Returns a net with the same predict_next interface so the
existing evaluator/baseline machinery compares it head-to-head.

Usage (k-fold compare vs one-step + linear):
    .venv/bin/python rollout_train.py --data ../run_compact --rollout-k 16 --epochs 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout  # noqa: E402
from baselines import build_baselines  # noqa: E402


def _traj_sequences(ds):
    """List of (T+1, n_targets) actual target sequences, one per trajectory.

    Compact panel only: features == targets (no drivers), so the obs sequence is
    X rows in order plus the final Y.
    """
    seqs = []
    for t in np.unique(ds.traj_id):
        sel = ds.traj_id == t
        X = ds.X[sel]
        Y = ds.Y[sel]
        seq = np.vstack([X, Y[-1:]])  # (T+1, n_features)
        seqs.append(seq[:, : ds.spec.n_targets].astype(np.float32))
    return seqs


def train_rollout(ds, hidden=(64, 64), epochs=400, lr=5e-3, rollout_k=16,
                  seed=0, val_traj=None):
    """Train a SurrogateNet with a K-step rollout loss. Compact panel only."""
    from pbg_torch import SurrogateNet
    from pbg_torch.spec import Normalizer

    assert ds.spec.n_features == ds.spec.n_targets, \
        "rollout_train assumes no exogenous drivers (compact panel)"
    torch.manual_seed(seed)

    # normalization fit on one-step training stats (features + per-step delta)
    feat_norm = Normalizer.from_data(ds.X)
    delta = ds.Y - ds.X[:, : ds.spec.n_targets]
    targ_norm = Normalizer.from_data(delta)
    net = SurrogateNet(ds.spec, feat_norm, targ_norm, hidden=hidden)

    seqs = _traj_sequences(ds)
    train_seqs = [torch.tensor(s) for i, s in enumerate(seqs) if i != val_traj]

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tstd = net.targ_std
    tmean = net.targ_mean

    def step_once(s):
        # s: (batch, n_targets) raw; returns next raw targets (delta param)
        norm_delta = net.forward_normalized(s)              # predicted normalized delta
        raw_delta = norm_delta * tstd + tmean
        return s + raw_delta

    for _ in range(epochs):
        net.train()
        total = 0.0
        nwin = 0
        for seq in train_seqs:
            T = seq.shape[0]
            if T <= rollout_k + 1:
                starts = [0]
            else:
                starts = list(range(0, T - rollout_k - 1, max(1, rollout_k // 2)))
            for st in starts:
                k = min(rollout_k, T - st - 1)
                s = seq[st: st + 1]                          # (1, n_targets)
                loss = 0.0
                for j in range(1, k + 1):
                    s = step_once(s)
                    target = seq[st + j: st + j + 1]
                    # loss in normalized target space for scale balance
                    loss = loss + torch.mean(((s - target) / tstd) ** 2)
                loss = loss / k
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                nwin += 1
    return net


# ----- k-fold comparison: one-step vs rollout-loss vs linear -----------------

def _median_col_r2(actual, pred):
    actual = actual if actual.ndim > 1 else actual[:, None]
    pred = pred if pred.ndim > 1 else pred[:, None]
    ss_res = np.sum((actual - pred) ** 2, axis=0)
    ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    v = ss_tot > 0
    return float(np.median(1 - ss_res[v] / ss_tot[v])) if v.any() else float("nan")


def _rollout(model, x0, n):
    s = np.asarray(x0, np.float64).copy()
    out = [s.copy()]
    for _ in range(n):
        s = model.predict_next(s[None, :])[0]
        out.append(s.copy())
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout-k", type=int, default=16)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()

    from pbg_torch import TransitionDataset, train_surrogate

    ds = TransitionDataset.load(os.path.join(args.data, "transitions.npz"))
    layout = PanelLayout.from_dict(json.load(open(os.path.join(args.data, "layout.json"))))
    mass_col = layout.labels.index(("mass", "cell_mass"))
    traj_ids = list(np.unique(ds.traj_id))

    res = {"onestep": [], "rollout_loss": [], "linear": []}
    inst = {"onestep": 0, "rollout_loss": 0, "linear": 0}
    for fi, held in enumerate(traj_ids):
        keep = ds.traj_id != held
        train_ds = TransitionDataset(X=ds.X[keep], Y=ds.Y[keep], DT=ds.DT[keep],
                                     traj_id=ds.traj_id[keep], spec=ds.spec)
        sel = ds.traj_id == held
        Xh, Yh = ds.X[sel], ds.Y[sel]
        seq = np.vstack([Xh[0][None, :], Yh])
        mass_actual = seq[:, mass_col]
        scale = float(np.std(mass_actual)) or 1.0
        n_roll = seq.shape[0] - 1

        one, _ = train_surrogate(train_ds, hidden=tuple(args.hidden), epochs=args.epochs,
                                 lr=args.lr, seed=fi)
        roll_net = train_rollout(train_ds, hidden=tuple(args.hidden), epochs=args.epochs,
                                 lr=args.lr, rollout_k=args.rollout_k, seed=fi)
        lin = build_baselines(ds.spec)["linear"].fit(train_ds.X, train_ds.Y)

        for name, model in (("onestep", one), ("rollout_loss", roll_net), ("linear", lin)):
            r = _rollout(model, seq[0], n_roll)[:, mass_col]
            nrmse = float(np.sqrt(np.nanmean((r - mass_actual) ** 2)) / scale)
            res[name].append(nrmse)
            if not np.isfinite(nrmse) or nrmse > 1.0:
                inst[name] += 1
        print(f"  fold {fi}: onestep={res['onestep'][-1]:.3f}  "
              f"rollout_loss={res['rollout_loss'][-1]:.3f}  linear={res['linear'][-1]:.3f}")

    out = {"rollout_k": args.rollout_k, "n_folds": len(traj_ids), "models": {}}
    for name in res:
        b = np.array(res[name])
        out["models"][name] = {
            "rollout_mass_nrmse_median": float(np.nanmedian(b)),
            "unstable_folds": int(inst[name]), "n_folds": int(b.size),
        }
        print(f"  {name:13s}: median rollout nRMSE={out['models'][name]['rollout_mass_nrmse_median']:.3f}"
              f"  unstable={inst[name]}/{b.size}")
    with open(os.path.join(args.data, "metrics_improvement.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  saved -> {args.data}/metrics_improvement.json")


if __name__ == "__main__":
    main()
