"""Rigor upgrades for the surrogate evaluation:

  --mode kfold   Leave-one-trajectory-out cross-validation comparing the neural
                 surrogate against trivial baselines (persistence, mean-delta,
                 linear) — one-step median per-column R^2 and autoregressive
                 rollout error, reported as mean±std across folds. Plus an
                 error-vs-horizon curve (rollout abs-error per step) from one
                 fold. This answers "is the surrogate's fidelity real, or would
                 a trivial model do as well?".

  --mode skill   Per-observable skill vs persistence on the held-out trajectory:
                 skill = 1 - MSE_model / MSE_persistence (per column). Positive =
                 the model beats persistence. Summarized per group. For the broad
                 panel this turns "2.5% of columns R^2>0.5" into "which
                 observables are learnable from the panel at all".

Writes metrics_rigor.json into the run dir. Reuses the sampled TransitionDataset
(no re-sampling).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout  # noqa: E402
from baselines import build_baselines  # noqa: E402


def _load(data_dir):
    from pbg_torch import TransitionDataset
    ds = TransitionDataset.load(os.path.join(data_dir, "transitions.npz"))
    layout = PanelLayout.from_dict(json.load(open(os.path.join(data_dir, "layout.json"))))
    return ds, layout


def _median_col_r2(actual, pred):
    """Median per-column R^2 over varying columns (robust for high-dim groups)."""
    actual = actual if actual.ndim > 1 else actual[:, None]
    pred = pred if pred.ndim > 1 else pred[:, None]
    ss_res = np.sum((actual - pred) ** 2, axis=0)
    ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    valid = ss_tot > 0
    if not valid.any():
        return float("nan")
    return float(np.median(1.0 - ss_res[valid] / ss_tot[valid]))


def _rollout(model, x0, n_steps):
    s = np.asarray(x0, dtype=np.float64).copy()
    out = [s.copy()]
    for _ in range(n_steps):
        s = model.predict_next(s[None, :])[0]
        out.append(s.copy())
    return np.asarray(out)


def _traj_seq(ds, layout, traj):
    """Reconstruct the actual observable sequence for one trajectory."""
    sel = ds.traj_id == traj
    Xh, Yh = ds.X[sel], ds.Y[sel]
    return Xh, Yh, np.vstack([Xh[0][None, :], Yh])


def _train_surrogate(train_ds, hidden, epochs, lr, seed):
    from pbg_torch import train_surrogate
    net, _ = train_surrogate(train_ds, hidden=tuple(hidden), epochs=epochs, lr=lr, seed=seed)
    return net


def mode_kfold(ds, layout, data_dir, hidden, epochs, lr):
    from pbg_torch import TransitionDataset

    traj_ids = list(np.unique(ds.traj_id))
    mass_col = layout.labels.index(("mass", "cell_mass")) if ("mass", "cell_mass") in layout.labels else 0
    model_names = ["surrogate", "persistence", "mean-delta", "linear"]
    per_fold = {m: {"one_step_med_r2": [], "rollout_mass_nrmse": []} for m in model_names}
    horizon = None  # filled from the first fold

    for fi, held in enumerate(traj_ids):
        keep = ds.traj_id != held
        train_ds = TransitionDataset(X=ds.X[keep], Y=ds.Y[keep], DT=ds.DT[keep],
                                     traj_id=ds.traj_id[keep], spec=ds.spec)
        Xh, Yh, seq = _traj_seq(ds, layout, held)

        models = {"surrogate": _train_surrogate(train_ds, hidden, epochs, lr, seed=fi)}
        for name, bl in build_baselines(ds.spec).items():
            models[name] = bl.fit(train_ds.X, train_ds.Y)

        n_roll = seq.shape[0] - 1
        mass_actual = seq[:, mass_col]
        mass_scale = float(np.std(mass_actual)) or 1.0
        for name, model in models.items():
            per_fold[name]["one_step_med_r2"].append(_median_col_r2(Yh, model.predict_next(Xh)))
            roll = _rollout(model, seq[0], n_roll)[:, mass_col]
            rmse = float(np.sqrt(np.nanmean((roll - mass_actual) ** 2)))
            per_fold[name]["rollout_mass_nrmse"].append(rmse / mass_scale)

        if horizon is None:
            horizon = {"steps": list(range(n_roll + 1)),
                       "actual_mass": mass_actual.tolist(), "models": {}}
            for name, model in models.items():
                roll = _rollout(model, seq[0], n_roll)[:, mass_col]
                horizon["models"][name] = np.abs(roll - mass_actual).tolist()
        print(f"  fold {fi} (held traj {held}): "
              + ", ".join(f"{m} R²={per_fold[m]['one_step_med_r2'][-1]:+.3f}" for m in model_names))

    summary = {}
    for m in model_names:
        a = np.array(per_fold[m]["one_step_med_r2"])
        b = np.array(per_fold[m]["rollout_mass_nrmse"])
        # rollout nRMSE can explode on some folds (autoregressive instability);
        # report the robust MEDIAN + how many folds were unstable (>1 = worse
        # than just predicting the trajectory mean, or non-finite).
        n_unstable = int(np.sum(~np.isfinite(b)) + np.sum(np.isfinite(b) & (b > 1.0)))
        summary[m] = {
            "one_step_med_r2_mean": float(np.nanmean(a)), "one_step_med_r2_std": float(np.nanstd(a)),
            "rollout_mass_nrmse_median": float(np.nanmedian(b)),
            "rollout_mass_nrmse_iqr": float(np.subtract(*np.nanpercentile(b, [75, 25]))),
            "rollout_folds_unstable": n_unstable,
            "n_folds": int(b.size),
        }
        print(f"  {m:11s}: one-step med R²={summary[m]['one_step_med_r2_mean']:+.3f}"
              f"±{summary[m]['one_step_med_r2_std']:.3f}  "
              f"rollout mass nRMSE median={summary[m]['rollout_mass_nrmse_median']:.3f}"
              f"  unstable folds={n_unstable}/{b.size}")

    out = {"mode": "kfold", "n_folds": len(traj_ids), "summary": summary, "horizon": horizon}
    with open(os.path.join(data_dir, "metrics_rigor.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  saved -> {data_dir}/metrics_rigor.json")


def mode_skill(ds, layout, data_dir):
    """Per-observable skill vs persistence on the existing held-out trajectory."""
    from pbg_torch import SurrogateNet

    hist = json.load(open(os.path.join(data_dir, "train_history.json")))
    holdout = hist["holdout_traj"]
    net = SurrogateNet.load(os.path.join(data_dir, "surrogate.pt"))
    persist = build_baselines(ds.spec)["persistence"]

    sel = ds.traj_id == holdout
    Xh, Yh = ds.X[sel], ds.Y[sel]
    nn_pred = net.predict_next(Xh)
    p_pred = persist.predict_next(Xh)

    mse_nn = np.mean((Yh - nn_pred) ** 2, axis=0)
    mse_p = np.mean((Yh - p_pred) ** 2, axis=0)
    valid = mse_p > 0
    skill = np.full(Yh.shape[1], np.nan)
    skill[valid] = 1.0 - mse_nn[valid] / mse_p[valid]  # >0 means NN beats persistence

    per_group = {}
    for g, (s, e) in layout.group_slices.items():
        sk = skill[s:e]
        sk = sk[np.isfinite(sk)]
        if sk.size == 0:
            continue
        per_group[g] = {
            "n_varying": int(sk.size),
            "frac_beats_persistence": float(np.mean(sk > 0)),
            "median_skill": float(np.median(sk)),
        }
        print(f"  {g:11s}: {per_group[g]['frac_beats_persistence']*100:5.1f}% beat persistence "
              f"(median skill {per_group[g]['median_skill']:+.3f}, n={sk.size})")

    allsk = skill[np.isfinite(skill)]
    out = {"mode": "skill", "holdout_traj": int(holdout),
           "overall_frac_beats_persistence": float(np.mean(allsk > 0)),
           "per_group": per_group,
           "skill_hist": np.clip(allsk, -1, 1).tolist()}
    with open(os.path.join(data_dir, "metrics_skill.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  overall: {out['overall_frac_beats_persistence']*100:.1f}% of varying columns beat persistence")
    print(f"  saved -> {data_dir}/metrics_skill.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kfold", "skill"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()
    ds, layout = _load(args.data)
    if args.mode == "kfold":
        mode_kfold(ds, layout, args.data, args.hidden, args.epochs, args.lr)
    else:
        mode_skill(ds, layout, args.data)


if __name__ == "__main__":
    main()
