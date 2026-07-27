"""Evaluate the emulators ACROSS a division discontinuity.

Reuses the investigation's existing emulator machinery verbatim — the trivial
baselines (persistence, mean-delta, linear) from ``baselines.py`` and the
rollout-loss neural surrogate from ``rollout_train.py`` — and applies them to the
through-division dataset produced by ``sample_through_division.py``. The whole
question sm-04 asks is whether these emulators, which win WITHIN a generation
(sm-01/sm-03), still hold when the trajectory crosses the cell_mass halving at
division, or whether they fail to represent the discontinuity.

Leave-one-trajectory-out CV (like sm-01's rigor_eval). For each held-out
lineage it reports, on the cell_mass observable:

  * within_gen_nrmse  — autoregressive rollout error over the pre-division
    (generation-0) portion only. This is the sm-01/sm-03 regime.
  * boundary_onestep  — TEACHER-FORCED one-step error at the division tick:
    feed the model the true mother's last pre-division state and compare its
    single-step prediction to the daughter's actual (halved) state. Directly
    answers "does the emulator follow the cell_mass halving?".
  * boundary_actual_drop — the true relative mass change at division
    (|daughter - mother| / mother ≈ 0.5, the halving), for reference.
  * full_rollout_nrmse — rollout error over the whole through-division sequence.

Writes metrics_through_division.json into the data dir.

Usage:
    .venv/bin/python eval_through_division.py --data <dir> \
        --hidden 64 64 --epochs 300 --rollout-k 16
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
    meta = np.load(os.path.join(data_dir, "meta.npz"))
    return ds, layout, meta


def _rollout(model, x0, n_steps):
    s = np.asarray(x0, dtype=np.float64).copy()
    out = [s.copy()]
    for _ in range(n_steps):
        s = model.predict_next(s[None, :])[0]
        out.append(s.copy())
    return np.asarray(out)


def _traj_arrays(ds, meta, held):
    """For one held-out trajectory: X, Y rows, per-row spans/gen flags, and the
    reconstructed absolute observable sequence seq (T+1, n_features)."""
    sel = ds.traj_id == held
    Xh, Yh = ds.X[sel], ds.Y[sel]
    spans = meta["spans_division"][sel]
    gens = meta["generation"][sel]
    seq = np.vstack([Xh[0][None, :], Yh])  # compact panel: features == targets
    return Xh, Yh, spans, gens, seq


def evaluate(ds, layout, meta, data_dir, hidden, epochs, lr, rollout_k):
    from pbg_torch import TransitionDataset, train_surrogate
    from rollout_train import train_rollout

    mass_col = layout.labels.index(("mass", "cell_mass"))
    traj_ids = list(np.unique(ds.traj_id))
    model_names = ["persistence", "mean-delta", "linear", "onestep_nn", "rollout_nn"]
    per_fold = {m: {"within_gen_nrmse": [], "boundary_onestep": [], "full_rollout_nrmse": []}
                for m in model_names}
    boundary_drop = []
    trace = None  # per-tick rollout of one representative fold, for the figure

    for fi, held in enumerate(traj_ids):
        keep = ds.traj_id != held
        train_ds = TransitionDataset(X=ds.X[keep], Y=ds.Y[keep], DT=ds.DT[keep],
                                     traj_id=ds.traj_id[keep], spec=ds.spec)
        Xh, Yh, spans, gens, seq = _traj_arrays(ds, meta, held)

        # locate the across-division transition row (spans==1) in this trajectory
        div_rows = np.where(spans == 1)[0]
        div_row = int(div_rows[0]) if div_rows.size else None

        mass_actual = seq[:, mass_col]
        mass_scale = float(np.std(mass_actual)) or 1.0

        # actual relative mass drop at division (the halving), for reference
        if div_row is not None:
            m_mother = Xh[div_row, mass_col]
            m_daughter = Yh[div_row, mass_col]
            boundary_drop.append(abs(m_daughter - m_mother) / (abs(m_mother) or 1.0))

        # --- build all models on the SAME training folds ---
        models = dict(build_baselines(ds.spec))
        for name in models:  # persistence/mean-delta/linear .fit
            models[name] = models[name].fit(train_ds.X, train_ds.Y)
        models["onestep_nn"], _ = train_surrogate(
            train_ds, hidden=tuple(hidden), epochs=epochs, lr=lr, seed=fi)
        models["rollout_nn"] = train_rollout(
            train_ds, hidden=tuple(hidden), epochs=epochs, lr=lr,
            rollout_k=rollout_k, seed=fi)

        # within-generation rollout: only the pre-division (gen 0) portion
        gen0_end = div_row if div_row is not None else seq.shape[0] - 1
        wg_actual = mass_actual[: gen0_end + 1]
        wg_scale = float(np.std(wg_actual)) or 1.0
        n_full = seq.shape[0] - 1

        for name, model in models.items():
            # within-gen nRMSE (roll from birth to just before division)
            wg_roll = _rollout(model, seq[0], gen0_end)[:, mass_col]
            wg_rmse = float(np.sqrt(np.nanmean((wg_roll - wg_actual) ** 2)))
            per_fold[name]["within_gen_nrmse"].append(wg_rmse / wg_scale)

            # full rollout nRMSE (through division)
            f_roll = _rollout(model, seq[0], n_full)[:, mass_col]
            f_rmse = float(np.sqrt(np.nanmean((f_roll - mass_actual) ** 2)))
            per_fold[name]["full_rollout_nrmse"].append(f_rmse / mass_scale)

            # boundary teacher-forced one-step error (does it follow the halving?)
            if div_row is not None:
                pred_next = model.predict_next(Xh[div_row][None, :])[0, mass_col]
                err = abs(pred_next - Yh[div_row, mass_col]) / mass_scale
                per_fold[name]["boundary_onestep"].append(err)

        # Save a rollout trace from the first fold that actually crosses a
        # division, so the figure can show the rollout missing the halving.
        if trace is None and div_row is not None:
            trace = {
                "held_traj": int(held),
                "div_index": int(div_row),
                "actual_mass": mass_actual.tolist(),
                "models": {name: _rollout(model, seq[0], n_full)[:, mass_col].tolist()
                           for name, model in models.items()},
            }

        print(f"  fold {fi} (held {held}, div_row={div_row}): "
              + ", ".join(f"{m} wg={per_fold[m]['within_gen_nrmse'][-1]:.3f}"
                          f"/bd={per_fold[m]['boundary_onestep'][-1]:.2f}"
                          for m in model_names if per_fold[m]['boundary_onestep']))

    def _agg(vals):
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        return {"median": float(np.median(a)) if a.size else float("nan"),
                "mean": float(np.mean(a)) if a.size else float("nan"),
                "max": float(np.max(a)) if a.size else float("nan"),
                "n": int(a.size)}

    summary = {}
    for m in model_names:
        wg = _agg(per_fold[m]["within_gen_nrmse"])
        bd = _agg(per_fold[m]["boundary_onestep"])
        fr = _agg(per_fold[m]["full_rollout_nrmse"])
        ratio = bd["median"] / wg["median"] if wg["median"] else float("nan")
        summary[m] = {
            "within_gen_nrmse": wg, "boundary_onestep": bd, "full_rollout_nrmse": fr,
            "boundary_over_within_ratio": ratio,
        }
        print(f"  {m:12s}: within-gen nRMSE median={wg['median']:.3f}  "
              f"boundary one-step median={bd['median']:.2f}  "
              f"(boundary/within = {ratio:.1f}x)")

    drop = _agg(boundary_drop)
    print(f"\n  actual mass drop at division: median={drop['median']:.3f} "
          f"(≈0.5 = clean halving), n={drop['n']}")

    out = {
        "n_folds": len(traj_ids),
        "boundary_actual_drop": drop,
        "summary": summary,
        "trace": trace,
        "note": ("boundary_onestep is a TEACHER-FORCED one-step error at the "
                 "division tick (true mother state in, prediction vs halved "
                 "daughter). within_gen_nrmse is the sm-01/sm-03 regime."),
    }
    with open(os.path.join(data_dir, "metrics_through_division.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  saved -> {data_dir}/metrics_through_division.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--rollout-k", type=int, default=16)
    args = ap.parse_args()
    ds, layout, meta = _load(args.data)
    evaluate(ds, layout, meta, args.data, args.hidden, args.epochs, args.lr, args.rollout_k)


if __name__ == "__main__":
    main()
