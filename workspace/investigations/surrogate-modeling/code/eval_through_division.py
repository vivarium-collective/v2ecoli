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


def _subsample(ds, meta, stride):
    """Return a temporally-subsampled copy of the dataset at dt=``stride``.

    Coarse growth is smooth over seconds (sm-01), so a dt=k emulator loses
    nothing but shrinks each ~2600-step lineage to sm-03 scale (~T/k), keeping
    the batch-size-1 rollout-loss NN tractable. The grid is anchored on the
    division so the cell_mass halving is preserved as exactly ONE dt=k
    transition (from k seconds pre-division to the daughter's first state);
    no within-generation transition straddles the boundary. stride<=1 is a
    no-op passthrough.
    """
    from pbg_torch import TransitionDataset
    if stride is None or stride <= 1:
        return ds, meta
    nt = ds.spec.n_targets
    spans_in = meta["spans_division"]
    Xn, Yn, spans_n, gens_n, tid_n = [], [], [], [], []
    for t in np.unique(ds.traj_id):
        sel = ds.traj_id == t
        Xh, Yh = ds.X[sel], ds.Y[sel]
        seq = np.vstack([Xh[0][None, :], Yh])            # (T+1, n_features)
        sp = spans_in[sel]
        dr = np.where(sp == 1)[0]
        boundary = int(dr[0]) + 1 if dr.size else seq.shape[0]  # seq idx of daughter_first
        back = list(range(boundary - stride, -1, -stride))[::-1]  # gen0 grid, ends at boundary-stride
        fwd = list(range(boundary, seq.shape[0], stride))          # daughter grid, starts at boundary
        kept = back + fwd
        for i in range(len(kept) - 1):
            a, b = kept[i], kept[i + 1]
            Xn.append(seq[a]); Yn.append(seq[b])
            spans_n.append(1 if (a < boundary <= b) else 0)
            gens_n.append(1 if a >= boundary else 0)
            tid_n.append(int(t))
    Xn = np.asarray(Xn); Yn = np.asarray(Yn)
    tid_n = np.asarray(tid_n, dtype=np.int64)
    ds2 = TransitionDataset(X=Xn, Y=Yn, DT=np.full(Xn.shape[0], float(stride)),
                            traj_id=tid_n, spec=ds.spec)
    meta2 = {"spans_division": np.asarray(spans_n, dtype=np.int64),
             "generation": np.asarray(gens_n, dtype=np.int64),
             "traj_id": tid_n}
    return ds2, meta2


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
    metrics = ["wg_onestep", "boundary_onestep", "wg_rollout_nrmse", "full_rollout_nrmse"]
    per_fold = {m: {k: [] for k in metrics} for m in model_names}
    boundary_drop = []
    trace = None  # per-step rollout of one representative fold, for the figure

    span_all = meta["spans_division"]

    for fi, held in enumerate(traj_ids):
        # TRAIN on WITHIN-GENERATION transitions only (exclude the across-
        # division rows): this is the regime sm-00/sm-01 established — a smooth
        # single-generation emulator. The test is whether that emulator holds
        # when APPLIED across the division boundary it never trained on. (A
        # secondary check confirmed that ADDING the 6 halving examples to
        # training does not help and corrupts the within-gen fit — the halving
        # is not predictable from this observable panel.)
        keep = (ds.traj_id != held) & (span_all == 0)
        train_ds = TransitionDataset(X=ds.X[keep], Y=ds.Y[keep], DT=ds.DT[keep],
                                     traj_id=ds.traj_id[keep], spec=ds.spec)
        Xh, Yh, spans, gens, seq = _traj_arrays(ds, meta, held)

        div_rows = np.where(spans == 1)[0]
        div_row = int(div_rows[0]) if div_rows.size else None
        mass_actual = seq[:, mass_col]
        # normalize by the WITHIN-GENERATION mass scale (the signal the emulator
        # actually operates on), so the boundary error is measured in units of
        # ordinary growth variation — not diluted by the halving itself.
        wg_mask = spans == 0
        wg_scale = float(np.std(mass_actual[:-1][wg_mask])) or 1.0

        if div_row is not None:
            m_mother, m_daughter = Xh[div_row, mass_col], Yh[div_row, mass_col]
            boundary_drop.append(abs(m_daughter - m_mother) / (abs(m_mother) or 1.0))

        models = dict(build_baselines(ds.spec))
        for name in models:
            models[name] = models[name].fit(train_ds.X, train_ds.Y)
        models["onestep_nn"], _ = train_surrogate(
            train_ds, hidden=tuple(hidden), epochs=epochs, lr=lr, seed=fi)
        models["rollout_nn"] = train_rollout(
            train_ds, hidden=tuple(hidden), epochs=epochs, lr=lr,
            rollout_k=rollout_k, seed=fi)

        gen0_end = div_row if div_row is not None else seq.shape[0] - 1
        n_full = seq.shape[0] - 1
        wg_rows = np.where(wg_mask)[0]

        for name, model in models.items():
            # (1) TEACHER-FORCED one-step error, within-generation (median over
            #     all within-gen transitions) — like-for-like with the boundary.
            preds = model.predict_next(Xh[wg_rows])[:, mass_col]
            wg_os = np.abs(preds - Yh[wg_rows, mass_col]) / wg_scale
            per_fold[name]["wg_onestep"].append(float(np.median(wg_os)))
            # (2) TEACHER-FORCED one-step error AT the division tick.
            if div_row is not None:
                pred_b = model.predict_next(Xh[div_row][None, :])[0, mass_col]
                per_fold[name]["boundary_onestep"].append(
                    float(abs(pred_b - Yh[div_row, mass_col]) / wg_scale))
            # (3) autoregressive rollout nRMSE, within generation 0 only.
            wg_roll = _rollout(model, seq[0], gen0_end)[:, mass_col]
            per_fold[name]["wg_rollout_nrmse"].append(
                float(np.sqrt(np.nanmean((wg_roll - mass_actual[:gen0_end + 1]) ** 2)) / wg_scale))
            # (4) autoregressive rollout nRMSE through the division.
            f_roll = _rollout(model, seq[0], n_full)[:, mass_col]
            per_fold[name]["full_rollout_nrmse"].append(
                float(np.sqrt(np.nanmean((f_roll - mass_actual) ** 2)) / wg_scale))

        if trace is None and div_row is not None:
            trace = {
                "held_traj": int(held), "div_index": int(div_row),
                "actual_mass": mass_actual.tolist(),
                "models": {name: _rollout(model, seq[0], n_full)[:, mass_col].tolist()
                           for name, model in models.items()},
            }

        print(f"  fold {fi} (held {held}): "
              + ", ".join(f"{m} wg1={per_fold[m]['wg_onestep'][-1]:.4f}"
                          f"/bd1={per_fold[m]['boundary_onestep'][-1]:.2f}"
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
        agg = {k: _agg(per_fold[m][k]) for k in metrics}
        wg1, bd1 = agg["wg_onestep"]["median"], agg["boundary_onestep"]["median"]
        ratio = bd1 / wg1 if wg1 else float("nan")
        agg["boundary_over_within_onestep_ratio"] = ratio
        summary[m] = agg
        print(f"  {m:12s}: within-gen one-step nRMSE median={wg1:.4f}  "
              f"boundary one-step median={bd1:.2f}  (boundary/within = {ratio:.0f}x)  "
              f"| wg-rollout={agg['wg_rollout_nrmse']['median']:.3f} "
              f"full-rollout={agg['full_rollout_nrmse']['median']:.3f}")

    drop = _agg(boundary_drop)
    print(f"\n  actual mass drop at division: median={drop['median']:.3f} "
          f"(≈0.5 = clean halving), n={drop['n']}")

    out = {
        "n_folds": len(traj_ids),
        "trained_on": "within-generation transitions only (across-division rows excluded)",
        "normalization": "errors normalized by within-generation cell_mass std (wg_scale)",
        "boundary_actual_drop": drop,
        "summary": summary,
        "trace": trace,
        "note": ("wg_onestep and boundary_onestep are both TEACHER-FORCED "
                 "one-step errors (true state in), so their ratio isolates the "
                 "division discontinuity from rollout compounding. wg_rollout / "
                 "full_rollout are autoregressive."),
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
    ap.add_argument("--stride", type=int, default=8,
                    help="temporal subsample (dt, seconds) for emulator train/eval; "
                         "1 = full dt=1 resolution")
    args = ap.parse_args()
    ds, layout, meta = _load(args.data)
    ds, meta = _subsample(ds, meta, args.stride)
    print(f"  dt={args.stride}s: {ds.X.shape[0]} transitions across "
          f"{len(np.unique(ds.traj_id))} lineages "
          f"({int(np.sum(meta['spans_division']))} across-division)")
    evaluate(ds, layout, meta, args.data, args.hidden, args.epochs, args.lr, args.rollout_k)


if __name__ == "__main__":
    main()
