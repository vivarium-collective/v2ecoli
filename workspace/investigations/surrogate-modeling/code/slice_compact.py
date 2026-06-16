"""Slice a broad-panel TransitionDataset down to a compact physiological panel
(growth/mass + chromosome) without re-sampling. Produces a new run dir with its
own transitions.npz + layout.json so train/evaluate run unchanged.

Usage:
    .venv/bin/python slice_compact.py --src ../run --dst ../run_compact \
        --groups mass chromosome
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout, build_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--groups", nargs="+", default=["mass", "chromosome"])
    args = ap.parse_args()

    from pbg_torch import TransitionDataset

    ds = TransitionDataset.load(os.path.join(args.src, "transitions.npz"))
    full = PanelLayout.from_dict(json.load(open(os.path.join(args.src, "layout.json"))))

    # column indices for the requested groups, in panel order
    cols = []
    for g in args.groups:
        s, e = full.group_slices[g]
        cols.extend(range(s, e))
    cols = np.array(cols, dtype=int)

    # rebuild a compact layout with recomputed slices
    labels = [full.labels[i] for i in cols]
    slices, start = {}, 0
    for g in args.groups:
        n = full.group_slices[g][1] - full.group_slices[g][0]
        slices[g] = (start, start + n)
        start += n
    compact = PanelLayout(groups=list(args.groups), labels=labels,
                          group_slices=slices, exchange_keys=full.exchange_keys
                          if "exchange" in args.groups else [])

    spec = build_spec(compact)
    out = TransitionDataset(X=ds.X[:, cols], Y=ds.Y[:, cols], DT=ds.DT,
                            traj_id=ds.traj_id, spec=spec)
    os.makedirs(args.dst, exist_ok=True)
    out.save(os.path.join(args.dst, "transitions.npz"))
    with open(os.path.join(args.dst, "layout.json"), "w") as fh:
        json.dump(compact.to_dict(), fh)
    print(f"Compact panel: {len(cols)} observables {args.groups} "
          f"x {out.n_transitions} transitions -> {args.dst}")


if __name__ == "__main__":
    main()
