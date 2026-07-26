"""Phase-2 demonstration: jump-process variance lives at the COUNT level, not
in aggregate mass/pool observables.

The pdmp-02 closeout found the cell_mass trajectory-distribution-match gate
FAILED and concluded cell_mass is the WRONG observable for jump-process
variance -- the consumption_matched homeostat washes per-tick variance out of
aggregate quantities by construction. This script quantifies that: across N
seeds it compares the cross-seed coefficient of variation (CV) of aggregate
observables (cell_mass, dry_mass, total monomer, ATP count) against a DIRECT
jump-event count (cumulative transcription-initiation events, rnap_data.
rna_init_event summed over ticks).

Result (N=4, 250 steps, M9-glucose baseline): aggregates all wash out to
~0.028% CV; cumulative rna_init shows ~0.85% CV -- ~30x higher. This is the
empirical basis for re-anchoring Phase-3 inference on count-level listeners.

Usage:  .venv/bin/python scripts/phase2_count_variance.py [--n-seeds 4] [--n-steps 250]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from v2ecoli import build_composite
from v2ecoli.library.quantity_helpers import fg_magnitude


def _total(x) -> float:
    try:
        return float(np.nansum(np.asarray(x, dtype=float)))
    except Exception:
        return float("nan")


def _atp(agent) -> float:
    bulk = agent.get("bulk")
    try:
        ids = list(bulk["id"])
        return float(bulk["count"][ids.index("ATP[c]")])
    except Exception:
        return float("nan")


def _cv(values) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return 100.0 * a.std() / a.mean() if (a.size and a.mean()) else float("nan")


def main(n_seeds: int, n_steps: int) -> None:
    cm, dm, mono, atp, cum_init = [], [], [], [], []
    for seed in range(n_seeds):
        c = build_composite("ecoli_baseline", cache_dir="out/cache", seed=seed)
        ci = 0.0
        for _ in range(n_steps):
            c.run(1)
            listeners = (((c.state.get("agents") or {}).get("0") or {})
                         .get("listeners") or {})
            rie = (listeners.get("rnap_data") or {}).get("rna_init_event")
            if rie is not None:
                ci += _total(rie)
        agent = (c.state.get("agents") or {}).get("0") or {}
        L = agent.get("listeners") or {}
        cm.append(fg_magnitude((L.get("mass") or {}).get("cell_mass")))
        dm.append(fg_magnitude((L.get("mass") or {}).get("dry_mass")))
        mono.append(_total(L.get("monomer_counts")))
        atp.append(_atp(agent))
        cum_init.append(ci)
        print(f"seed {seed}: cell_mass={cm[-1]:.2f} dry={dm[-1]:.2f} "
              f"monomer={mono[-1]:.4g} ATP={atp[-1]:.0f} cum_rna_init={ci:.0f}",
              flush=True)

    print(f"\nCV across {n_seeds} seeds at {n_steps} steps:")
    print(f"  cell_mass            CV = {_cv(cm):.4f}%   (aggregate -- washed out)")
    print(f"  dry_mass             CV = {_cv(dm):.4f}%   (aggregate)")
    print(f"  total monomer        CV = {_cv(mono):.4f}%   (aggregate)")
    print(f"  ATP[c] count         CV = {_cv(atp):.4f}%   (aggregate)")
    print(f"  cumulative rna_init  CV = {_cv(cum_init):.4f}%   <- jump-event count")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=250)
    args = p.parse_args()
    main(args.n_seeds, args.n_steps)
