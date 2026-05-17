"""Extension of studies/dnaa-01-expression-dynamics/sims/run_baseline.py that
adds a `--dnaa_autorep_multiplier` flag for the joint (TE × fc) sweep.

Why a new file rather than editing run_baseline.py:
  - Keeps the existing dnaa-01 setup untouched.
  - Lets the user review and promote later if desired.

The new flag scales the deltaV entries in delta_prob (sparse matrix) for
all rows where deltaJ == 12 (i.e., autorepression of TUs by DnaA-TF).
fc=0.5 halves the per-bound-DnaA suppression strength; fc=2.0 doubles it.

Writes to:
  studies/dnaa-01-expression-dynamics/runs.db  (same DB as the parent's
  existing sweep — sim names are tagged baseline-te{N}x-fc{F}-seed{S}).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

V2ECOLI_DIR = Path(__file__).resolve().parents[3]  # …/v2ecoli (skip overnight/dnaa-rep/investigations)
sys.path.insert(0, str(V2ECOLI_DIR))

import numpy as np
import dill  # noqa: F401
from process_bigraph import Composite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration_min', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, required=True,
                        help='Sim name (e.g. baseline-te20x-fc0.5-seed0).')
    parser.add_argument('--cache_dir', type=str,
                        default=str(V2ECOLI_DIR / 'out' / 'cache'))
    parser.add_argument('--dnaa_te_multiplier', type=float, default=1.0)
    parser.add_argument('--dnaa_autorep_multiplier', type=float, default=1.0,
                        help='Scale delta_prob.deltaV[deltaJ==12] by this factor. '
                             'fc<1 weakens autorepression; fc>1 strengthens it.')
    parser.add_argument('--heavy_tf_listener', action='store_true', default=True,
                        help='Enabled by default — autorepression evaluation needs it.')
    args = parser.parse_args()

    STUDY_DIR = V2ECOLI_DIR / 'studies' / 'dnaa-01-expression-dynamics'

    from v2ecoli.composites.baseline import baseline
    from v2ecoli.composites._helpers import sqlite_emitter
    from v2ecoli.core import build_core, load_cache_bundle

    core = build_core()
    print(f'[run+fc] core built', flush=True)

    bundle = load_cache_bundle(args.cache_dir)

    if args.heavy_tf_listener:
        tf_cfg = bundle['configs'].get('ecoli-tf-binding')
        if tf_cfg is not None:
            tf_cfg['emit_n_bound_TF_per_TU'] = True
            print('[run+fc] tf_binding.emit_n_bound_TF_per_TU = True', flush=True)

    if args.dnaa_te_multiplier != 1.0:
        pi_cfg = bundle['configs']['ecoli-polypeptide-initiation']
        te = pi_cfg['translation_efficiencies']
        old = float(te[3861])
        te[3861] = old * args.dnaa_te_multiplier
        print(f'[run+fc] dnaA TE: {old:.3e} → {te[3861]:.3e}  (× {args.dnaa_te_multiplier})',
              flush=True)

    if args.dnaa_autorep_multiplier != 1.0:
        tf_cfg = bundle['configs']['ecoli-tf-binding']
        dp = tf_cfg['delta_prob']
        deltaV = np.asarray(dp['deltaV'])
        deltaJ = np.asarray(dp['deltaJ'])
        mask = (deltaJ == 12)
        old_values = deltaV[mask].copy()
        deltaV[mask] = deltaV[mask] * args.dnaa_autorep_multiplier
        dp['deltaV'] = deltaV
        print(f'[run+fc] autorep multiplier × {args.dnaa_autorep_multiplier} applied to '
              f'{mask.sum()} TU entries in delta_prob[:, 12] (DnaA col). '
              f'sum old |deltaV|={np.abs(old_values).sum():.3e}, '
              f'sum new={np.abs(deltaV[mask]).sum():.3e}', flush=True)

    duration_s = args.duration_min * 60.0
    with sqlite_emitter(file_path=str(STUDY_DIR),
                        db_file='runs.db',
                        name=args.name,
                        subsample=1):
        t_build = time.time()
        doc = baseline(core=core, seed=args.seed, cache_dir=args.cache_dir)
        print(f'[run+fc] composite built in {time.time()-t_build:.1f}s', flush=True)
        comp = Composite(doc, core=core)
        print(f'[run+fc] composite instantiated; emitting to {STUDY_DIR}/runs.db',
              flush=True)
        t_sim = time.time()
        comp.update({}, duration_s)
        print(f'[run+fc] sim done: {duration_s:.0f}s simulated in '
              f'{time.time()-t_sim:.1f}s wall', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
