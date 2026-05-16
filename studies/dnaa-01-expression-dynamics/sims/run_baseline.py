"""Run one v2ecoli baseline simulation and emit per-step history to SQLite.

Usage:
    python studies/dnaa-01-expression-dynamics/sims/run_baseline.py \
        [--duration_min 60] [--seed 0] [--name baseline-steady-state]

Writes:
    studies/dnaa-01-expression-dynamics/runs.db    (SQLite, append-safe)

The DB is the source-of-truth that conftest.py reads via the
``baseline_history`` fixture. Multiple invocations append rows under
distinct ``simulation_id`` UUIDs; the ``simulations`` table records each
run's ``name`` (= the ``simulation_set`` entry name in study.yaml).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import dill  # noqa: F401  (process_bigraph emitters serialize state via dill)
from process_bigraph import Composite


STUDY_DIR = Path(__file__).resolve().parents[1]
V2ECOLI_DIR = STUDY_DIR.parents[1]
sys.path.insert(0, str(V2ECOLI_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration_min', type=float, default=60.0,
                        help='Simulated duration in minutes (default: 60).')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for stochastic initialisation.')
    parser.add_argument('--name', type=str, default='baseline-steady-state',
                        help='Run name persisted to simulations.name.')
    parser.add_argument('--cache_dir', type=str,
                        default=str(V2ECOLI_DIR / 'out' / 'cache'),
                        help='ParCa cache directory.')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Record every Nth step (default: 1).')
    parser.add_argument('--heavy_tf_listener', action='store_true',
                        help='Flip tf_binding.emit_n_bound_TF_per_TU=True so '
                             'the (n_TU x n_TF) binding matrix is emitted. '
                             '~900KB/tick — needed for autorepression-correlation '
                             'but expensive. Off by default to match the v2ecoli '
                             'default.')
    parser.add_argument('--dnaa_te_multiplier', type=float, default=1.0,
                        help='Multiply dnaA (PD03831[c]) translation_efficiency '
                             'by this factor before composite build. Used to '
                             'test the F-01 calibration hypothesis — default '
                             'is 1.0 (no change).')
    args = parser.parse_args()

    duration_s = args.duration_min * 60.0
    runs_db = STUDY_DIR / 'runs.db'

    from v2ecoli.composites.baseline import baseline
    from v2ecoli.composites._helpers import sqlite_emitter
    from v2ecoli.core import build_core, load_cache_bundle

    core = build_core()
    print(f"[run_baseline] core built", flush=True)

    # Pre-build patches to the cached config. Both flags are no-ops in their
    # default state so a vanilla run is byte-identical to the canonical
    # v2ecoli baseline.
    needs_bundle = args.heavy_tf_listener or args.dnaa_te_multiplier != 1.0
    if needs_bundle:
        bundle = load_cache_bundle(args.cache_dir)
        if args.heavy_tf_listener:
            tf_cfg = bundle['configs'].get('ecoli-tf-binding')
            if tf_cfg is not None:
                tf_cfg['emit_n_bound_TF_per_TU'] = True
                print("[run_baseline] tf_binding.emit_n_bound_TF_per_TU = True (heavy listener)",
                      flush=True)
        if args.dnaa_te_multiplier != 1.0:
            # PD03831[c] is monomer_ids[3861]; translation_efficiencies is
            # aligned with monomer_ids, so the same index applies.
            pi_cfg = bundle['configs'].get('ecoli-polypeptide-initiation')
            if pi_cfg is not None and 'translation_efficiencies' in pi_cfg:
                import numpy as np
                te = pi_cfg['translation_efficiencies']
                old = float(te[3861])
                new = old * args.dnaa_te_multiplier
                te[3861] = new
                print(f"[run_baseline] dnaA TE: {old:.3e} → {new:.3e}  "
                      f"(× {args.dnaa_te_multiplier})", flush=True)

    with sqlite_emitter(file_path=str(STUDY_DIR),
                        db_file='runs.db',
                        name=args.name,
                        subsample=args.subsample):
        t_build = time.time()
        doc = baseline(core=core, seed=args.seed, cache_dir=args.cache_dir)
        print(f"[run_baseline] composite built in {time.time()-t_build:.1f}s",
              flush=True)

        comp = Composite(doc, core=core)
        print(f"[run_baseline] composite instantiated; "
              f"emitting to {runs_db}", flush=True)

        t_sim = time.time()
        comp.update({}, duration_s)
        elapsed = time.time() - t_sim
        print(f"[run_baseline] sim done: {duration_s:.0f}s simulated "
              f"in {elapsed:.1f}s wall", flush=True)

    print(f"[run_baseline] runs.db at {runs_db}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
