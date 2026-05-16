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
    args = parser.parse_args()

    duration_s = args.duration_min * 60.0
    runs_db = STUDY_DIR / 'runs.db'

    from v2ecoli.composites.baseline import baseline
    from v2ecoli.composites._helpers import sqlite_emitter
    from v2ecoli.core import build_core, load_cache_bundle

    core = build_core()
    print(f"[run_baseline] core built", flush=True)

    # Heavy-listener opt-in: monkey-patch the cached tf_binding config to flip
    # emit_n_bound_TF_per_TU before baseline() reads the bundle. This lets the
    # autorepression-correlation behavior_test see the per-TU x per-TF matrix
    # without changing the v2ecoli composite default.
    if args.heavy_tf_listener:
        bundle = load_cache_bundle(args.cache_dir)
        tf_cfg = bundle['configs'].get('ecoli-tf-binding')
        if tf_cfg is not None:
            tf_cfg['emit_n_bound_TF_per_TU'] = True
            print("[run_baseline] tf_binding.emit_n_bound_TF_per_TU = True (heavy listener)",
                  flush=True)

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
