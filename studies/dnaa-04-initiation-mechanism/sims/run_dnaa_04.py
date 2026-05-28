"""dnaa-04 runner — shadow-observer composite for the DnaA-driven initiation trigger.

Builds ``dnaa_04_with_dnaa_initiation_trigger`` which stacks the
full dnaa-02 + dnaa-03 chain and appends a DnaaInitiationMechanism
Step that emits ``listeners.dnaA_initiation.*``. The observer is
non-destructive — it does NOT modify v2ecoli's existing replication
initiation; it runs alongside and produces a comparable signal.

Captures per-tick:
  - dnaA_cycle.*       (inherited from dnaa-02)
  - dnaA_binding.*     (inherited from dnaa-03)
  - dnaA_initiation.*  (new): would_fire, oric_high_obs, atp_fraction_obs,
                       in_refractory, t_since_last_fire_s, cumulative_fires
  - v2ecoli's actual replisome state (n_forks) — for comparing the
    mechanistic trigger against the existing heuristic.

Usage:
    python studies/dnaa-04-initiation-mechanism/sims/run_dnaa_04.py \\
        <duration_s> <interval_s> <out_json> \\
        [--seed N] [--initial-dnaA 500] \\
        [--oric-threshold 0.8] [--atp-threshold 0.3] [--refractory 600]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np

from v2ecoli import build_composite
from v2ecoli.composites._helpers import flush_parquet, parquet_emitter
from pbg_superpowers.runner import pbg_runner

STUDY_SLUG = "dnaa-04-initiation-mechanism"
INVESTIGATION_SLUG = "dnaa-replication"


def _count_replisomes(cell):
    """v2ecoli's existing initiation produces active_replisome instances."""
    unique = cell.get("unique", {})
    rep = unique.get("active_replisome") if isinstance(unique, dict) else None
    if rep is None or not hasattr(rep, "dtype"):
        return 0
    if "_entryState" in (rep.dtype.names or ()):
        return int(rep["_entryState"].sum())
    return 0


def snap(t, cell):
    listeners = cell.get("listeners", {}) or {}
    cyc  = listeners.get("dnaA_cycle") or {}
    bind = listeners.get("dnaA_binding") or {}
    init = listeners.get("dnaA_initiation") or {}

    return {
        "time":                  t,
        "atp_count":             cyc.get("atp_count"),
        "adp_count":             cyc.get("adp_count"),
        "atp_fraction":          cyc.get("atp_fraction"),
        "oric_high_occupied":    bind.get("oric_high_occupied"),
        "chrom_occupied_fraction": bind.get("chrom_occupied_fraction"),
        "would_fire":            init.get("would_fire"),
        "oric_high_obs":         init.get("oric_high_obs"),
        "in_refractory":         init.get("in_refractory"),
        "t_since_last_fire_s":   init.get("t_since_last_fire_s"),
        "cumulative_fires":      init.get("cumulative_fires"),
        # v2ecoli's existing heuristic: count of active forks
        "actual_n_replisomes":   _count_replisomes(cell),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("duration", type=int)
    p.add_argument("interval", type=int)
    p.add_argument("out_path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-dnaA", type=int, default=500,
                   help="Override initial DnaA count per cell (default 500 = "
                        "literature target to test mechanism in isolation of "
                        "dnaa-01's calibration shortfall).")
    p.add_argument("--oric-threshold", type=float, default=0.8)
    p.add_argument("--atp-threshold", type=float, default=0.3)
    p.add_argument("--refractory", type=float, default=600.0,
                   help="SeqA-v0 refractory window (s) after each would_fire.")
    p.add_argument("--cache-dir", default="out/cache")
    p.add_argument("--sim-name", default=None)
    args = p.parse_args()
    sim_name = args.sim_name or (
        f"dnaa04-seed{args.seed}-θoric{args.oric_threshold}-θatp{args.atp_threshold}-"
        f"ref{int(args.refractory)}")
    params = {
        "seed": args.seed,
        "duration_s": args.duration,
        "interval_s": args.interval,
        "initial_dnaA": args.initial_dnaA,
        "oric_threshold": args.oric_threshold,
        "atp_threshold": args.atp_threshold,
        "refractory_seconds": args.refractory,
    }

    with pbg_runner(study=STUDY_SLUG, name=sim_name, params=params) as run:
        with parquet_emitter(
            out_dir=f"studies/{STUDY_SLUG}/parquet-runs",
            experiment_id=run.run_id,
            study_slug=STUDY_SLUG,
            investigation_slug=INVESTIGATION_SLUG,
        ):
            t0 = time.time()
            composite = build_composite(
                "dnaa_04_with_dnaa_initiation_trigger",
                cache_dir=args.cache_dir,
                seed=args.seed,
                initial_dnaA_count_per_cell=args.initial_dnaA,
                oric_high_threshold=args.oric_threshold,
                atp_fraction_threshold=args.atp_threshold,
                refractory_seconds=args.refractory,
            )
            load_time = time.time() - t0

            cell = composite.state["agents"]["0"]
            snapshots = [snap(0, cell)]
            t_run = time.time()
            total = 0
            divided = False
            while total < args.duration:
                chunk = min(args.interval, args.duration - total)
                try:
                    composite.run(chunk)
                except Exception:
                    divided = True
                    break
                total += chunk
                cell = composite.state.get("agents", {}).get("0")
                if cell is None:
                    divided = True
                    break
                snapshots.append(snap(total, cell))
                run.heartbeat(len(snapshots))
            wall_time = time.time() - t_run
            flush_parquet(composite, success=True)
            run.n_steps = len(snapshots)

            # Comparison metrics
            would_fire_events = [s["time"] for s in snapshots if s.get("would_fire") == 1]
            initial_forks = snapshots[0].get("actual_n_replisomes", 0)
            final_forks   = snapshots[-1].get("actual_n_replisomes", 0)
            heuristic_fires = max(0, final_forks - initial_forks)  # crude estimate
            cumulative_would_fires = snapshots[-1].get("cumulative_fires", 0)

            summary = {
                "would_fire_event_times":    would_fire_events,
                "cumulative_would_fires":    cumulative_would_fires,
                "heuristic_fires_estimate":  heuristic_fires,
                "initial_n_replisomes":      initial_forks,
                "final_n_replisomes":        final_forks,
                "duration_min":              args.duration / 60.0,
                "fires_per_min_mechanism":   (cumulative_would_fires /
                                              (args.duration / 60.0)
                                              if args.duration > 0 else 0),
            }

            result = {
                "composite":         "dnaa_04_with_dnaa_initiation_trigger",
                "seed":              args.seed,
                "params":            params,
                "load_time":         load_time,
                "wall_time":         wall_time,
                "sim_time":          total,
                "interval":          args.interval,
                "divided":           divided,
                "run_id":            run.run_id,
                "db_path":           str(run.db_path),
                "snapshots":         snapshots,
                "summary":           summary,
            }
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            with open(args.out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

    print(f"\nWrote {args.out_path}")
    print(f"  sim_time={total}s  wall={wall_time:.1f}s")
    print(f"  cumulative would_fires (mechanism): {summary['cumulative_would_fires']}")
    print(f"  heuristic firings (≈ Δ n_replisomes / 2): {summary['heuristic_fires_estimate']}")
    print(f"  initial n_replisomes: {summary['initial_n_replisomes']}, "
          f"final: {summary['final_n_replisomes']}")
    if would_fire_events:
        print(f"  would_fire times (s): {would_fire_events[:5]}"
              + (" …" if len(would_fire_events) > 5 else ""))


if __name__ == "__main__":
    main()
