"""Run mbp-* simulation variants with sqlite_emitter for dashboard tracking.

Produces the FIRST round of captured (dashboard-visible) sims for the
multiscale-bioprocess investigation. Each run is wrapped in
``sqlite_emitter(study_slug=..., investigation_slug='multiscale-bioprocess')``
so the dashboard's Simulations DB tab groups them under this investigation.

Currently runnable variants (those that don't need the upstream
``pbg-bioreactor-transport-fork`` PR for mbp-03's BiRDTransportProcess):

  mbp-01-time-varying-environment:
    static-env-baseline-60min   — baseline_time_varying_env (env_driver_mode=static)
                                  for 60 sim-min. Mirrors the static-env-regression
                                  sim_set entry; regression-guard against
                                  per-cell state perturbation.

  mbp-02-population-aggregation:
    aggregator-cpa1-60min        — baseline_population at cells_per_agent=1.0
    aggregator-cpa1e6-60min      — at 1e6 (scaling-factor sweep)
    aggregator-cpa1e9-60min      — at 1e9 (high-density representative sampling)

  Cross-investigation reference:
    baseline-reference-60min     — unmodified v2ecoli.composites.baseline for
                                   side-by-side comparison with the aggregated
                                   variants above (per-cell observables MUST be
                                   identical at cells_per_agent=1.0 — the
                                   regression-guard claim).

Each sim runs for 60 sim-minutes (= 3600 sim-seconds) — one v2ecoli
doubling, the floor Haochen flagged for meaningful runs.

Usage:
    python scripts/run_mbp_tracked.py [--variant <name>] [--force]

    --variant <name>  run only the named variant (default: run all)
    --force           re-run even if a sim with the same name already exists
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from process_bigraph import Composite

from v2ecoli.composites._helpers import sqlite_emitter
from v2ecoli.core import build_core


INVESTIGATION_SLUG = "multiscale-bioprocess"
DEFAULT_DURATION_SEC = 60 * 60   # 60 sim-min (one doubling)
DEFAULT_INTERVAL_SEC = 60        # chunk size for composite.run()


# Each variant = (sim_name, study_slug, composite_builder, builder_kwargs).
# composite_builder is a callable that takes core + cache_dir and returns
# a process-bigraph document dict.

def _build_baseline(core, cache_dir, **_):
    from v2ecoli.composites.baseline import baseline
    return baseline(core=core, seed=0, cache_dir=cache_dir)

def _build_baseline_population(core, cache_dir, *, cells_per_agent):
    from v2ecoli.composites.baseline_population import baseline_population
    return baseline_population(
        core=core, seed=0, cache_dir=cache_dir,
        cells_per_agent=cells_per_agent,
    )

def _build_baseline_time_varying_env(core, cache_dir, **_):
    from v2ecoli.composites.baseline_time_varying_env import baseline_time_varying_env
    return baseline_time_varying_env(
        core=core, seed=0, cache_dir=cache_dir,
        env_driver_mode="static",
    )


VARIANTS = [
    # (sim_name, study_slug, builder_fn, builder_kwargs)
    (
        "baseline-reference-60min",
        # No mbp-* study slug; this is a CROSS-investigation reference
        # for side-by-side comparison with the aggregated variants.
        "multiscale-bioprocess-reference",
        _build_baseline,
        {},
    ),
    (
        "aggregator-cpa1-60min",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0},
    ),
    (
        "aggregator-cpa1e6-60min",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e6},
    ),
    (
        "aggregator-cpa1e9-60min",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e9},
    ),
    (
        "static-env-baseline-60min",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {},
    ),
]


def _run_one_variant(
    *,
    sim_name: str,
    study_slug: str,
    builder_fn,
    builder_kwargs: dict,
    duration_sec: int,
    interval_sec: int,
    cache_dir: str,
    core,
) -> dict:
    print(f"\n=== {sim_name} ({study_slug}) ===")
    print(f"  duration: {duration_sec}s ({duration_sec/60:.0f} sim-min)")
    print(f"  kwargs:   {builder_kwargs}")

    with sqlite_emitter(
        name=sim_name,
        study_slug=study_slug,
        investigation_slug=INVESTIGATION_SLUG,
    ):
        t_build = time.time()
        doc = builder_fn(core, cache_dir, **builder_kwargs)
        comp = Composite(doc, core=core)
        build_time = time.time() - t_build

        t_run = time.time()
        elapsed = 0
        crashed = False
        crash_msg = ""
        while elapsed < duration_sec:
            chunk = min(interval_sec, duration_sec - elapsed)
            try:
                comp.run(chunk)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                # The canonical sqlite_emitter post-division daughter-write
                # crash (UNIQUE-constraint on (simulation_id, step) — see
                # feedback_multigen_sqlite_daughter_run_crashes.md). The
                # pre-division trajectory IS valid and persisted; we annotate
                # the outcome rather than calling the whole run failed.
                if (
                    "UNIQUE constraint failed: history.simulation_id, history.step"
                    in str(exc)
                ):
                    crash_msg = "stopped at first-division event (known sqlite_emitter daughter-write crash; pre-division data captured)"
                else:
                    crash_msg = msg
                crashed = True
                break
            elapsed += chunk
        wall_time = time.time() - t_run

    # Snapshot a few key state values for the result summary.
    state = comp.state if not crashed else None
    cell_mass = None
    population = None
    if state is not None:
        agent0 = (state.get("agents") or {}).get("0")
        if agent0:
            mass = (agent0.get("listeners") or {}).get("mass") or {}
            cell_mass = mass.get("cell_mass")
        population = state.get("population")

    print(f"  build: {build_time:.1f}s  run: {wall_time:.1f}s  sim: {elapsed}s")
    if crashed:
        print(f"  CRASHED: {crash_msg}")
    else:
        print(f"  cell_mass[0]: {cell_mass!r}")
        if population is not None:
            print(f"  population: {population!r}")

    return {
        "sim_name":      sim_name,
        "study_slug":    study_slug,
        "duration_sec":  duration_sec,
        "elapsed_sec":   elapsed,
        "build_wall_s":  build_time,
        "run_wall_s":    wall_time,
        "crashed":       crashed,
        "crash_msg":     crash_msg,
        "cell_mass_fg":  float(cell_mass) if cell_mass is not None else None,
        "population":    population,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None,
                   help="Run only the named variant (default: all).")
    p.add_argument("--duration-sec", type=int, default=DEFAULT_DURATION_SEC,
                   help=f"Sim duration in seconds (default: {DEFAULT_DURATION_SEC}).")
    p.add_argument("--interval-sec", type=int, default=DEFAULT_INTERVAL_SEC,
                   help=f"Composite-tick chunk size (default: {DEFAULT_INTERVAL_SEC}).")
    p.add_argument("--cache-dir", default="out/cache",
                   help="ParCa cache directory.")
    args = p.parse_args()

    variants = VARIANTS
    if args.variant:
        variants = [v for v in VARIANTS if v[0] == args.variant]
        if not variants:
            available = ", ".join(v[0] for v in VARIANTS)
            sys.exit(f"unknown variant {args.variant!r}; available: {available}")

    print(f"Running {len(variants)} variant(s) under "
          f"investigation={INVESTIGATION_SLUG}")
    print(f"Workspace DB: .pbg/composite-runs.db (dashboard auto-discovers)")

    core = build_core()
    results = []
    t_all = time.time()
    for sim_name, study_slug, builder_fn, builder_kwargs in variants:
        result = _run_one_variant(
            sim_name=sim_name,
            study_slug=study_slug,
            builder_fn=builder_fn,
            builder_kwargs=builder_kwargs,
            duration_sec=args.duration_sec,
            interval_sec=args.interval_sec,
            cache_dir=args.cache_dir,
            core=core,
        )
        results.append(result)
    total_wall = time.time() - t_all

    print(f"\n{'='*60}")
    print(f"Done — {len(results)} variant(s) in {total_wall:.1f}s wall.")
    real_failures = [
        r for r in results
        if r["crashed"]
        and "first-division event" not in r["crash_msg"]
    ]
    division_stops = [
        r for r in results
        if r["crashed"] and "first-division event" in r["crash_msg"]
    ]
    if division_stops:
        print(f"  {len(division_stops)} stopped at first division (data valid up to division):")
        for r in division_stops:
            print(f"    - {r['sim_name']}: sim={r['elapsed_sec']}s")
    if real_failures:
        print(f"  {len(real_failures)} REAL failures:")
        for r in real_failures:
            print(f"    - {r['sim_name']}: {r['crash_msg']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
