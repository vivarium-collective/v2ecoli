"""Run mbp-* simulation variants past first division, tracked in the
workspace-shared sqlite registry.

This is the v2 runner. v1 used ``sqlite_emitter()`` as a context manager
which injects a SQLiteEmitter Step into the composite — that path hits
``UNIQUE constraint failed: history.simulation_id, history.step`` at the
first division (both daughters try to emit at the same step). v2 uses
:func:`v2ecoli.library.sqlite_run.run_multigen_sqlite` with external
emitter management + lineage-following (``single_daughters=True``) so
runs continue past division.

Each run is also tagged with ``study_slug`` + ``investigation_slug`` so
the dashboard's Simulations DB tab groups them under the
``multiscale-bioprocess`` investigation.

Currently runnable variants (those that don't need the upstream
``pbg-bioreactor-transport-fork`` PR for mbp-03's BiRDTransportProcess):

  mbp-01-time-varying-environment:
    static-env-baseline   — baseline_time_varying_env (env_driver_mode=static)
                            single-cell + division, regression-guard

  mbp-02-population-aggregation:
    aggregator-cpa1       — baseline_population at cells_per_agent=1.0
    aggregator-cpa1e6     — at 1e6 (scaling-factor sweep)
    aggregator-cpa1e9     — at 1e9 (high-density representative sampling)

  Cross-investigation reference:
    baseline-reference    — unmodified v2ecoli.composites.baseline

Default duration is 120 sim-min (~2 doublings); --duration-sec overrides.

Usage:
    python scripts/run_mbp_tracked.py [--variant <name>] [--duration-sec N]
                                      [--max-generations N]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli.composites._helpers import (
    _ensure_study_columns,
    _stamp_study_metadata,
)
from v2ecoli.core import build_core
from v2ecoli.library.sqlite_run import run_multigen_sqlite
from v2ecoli.library.parquet_run import run_multigen_parquet


INVESTIGATION_SLUG = "multiscale-bioprocess"
DEFAULT_DURATION_SEC = 120 * 60   # 120 sim-min (~2 doublings)
DEFAULT_MAX_GENERATIONS = 3
DEFAULT_CHUNK = 1   # per-tick emission (~80 ms/tick on this machine)
DEFAULT_EMITTER = "parquet"   # workspace default (see workspace.yaml.runtime.default_emitter)
DB_PATH = REPO_ROOT / ".pbg" / "composite-runs.db"
# Per-study parquet roots: studies/<study_slug>/parquet-runs/<experiment_id>/...
# This is the convention vivarium-dashboard's _latest_parquet_for_study reads
# from (vivarium_dashboard/lib/study_charts.py:_latest_parquet_for_study).
# The cross-investigation reference variant (study_slug not a real study)
# also gets a per-slug directory so the dashboard can still discover it via
# the same code path, even though no study.yaml lives there.
STUDIES_ROOT = REPO_ROOT / "studies"

def _parquet_root_for(study_slug: str) -> Path:
    return STUDIES_ROOT / study_slug / "parquet-runs"

# Agent-rooted emit paths (under agents/<id>/) shared by all variants.
COMMON_AGENT_PATHS = [
    "listeners/mass/cell_mass",
    "listeners/mass/dry_mass",
    "listeners/mass/instantaneous_growth_rate",
    "listeners/monomer_counts/monomerCounts",
    # boundary.external.GLC is per-agent (rooted under agents/<id>/),
    # so it belongs in emit_paths (agent-level), NOT extra_root_paths.
    # Captures the driver-supplied glucose trajectory at the cell
    # boundary for mbp-01's driven-env runs; trivial overhead for
    # other sims (one float per emit).
    "boundary/external/GLC",
]


def _build_baseline(core, cache_dir, *, seed=0):
    from v2ecoli.composites.baseline import baseline
    return baseline(core=core, seed=seed, cache_dir=cache_dir)

def _build_baseline_population(core, cache_dir, *, cells_per_agent, seed=0):
    from v2ecoli.composites.baseline_population import baseline_population
    return baseline_population(
        core=core, seed=seed, cache_dir=cache_dir,
        cells_per_agent=cells_per_agent,
    )

def _build_baseline_time_varying_env(
    core, cache_dir, *,
    seed=0,
    env_driver_mode="static",
    synthetic_trajectory_spec=None,
):
    from v2ecoli.composites.baseline_time_varying_env import baseline_time_varying_env
    return baseline_time_varying_env(
        core=core, seed=seed, cache_dir=cache_dir,
        env_driver_mode=env_driver_mode,
        synthetic_trajectory_spec=synthetic_trajectory_spec or {},
    )


# (sim_name, study_slug, builder_fn, builder_kwargs, extra_root_paths)
VARIANTS = [
    (
        "baseline-reference-multigen",
        # Cross-investigation reference (not a study); kept under a
        # named pseudo-slug so the dashboard groups it visibly with the
        # other mbp comparison entries without polluting any real study.
        "multiscale-bioprocess-reference",
        _build_baseline,
        {},
        [],
    ),
    (
        "aggregator-cpa1-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "aggregator-cpa1e6-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e6},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "aggregator-cpa1e9-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e9},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "static-env-baseline-multigen",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {},
        [],   # static mode: env-driver doesn't write external_concentrations
    ),
]


# Multi-seed sweep over the cpa scaling axis. Chris's 2026-05-28 Pass 2
# §1.a confirmation validated cpa scaling AT seed=0; this sweep extends
# the evidence to seeds {0, 1, 2} so the report can show variance bands
# across stochastic replicates and answer "is single-seed scaling a
# coincidence or a robust property of representative-sampling under the
# 0D well-mixed assumption?"
#
# 3 cpa × 3 seeds = 9 runs. Each run targets ~120 sim-min (2 doublings)
# and writes into studies/mbp-02-population-aggregation/parquet-runs/
# alongside the existing 3 single-seed runs. Sim names follow
# `aggregator-cpa<N>-seed<S>-multigen` for unambiguous artifact grouping.
_MBP_02_MULTISEED_CPA_VALUES = (1.0, 1.0e6, 1.0e9)
_MBP_02_MULTISEED_SEEDS = (0, 1, 2)

for _cpa in _MBP_02_MULTISEED_CPA_VALUES:
    for _seed in _MBP_02_MULTISEED_SEEDS:
        # Skip the (cpa, seed=0) triple since those already live as the
        # single-seed `aggregator-cpa<N>-multigen` runs above — re-running
        # would duplicate without adding signal.
        if _seed == 0:
            continue
        _cpa_label = "1" if _cpa == 1.0 else ("1e6" if _cpa == 1.0e6 else "1e9")
        VARIANTS.append((
            f"aggregator-cpa{_cpa_label}-seed{_seed}-multigen",
            "mbp-02-population-aggregation",
            _build_baseline_population,
            {"cells_per_agent": _cpa, "seed": _seed},
            [
                "population/total_biomass_gDW",
                "population/cell_count",
                "population/biomass_concentration_gL",
                "population/OD600",
            ],
        ))
del _cpa, _seed, _cpa_label  # housekeeping


# mbp-01 driven-env multigen variants — first time the EnvironmentDriver /
# EnvironmentMirror path runs at multigen scale (unblocked by 60cdbd3).
# Chris's 2026-05-28 review didn't see any env-driven trajectory; these
# produce two: linear-decline (glucose drops over the run) and zero-clamp
# (instant deprivation). Each goes into mbp-01's parquet-runs/ and feeds
# the per-study Charts panel.
_MBP_01_DRIVEN_EXTRA_PATHS: list[str] = []  # boundary.external.GLC moved to COMMON_AGENT_PATHS
VARIANTS.extend([
    # Long-duration batch-phase prefix sim (2026-05-29). Runs the cpa=1e9
    # composite at sim_seconds=21600 (6 h, the Beulig batch-phase
    # endpoint) with single_daughters and max_generations bumped to 12
    # via the runner's --max-generations flag (default cap is 3).
    # Produces the data that mbp-05's expectation-setting preliminary
    # chart needs to know whether v2ecoli's per-cell mass accumulates
    # enough over 6 h to materially close the Beulig density gap, or
    # whether even at this duration the residual gap persists.
    (
        "aggregator-cpa1e9-batch-prefix-multigen",
        "mbp-02-population-aggregation",
        _build_baseline_population,
        {"cells_per_agent": 1.0e9},
        [
            "population/total_biomass_gDW",
            "population/cell_count",
            "population/biomass_concentration_gL",
            "population/OD600",
        ],
    ),
    (
        "linear-decline-glc-multigen",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {
            "env_driver_mode": "synthetic_trajectory",
            "synthetic_trajectory_spec": {
                # 5 mM -> 0 over 60 sim-min (~1 doubling). Bare-name
                # convention per chris_feedback_2026_05_28 §11 resolution.
                "GLC": {
                    "kind": "linear_decline",
                    "start_gL": 5.0, "end_gL": 0.0, "duration_min": 60.0,
                },
            },
        },
        _MBP_01_DRIVEN_EXTRA_PATHS,
    ),
    (
        "zero-clamp-glc-multigen",
        "mbp-01-time-varying-environment",
        _build_baseline_time_varying_env,
        {
            "env_driver_mode": "synthetic_trajectory",
            "synthetic_trajectory_spec": {
                "GLC": {"kind": "clamp_to_value", "value_mmolL": 0.0},
            },
        },
        _MBP_01_DRIVEN_EXTRA_PATHS,
    ),
])


def _register_simulation_row(
    db_path: Path, *, simulation_id: str, name: str,
    study_slug: str, investigation_slug: str, started_at: float,
) -> None:
    """Insert a row into the simulations table that the dashboard reads.

    SQLiteEmitter writes history rows but does NOT itself add a
    simulations table row when constructed externally (the in-composite
    Step path does that automatically). We insert it here so the
    dashboard's /api/simulations endpoint surfaces the run.
    """
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        # Schema is created by SQLiteEmitter on first connect; the table
        # always exists after the first em.update(). Use INSERT OR IGNORE
        # so re-runs of the registration step don't conflict on the
        # PRIMARY KEY (simulation_id).
        conn.execute(
            """
            INSERT OR IGNORE INTO simulations
              (simulation_id, name, started_at, study_slug, investigation_slug)
            VALUES (?, ?, ?, ?, ?)
            """,
            (simulation_id, name, iso, study_slug, investigation_slug),
        )
        conn.commit()
    finally:
        conn.close()


def _count_parquet_rows(out_dir: Path, experiment_id: str) -> int:
    """Count rows emitted under one experiment_id by scanning the
    hive-partitioned parquet tree. Cheap polars scan; column not needed."""
    try:
        import polars as pl
        hive_root = out_dir / experiment_id / "history"
        if not hive_root.exists():
            return 0
        # Lazy scan + count; doesn't materialise data.
        return int(pl.scan_parquet(str(hive_root / "**" / "*.pq")).select(
            pl.len()
        ).collect().item())
    except Exception:
        return 0


def _run_one_variant(
    *, sim_name, study_slug, builder_fn, builder_kwargs, extra_root_paths,
    duration_sec, max_generations, chunk, cache_dir, core, emitter,
) -> dict:
    print(f"\n=== {sim_name} ({study_slug}) ===")
    print(f"  emitter: {emitter}")
    print(f"  duration: {duration_sec}s ({duration_sec/60:.0f} sim-min)")
    print(f"  max_generations: {max_generations}")
    print(f"  kwargs: {builder_kwargs}")

    simulation_id = str(uuid.uuid4())

    t_build = time.time()
    doc = builder_fn(core, cache_dir, **builder_kwargs)
    from process_bigraph import Composite
    composite = Composite(doc, core=core)
    build_time = time.time() - t_build

    t_start = time.time()

    if emitter == "sqlite":
        # Workspace-shared sqlite registry; tag for dashboard grouping.
        _ensure_study_columns(str(DB_PATH))
        _register_simulation_row(
            DB_PATH, simulation_id=simulation_id, name=sim_name,
            study_slug=study_slug, investigation_slug=INVESTIGATION_SLUG,
            started_at=t_start,
        )
        _stamp_study_metadata(str(DB_PATH), simulation_id, study_slug, INVESTIGATION_SLUG)

        t_run = time.time()
        result = run_multigen_sqlite(
            composite,
            run_id=simulation_id,
            db_file=str(DB_PATH),
            emit_paths=COMMON_AGENT_PATHS,
            extra_root_paths=extra_root_paths,
            max_steps=duration_sec,
            max_generations=max_generations,
            chunk=chunk,
            single_daughters=True,
            core=core,
        )
        wall_time = time.time() - t_run

        conn = sqlite3.connect(str(DB_PATH))
        n_rows, max_step = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(step), 0) FROM history WHERE simulation_id = ?",
            (simulation_id,),
        ).fetchone()
        conn.close()

        artifact = str(DB_PATH.relative_to(REPO_ROOT))

    elif emitter == "parquet":
        # Workspace-default emitter (workspace.yaml.runtime.default_emitter).
        # Hive-partitioned per experiment_id/variant/lineage_seed/generation/agent_id.
        # Written under studies/<study_slug>/parquet-runs/ so vivarium-dashboard's
        # _latest_parquet_for_study can discover them per-study.
        parquet_root = _parquet_root_for(study_slug)
        parquet_root.mkdir(parents=True, exist_ok=True)

        t_run = time.time()
        result = run_multigen_parquet(
            composite,
            experiment_id=simulation_id,
            out_dir=str(parquet_root),
            emit_paths=COMMON_AGENT_PATHS,
            extra_root_paths=extra_root_paths,
            max_steps=duration_sec,
            max_generations=max_generations,
            chunk=chunk,
            single_daughters=True,
            core=core,
            study_slug=study_slug,
            investigation_slug=INVESTIGATION_SLUG,
        )
        wall_time = time.time() - t_run

        n_rows = _count_parquet_rows(parquet_root, simulation_id)
        max_step = n_rows - 1 if n_rows > 0 else 0
        artifact = str(
            (parquet_root / simulation_id).relative_to(REPO_ROOT)
        )

    else:
        raise ValueError(f"unknown emitter {emitter!r}; expected sqlite|parquet")

    print(f"  build: {build_time:.1f}s  run: {wall_time:.1f}s")
    print(f"  result: {result}")
    print(f"  artifact: {artifact}")
    print(f"  rows: {n_rows}  max_step: {max_step}")

    return {
        "simulation_id":   simulation_id,
        "sim_name":        sim_name,
        "study_slug":      study_slug,
        "emitter":         emitter,
        "artifact":        artifact,
        "duration_sec":    duration_sec,
        "build_wall_s":    build_time,
        "run_wall_s":      wall_time,
        "result_steps":    result.get("steps"),
        "result_gens":     result.get("generations"),
        "n_history_rows":  n_rows,
        "max_step":        max_step,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None,
                   help="Run only the named variant (default: all).")
    p.add_argument("--emitter", choices=["parquet", "sqlite"], default=DEFAULT_EMITTER,
                   help=(f"Emitter to capture history with (default: {DEFAULT_EMITTER}, "
                         "the workspace default per workspace.yaml.runtime.default_emitter)."))
    p.add_argument("--duration-sec", type=int, default=DEFAULT_DURATION_SEC)
    p.add_argument("--max-generations", type=int, default=DEFAULT_MAX_GENERATIONS)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                   help=(f"Composite-tick chunk (default {DEFAULT_CHUNK} → "
                         "per-tick emit; larger = sparser emit, faster runtime)."))
    p.add_argument("--cache-dir", default="out/cache")
    args = p.parse_args()

    variants = VARIANTS
    if args.variant:
        variants = [v for v in VARIANTS if v[0] == args.variant]
        if not variants:
            available = ", ".join(v[0] for v in VARIANTS)
            sys.exit(f"unknown variant {args.variant!r}; available: {available}")

    print(f"Running {len(variants)} variant(s) under investigation={INVESTIGATION_SLUG}")
    if args.emitter == "sqlite":
        print(f"Workspace DB: {DB_PATH.relative_to(REPO_ROOT)}")
    else:
        print(f"Parquet roots: studies/<study_slug>/parquet-runs/<simulation_id>/history/...")
    print(f"Per-variant: emitter={args.emitter}  max_steps={args.duration_sec}s "
          f"({args.duration_sec/60:.0f} min), max_generations={args.max_generations}, "
          f"chunk={args.chunk}")

    core = build_core()
    results = []
    t_all = time.time()
    for sim_name, study_slug, builder_fn, builder_kwargs, extra_root in variants:
        result = _run_one_variant(
            sim_name=sim_name, study_slug=study_slug,
            builder_fn=builder_fn, builder_kwargs=builder_kwargs,
            extra_root_paths=extra_root,
            duration_sec=args.duration_sec,
            max_generations=args.max_generations,
            chunk=args.chunk,
            emitter=args.emitter,
            cache_dir=args.cache_dir, core=core,
        )
        results.append(result)
    total_wall = time.time() - t_all

    print(f"\n{'='*60}")
    print(f"Done — {len(results)} variant(s) in {total_wall:.1f}s wall "
          f"({total_wall/60:.1f} min).")
    for r in results:
        sim_min = r["max_step"] / 60.0
        gens = r["result_gens"]
        print(f"  {r['sim_name']:35s} sim={sim_min:5.1f}min  gens={gens}  rows={r['n_history_rows']}")


if __name__ == "__main__":
    main()
