"""Multi-generation Phase 0 ensemble — XArray emit + Ray parallelism.

PDMP-specific parallel runner. Each seed runs a full v2ecoli baseline lineage
(N generations past division) and emits per-generation listener trajectories to
its own zarr store via ``v2ecoli.library.xarray_run.run_multigen_xarray``
(the XArrayEmitter, zarr-partitioned by generation). Seeds are INDEPENDENT, so
they fan out across Ray workers (one worker process per seed, BLAS/OpenMP
threads balanced to ``cores // n_seeds``) via
``v2ecoli.library.parallel_seeds.run_seeds_parallel`` — wall time collapses to
~one seed's run instead of N×, with identical results.

This is the multi-generation counterpart of ``run_phase0_xarray_ensemble.py``
(single generation) and the XArray/parallel successor of
``run_phase0_multigen.py`` (sqlite, sequential — kept as a fallback).

Output per seed:
  .pbg/runs/phase0-multigen-xarray/seed_<NN>/store.zarr/   (zarr group / generation)
  .pbg/runs/phase0-multigen-xarray/seed_<NN>/summary.json
Plus an ensemble-level summary.json (per-seed wall + generations + parallel mode).

Goal: unblock pdmp-02's multi-gen-inheritance-binomial primary test — variance
of daughter1−daughter2 should be consistent with binomial(mother, 0.5) partition
across many divisions, with the trajectories on the same zarr format every
later phase validates against.

Run from worktree root:
    python scripts/run_phase0_multigen_xarray.py \
        [--n-seeds 4] [--max-steps 2100] [--max-generations 3] \
        [--chunk 60] [--parallel ray|off]

``--parallel`` overrides workspace.yaml ``runtime.parallel`` for this run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli import build_composite
from v2ecoli.composites._helpers import set_null_emitter_override
from v2ecoli.library.parallel_seeds import run_seeds_parallel
from v2ecoli.library.xarray_run import run_multigen_xarray, view_from_emit_paths


CACHE_DIR = "out/cache"
OUT_ROOT = Path(".pbg/runs/phase0-multigen-xarray")

# Listeners-only per v2ecoli convention; scalar mass observables (no vector
# coord arrays needed) — covers what pdmp-02's inheritance/division tests need.
EMIT_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.volume",
    "listeners.mass.growth",
    "listeners.mass.instantaneous_growth_rate",
]


def run_one(seed: int, max_steps: int, max_generations: int, chunk: int) -> dict:
    """Single-seed multi-generation lineage emitted to a per-seed zarr store.

    Module-level + self-contained so Ray can pickle it into a fresh worker:
    builds its own composite, does its own emit wiring, returns a plain dict.
    """
    # Per-worker setup (runs in each fresh Ray process): suppress the baseline
    # composite's generator-declared ParquetEmitter so it does NOT write to the
    # shared .pbg/parquet-runs/default/ path — N parallel workers would collide
    # there (FileExistsError). The composite then attaches a minimal RAMEmitter
    # (global_time only) and WE drive the per-seed XArray store out of band.
    set_null_emitter_override(True)

    out_dir = OUT_ROOT / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "store.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    # Scalar-only view (no vector coord arrays). run_multigen_xarray warms the
    # composite + filters the view to existing leaves itself.
    view = view_from_emit_paths(EMIT_PATHS, include_vectors=False)
    metadata_base = {
        "experiment_id": f"phase0-multigen-xarray-seed{seed:02d}",
        "variant": 0,  # 0 = baseline-M9-glucose
        "lineage_seed": seed,
        "time_step": 1.0,
        "max_duration": float(max_steps),
    }
    try:
        result = run_multigen_xarray(
            composite,
            store_path=store_path,
            view=view,
            metadata_base=metadata_base,
            max_steps=max_steps,
            max_generations=max_generations,
            chunk=chunk,
            initial_agent_id="0",
            overwrite=False,  # we already cleared store_path above
        )
    except Exception as e:  # noqa: BLE001
        print(f"  seed={seed:02d} FAILED: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return {"seed": seed, "error": str(e), "type": type(e).__name__,
                "wall_seconds": round(time.time() - t0, 2)}

    wall = time.time() - t0
    summary = {
        "seed": seed,
        "max_steps": max_steps,
        "max_generations": max_generations,
        "actual_steps": result.get("steps"),
        "actual_generations_seen": result.get("generations", []),
        "wall_seconds": round(wall, 2),
        "xarray_store": str(store_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  seed={seed:02d}: wall={wall:6.1f}s  steps={result.get('steps')}  "
          f"generations={result.get('generations')}  store={store_path}", flush=True)
    return summary


def _parallel_mode(cli_override: str | None) -> str | None:
    """Resolve parallel mode: CLI override wins, else workspace.yaml
    runtime.parallel ('ray' | None)."""
    if cli_override is not None:
        return None if cli_override == "off" else cli_override
    try:
        import yaml
        ws = REPO_ROOT / "workspace.yaml"
        if ws.is_file():
            rt = (yaml.safe_load(ws.read_text(encoding="utf-8")) or {}).get("runtime") or {}
            return rt.get("parallel")
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=2100)
    p.add_argument("--max-generations", type=int, default=3)
    p.add_argument("--chunk", type=int, default=60)
    p.add_argument("--parallel", choices=["ray", "off"], default=None,
                   help="override workspace.yaml runtime.parallel for this run")
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mode = _parallel_mode(args.parallel)
    print(f"Phase 0 multi-gen XArray ensemble: N={args.n_seeds} seeds × "
          f"{args.max_steps} steps × ≤{args.max_generations} generations "
          f"(chunk={args.chunk}, parallel={mode or 'sequential'})")

    # Fan seeds across Ray workers when runtime.parallel=ray (one worker/seed,
    # thread-balanced); sequential otherwise. Results are identical either way.
    run = run_seeds_parallel(
        range(args.n_seeds), run_one, mode=mode,
        run_kwargs={"max_steps": args.max_steps,
                    "max_generations": args.max_generations,
                    "chunk": args.chunk})
    results = run.results
    total = run.wall_s
    if run.parallel:
        print(f"  [ray] {run.n_seeds} seeds, {run.n_threads_per_worker} thread(s)/worker")

    successful = [r for r in results if "error" not in r]
    ensemble = {
        "n_seeds_requested": args.n_seeds,
        "n_seeds_successful": len(successful),
        "max_steps": args.max_steps,
        "max_generations": args.max_generations,
        "total_wall_seconds": round(total, 2),
        "parallel_mode": run.mode,
        "n_threads_per_worker": run.n_threads_per_worker,
        "per_seed": results,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(ensemble, indent=2))
    print(f"\nDone: {len(successful)}/{args.n_seeds} runs, total wall {total/60:.1f} min "
          f"({run.mode})")
    for r in successful:
        print(f"  seed_{r['seed']:02d}: {len(r.get('actual_generations_seen', []))} generations, "
              f"{r.get('actual_steps')} steps")


if __name__ == "__main__":
    main()
