"""Multi-generation baseline ensemble via v2ecoli's LineageProcess/
batch_baseline_runner pipeline -- real cell division across generations,
unlike scripts/run_phase0_xarray_ensemble.py's single-generation dispatch
(which silently ignores any requested generation count).

This pipeline (v2ecoli.steps.batch_baseline_runner.dispatch_batch) has never
been dispatched on real GovCloud/Batch/Ray infrastructure before this script
existed -- previously only unit-tested against a fully stubbed run_workflow.

Output is restructured after dispatch_batch() returns into the SAME
seed_NN/store.zarr + seed_NN/summary.json convention
run_phase0_xarray_ensemble.py already produces (rather than changing
LineageProcess's own internal zarr-naming, which many other callers depend
on), so the existing landing/analysis machinery
(vivarium_workbench.lib.remote_run_landing, scripts/run_standalone_analysis.py)
works completely unmodified.

Usage:
    python scripts/run_batch_baseline_ray.py --n-seeds 2 --n-generations 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli.steps.batch_baseline_runner import dispatch_batch

# Same fixed directory simulation_service_ray.py's SIM_OUT_DIR points at and
# the Ray-batch entrypoint syncs to S3 -- matches run_phase0_xarray_ensemble.py
# exactly, so this pipeline's output reaches S3 through the same, already-
# working mechanism regardless of which script actually ran.
CACHE_DIR = str(REPO_ROOT / "out" / "cache")
OUT_ROOT = REPO_ROOT / ".pbg" / "runs" / "phase0-xarray"


def restructure_seed_stores(result: dict, out_root: Path) -> list[dict]:
    """Move each seed's <experiment_id>_v<variant>_s<seed>.zarr (LineageProcess's
    own naming, mirrored by batch_baseline_runner._lineage_store_path) into
    seed_NN/store.zarr + a seed_NN/summary.json."""
    entries = []
    for seed_key, entry in (result.get("seeds") or {}).items():
        seed = int(seed_key)
        seed_dir = out_root / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        store_path = entry.get("store_path")
        if store_path and Path(store_path).is_dir():
            dest = seed_dir / "store.zarr"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(store_path, str(dest))
        summary = {
            "seed": seed,
            "generations_reached": entry.get("generations_reached", 0),
            "complete": bool(entry.get("complete", False)),
        }
        if "error" in entry:
            summary["error"] = entry["error"]
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        entries.append(summary)
    return entries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--n-generations", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-duration", type=float, default=3600.0,
                   help="per-generation sim-time cap in seconds")
    p.add_argument("--experiment-id", default="batch_baseline")
    p.add_argument("--parallel", choices=["ray", "off"], default="ray")
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Batch baseline: N={args.n_seeds} seeds x {args.n_generations} generations "
          f"(parallel={args.parallel})")
    try:
        result = dispatch_batch(
            n_seeds=args.n_seeds,
            n_generations=args.n_generations,
            base_seed=args.base_seed,
            max_duration=args.max_duration,
            cache_dir=CACHE_DIR,
            out_dir=str(OUT_ROOT),
            experiment_id=args.experiment_id,
            # "both" (not "xarray") so the sweep ALSO gets hive-parquet under
            # OUT_ROOT/history/ -- what the DuckDB-based Analysis family
            # (v2ecoli.workflow.analysis.Analysis, e.g. the cd1/ptools suite)
            # reads via analysis_runner.run_analyses(). Without this, an
            # xarray-only sweep has nothing for those analyses to query.
            emitter="both",
            parallel=None if args.parallel == "off" else args.parallel,
            # Post-hoc standalone analysis (scripts/run_standalone_analysis.py)
            # handles analyses separately over the landed S3 output -- skip
            # dispatch_batch's own inline analysis dispatch (this pipeline's
            # local-only flush; the standalone script now runs the same
            # analysis_runner.run_analyses() directly against the S3 sweep).
            analyses="none",
        )
    except Exception as e:  # noqa: BLE001 -- must still report failure, not crash silently
        sys.exit(f"batch dispatch raised {type(e).__name__}: {e}")

    entries = restructure_seed_stores(result, OUT_ROOT)
    successful = [e for e in entries if e.get("complete") and "error" not in e]

    ensemble = {
        "n_seeds_requested": args.n_seeds,
        "n_seeds_successful": len(successful),
        "n_generations_requested": args.n_generations,
        "total_wall_seconds": result.get("wall_s"),
        "per_seed": entries,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(ensemble, indent=2))
    print(f"Done: {len(successful)}/{args.n_seeds} seeds complete "
          f"({args.n_generations} generations requested)")

    if not successful:
        # dispatch_batch/run_workflow never raises and never signals failure via
        # exit code on its own (confirmed: v2ecoli.workflow.run.main() only
        # warnings.warn()s on incomplete results) -- the same class of bug fixed
        # in run_phase0_xarray_ensemble.py. This is the fix for this pipeline.
        sys.exit(f"all {args.n_seeds} seeds failed -- see per_seed errors in summary.json")


if __name__ == "__main__":
    main()
