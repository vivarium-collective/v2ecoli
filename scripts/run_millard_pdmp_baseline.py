"""Run the millard_pdmp_baseline composite for 600 s with XArrayEmitter capture.

Modeled after scripts/run_phase0_xarray_ensemble.py — single-replicate variant
that drives an XArrayEmitter writing per-tick listener observables to
``.pbg/runs/millard-pdmp-baseline/store.zarr/``.

Goal: demonstrate the Millard-PDMP composite runs end-to-end for 600 s without
crashing. NOT a biological validation pass — W2-vs-Phase-0 ensemble
comparison is out of scope here.

Usage:
    python scripts/run_millard_pdmp_baseline.py [--n-steps 600] [--chunk 60] [--seed 0]
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
from v2ecoli.library.xarray_run import (
    view_from_emit_paths,
    filter_view_to_existing_leaves,
    extract_output_metadata_from_state,
    _filter_agent_state,
)
from v2ecoli.library.xarray_emitter import XArrayEmitter


CACHE_DIR = "out/cache"
OUT_DIR = Path(".pbg/runs/millard-pdmp-baseline")

EMIT_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.water_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.rRna_mass",
    "listeners.mass.mRna_mass",
    "listeners.mass.tRna_mass",
    "listeners.mass.volume",
    "listeners.mass.growth",
    "listeners.mass.instantaneous_growth_rate",
    "listeners.mass.dry_mass_fold_change",
]


def _extract_endpoint_summary(state: dict, n_steps_done: int, wall: float, seed: int) -> dict:
    summary = {"seed": seed, "n_steps": n_steps_done, "wall_seconds": round(wall, 2)}
    agents = (state or {}).get("agents") or {}
    agent_id = next(iter(agents), None)
    agent = agents.get(agent_id, {}) if agent_id else {}
    m = agent.get("listeners", {}).get("mass", {})
    try:
        summary["cell_mass_fg"] = float(m.get("cell_mass", float("nan")))
        summary["dry_mass_fg"] = float(m.get("dry_mass", float("nan")))
    except Exception:
        pass
    summary["global_time"] = float((state or {}).get("global_time", float("nan")))
    shared = agent.get("shared", {})
    summary["central_metabolites_count"] = len(shared.get("central_metabolites", {}) or {})
    summary["bridge_shared_pool_count"] = shared.get("bridge_diagnostics", {}).get("shared_pool_count")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps", type=int, default=600,
                   help="Total simulation seconds (default 600).")
    p.add_argument("--chunk", type=int, default=60,
                   help="Seconds per composite.run() call (default 60).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tick-s", type=float, default=1.0,
                   help="Millard/LQR/bridge update interval (default 1.0).")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store_path = OUT_DIR / "store.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    print(f"Building composite (seed={args.seed}, tick_s={args.tick_s})...")
    t_build = time.time()
    composite = build_composite(
        "millard_pdmp_baseline",
        cache_dir=CACHE_DIR,
        seed=args.seed,
        tick_s=args.tick_s,
    )
    print(f"  built in {time.time() - t_build:.1f}s")

    view = view_from_emit_paths(EMIT_PATHS, include_vectors=False)
    metadata_base = {
        "experiment_id": f"millard-pdmp-baseline-seed{args.seed:02d}",
        "variant": 0,
        "lineage_seed": args.seed,
        "time_step": 1.0,
        "max_duration": float(args.n_steps),
        "agent_id": "0",
        "generation": 1,
    }

    # Warm-up tick so listener vectors materialize for coord-array discovery.
    print("Warm-up tick...")
    t_warm = time.time()
    try:
        composite.run(1)
    except Exception as e:
        print(f"  warm-up FAILED: {e}")
        raise
    print(f"  warm-up in {time.time() - t_warm:.1f}s")

    state_after_warmup = composite.state or {}
    filtered_view = filter_view_to_existing_leaves(state_after_warmup, view)
    if not filtered_view:
        print(f"  WARNING: no view leaves remain after filtering — emitter would crash. "
              f"Falling back to global_time-only capture.")
        filtered_view = []
    output_metadata = (
        extract_output_metadata_from_state(state_after_warmup, filtered_view)
        if filtered_view else {}
    )

    em = None
    if filtered_view:
        em = XArrayEmitter(config={
            "emit": {"global_time": "node"},
            "out_uri": str(store_path),
            "transducer": {
                "predicate": [[{"subsample": {"interval": 1}}]],
                "buffer": {"size": 3},
            },
            "view": filtered_view,
            "writer": {
                "backend": "zarr",
                "store": str(store_path),
                "buffers_per_chunk": 1,
                "backend_config": {"format": 3},
            },
            "metadata": metadata_base,
            "metadata_keys": [],
            "metadata_validators": {},
            "output_metadata": output_metadata or {},
            "debug": False,
        }, core=composite.core)

    followed = "0"
    done = 1
    print(f"Running {args.n_steps - done} more seconds in chunks of {args.chunk}s...")
    t_run = time.time()
    crashed = False
    err_msg = None
    while done < args.n_steps:
        step = min(args.chunk, args.n_steps - done)
        t_chunk = time.time()
        try:
            composite.run(step)
        except Exception as e:
            crashed = True
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  CRASH at t={done}s: {err_msg[:200]}")
            break
        done += step
        print(f"  t={done}s ({time.time() - t_chunk:.1f}s wall)")
        agents = (composite.state or {}).get("agents") or {}
        if followed in agents and em is not None:
            payload = _filter_agent_state(agents[followed], filtered_view)
            try:
                em.update({
                    "time": float(done),
                    "global_time": float(done),
                    "agents": {followed: payload},
                })
            except Exception as e:
                print(f"  emitter update FAILED at t={done}: {str(e)[:200]}")
        if agents.get(followed) is None:
            print(f"  cell divided at t={done}s — stopping")
            break

    wall = time.time() - t_run
    if em is not None:
        try:
            em.close(success=True)
        except Exception as e:
            print(f"  emitter close FAILED: {str(e)[:200]}")

    summary = _extract_endpoint_summary(composite.state, done, wall, args.seed)
    summary["completed"] = not crashed
    summary["error"] = err_msg
    summary["zarr_store"] = str(store_path)
    summary["target_n_steps"] = args.n_steps

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"DONE: completed={not crashed}  wall={wall:.1f}s  "
          f"sim_time={done}s  dry_mass_fg={summary.get('dry_mass_fg')}  "
          f"cell_mass_fg={summary.get('cell_mass_fg')}")
    if crashed:
        print(f"  ERROR: {err_msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
