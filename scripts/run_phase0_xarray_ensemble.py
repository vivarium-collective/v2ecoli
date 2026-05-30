"""Phase 0 ensemble with FULL trajectory capture via XArrayEmitter.

Successor to scripts/run_phase0_pilot_ensemble.py — that one captured only
endpoint state because composite.run() doesn't auto-attach an emitter. This
one uses v2ecoli.library.xarray_run.run_multigen_xarray (max_generations=1)
to drive an XArrayEmitter per replicate, writing per-step listener
observables to a per-seed zarr store under .pbg/runs/phase0-xarray/seed_NN/.

Output per seed:
  .pbg/runs/phase0-xarray/seed_<NN>/store.zarr/
  .pbg/runs/phase0-xarray/seed_<NN>/summary.json   (endpoint state, same shape as pilot)

Plus an ensemble-level summary.json with per-seed wall + endpoint stats.

Wall time projection: prior pilot was 48s/seed for endpoint-only; with
trajectory capture expect 60-90s/seed (emitter buffer + zarr writes add
~25% overhead). N=64 → ~80 min.

Usage:
    python scripts/run_phase0_xarray_ensemble.py [--n-seeds 64] [--n-steps 600] [--chunk 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from v2ecoli import build_composite
from v2ecoli.library.xarray_run import (
    view_from_emit_paths,
    filter_view_to_existing_leaves,
    extract_output_metadata_from_state,
    _filter_agent_state,
)
from v2ecoli.library.xarray_emitter import XArrayEmitter


CACHE_DIR = "out/cache"
OUT_ROOT = Path(".pbg/runs/phase0-xarray")

# Observables to capture. Listeners-only per v2ecoli convention; covers
# what every Phase 1+ phase needs to validate against.
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


def _extract_endpoint_summary(composite_state: dict, seed: int, wall: float, n_steps: int) -> dict:
    summary: dict = {"seed": seed, "n_steps": n_steps, "wall_seconds": round(wall, 2)}
    try:
        agents = composite_state.get("agents") or {}
        agent_id = next(iter(agents), None)
        agent = agents.get(agent_id, {}) if agent_id else {}
    except Exception:
        agent = {}
    try:
        m = agent.get("listeners", {}).get("mass", {})
        summary["dry_mass_fg"] = float(m.get("dry_mass", float("nan")))
        summary["cell_mass_fg"] = float(m.get("cell_mass", float("nan")))
    except Exception:
        pass
    try:
        bulk = agent.get("bulk")
        if bulk is not None:
            ids = list(bulk["id"]) if hasattr(bulk, "dtype") else [row[0] for row in bulk]
            counts = list(bulk["count"]) if hasattr(bulk, "dtype") else [row[1] for row in bulk]
            for needle in ("ATP[c]", "WATER[c]"):
                try:
                    idx = ids.index(needle)
                    summary[f"{needle}_count"] = int(counts[idx])
                except ValueError:
                    pass
    except Exception:
        pass
    summary["global_time"] = float(composite_state.get("global_time", float("nan")))
    return summary


def run_one(seed: int, n_steps: int, chunk: int) -> dict:
    """Single-generation run with explicit emitter wiring (buffer_size=1
    so every chunk flushes — avoids run_multigen_xarray's trailing-buffer
    drop, which loses data for runs whose update count isn't a multiple
    of the default buffer size of 4)."""
    import shutil
    out_dir = OUT_ROOT / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "store.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    view = view_from_emit_paths(EMIT_PATHS, include_vectors=False)
    metadata_base = {
        "experiment_id": f"phase0-xarray-seed{seed:02d}",
        "variant": 0,  # 0 = baseline-M9-glucose (only condition this slice)
        "lineage_seed": seed,
        "time_step": 1.0,
        "max_duration": float(n_steps),
        "agent_id": "0",
        "generation": 1,
    }

    # Warm-up tick so listeners materialize + we can read coord arrays.
    try:
        composite.run(1)
    except Exception as e:
        print(f"  seed={seed:02d}: warmup FAILED -- {e}")
    state_after_warmup = composite.state or {}
    filtered_view = filter_view_to_existing_leaves(state_after_warmup, view)
    if not filtered_view:
        raise RuntimeError(f"No view leaves remain after filtering against composite state. View: {view}")
    output_metadata = extract_output_metadata_from_state(state_after_warmup, filtered_view)

    # Build emitter with buffer_size=1 so every update flushes.
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
    while done < n_steps:
        step = min(chunk, n_steps - done)
        try:
            composite.run(step)
        except Exception as e:
            print(f"  seed={seed:02d}: composite stopped at {done}s: {str(e)[:80]}")
            break
        done += step
        agents = (composite.state or {}).get("agents") or {}
        if followed in agents:
            payload = _filter_agent_state(agents[followed], filtered_view)
            em.update({
                "time": float(done),
                "global_time": float(done),
                "agents": {followed: payload},
            })

    try:
        em.close(success=True)
    except Exception as e:
        print(f"  seed={seed:02d}: close failed: {str(e)[:80]}")

    wall = time.time() - t0
    summary = _extract_endpoint_summary(composite.state, seed, wall, n_steps)
    summary["xarray_store"] = str(store_path)
    summary["xarray_steps"] = done

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  seed={seed:02d}: wall={wall:6.1f}s  ATP={summary.get('ATP[c]_count', '?')}  "
        f"dry_mass_fg={summary.get('dry_mass_fg', '?')}  "
        f"xarray_steps={summary.get('xarray_steps')}"
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=600)
    p.add_argument("--chunk", type=int, default=60)
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0 XArray ensemble: N={args.n_seeds} seeds x {args.n_steps} steps (chunk={args.chunk})")
    t0 = time.time()
    results = []
    for seed in range(args.n_seeds):
        try:
            results.append(run_one(seed, args.n_steps, args.chunk))
        except Exception as e:
            print(f"  seed={seed:02d}: FAILED -- {type(e).__name__}: {e}")
            results.append({"seed": seed, "error": str(e), "type": type(e).__name__})
    total_wall = time.time() - t0

    successful = [r for r in results if "error" not in r]
    atp_counts = [r.get("ATP[c]_count") for r in successful if "ATP[c]_count" in r]
    mass_vals = [r.get("dry_mass_fg") for r in successful if "dry_mass_fg" in r]

    ensemble = {
        "n_seeds_requested": args.n_seeds,
        "n_seeds_successful": len(successful),
        "n_steps": args.n_steps,
        "chunk": args.chunk,
        "total_wall_seconds": round(total_wall, 2),
        "per_seed": results,
    }
    if atp_counts:
        a = np.array(atp_counts, dtype=float)
        ensemble["ATP[c]_count_stats"] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "cv_pct": float(a.std() / a.mean() * 100) if a.mean() > 0 else None,
        }
    if mass_vals:
        m = np.array(mass_vals, dtype=float)
        ensemble["dry_mass_fg_stats"] = {
            "mean": float(m.mean()),
            "std": float(m.std()),
            "cv_pct": float(m.std() / m.mean() * 100) if m.mean() > 0 else None,
        }

    (OUT_ROOT / "summary.json").write_text(json.dumps(ensemble, indent=2))
    print()
    print(f"Done: {len(successful)}/{args.n_seeds} runs, total wall {total_wall/60:.1f} min")
    if "ATP[c]_count_stats" in ensemble:
        s = ensemble["ATP[c]_count_stats"]
        print(f"ATP[c] across seeds: mean={s['mean']:.2e} std={s['std']:.2e} CV={s['cv_pct']:.2f}%")


if __name__ == "__main__":
    main()
