"""Phase 3 sprint 3+4: XArray-based likelihood ensemble.

Runs N PDMP+poisson simulations under XArrayEmitter, each writing per-step
likelihood observables to its own zarr store. Then loads all replicates back
into a single xarray.Dataset with dim ``replicate × time × observable`` — the
canonical inference shape that downstream ABC-SMC / SBC drivers consume.

Replaces sprint 3's misstep on SQLiteEmitter. The investigation YAML calls
for XArrayEmitter as the standard for this branch (workspace.yaml's
default_emitter remains parquet for non-PDMP work).

Per-seed output:
  .pbg/runs/pdmp-03-likelihood/seed_<NN>/store.zarr/

Ensemble output (printed + optionally returned):
  - per-time mean and σ across replicates
  - per-replicate totals (sum over time)
  - collector arithmetic check (total ≈ transcript + polypeptide)

Usage::

    .venv/bin/python scripts/phase3_likelihood_xarray_ensemble.py [--n 8] [--duration 60]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from v2ecoli import build_composite
from v2ecoli.library.xarray_emitter import XArrayEmitter
from v2ecoli.library.xarray_run import (
    view_from_emit_paths,
    filter_view_to_existing_leaves,
    extract_output_metadata_from_state,
    _filter_agent_state,
)


CACHE_DIR = "out/cache"
OUT_ROOT = Path(".pbg/runs/pdmp-03-likelihood")

EMIT_PATHS = [
    # Collector outputs — sprint 2 LikelihoodCollector exposes both
    # per-process likelihoods AND their sum, so we DON'T duplicate by
    # also requesting `listeners.rnap_data.log_likelihood` /
    # `listeners.ribosome_data.log_likelihood`. view_from_emit_paths
    # collapses by leaf NAME, which would alias both per-process
    # log_likelihoods to the same `log_likelihood` zarr leaf.
    "listeners.likelihood.transcript_init",
    "listeners.likelihood.polypeptide_init",
    "listeners.likelihood.total",
    # Reference observables for cross-comparison with Phase 2's
    # cell_mass-based ensemble validation.
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
]


def run_one(seed: int, duration_s: int, chunk: int = 30,
            transcript_scale: float = 1.0,
            out_root: Path | None = None) -> dict:
    root = out_root or OUT_ROOT
    out_dir = root / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "store.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    t0 = time.time()
    composite = build_composite(
        "millard_pdmp_baseline",
        cache_dir=CACHE_DIR,
        seed=seed,
        with_ref_growth=True,
        ref_growth_flux_source="consumption_matched",
        transcript_initiation_mode="poisson",
        polypeptide_initiation_mode="poisson",
        transcript_init_prob_scale=transcript_scale,
    )
    view = view_from_emit_paths(EMIT_PATHS, include_vectors=False)
    metadata_base = {
        "experiment_id": f"pdmp-03-scale{transcript_scale:.3f}-seed{seed:02d}",
        "variant": 0,
        "lineage_seed": seed,
        "time_step": 1.0,
        "max_duration": float(duration_s),
        "agent_id": "0",
        "generation": 1,
    }

    # Warm-up tick so listeners materialize.
    try:
        composite.run(1)
    except Exception as e:
        print(f"  seed={seed:02d}: warmup FAILED — {e}")
        return {"seed": seed, "error": str(e)}

    state_after_warmup = composite.state or {}
    filtered_view = filter_view_to_existing_leaves(state_after_warmup, view)
    if not filtered_view:
        return {"seed": seed,
                "error": "no view leaves remained after warmup filter"}
    output_metadata = extract_output_metadata_from_state(
        state_after_warmup, filtered_view)

    em = XArrayEmitter(config={
        "emit": {"global_time": "node"},
        "out_uri": str(store_path),
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            # Min buffer size accepted by XarrayTransducer is 3
            # (validate_config rejects buf_size <= 2). Chunk=1 below
            # produces enough per-tick emits that the buffer fills
            # cleanly and only the trailing partial — handled by
            # explicit close(success=True) — risks loss.
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
    while done < duration_s:
        step = min(chunk, duration_s - done)
        composite.run(step)
        done += step
        agents = (composite.state or {}).get("agents") or {}
        if followed in agents:
            payload = _filter_agent_state(agents[followed], filtered_view)
            em.update({
                "time": float(done),
                "global_time": float(done),
                "agents": {followed: payload},
            })

    # XArrayEmitter min buffer_size is 3 (XarrayTransducer rejects <=2).
    # The trailing 1-2 emits that don't fill a final buffer are NOT
    # flushed by close() — they're silently lost. Pad with up to
    # ``buf_size - 1`` synthetic emits using FORWARD time stamps so the
    # emitter doesn't dedupe them against the last real emit. The
    # synthetic rows duplicate the last real state — readers should
    # trim to ``time <= duration_s`` to drop them.
    BUF = 3
    leftover = (done - 1) % BUF  # 1 warmup tick not counted in emits
    if leftover != 0:
        agents = (composite.state or {}).get("agents") or {}
        if followed in agents:
            payload = _filter_agent_state(agents[followed], filtered_view)
            for i in range(BUF - leftover):
                synth_t = float(done + i + 1)
                em.update({
                    "time": synth_t,
                    "global_time": synth_t,
                    "agents": {followed: payload},
                })

    try:
        em.close(success=True)
    except Exception as e:
        print(f"  seed={seed:02d}: close failed: {str(e)[:80]}")

    wall = time.time() - t0
    return {
        "seed": seed,
        "wall_s": round(wall, 2),
        "store": str(store_path),
    }


def load_ensemble(n: int, max_time: float | None = None):
    """Open all seed_*/store.zarr; stack along replicate dim.

    XArrayEmitter writes hive groups (``experiment_id=.../variant=0/
    lineage_seed=N/``), and under that ONE child group per observable
    leaf name (``cell_mass``, ``transcript_init``, …). Each leaf
    group's actual array is a single data_var named ``generation=1``
    along dim ``emitstep_gen=1``. We walk to the per-seed lineage
    node, then collect each child's ``generation=1`` array under its
    leaf-group name.
    """
    import xarray as xr

    datasets = []
    for seed in range(n):
        store = OUT_ROOT / f"seed_{seed:02d}" / "store.zarr"
        if not store.is_dir():
            continue
        dt = xr.open_datatree(store, engine="zarr", consolidated=False)
        # Walk down the (single-child) hive chain to the lineage node.
        node = dt
        while node.children and "=" in next(iter(node.children)):
            node = next(iter(node.children.values()))
        # `node` is now the lineage_seed=N group; its children are the
        # per-leaf-name observable groups.
        data_vars = {}
        for leaf_name, leaf_node in node.children.items():
            if "generation=1" in leaf_node.data_vars:
                data_vars[leaf_name] = leaf_node["generation=1"].rename(
                    {"emitstep_gen=1": "time"})
        if not data_vars:
            continue
        # Time coord from the lineage group's `time_gen=1` if present.
        coords = {}
        if "time_gen=1" in node.data_vars:
            coords["time"] = node["time_gen=1"].rename(
                {"emitstep_gen=1": "time"}).values
        ds = xr.Dataset(data_vars=data_vars, coords=coords).expand_dims(
            replicate=[seed])
        datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"No zarr stores found under {OUT_ROOT}")
    out = xr.concat(datasets, dim="replicate")
    if max_time is not None and "time" in out.coords:
        out = out.where(out["time"] <= max_time, drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=30)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 sprint 4 ensemble: N={args.n} × {args.duration}s, "
          f"XArrayEmitter zarr stores under {OUT_ROOT}/")
    t0 = time.time()
    summaries = []
    for seed in range(args.n):
        s = run_one(seed, args.duration, args.chunk)
        summaries.append(s)
        if "error" in s:
            print(f"  seed={seed:02d}: ERROR {s['error']}")
        else:
            print(f"  seed={seed:02d}: wall={s['wall_s']:.1f}s "
                  f"→ {s['store']}", flush=True)
    print(f"\nTotal wall: {time.time() - t0:.1f}s")

    print("\nLoading ensemble back as xarray...")
    ds = load_ensemble(args.n, max_time=float(args.duration))
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  vars: {list(ds.data_vars)}")

    # Inspect each observable. Leaf-group names are: transcript_init,
    # polypeptide_init, total, cell_mass, dry_mass.
    for name in ("transcript_init", "polypeptide_init", "total",
                 "cell_mass", "dry_mass"):
        if name not in ds.data_vars:
            continue
        arr = ds[name]
        per_rep_total = arr.sum(dim="time").values
        print(f"\n  {name}:")
        print(f"    shape: {arr.shape}, dtype: {arr.dtype}")
        if name in ("transcript_init", "polypeptide_init", "total"):
            print(f"    per-replicate Σ_t: μ={per_rep_total.mean():.2f} "
                  f"σ={per_rep_total.std():.2f}")
        else:
            ends = arr.isel(time=-1).values
            print(f"    cm[-1]/dm[-1]: μ={ends.mean():.2f} σ={ends.std():.2f}")
        print(f"    per-time σ across replicates: range "
              f"{arr.std(dim='replicate').min().item():.3f} → "
              f"{arr.std(dim='replicate').max().item():.3f}")

    # Collector arithmetic check.
    if all(k in ds.data_vars for k in ("total", "transcript_init",
                                       "polypeptide_init")):
        diff = float(np.abs(
            (ds["total"] - (ds["transcript_init"]
                            + ds["polypeptide_init"])).values).max())
        print(f"\n  collector arithmetic check: "
              f"max|total − (ti + pi)| across (replicate × time) = {diff:.6f}")
    print("\nEnsemble persisted ✓ — query as `xr.open_zarr(<store>)`.")


if __name__ == "__main__":
    main()
