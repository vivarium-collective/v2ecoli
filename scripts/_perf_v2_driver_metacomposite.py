#!/usr/bin/env python3
"""Bridged-parallel-composite v2ecoli driver — the process-bigraph-native
multiseed variant.

Instead of an external @ray.remote loop (scripts/_perf_v2_driver_ray.py), this
expresses the whole N-seed sweep as ONE process-bigraph composite: each seed is
a `LineageProcess` node (a bridged composite-process — it holds the inner WCM
Composite and exposes only a narrow summary/complete bridge, with its OWN
streaming parquet emitter inside it). With ``--mode ray`` each branch gets a
``ray:LineageProcess`` address and the composite runs with
``parallel_processes=True``, so the engine dispatches the N branch updates to
Ray actors concurrently each tick. Because the bridge in/out is tiny
(``inputs()=={}``), only a trivial payload crosses per tick — the heavy WCM
state stays inside each actor.

``--mode local`` runs the same composite sequentially (the existing
workflow path) — useful to confirm correctness before measuring the parallel
version.

Mirrors the summary.json shape of the other perf drivers for apples-to-apples
comparison against the external fan-out.

  python scripts/_perf_v2_driver_metacomposite.py --n-seeds 2 --generations 2 --mode ray
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# pbg-emitters' editable-install hook is broken in this venv (uv quirk — the
# package imports only via explicit path), so put the checkout on the path for
# the driver AND propagate it to Ray workers via PYTHONPATH (set before ray.init).
PBG_EMITTERS = "/Users/eranagmon/code/pbg-emitters"
if Path(PBG_EMITTERS).is_dir():
    sys.path.insert(0, PBG_EMITTERS)
OUT_ROOT = Path(".pbg/runs/perf-v2-metacomposite")
CACHE_DIR = "out/cache"


def _build_doc(n_seeds: int, generations: int, max_steps: int, mode: str) -> dict:
    """Meta-composite doc: N LineageProcess branches; ray: addresses when parallel."""
    from v2ecoli.workflow.variants import expand_branches  # noqa
    address = "ray:LineageProcess" if mode == "ray" else "local:LineageProcess"
    # One generation of v2ecoli ≈ a few thousand steps; cap per-gen wall by a
    # generous sim-time so generations end at natural division, matching the
    # other drivers' single_daughters lineage.
    branches = {}
    for seed in range(n_seeds):
        key = f"variant=0/seed={seed}"
        branches[key] = {
            "lineage": {
                "_type": "process",
                "address": address,
                "interval": 1.0,
                "config": {
                    "cache_dir": CACHE_DIR, "seed": seed, "lineage_seed": seed,
                    "variant_index": 0, "variant_name": "baseline",
                    "config_overrides": {}, "generations": generations,
                    "single_daughters": True,
                    "experiment_id": f"perf-metacomp-seed{seed:02d}",
                    "out_dir": str(OUT_ROOT / f"seed_{seed:02d}"),
                    "max_duration_per_gen": float(max_steps),
                    "time_step": 1.0, "emitter": "parquet", "emitter_arg": {},
                },
                "inputs": {},
                "outputs": {"summary": ["summary"], "complete": ["complete"]},
            },
            "summary": {}, "complete": False,
        }
    return {"state": {"global_time": 0.0, "branches": branches},
            "skip_initial_steps": True, "sequential_steps": False,
            "parallel_processes": mode == "ray"}


def _all_complete(composite) -> bool:
    branches = composite.state.get("branches", {})
    return bool(branches) and all(b.get("complete") for b in branches.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--mode", choices=["local", "ray"], default="ray")
    a = ap.parse_args()
    if not (REPO / CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found under {REPO}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.workflow.meta_composite import register_workflow_processes

    core = build_core()
    register_workflow_processes(core)               # local:LineageProcess
    if a.mode == "ray":
        # Register the process protocols (adds the `ray:` address handler) and
        # single-thread BLAS in the actors so N branches don't oversubscribe.
        import os
        threads = max(1, (os.cpu_count() or 1) // max(1, a.n_seeds))
        import ray
        _env = {k: str(threads) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
        # Workers need pbg-emitters (broken editable hook) + the repo on PYTHONPATH.
        _env["PYTHONPATH"] = f"{REPO}:{PBG_EMITTERS}:" + os.environ.get("PYTHONPATH", "")
        ray.init(ignore_reinit_error=True, log_to_driver=False,
                 runtime_env={"env_vars": _env})
        from process_bigraph.protocols import register_types as register_protocols
        register_protocols(core)
        # The ray protocol resolves the class by name from this registry.
        from process_bigraph.protocols.ray import register_process_class
        from v2ecoli.workflow.lineage import LineageProcess
        register_process_class("LineageProcess", LineageProcess)
        print(f"meta-composite RAY: {a.n_seeds} branches, {threads} thread(s) each",
              flush=True)
    else:
        print(f"meta-composite LOCAL (sequential): {a.n_seeds} branches", flush=True)

    doc = _build_doc(a.n_seeds, a.generations, a.max_steps, a.mode)
    composite = Composite(doc, core=core)

    t0 = time.time()
    dt, elapsed, cap = 1.0, 0.0, float(a.max_steps * a.generations + 100)
    while not _all_complete(composite) and elapsed < cap:
        composite.run(dt)
        elapsed += dt
    total = time.time() - t0

    # Pull per-branch summaries from the live LineageProcess instances.
    per_seed = []
    for path_tuple, edge in composite.process_paths.items():
        inst = edge.get("instance")
        base = getattr(inst, "instance", inst)   # unwrap ray shadow if present
        if base is not None and hasattr(base, "_summaries"):
            if len(path_tuple) >= 2 and path_tuple[0] == "branches":
                seed = int(path_tuple[1].split("seed=")[-1])
                gens = [s.get("generation") for s in base._summaries]
                per_seed.append({"seed": seed,
                                 "generations_done": len(base._summaries),
                                 "actual_generations_seen": gens})
    per_seed.sort(key=lambda r: r["seed"])
    if a.mode == "ray":
        import ray
        ray.shutdown()

    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "mode": a.mode, "parallel": a.mode == "ray",
        "n_seeds_requested": a.n_seeds, "n_seeds_successful": len(per_seed),
        "generations": a.generations, "max_steps": a.max_steps,
        "single_daughters": True, "sim_time_elapsed": elapsed,
        "total_wall_seconds": round(total, 2), "per_seed": per_seed,
    }, indent=2))
    print(f"Done ({a.mode}): {len(per_seed)}/{a.n_seeds} branches | "
          f"total wall {total/60:.1f} min | sim-time {elapsed:.0f}s", flush=True)
    for r in per_seed:
        print(f"  seed={r['seed']:02d}: generations={r['actual_generations_seen']}",
              flush=True)


if __name__ == "__main__":
    main()
