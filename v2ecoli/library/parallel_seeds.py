"""Run a multiseed job with optional Ray parallelism — one worker per seed.

This is the reusable form of the fan-out proven in the v2ecoli↔vEcoli
comparison (`scripts/_perf_v2_driver_ray.py`, PR #151): independent seeds are
embarrassingly parallel, so each runs in its own Ray worker PROCESS (no GIL
contention), with per-worker BLAS/OpenMP threads capped to `cores // n_seeds`
so the workers don't oversubscribe the machine. Step-matched, this reaches
parity with vEcoli's Nextflow fan-out — see
`docs/perf/v2ecoli-vs-vecoli-performance.md`.

Use it anywhere you have a `for seed in seeds: run_one(seed)` loop over
INDEPENDENT seeds (ensemble runners, sweeps). It is a drop-in:

    from v2ecoli.library.parallel_seeds import run_seeds_parallel

    def run_one(seed):                      # MUST be a top-level function
        composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
        ...                                 # build + run + emit; do worker setup here
        return {"seed": seed, ...}

    result = run_seeds_parallel(range(n_seeds), run_one, mode=parallel_mode)
    per_seed = result.results               # in seed order

`mode` comes from `workspace.yaml` `runtime.parallel` ("ray" | None). It is
always SAFE: if `mode != "ray"` or Ray isn't importable, it runs the seeds
sequentially with identical results — parallelism never changes the science,
only the wall time.

IMPORTANT: `run_one` must be a module-level (importable) function and must do
its own per-worker setup (e.g. ``set_null_emitter_override``), because Ray
pickles it into a fresh worker process. Closures / locals won't serialize.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# BLAS/OpenMP knobs that must be set in the worker env BEFORE numpy imports,
# so N concurrent workers don't each grab every core and thrash.
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class ParallelResult:
    """Outcome of a multiseed run."""
    results: list[Any]                 # per-seed return values, in seed order
    wall_s: float                      # total wall (critical path when parallel)
    mode: str                          # "ray" | "sequential"
    n_seeds: int
    n_threads_per_worker: int | None = None
    detail: dict = field(default_factory=dict)

    @property
    def parallel(self) -> bool:
        return self.mode == "ray"


def _resolve_threads(n_seeds: int, num_threads: int | None) -> int:
    if num_threads and num_threads > 0:
        return int(num_threads)
    return max(1, (os.cpu_count() or 1) // max(1, n_seeds))


# Per-seed RAM reservation so Ray bounds seed concurrency by memory, not just CPUs.
# A single-generation seed resides ~4 GB, but a MULTI-generation lineage grows well
# past that: the full 1000×10 sweep saw _run_seed_worker reach 7–9 GB by generation
# ~3–4 and OOM the nodes (16 seeds × ~8 GB > 120 GB). Reserve 12 GB by default to
# cover a deep lineage (10 generations); tune per run via V2E_RAY_SEED_MEMORY_GB
# (0 disables). Higher reservation → fewer concurrent seeds per node, but no OOM.
_DEFAULT_SEED_MEMORY_GB = 12.0


def _resolve_seed_memory(memory_per_worker: int | None) -> int:
    """Bytes of Ray logical ``memory`` to reserve per seed task (0 = disabled)."""
    if memory_per_worker is not None:
        return max(0, int(memory_per_worker))
    env = os.environ.get("V2E_RAY_SEED_MEMORY_GB")
    gb = float(env) if env not in (None, "") else _DEFAULT_SEED_MEMORY_GB
    return int(gb * 1024**3)


def run_seeds_parallel(
    seeds: Iterable[int],
    run_one: Callable[..., Any],
    *,
    mode: str | None = "ray",
    run_kwargs: dict | None = None,
    num_threads: int | None = None,
    ray_env: dict | None = None,
    memory_per_worker: int | None = None,
) -> ParallelResult:
    """Run ``run_one(seed, **run_kwargs)`` for every seed; fan out across Ray
    workers when ``mode == "ray"``, else run sequentially. Always falls back to
    sequential if Ray is unavailable — so callers can pass the workspace flag
    verbatim.

    Args:
      seeds: iterable of seed ints.
      run_one: top-level function ``run_one(seed, **run_kwargs) -> result`` (does
        its own per-worker setup; must be picklable / importable). NOTE: Ray
        re-imports the module in a fresh worker, so ``run_one`` must NOT rely on
        module globals mutated at runtime — pass everything via ``run_kwargs``
        and use absolute paths.
      mode: "ray" to parallelize, anything else (or None) to run sequentially.
      run_kwargs: extra keyword args passed to every ``run_one`` call
        (serialized per-call by Ray).
      num_threads: BLAS/OpenMP threads per worker; default ``cores // n_seeds``.
      ray_env: extra ``env_vars`` to merge into the Ray worker environment
        (e.g. ``{"PYTHONPATH": ...}``). Thread caps are added automatically.
      memory_per_worker: bytes of Ray logical ``memory`` to reserve per seed task
        so concurrency is bounded by RAM as well as CPUs (prevents OOM when many
        ~4 GB whole-cell seeds share one node). Default: ``V2E_RAY_SEED_MEMORY_GB``
        env or ~6 GiB; pass ``0`` to disable.

    Returns:
      ParallelResult with per-seed ``results`` (seed order) + timing.
    """
    seeds = list(seeds)
    n = len(seeds)
    kw = dict(run_kwargs or {})

    want_ray = str(mode).lower() == "ray"
    ray = None
    if want_ray:
        try:
            import ray as _ray
            ray = _ray
        except ImportError:
            warnings.warn(
                "run_seeds_parallel: runtime.parallel='ray' but `ray` is not "
                "installed (pip install 'v2ecoli[ray]'); running sequentially.",
                stacklevel=2,
            )

    # ---- sequential path (default + safe fallback) ----------------------
    if ray is None or n <= 1:
        t0 = time.time()
        results = [run_one(s, **kw) for s in seeds]
        return ParallelResult(results=results, wall_s=round(time.time() - t0, 2),
                              mode="sequential", n_seeds=n,
                              detail={"reason": "mode!=ray" if not want_ray
                                      else ("ray-unavailable" if ray is None
                                            else "single-seed")})

    # ---- Ray fan-out: one worker process per seed -----------------------
    threads = _resolve_threads(n, num_threads)
    mem_bytes = _resolve_seed_memory(memory_per_worker)
    env_vars = {k: str(threads) for k in _THREAD_ENV_KEYS}
    if ray_env:
        env_vars.update({k: str(v) for k, v in ray_env.items()})

    # Nest-safe: only bootstrap Ray if we're not ALREADY inside a cluster. The
    # compose / Ray-on-Batch entrypoint forms a cluster before running run_pbg,
    # and tearing it down (ray.shutdown below) mid-run killed the workers — the
    # batch came back empty in ~1s. When a cluster already exists we reuse it
    # (fan out across its nodes) and leave it running.
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(ignore_reinit_error=True, log_to_driver=False,
                 runtime_env={"env_vars": env_vars})
    try:
        remote = ray.remote(run_one)
        t0 = time.time()
        # Pass the thread env per task so workers get it whether or not we started
        # Ray (a reused cluster keeps its own top-level runtime_env). `memory`
        # bounds seed concurrency by RAM so N cpus don't schedule N ~4 GB workers
        # and OOM the node (see _resolve_seed_memory).
        opts: dict[str, Any] = {"num_cpus": threads, "runtime_env": {"env_vars": env_vars}}
        if mem_bytes > 0:
            opts["memory"] = mem_bytes
        futures = [remote.options(**opts).remote(s, **kw) for s in seeds]
        results = ray.get(futures)           # critical-path wall = slowest seed
        wall = time.time() - t0
    finally:
        if started_ray:
            ray.shutdown()               # never tear down a cluster we didn't start

    return ParallelResult(
        results=results, wall_s=round(wall, 2), mode="ray", n_seeds=n,
        n_threads_per_worker=threads,
        detail={"cpu_total": os.cpu_count(),
                "mem_per_worker_gb": round(mem_bytes / 1024**3, 2) if mem_bytes else 0},
    )
