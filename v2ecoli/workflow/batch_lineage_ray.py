"""Process-bigraph-native multiseed batch dispatch — item 101.

``batch_baseline``/``BatchBaselineRunner`` already wires N seed-lineages together, but does it
by calling Ray's raw Python API directly from inside one opaque Step (``run_workflow``) — the
seed-level fan-out is invisible to process-bigraph's own object model, structurally identical to
chain-dispatch's own external-orchestration defect, one layer down (see backlog item 98's own
memory record for the full trace).

This module builds the same seeds-x-generations shape a different way: N real ``LineageProcess``
nodes, each a genuine node in the composite's own state tree, each addressed via process-bigraph's
real ``ray:`` protocol (the same mechanism the colony composite already proves works across real
physical AWS nodes — see ``v2ecoli/colony.py``). ``LineageProcess`` itself needs no changes: it
already runs a lineage's own generation-to-generation progression using real structural deltas
(``_add``/``_remove`` at each division) — that part was already pbg-native. What's new here is
composing many of them, natively, instead of hiding the fan-out inside a Step's own Python loop.

Pool sizing is NOT left to the ``ray:`` protocol's own bare default. Empirically confirmed
(2026-09-01, local test, ``process_bigraph.protocols.ray``): the actor pool defaults to
``os.cpu_count()`` evaluated on whichever node the driver runs on -- on this ecosystem's AWS Batch
MNP topology that is the HEAD node specifically (which itself runs with ``--num-cpus=0`` in Ray's
own resource accounting, so it never RUNS an actor, but the pool-size CALCULATION still uses its
real hardware core count -- a number with zero relationship to how many workers/cores exist
elsewhere in the cluster). A real local test (8 actors, 3s each) measured pool_size=2 -> 80.75s
wall vs. pool_size=8 -> 9.09s wall, a ~9x difference from that one setting. This is the exact
mechanism that produces a "whole seeds x generations matrix running one square at a time"
bottleneck if left unsized.

``n_workers`` therefore defaults to ``None``, NOT a concrete small int (a real item-101 incident,
2026-09-01: an earlier concrete default of ``2`` silently shadowed the cluster-derived value below,
since ``RayProtocolRuntime`` only reads its env var when given ``None`` explicitly). With ``None``,
``prewarm_lineage_pool`` passes it straight through to ``get_or_create_runtime``, which falls
through to the ``RAY_SHARDS_DEFAULT`` env var -- viva-api's own dispatch code already computes this
correctly from real per-node vCPUs x real node count for every multi-node dispatch. Override
``n_workers`` explicitly only when deliberately capping concurrency below the cluster's real
capacity (e.g. fewer lineages than available shards).
"""

from __future__ import annotations

from typing import Any


def register_ray_lineage(core: Any) -> Any:
    """Register ``LineageProcess`` for the ``ray:`` address protocol and register the
    protocol's own types on ``core``. Mirrors ``v2ecoli/colony.py::make_colony``'s own
    ``transport == 'ray'`` branch -- same registration calls, applied to ``LineageProcess``
    instead of ``EcoliWCM``.

    Returns ``core`` (mutated in place; returned for chaining, matching the rest of this
    module's own functions).
    """
    from process_bigraph.protocols import ray as ray_protocol
    from v2ecoli.workflow.lineage import LineageProcess

    ray_protocol.register_types(core)
    ray_protocol.register_process_class("LineageProcess", LineageProcess)
    return core


def prewarm_lineage_pool(core: Any, n_workers: int | None) -> Any:
    """Size the ``ray:`` actor pool BEFORE any ``ray:LineageProcess`` address is resolved.

    ``RayProtocolRuntime`` sizes its pool for a (class_name, config) key on FIRST creation only
    -- "``n_shards_default`` ... honored only on creation" (``process_bigraph/protocols/ray.py``
    docstring). ``Composite.__init__`` resolves every ``ray:`` address as it builds the state
    tree, which would otherwise create the runtime with the protocol's own default pool size
    (``os.cpu_count()``) before this module gets a chance to size it correctly. Calling this
    FIRST, with the real target concurrency, wins that race deliberately.

    ``n_workers=None`` (the recommended default -- see this module's own docstring) passes
    straight through to ``get_or_create_runtime``, which falls through to the cluster-derived
    ``RAY_SHARDS_DEFAULT`` env var. Pass a concrete int only to deliberately override that.
    """
    from process_bigraph.protocols.ray import get_or_create_runtime

    if n_workers is not None and n_workers < 1:
        raise ValueError(f"prewarm_lineage_pool: n_workers must be >= 1, got {n_workers}")
    get_or_create_runtime(core, n_shards_default=n_workers)
    return core


def build_lineage_ray_batch_document(
    *,
    n_seeds: int,
    n_generations: int,
    base_seed: int = 0,
    cache_dir: str = "out/cache",
    out_dir: str = "",
    experiment_id: str = "batch_lineage_ray",
    emitter: str = "both",
    max_duration_per_gen: float = 3600.0,
    time_step: float = 1.0,
    media: str = "minimal",
    variants: dict | None = None,
    injected_processes: dict | None = None,
    config_overrides: dict | None = None,
    emitter_arg: dict | None = None,
) -> dict:
    """Build a document with N real ``ray:LineageProcess`` nodes, one per seed.

    Unlike ``_build_batch_document`` (``v2ecoli/composites/ecoli_baseline.py``), which returns a
    single-Step document whose ``BatchBaselineRunner`` fans seeds out internally at run time, this
    returns N real state-tree nodes -- the seeds x generations shape is represented in the
    document itself, not hidden inside one Step's own Python loop.

    Each lineage's ``interval`` is set to ``max_duration_per_gen``. ``Composite.run(interval)``
    takes TOTAL SIMULATED TIME to advance, not a tick/step count (confirmed directly from its own
    source, 2026-09-01 -- an easy, real mistake to make from the CLI's own ``-n/--steps`` naming
    in ``run_pbg.py``, which is a different, unrelated entrypoint). The caller is expected to call
    ``composite.run(n_generations * max_duration_per_gen)`` -- enough total simulated time for
    each lineage to reach its own division point (or its own per-generation timeout) once per
    generation, matching ``LineageProcess.update()``'s own one-call-per-generation contract.

    Per-seed lineage_seed follows ``base_seed + i`` (mirrors ``BatchBaselineRunner``'s own
    ``seeds = list(range(base_seed, base_seed + n_seeds))`` convention, so results stay directly
    comparable against the existing mechanism).
    """
    if n_seeds < 1:
        raise ValueError(f"build_lineage_ray_batch_document: n_seeds must be >= 1, got {n_seeds}")
    if n_generations < 1:
        raise ValueError(
            f"build_lineage_ray_batch_document: n_generations must be >= 1, got {n_generations}"
        )

    from v2ecoli.steps.batch_baseline_runner import resolve_out_dir

    resolved_out_dir = resolve_out_dir(out_dir)

    state: dict[str, Any] = {"lineages": {}}
    for i in range(n_seeds):
        seed = base_seed + i
        node_name = f"lineage_{seed:04d}"
        config: dict[str, Any] = {
            "cache_dir": cache_dir,
            "seed": seed,
            "lineage_seed": seed,
            "generations": int(n_generations),
            "single_daughters": True,
            "experiment_id": experiment_id,
            "out_dir": resolved_out_dir,
            "max_duration_per_gen": float(max_duration_per_gen),
            "time_step": float(time_step),
            "media": media,
            "emitter": emitter,
        }
        if variants:
            # LineageProcess itself has no variant concept (that's applied one layer up in
            # BatchBaselineRunner today, via _apply_config_variant before dispatch) -- fold any
            # caller-supplied override in via config_overrides for now. A real variant sweep
            # across ray:-distributed lineages is real, separate, not-yet-scoped work.
            config["config_overrides"] = {**(config_overrides or {}), **variants}
        elif config_overrides:
            config["config_overrides"] = dict(config_overrides)
        if injected_processes:
            config["injected_processes"] = dict(injected_processes)
        if emitter_arg:
            config["emitter_arg"] = dict(emitter_arg)

        state[node_name] = {
            "_type": "process",
            "address": "ray:LineageProcess",
            "config": config,
            "interval": float(max_duration_per_gen),
            "inputs": {},
            "outputs": {
                "summary": ["lineages", node_name, "summary"],
                "complete": ["lineages", node_name, "complete"],
            },
        }

    return {"state": state}


def build_lineage_ray_composite(
    *,
    n_seeds: int,
    n_generations: int,
    n_workers: int,
    **document_kwargs: Any,
) -> Any:
    """Build a ready-to-run ``Composite`` wiring ``n_seeds`` real ``ray:LineageProcess`` nodes.

    ``n_workers`` is the real target concurrency (see this module's own docstring on why this is
    required, never defaulted) -- typically the real number of Ray worker slots available across
    the whole MNP cluster's non-head nodes, NOT ``os.cpu_count()`` on whichever single node the
    driver happens to run on.

    Caller is responsible for ``composite.run(n_generations)`` and for tearing down the pool
    afterward (``process_bigraph.protocols.ray.shutdown_all_runtimes()``) -- this function only
    builds, it does not run or clean up, mirroring ``v2ecoli/colony.py::make_colony``'s own
    build-only contract.
    """
    from process_bigraph import Composite
    from v2ecoli.core import build_core

    core = build_core()
    register_ray_lineage(core)
    # MUST happen before Composite() resolves any ray: address below -- see
    # prewarm_lineage_pool's own docstring for why order matters here.
    prewarm_lineage_pool(core, n_workers)

    doc = build_lineage_ray_batch_document(
        n_seeds=n_seeds, n_generations=n_generations, **document_kwargs
    )
    return Composite({**doc, "parallel_processes": True}, core=core)
