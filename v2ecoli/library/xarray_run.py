"""Multi-generation XArray run helper for v2ecoli composites.

Wraps the validated emitter-per-generation pattern (see
``docs/superpowers/notes/2026-05-22-xarray-multigen-emit-validated.py``) so the
three non-obvious gotchas of driving ``XArrayEmitter`` manually are handled
once, in one place:

  1. Buffer must flush before close, else ``flush`` asserts. We force a small
     buffer + ``subsample(1)`` so every update produces an emit-step and a
     flush happens within a few ticks per generation.

  2. Input must EXACTLY match the view subtree. We filter the agent's state
     down to the leaves declared in the view config before each ``update``.

  3. vEcoli daughters can re-use the parent's id at division (e.g. ``00`` ->
     ``00``, ``01``). We detect division by agent-count growth in the colony,
     not by ``followed in agents`` (which keeps being True across some
     divisions). When a division is seen, we close the current emitter and
     open a new one for the next generation, following a daughter.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from v2ecoli.library.xarray_emitter import XArrayEmitter


def view_from_emit_paths(
    emit_paths: list[str], *, default_dtype: str = "<f8",
) -> list[dict]:
    """Build a v2ecoli-shaped XArray view config from a flat list of dotted emit paths.

    For dashboard-driven runs, each study declares its observables as dotted
    paths like ``"listeners.replication_data.number_of_oric"``. The XArray
    view config is nested by root + variable path. This helper builds it.

    Only handles ``listeners.<...>`` paths for v0 (the v2ecoli convention).
    Drops other roots (``bulk``, etc.) with a warning.

    Default dtype is ``<f8`` (float64); pass an explicit ``dtype_overrides``
    map at the call site if int dtypes are needed for specific leaves.
    """
    variables: dict = {}
    for p in emit_paths:
        parts = p.split(".")
        if len(parts) < 2 or parts[0] != "listeners":
            continue
        rest = parts[1:]
        leaf_name = rest[-1]
        cursor = variables
        for k in rest[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[leaf_name] = [{"path": leaf_name, "dtype": default_dtype}]
    if not variables:
        return []
    return [{"root": ("listeners",), "variables": variables}]


def _view_leaves(view: list[dict]) -> list[tuple[tuple[str, ...], str]]:
    """Walk a view config; return [(input_path_under_root, output_name), ...]."""
    leaves: list[tuple[tuple[str, ...], str]] = []

    def walk(node: Any, prefix: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for spec in node:
                leaves.append((prefix, spec["path"]))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, prefix + (k,))

    for entry in view:
        walk(entry.get("variables", {}), ())
    return leaves


def _filter_agent_state(agent_state: dict, view: list[dict]) -> dict:
    """Filter agent state to exactly the leaves declared by the view.

    Returns a dict shaped like ``{<root[0]>: {<root[1]>: ...{<leaf>: value}...}}``
    for each view entry. Unrelated keys (other listeners, bulk, etc.) are dropped.
    """
    out: dict[str, Any] = {}
    for entry in view:
        root: tuple[str, ...] = tuple(entry["root"])
        # Start at the agent and walk down to the root.
        cur: Any = agent_state
        for k in root:
            cur = (cur or {}).get(k, {})
        # Now `cur` is the subtree at the view root. Filter it to declared paths.
        for leaf_path, _out_name in _view_leaves([entry]):
            value: Any = cur
            for k in leaf_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            # Place value into out at full path: root/.../leaf_path
            cursor = out
            for k in root + leaf_path[:-1]:
                cursor = cursor.setdefault(k, {})
            cursor[leaf_path[-1]] = value
    return out


def _build_emitter(
    *,
    core: Any,
    store_path: Path,
    view: list[dict],
    metadata_base: dict,
    generation: int,
    agent_id: str,
    buffer_size: int = 4,
) -> XArrayEmitter:
    md = dict(metadata_base)
    md["agent_id"] = agent_id
    md["generation"] = generation
    cfg = {
        "emit": {"global_time": "node"},
        "out_uri": str(store_path),
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": buffer_size},
        },
        "view": view,
        "writer": {
            "backend": "zarr",
            "store": str(store_path),
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": md,
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }
    return XArrayEmitter(config=cfg, core=core)


def run_multigen_xarray(
    composite: Any,
    *,
    store_path: str | Path,
    view: list[dict],
    metadata_base: dict,
    max_steps: int,
    max_generations: int = 3,
    chunk: int = 60,
    initial_agent_id: str = "0",
    overwrite: bool = True,
    division_detector: Callable[[set[str], set[str]], tuple[bool, str | None]] | None = None,
) -> dict:
    """Run a v2ecoli composite past divisions, swapping XArrayEmitters per generation.

    Args:
      composite: a process-bigraph ``Composite`` with v2ecoli agent state.
      store_path: zarr store path; partitions accumulate by generation.
      view: XArray view config (list of ``{"root": tuple, "variables": dict}``).
      metadata_base: base metadata for the emitter; ``agent_id`` and
        ``generation`` are filled per-generation. Must contain ``experiment_id``,
        ``variant``, ``lineage_seed``, ``time_step``, ``max_duration``.
      max_steps: stop after this many composite ticks.
      max_generations: cap on how many generations to follow.
      chunk: how many ticks between emitter updates.
      initial_agent_id: agent_id to start following.
      overwrite: if True, delete ``store_path`` before starting.
      division_detector: optional ``(prev_ids, curr_ids) -> (divided?, daughter_id|None)``.
        Default: detect division when ``len(curr) > len(prev)`` and pick the
        first new agent_id sorted.

    Returns: ``{"steps": int, "generations": list[int], "store": str}``.
    """
    store_path = Path(store_path)
    if overwrite and store_path.exists():
        shutil.rmtree(store_path)
    core = composite.core

    if division_detector is None:
        def division_detector(prev, curr):
            new = sorted(curr - prev)
            if len(curr) > len(prev) and new:
                return True, new[0]
            return False, None

    followed = initial_agent_id
    gen = 1
    em = _build_emitter(
        core=core, store_path=store_path, view=view,
        metadata_base=metadata_base, generation=gen, agent_id=followed,
    )
    done = 0
    gens_seen = [gen]
    prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    while done < max_steps and gen <= max_generations:
        try:
            composite.run(chunk)
        except Exception as e:
            print(f"[multigen_xarray] composite stopped at {done}s: {str(e)[:80]}")
            break
        done += chunk
        agents = (composite.state or {}).get("agents") or {}
        curr_ids = set(agents.keys())

        if followed in agents:
            payload = _filter_agent_state(agents[followed], view)
            em.update({
                "time": float(done),
                "global_time": float(done),
                "agents": {followed: payload},
            })

        divided, daughter = division_detector(prev_ids, curr_ids)
        if divided and daughter:
            em.close(success=True)
            followed = daughter
            gen += 1
            gens_seen.append(gen)
            if gen > max_generations:
                break
            em = _build_emitter(
                core=core, store_path=store_path, view=view,
                metadata_base=metadata_base, generation=gen, agent_id=followed,
            )

        prev_ids = curr_ids

    try:
        em.close(success=True)
    except Exception as e:
        print(f"[multigen_xarray] close failed: {str(e)[:80]}")

    return {"steps": done, "generations": gens_seen, "store": str(store_path)}
