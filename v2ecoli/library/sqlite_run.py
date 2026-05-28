"""Multi-generation SQLite run helper for v2ecoli composites.

Mirrors :func:`v2ecoli.library.xarray_run.run_multigen_xarray` for the SQLite
emitter branch. Workspace-side (v2ecoli) because the daughter-walk rule
("first sorted new agent_id after parent disappears") and the
``agents/<id>/`` lineage convention are biological/v2ecoli choices, not
framework contracts — keeping them out of vivarium-dashboard avoids the
anti-pattern flagged in dashboard PR #104.

**Why external emitter driving (not in-composite injection)**

The dashboard's default sqlite path injects a SQLiteEmitter into the
composite state with `inputs` wired to `agents/0/listeners/...` — when
the composite divides (parent `'0'` removed, daughters `'00'` + `'01'`
appear), the injected emitter's static paths don't resolve to anything,
so it writes empty rows for the rest of the run. Mirrors the same
problem the xarray runner solves by managing the emitter externally
and feeding it the followed agent's state per-chunk.

For multi-gen, we construct the SQLiteEmitter externally, run the
composite in chunks, extract the *currently-followed* agent's state
after each chunk, and call `emitter.update(...)` directly. On division
the runner picks the first sorted new agent_id and continues — the
emitter keeps writing to the same db under the same simulation_id, with
the changing agent_id encoded in the JSON `state` column.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def _normalize_emit_paths(emit_paths: list[str]) -> list[tuple[str, ...]]:
    """Convert dotted/slash emit paths to relative tuple paths.

    Each path may carry an `agents/<id>/` prefix (the legacy form);
    strip it so the result is rooted at the agent's subtree. Drop empty
    paths. Dedupe.
    """
    out: set[tuple[str, ...]] = set()
    for p in emit_paths or []:
        parts = [x for x in str(p).replace(".", "/").split("/") if x]
        if not parts:
            continue
        if parts[0] == "agents" and len(parts) >= 3:
            parts = parts[2:]
        out.add(tuple(parts))
    return sorted(out)


def _filter_agent_state(agent_state: dict, leaves: list[tuple[str, ...]]) -> dict:
    """Pull declared leaves out of one agent's state into a nested dict.

    Each `leaf` is a tuple path under the agent (e.g.
    `('listeners', 'dnaA_cycle', 'atp_count')`). Missing leaves are left
    out so the SQLiteEmitter's `emit` schema records JSON `null` rather
    than crashing.
    """
    out: dict = {}
    for leaf in leaves:
        if not leaf:
            continue
        value: Any = agent_state
        for k in leaf:
            value = (value or {}).get(k) if isinstance(value, dict) else None
            if value is None:
                break
        if value is None:
            continue
        cursor = out
        for k in leaf[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[leaf[-1]] = value
    return out


def _build_emit_schema(leaves: list[tuple[str, ...]]) -> dict:
    """Build a process-bigraph emit schema mirroring the leaf structure.

    SQLiteEmitter's `config.emit` declares which paths it writes. We use
    `'any'` for the leaf type since the captured values are
    JSON-serialised; the schema's job here is structural, not type-
    checking.

    Conflict handling: when a leaf is a strict prefix of another leaf
    (e.g. `agents/0/listeners/monomer_counts` AND
    `agents/0/listeners/monomer_counts/monomerCounts`), drop the
    shorter one and keep the more specific. Mixing them yields a
    'str does not support item assignment' crash on `cursor.setdefault`
    when the second-pass walks into a key that was already set to 'any'.
    Surfaced 2026-05-25 wiring dnaa-01's `agents/0/listeners/monomer_counts`
    readout alongside an existing `monomer_counts/monomerCounts` test
    measure path.
    """
    leaf_set = {tuple(l) for l in leaves if l}
    # Drop any leaf that is a strict prefix of a longer leaf in the set.
    keep: list[tuple[str, ...]] = []
    for leaf in leaf_set:
        is_prefix_of_other = any(
            other != leaf and len(other) > len(leaf) and other[:len(leaf)] == leaf
            for other in leaf_set
        )
        if not is_prefix_of_other:
            keep.append(leaf)

    schema: dict = {}
    for leaf in keep:
        cursor = schema
        for k in leaf[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[leaf[-1]] = "any"
    return schema


def prune_to_followed_lineage(composite: Any, followed_id: str) -> int:
    """Drop all agents from ``composite.state['agents']`` except ``followed_id``.

    Used by :func:`run_multigen_sqlite` when ``single_daughters=True`` to
    bound peak memory at the one-cell footprint instead of 2^N at
    generation N (v2ecoli's composite is single-cell-by-design but
    follows all daughters in state by default).

    CRITICAL behaviour: process-bigraph's Composite caches
    ``process_paths`` / ``step_paths`` / ``front`` /
    ``_active_protocol_runtimes`` from the most recent
    :meth:`find_instance_paths` call. Simply deleting
    ``state['agents']['01']`` leaves those caches with dangling entries
    that resolve to ``None`` on the next tick — ``composite.run_process``
    then crashes with::

        TypeError: 'NoneType' object is not subscriptable
        at process_bigraph/composite.py:2834
            state_interval = process['interval']

    The fix: call ``composite.find_instance_paths(composite.state)``
    after the agent deletes. That rebuilds ``process_paths`` /
    ``step_paths`` against the pruned state, and also pops stale entries
    from ``self.front`` (composite.py:1789–1791).

    Args:
        composite: process-bigraph Composite. Must expose ``state`` (a dict
            with optional ``"agents"`` key) and ``find_instance_paths(state)``.
        followed_id: agent key to keep. All other agents are deleted.

    Returns:
        Number of sibling agents dropped.

    Bug history: 2026-05-25 multi-gen sqlite reproduction (commit 7f3d0a2)
    captured the dangling-ref crash. Memory note:
    ``multigen-sqlite-daughter-run-crashes``.
    """
    state = composite.state or {}
    agents = state.get("agents") or {}
    to_drop = [aid for aid in list(agents.keys()) if aid != followed_id]
    for aid in to_drop:
        try:
            del agents[aid]
        except KeyError:
            pass
    composite.find_instance_paths(composite.state)
    return len(to_drop)


def _normalize_root_paths(root_paths: list[str]) -> list[tuple[str, ...]]:
    """Like :func:`_normalize_emit_paths` but does NOT strip an
    ``agents/<id>/`` prefix — these paths are rooted at the composite
    state root (e.g. ``population/total_biomass_gDW``,
    ``environment/external_concentrations/GLC[p]``).
    """
    out: set[tuple[str, ...]] = set()
    for p in root_paths or []:
        parts = [x for x in str(p).replace(".", "/").split("/") if x]
        if parts:
            out.add(tuple(parts))
    return sorted(out)


def _filter_root_state(state: dict, leaves: list[tuple[str, ...]]) -> dict:
    """Pull declared leaves out of the composite state root into a nested dict.

    Mirror of :func:`_filter_agent_state` but rooted at the composite
    state (not at one agent). Missing leaves are skipped so the
    SQLiteEmitter records JSON ``null`` rather than crashing.
    """
    out: dict = {}
    for leaf in leaves:
        if not leaf:
            continue
        value: Any = state
        for k in leaf:
            value = (value or {}).get(k) if isinstance(value, dict) else None
            if value is None:
                break
        if value is None:
            continue
        cursor = out
        for k in leaf[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[leaf[-1]] = value
    return out


def _merge_into(target: dict, addition: dict) -> dict:
    """Recursively merge `addition` into `target` (target wins on conflict).

    Used to union the agent payload and the root-paths payload before
    passing to em.update().
    """
    for k, v in addition.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _merge_into(target[k], v)
        elif k not in target:
            target[k] = v
    return target


def run_multigen_sqlite(
    composite: Any,
    *,
    run_id: str,
    db_file: str | Path,
    emit_paths: list[str],
    max_steps: int,
    max_generations: int = 1,
    chunk: int = 100,
    initial_agent_id: str = "0",
    division_detector: Callable[[set[str], set[str]], tuple[bool, str | None]] | None = None,
    core: Any = None,
    single_daughters: bool = False,
    extra_root_paths: list[str] | None = None,
) -> dict:
    """Run a v2ecoli composite past divisions, externally-driven SQLiteEmitter.

    Args:
      composite: process-bigraph ``Composite`` with v2ecoli agent state.
        Should NOT have a SQLiteEmitter already injected — this runner
        owns the emitter lifecycle.
      run_id: simulation_id stamp for runs_meta + history rows.
      db_file: path to the sqlite db. Schema is set up by
        ``vivarium_dashboard.lib.composite_runs.connect``; this runner
        only writes history rows.
      emit_paths: list of dotted/slash paths to capture from the
        followed agent. Optional `agents/<id>/` prefix is stripped.
      max_steps: hard cap on total ticks to run.
      max_generations: cap on how many generations to follow.
        ``1`` (default) = single-generation behaviour, same as legacy
        ``run_with_division``.
      chunk: how many ticks between emitter updates / division checks.
      initial_agent_id: agent_id to start following.
      division_detector: optional ``(prev_ids, curr_ids) -> (divided?, daughter_id|None)``
        override. Default: when the followed agent is absent, pick the
        first sorted new agent_id (matches
        :func:`v2ecoli.library.xarray_run.run_multigen_xarray`'s rule).
      core: process-bigraph core for the SQLiteEmitter. Defaults to
        ``composite.core``.
      single_daughters: if True (default False), after each division
        drop the sibling daughter(s) from ``composite.state['agents']``
        so only the followed lineage continues. Bounds peak memory at
        ~one-cell footprint regardless of ``max_generations``. Without
        this, every cell tracked by the composite ticks independently;
        v2ecoli's composite is single-cell-by-design so after N
        divisions you have 2^N cells in state and the composite runs
        ALL of them every tick — observed at 62 GB RSS during the
        2026-05-24 multi-gen sqlite shakedown. With ``single_daughters=True``
        peak memory tracks one cell, not 2^N. Trade-off: throwing away
        the unfollowed lineage means no cross-lineage comparison; for
        single-cell-trajectory studies that's the desired behaviour.
      extra_root_paths: optional dotted/slash paths rooted at the
        composite state (NOT under any agent) to also emit at each chunk.
        Examples: ``"population/total_biomass_gDW"``,
        ``"environment/external_concentrations/GLC[p]"``. Used by mbp-*
        studies whose composites (baseline_population,
        baseline_time_varying_env) inject top-level data stores alongside
        ``agents``. The captured values appear as additional keys in the
        emitted JSON state column. Missing paths are skipped (no crash).

    Returns: ``{"steps": int, "generations": list[int]}``.

    Notes:
      * Composite that *raises* (not just division-via-state-mutation)
        is terminal regardless of ``max_generations`` — that signal
        means a real error. v2ecoli's ``dnaa_atp_fraction_clamp`` used
        to raise on division (band={} on daughter re-init); fixed
        2026-05-23 so no v2ecoli composite should raise on normal
        division any more.
      * No `gather_emitter_results(composite)` is needed downstream —
        the SQLiteEmitter writes durably to ``db_file`` as we go.
        Caller should query the db directly.
    """
    from process_bigraph.emitter import SQLiteEmitter

    if division_detector is None:
        def division_detector(prev: set[str], curr: set[str]) -> tuple[bool, str | None]:
            new = sorted(curr - prev)
            if len(curr) > len(prev) and new:
                return True, new[0]
            return False, None

    leaves = _normalize_emit_paths(emit_paths)
    root_leaves = _normalize_root_paths(extra_root_paths or [])
    emit_schema = _build_emit_schema(leaves)

    # Merge top-level paths into emit_schema. _build_emit_schema's
    # prefix-conflict dedup is per-tree, so building a separate root
    # schema and merging it in is safer than passing root_leaves through
    # the agent-rooted path that strips prefixes.
    if root_leaves:
        _merge_into(emit_schema, _build_emit_schema(root_leaves))

    db_path = Path(db_file)

    em = SQLiteEmitter(config={
        "file_path": str(db_path.parent),
        "db_file": db_path.name,
        "simulation_id": run_id,
        "emit": emit_schema,
        "subsample": 1,
        "batch_size": 1,
    }, core=core or composite.core)

    import gc

    max_steps = int(max_steps)
    followed = initial_agent_id
    gen = 1
    done = 0
    gens_seen = [gen]
    prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    while done < max_steps:
        n = min(chunk, max_steps - done)
        try:
            composite.run(n)
        except Exception as e:
            print(f"[multigen_sqlite] composite stopped at tick {done}: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            break
        done += n
        agents = (composite.state or {}).get("agents") or {}
        curr_ids = set(agents.keys())

        if followed in agents:
            payload = _filter_agent_state(agents[followed], leaves)
            update_state = {
                "time": float(done),
                "global_time": float(done),
                "agents": {followed: payload},
            }
            if root_leaves:
                _merge_into(
                    update_state,
                    _filter_root_state(composite.state or {}, root_leaves),
                )
            try:
                em.update(update_state)
            except Exception as e:
                print(f"[multigen_sqlite] emit failed at tick {done}: "
                      f"{type(e).__name__}: {str(e)[:120]}")
        else:
            # The followed agent disappeared — division. Pick daughter
            # if we have generations left; else stop.
            if gen >= max_generations:
                break
            divided, daughter = division_detector(prev_ids, curr_ids)
            if not divided or daughter is None:
                remaining = sorted(curr_ids)
                if not remaining:
                    break
                daughter = remaining[0]
            followed = daughter
            gen += 1
            gens_seen.append(gen)
            # MEMORY guard: optional sibling prune. Off by default so
            # the runner mirrors the xarray-runner's "keep everything"
            # semantics; opt in via `single_daughters=True` when memory
            # bounds matter (single-lineage multi-gen studies).
            if single_daughters:
                dropped = prune_to_followed_lineage(composite, followed)
                gc.collect()
                print(f"[multigen_sqlite] gen {gen} → following agent "
                      f"{followed!r} at tick {done} (single_daughters: "
                      f"dropped {dropped} sibling agent(s) + ran gc)")
            else:
                print(f"[multigen_sqlite] gen {gen} → following agent "
                      f"{followed!r} at tick {done}")
            # Emit immediately so the gen handoff has a marker row.
            if followed in agents:
                payload = _filter_agent_state(agents[followed], leaves)
                update_state = {
                    "time": float(done),
                    "global_time": float(done),
                    "agents": {followed: payload},
                }
                if root_leaves:
                    _merge_into(
                        update_state,
                        _filter_root_state(composite.state or {}, root_leaves),
                    )
                try:
                    em.update(update_state)
                except Exception:
                    pass
        prev_ids = curr_ids
        # Light gc per chunk only when single_daughters is on — that's
        # the mode with memory pressure to fight. Cheap insurance
        # against cyclic refs in composite scheduler queues.
        if single_daughters and n >= 50:
            gc.collect()

    return {"steps": done, "generations": gens_seen}
