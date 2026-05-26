"""Minimal repro for the multi-gen sqlite daughter crash.

Builds dnaa_stage1_constitutive on the Stage-1 glycerol cache, runs to
first division (~5000 ticks ≈ 83 min on glycerol), then applies the
same prune that `_prune_to_followed_lineage` does in
`v2ecoli/library/sqlite_run.py`, then attempts `composite.run(1)` to
capture the exception.

Goal: see the actual exception type + traceback so we can fix the prune
or the daughter-state init.

Usage:
    .venv/bin/python scripts/debug_multigen_daughter_crash.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline_recipes import dnaa_stage1_constitutive
    from process_bigraph import Composite

    print("[1/4] building composite on out/cache-stage1-glycerol ...", flush=True)
    core = build_core()
    doc = dnaa_stage1_constitutive(
        core=core, seed=0, cache_dir="out/cache-stage1-glycerol")
    composite = Composite({"state": doc.get("state", doc)}, core=core)
    initial_agents = sorted((composite.state.get("agents") or {}).keys())
    print(f"  initial agents: {initial_agents}", flush=True)

    print("[2/4] running until first division (max 6000 ticks, chunk 100) ...",
          flush=True)
    done = 0
    while done < 6000:
        composite.run(100)
        done += 100
        agents = composite.state.get("agents") or {}
        agent_ids = sorted(agents.keys())
        if set(agent_ids) != set(initial_agents):
            print(f"  DIVISION at tick {done}: {initial_agents} → {agent_ids}",
                  flush=True)
            break
    else:
        print(f"  no division within 6000 ticks; last agents={agent_ids}", flush=True)
        return

    # Daughter selection: smallest agent_id (matches sqlite_run's default).
    daughter = sorted(set(agent_ids) - set(initial_agents))[0]
    print(f"[3/4] pruning to followed lineage = {daughter!r} (mirrors "
          f"_prune_to_followed_lineage)", flush=True)
    agents = composite.state["agents"]
    to_drop = [aid for aid in list(agents.keys()) if aid != daughter]
    for aid in to_drop:
        del agents[aid]
    # APPLY THE FIX from sqlite_run.py: rebuild composite path caches
    # to drop stale refs to the deleted agent.
    print(f"  before find_instance_paths: process_paths has "
          f"{len(composite.process_paths)} entries; front has {len(composite.front)}",
          flush=True)
    composite.find_instance_paths(composite.state)
    print(f"  after  find_instance_paths: process_paths has "
          f"{len(composite.process_paths)} entries; front has {len(composite.front)}",
          flush=True)
    import gc
    gc.collect()
    print(f"  pruned {len(to_drop)} sibling(s); state['agents'] now = "
          f"{sorted(agents.keys())}", flush=True)

    print(f"[4/4] attempting composite.run(1) — expecting the crash ...",
          flush=True)
    try:
        composite.run(1)
    except Exception as e:
        print(f"\n*** CAPTURED EXCEPTION ***", flush=True)
        print(f"  type: {type(e).__name__}", flush=True)
        print(f"  message: {e}", flush=True)
        print(f"\n--- full traceback ---", flush=True)
        traceback.print_exc()
        return
    # If we get here, no crash. Try a few more ticks.
    print(f"  composite.run(1) succeeded! Trying 100 more ticks...", flush=True)
    try:
        composite.run(100)
        print(f"  composite.run(100) also succeeded — bug not reproduced!", flush=True)
    except Exception as e:
        print(f"\n*** CAPTURED EXCEPTION on 100 ticks ***", flush=True)
        print(f"  type: {type(e).__name__}", flush=True)
        print(f"  message: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
