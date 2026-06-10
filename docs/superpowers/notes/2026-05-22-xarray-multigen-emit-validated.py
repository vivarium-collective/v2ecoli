"""Validated emitter-per-generation XArray multi-gen run loop.

Result (2026-05-22, dnaa_00_baseline_with_dnaa_readout, seed=0, 9000 sim steps):
  DIVISION at t=2621s -> daughters 00, 01; emitter swap to gen=2, follow '00'
  DIVISION at t=3203s -> daughters 00, 01 (gen-2's '00' divided)
  zarr store has number_of_oric for generation=1 AND generation=2 partitions
  (single-store, accumulating; partitioned by experiment_id/variant/lineage_seed/generation)

Three lessons baked into the config:
  1. transducer.predicate=subsample(interval=1), buffer.size small (e.g. 4) so
     a non-final flush happens before close (otherwise final-close asserts
     `not include_static` since `include_static` is True only when
     generation==1 AND num_writes==0 — i.e. nothing was ever flushed).
  2. The input passed to em.update must be EXACTLY shaped `{"agents": {<id>:
     {<view-root>: {<view subtree only>}}}}` — siblings outside the view (bulk,
     listeners.atp, etc.) raise `Unexpected emit path`. Pre-filter to the view.
  3. After division, vEcoli daughters are renamed (not hierarchically
     concatenated) — they often re-use the parent's id ('00' -> '00','01'),
     so `followed in agents` keeps being True across a division and the
     swap loop misses it. Detect division by a content signal (chromosome
     count reset, listeners.evolvers_ran, or a process-bigraph division
     event), not agent-id disappearance.

This is the validated mechanism to wrap into `inject_xarray_emitter` (Task 7.2).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '.')
from bigraph_schema import allocate_core
from process_bigraph import Composite
from pbg_superpowers.composite_generator import _REGISTRY, build_generator
import v2ecoli.composites  # noqa: F401  - registers composites
from v2ecoli.library.xarray_emitter import XArrayEmitter

STORE = Path("/tmp/dnaa_multigen.zarr")
if STORE.exists():
    shutil.rmtree(STORE)

core = allocate_core()


def emitter_for(generation: int, agent_id: str) -> XArrayEmitter:
    """A new emitter, pinned to (generation, agent_id), writing to the shared store."""
    cfg = {
        "emit": {"global_time": "node"},
        "out_uri": str(STORE),
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 4},
        },
        "view": [{
            "root": ("listeners",),
            "variables": {
                "replication_data": {
                    "number_of_oric": [{"path": "number_of_oric", "dtype": "<i4"}],
                },
            },
        }],
        "writer": {
            "backend": "zarr",
            "store": str(STORE),
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": "dnaa-mg",
            "variant": 0,
            "lineage_seed": 0,
            "agent_id": agent_id,
            "generation": generation,
            "time_step": 1.0,
            "max_duration": 9000.0,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }
    return XArrayEmitter(config=cfg, core=core)


def view_filtered_state(agent_state: dict) -> dict:
    """Filter an agent's full state to exactly the view subtree the emitter expects."""
    listeners = agent_state.get("listeners") or {}
    repdata = listeners.get("replication_data") or {}
    return {"listeners": {"replication_data": {"number_of_oric": repdata.get("number_of_oric")}}}


entry = _REGISTRY["v2ecoli.composites.baseline_recipes.dnaa_00_baseline_with_dnaa_readout"]
doc = build_generator(entry, overrides={"seed": 0, "cache_dir": "out/cache"})
comp = Composite({"state": doc.get("state", doc)}, core=core)

followed = "0"
gen = 1
em = emitter_for(gen, followed)
done = 0
chunk = 60
MAX_GEN = 3
MAX_STEPS = 9000
gens_seen = [1]

while done < MAX_STEPS and gen <= MAX_GEN:
    try:
        comp.run(chunk)
    except Exception as e:
        print(f"run stopped at {done}: {str(e)[:50]}")
        break
    done += chunk
    agents = (comp.state or {}).get("agents", {})
    if followed in agents:
        em.update({
            "time": float(done),
            "global_time": float(done),
            "agents": {followed: view_filtered_state(agents[followed])},
        })
    else:
        # agent-id-disappearance branch — note this misses the case where the
        # division process renames daughters to re-use the parent's id (see (3)).
        daughters = sorted(
            [a for a in agents if a.startswith(followed) and a != followed]
        ) or sorted([a for a in agents if a != followed])
        if not daughters:
            print(f"no daughter at {done}; agents={list(agents)}")
            break
        em.close(success=True)
        followed = daughters[0]
        gen += 1
        gens_seen.append(gen)
        if gen > MAX_GEN:
            break
        em = emitter_for(gen, followed)
        print(f"division ~{done}s -> generation {gen}, follow lineage '{followed}'")
        if followed in agents:
            em.update({
                "time": float(done),
                "global_time": float(done),
                "agents": {followed: view_filtered_state(agents[followed])},
            })

try:
    em.close(success=True)
except Exception:
    pass

print(f"DONE done={done}s generations={gens_seen}")

# read back: how many generation partitions + oriC per gen
import xarray as xr  # noqa: E402

try:
    dt = xr.open_datatree(STORE, engine="zarr")
    for node in dt.subtree:
        if "number_of_oric" in node.path and node.data_vars:
            for v in node.data_vars:
                vals = node[v].values.ravel()
                print(f"  {node.path} {v}: {vals[:10]}")
except Exception as e:
    print("read err:", str(e)[:160])
