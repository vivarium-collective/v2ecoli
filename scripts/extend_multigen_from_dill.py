"""Extend a multigen lineage from a previously-dumped gen{N}.dill.

Picks up where ``run_condition_multigen_parquet.py`` left off — writes new
gens into the same parquet root (different generation= partitions).

Usage:
    python scripts/extend_multigen_from_dill.py \
        --cache-dir out/cache_succinate_default \
        --out-dir   out/succinate_5gen_default_parquet \
        --experiment-id succinate_5gen_default \
        --resume-dill out/succinate_5gen_default/gen_dills/gen5.dill \
        --start-gen 6 \
        --generations 5
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import dill

from v2ecoli import build_composite
from v2ecoli.composites._helpers import parquet_emitter
from v2ecoli.composites.baseline import baseline as baseline_doc
from v2ecoli.core import build_core
from v2ecoli.library.division import divide_cell
from v2ecoli.library.parquet_emitter import ParquetEmitter
from process_bigraph import Composite


def _run_gen(comp, max_duration, gen_idx):
    """Lifted from run_condition_multigen_parquet.py."""
    t_sim = 0
    last_state = None
    divided = False
    while t_sim < max_duration:
        comp.run(1)
        t_sim += 1
        agents = comp.state.get("agents", {})
        if len(agents) == 0:
            print(f"  ! gen {gen_idx}: no agents — composite empty")
            break
        if len(agents) > 1:
            # division happened mid-tick → keep the (lexicographically) first daughter
            # actually, daughters get used for the NEXT gen; we just record + break
            divided = True
            # The "this generation"'s last state was the parent — already captured
            # below as last_state on the previous tick. Just break.
            break
        agent_id = next(iter(agents))
        last_state = agents[agent_id]

    n_closed = ParquetEmitter.flush_all_in_composite(comp, success=divided)
    if n_closed:
        print(f"    gen {gen_idx}: flushed {n_closed} ParquetEmitter instance(s)")
    return t_sim, divided, last_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--resume-dill", required=True, help="path to gen{N}.dill from prior run")
    ap.add_argument("--start-gen", type=int, required=True, help="N+1 of the dill's gen")
    ap.add_argument("--generations", type=int, default=5, help="how many more gens to run")
    ap.add_argument("--max-min", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dill-dir", default=None,
                    help="defaults to out/<experiment-id>/gen_dills (same as base run)")
    args = ap.parse_args()

    max_duration = int(args.max_min * 60)
    dill_dir = args.dill_dir or f"out/{args.experiment_id}/gen_dills"
    os.makedirs(dill_dir, exist_ok=True)

    print("=" * 60)
    print(f"Multi-gen extension — gens {args.start_gen}..{args.start_gen + args.generations - 1}")
    print(f"  cache:        {args.cache_dir}")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  experiment:   {args.experiment_id}")
    print(f"  resume_dill:  {args.resume_dill}")
    print("=" * 60)

    with open(os.path.join(args.cache_dir, "sim_data_cache.dill"), "rb") as f:
        cache = dill.load(f)
    try:
        from v2ecoli.library.unit_bridge import rebind_cache_quantities
        rebind_cache_quantities(cache)
    except ImportError:
        pass
    configs = cache.get("configs", {})
    unique_names = cache.get("unique_names", [])
    dry_mass_inc = cache.get("dry_mass_inc_dict", {})

    core = build_core()

    # Load the resume-from state
    with open(args.resume_dill, "rb") as f:
        prev_cell_data = dill.load(f)

    t_pipeline = time.time()
    summary = []

    for offset in range(args.generations):
        gen_idx = args.start_gen + offset
        agent_id = "0" * gen_idx
        print(f"\n[{time.strftime('%H:%M:%S')}] gen {gen_idx} (agent_id={agent_id})")
        t0 = time.time()

        with parquet_emitter(
            out_dir=args.out_dir,
            experiment_id=args.experiment_id,
            lineage_seed=args.seed,
            agent_id=agent_id,
            generation=gen_idx,
        ):
            t_build = time.time()
            d1_state, _d2_state = divide_cell(prev_cell_data)
            bundle = {
                "initial_state": d1_state,
                "configs": configs,
                "unique_names": unique_names,
                "dry_mass_inc_dict": dry_mass_inc,
            }
            doc = baseline_doc(core=core, seed=args.seed + gen_idx,
                               cache_dir=args.cache_dir, bundle=bundle)
            comp = Composite(doc, core=core)
            print(f"    composite built in {time.time()-t_build:.1f}s")

            duration, divided, last_state = _run_gen(comp, max_duration, gen_idx)

        dry_f = float((last_state or {}).get("listeners", {}).get("mass", {}).get("dry_mass", 0))
        wall = time.time() - t0
        result = {
            "gen": gen_idx,
            "agent_id": agent_id,
            "duration_min": duration / 60.0,
            "divided": divided,
            "final_dry_mass_fg": dry_f,
            "wall_seconds": wall,
        }
        summary.append(result)
        print(f"    gen {gen_idx} summary: tau={duration/60:.1f} min  "
              f"final_dry={dry_f:.1f} fg  divided={divided}  wall={wall:.0f}s")

        prev_cell_data = last_state
        if last_state is not None:
            dill_path = os.path.join(dill_dir, f"gen{gen_idx}.dill")
            # Filter to data-only keys before dilling. The resume-built
            # composite includes live process instances at the top level
            # of agents/<id>, and their internal WeakValueDictionary stores
            # break dill via KeyedRef. The baseline run's gen5.dill uses
            # this 11-key shape plus a few schema-level extras.
            # NOTE: `exchange_data` looks like a data store but is actually
            # a process schema entry with an `instance` field holding an
            # ExchangeData process. Its internals carry a WeakValueDictionary
            # → KeyedRef that breaks dill. Excluded here.
            DATA_KEYS = {
                "bulk", "unique", "listeners", "global_time", "timestep",
                "boundary", "environment", "allocator_rng", "process_state",
                "divide", "division_threshold", "exchange",
                "next_update_time", "media_id", "ppgpp_state",
                "attenuation_config", "request", "allocate",
            }
            sanitized = {k: v for k, v in last_state.items() if k in DATA_KEYS}
            try:
                with open(dill_path, "wb") as f:
                    dill.dump(sanitized, f)
            except (TypeError, pickle.PicklingError) as e:
                print(f"    ! dill dump of gen {gen_idx} STILL failed "
                      f"({type(e).__name__}: {str(e)[:80]}); skipping checkpoint")
                if os.path.exists(dill_path):
                    os.remove(dill_path)

    total = time.time() - t_pipeline
    print(f"\n[done] {args.generations} gens in {total:.0f}s")
    for r in summary:
        print(f"  gen {r['gen']}: τ={r['duration_min']:.1f} min  "
              f"final_dry={r['final_dry_mass_fg']:.1f} fg")


if __name__ == "__main__":
    main()
