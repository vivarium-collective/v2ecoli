"""Multiseed × multigen runner for controlled mechanistic plasmid simulations.

Runs N seeds × M generations with BP1993 RNA control ON (default) and the
mechanistic replisome gate active for both chromosome and plasmid
replication (via ``out/cache_plasmid_mechanistic``). Expected phenotype:
stable plasmid copy number around ~28 across generations, chromosome
re-initiation proceeds, cell divides — the opposite of the uncontrolled
runaway scenario reported in ``multiseed_timeseries_v2.json``.

Output: ``out/plasmid/multiseed_multigen_timeseries.json`` — one entry
per seed with a list of generations. Snapshot schema matches
``run_plasmid_multigen.py`` plus all 6 replisome subunit bulk counts as
``bulk_<name>`` so DnaG depletion (or the lack of it, under control) is
plottable gen-over-gen.

    uv run python scripts/run_plasmid_multiseed_multigen.py \\
        --seeds 10 --generations 6 \\
        --cache-dir out/cache_plasmid_mechanistic
"""
from __future__ import annotations

import argparse
import binascii
import copy
import faulthandler
import json
import os
import sys
import time

import dill

faulthandler.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli.composite import _build_core, _load_cache_bundle
from v2ecoli.library.division import divide_cell
from v2ecoli.generate import build_document
from process_bigraph import Composite

from scripts.run_plasmid_experiment import (
    DNAG_INVESTIGATION_IDS, _bulk_count, _count_unique,
)
from scripts.run_plasmid_multiseed import REPLISOME_SUBUNIT_IDS


OUT_JSON_DEFAULT = "out/plasmid/multiseed_multigen_timeseries.json"
CHECKPOINT_DIR_DEFAULT = "out/plasmid/multiseed_multigen_checkpoints"
SNAPSHOT_INTERVAL = 1
MAX_GEN_DURATION_DEFAULT = 5400


def _ckpt_path(ckpt_dir: str, seed: int, gen: int) -> str:
    return os.path.join(ckpt_dir, f"seed{seed}_gen{gen}.dill")


def _latest_checkpoint(ckpt_dir: str, seed: int,
                       max_gen: int) -> tuple[int, str] | tuple[None, None]:
    """Return (gen, path) of the highest-gen checkpoint for this seed,
    or (None, None) if no checkpoint exists. Caller decides whether to
    resume from it."""
    if not os.path.isdir(ckpt_dir):
        return None, None
    for g in range(max_gen, 0, -1):
        p = _ckpt_path(ckpt_dir, seed, g)
        if os.path.exists(p):
            return g, p
    return None, None


def _save_checkpoint(ckpt_dir: str, seed: int, gen: int,
                     cell_data: dict) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = _ckpt_path(ckpt_dir, seed, gen) + ".tmp"
    with open(tmp, "wb") as f:
        dill.dump(cell_data, f)
    os.replace(tmp, _ckpt_path(ckpt_dir, seed, gen))


def _strip_emitter(doc: dict) -> dict:
    agent = doc.get("state", {}).get("agents", {}).get("0", {})
    if "emitter" in agent:
        del agent["emitter"]
    return doc


def _seed_override(configs: dict, seed: int) -> dict:
    """Apply the CRC32-XOR-name per-process seed pattern used in
    run_plasmid_multiseed.py. Without this, all seeds produce identical
    trajectories because per-process seeds are baked into the cache at
    build time."""
    for proc_name, proc_cfg in configs.items():
        if isinstance(proc_cfg, dict) and "seed" in proc_cfg:
            proc_cfg["seed"] = binascii.crc32(
                proc_name.encode("utf-8"), seed) & 0xFFFFFFFF
    return configs


def capture(t: float, cell: dict) -> dict:
    """Multigen snapshot + all 6 replisome subunit bulk counts. Matches
    the field naming used by run_plasmid_multiseed.py so existing plotting
    helpers can be reused."""
    mass = cell.get("listeners", {}).get("mass", {}) or {}
    unique = cell.get("unique", {}) or {}
    rna_control = cell.get("process_state", {}).get(
        "plasmid_rna_control", {}) or {}
    repl_data = cell.get("listeners", {}).get("replication_data", {}) or {}

    snap = {
        "time": float(t),
        "dry_mass": float(mass.get("dry_mass", 0.0) or 0.0),
        "cell_mass": float(mass.get("cell_mass", 0.0) or 0.0),
        "protein_mass": float(mass.get("protein_mass", 0.0) or 0.0),
        "dna_mass": float(mass.get("dna_mass", 0.0) or 0.0),
        "n_full_chromosomes": _count_unique(unique, "full_chromosome"),
        "n_active_replisomes": _count_unique(unique, "active_replisome"),
        "n_full_plasmids": _count_unique(unique, "full_plasmid"),
        "n_oriV": _count_unique(unique, "oriV"),
        "n_oriC": _count_unique(unique, "oriC"),
        "n_plasmid_active_replisomes": _count_unique(
            unique, "plasmid_active_replisome"),
        # BP1993 ODE state
        "D":       float(rna_control.get("D", 0.0) or 0.0),
        "D_tII":   float(rna_control.get("D_tII", 0.0) or 0.0),
        "D_lII":   float(rna_control.get("D_lII", 0.0) or 0.0),
        "D_p":     float(rna_control.get("D_p", 0.0) or 0.0),
        "D_starc": float(rna_control.get("D_starc", 0.0) or 0.0),
        "D_c":     float(rna_control.get("D_c", 0.0) or 0.0),
        "D_M":     float(rna_control.get("D_M", 0.0) or 0.0),
        "R_I":     float(rna_control.get("R_I", 0.0) or 0.0),
        "R_II":    float(rna_control.get("R_II", 0.0) or 0.0),
        "M":       float(rna_control.get("M", 0.0) or 0.0),
        "repl_accum":        float(rna_control.get("repl_accum", 0.0) or 0.0),
        "n_rna_initiations": int(rna_control.get("n_rna_initiations", 0) or 0),
        "critical_initiation_mass": float(
            repl_data.get("critical_initiation_mass", 0.0) or 0.0),
        "critical_mass_per_oriC": float(
            repl_data.get("critical_mass_per_oriC", 0.0) or 0.0),
    }
    for key, mid in DNAG_INVESTIGATION_IDS.items():
        c = _bulk_count(cell, mid)
        snap[key] = int(c) if c is not None else 0
    for key, mid in REPLISOME_SUBUNIT_IDS.items():
        c = _bulk_count(cell, mid)
        snap[f"bulk_{key}"] = int(c) if c is not None else 0
    return snap


def run_generation(composite: Composite, gen_idx: int,
                   max_duration: float) -> dict:
    cell = composite.state["agents"]["0"]
    snaps = [capture(0.0, cell)]
    t_wall0 = time.time()
    total_run = 0.0
    divided = False
    last_cell_data: dict | None = None

    data_keys = {
        "bulk", "unique", "listeners", "environment", "boundary",
        "global_time", "timestep", "divide", "division_threshold",
        "process_state", "allocator_rng",
    }

    while total_run < max_duration:
        chunk = min(SNAPSHOT_INTERVAL, max_duration - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            total_run += chunk
            err = str(e)
            if ("divide" in err.lower() or "_add" in err or "_remove" in err):
                divided = True
                break
            if composite.state.get("agents", {}).get("0") is None:
                divided = True
                break
            print(f"    gen {gen_idx} warning at t={total_run:.0f}: "
                  f"{type(e).__name__}: {err[:160]}")
            continue
        total_run += chunk
        cur_cell = composite.state.get("agents", {}).get("0")
        if cur_cell is None:
            divided = True
            break
        snaps.append(capture(total_run, cur_cell))
        last_cell_data = {
            k: v for k, v in cur_cell.items()
            if k in data_keys
            or k.startswith("request_")
            or k.startswith("allocate_")
        }

    return {
        "index": gen_idx,
        "duration": total_run,
        "wall_time": time.time() - t_wall0,
        "divided": divided,
        "snapshots": snaps,
        "cell_data_after": last_cell_data,
    }


def run_seed_lineage(seed: int, n_gens: int, cache_dir: str,
                     max_duration: float, core,
                     ckpt_dir: str | None = None,
                     prior_gens: list | None = None) -> dict:
    """Run one seed's lineage for up to n_gens generations.

    ``cell_data_after`` is stripped from the returned gens dicts to keep
    the aggregate JSON small, but is ALSO persisted as a per-(seed, gen)
    dill sidecar in ``ckpt_dir`` so a later run can resume from the
    highest-gen checkpoint without re-simulating completed gens.

    If ``prior_gens`` is supplied (list of already-completed gen dicts
    from an earlier run), the runner attempts to resume by loading the
    highest-gen checkpoint and continuing from the next generation.
    """
    initial_state, cache = _load_cache_bundle(cache_dir)
    configs = copy.deepcopy(cache["configs"])
    _seed_override(configs, seed)
    unique_names = cache.get("unique_names", [])
    dry_mass_inc = cache.get("dry_mass_inc_dict", {})

    t_wall0 = time.time()
    gens: list[dict] = list(prior_gens) if prior_gens else []
    stopped: str | None = None

    # Decide whether to resume. Need (a) checkpoint file, (b) matching
    # prior_gens entry so the JSON generations list stays consistent.
    resume_from: int = 0
    prev_cell: dict | None = None
    if ckpt_dir is not None:
        ckpt_gen, ckpt_path = _latest_checkpoint(ckpt_dir, seed, n_gens)
        # Only resume if the matching gen is present in prior_gens.
        if ckpt_gen and len(gens) >= ckpt_gen:
            try:
                print(f"[{time.strftime('%H:%M:%S')}] seed {seed}: "
                      f"resuming from checkpoint gen {ckpt_gen}")
                with open(ckpt_path, "rb") as f:
                    prev_cell = dill.load(f)
                resume_from = ckpt_gen
                # Truncate gens to ckpt_gen in case prior_gens had more
                # (shouldn't happen, but be conservative).
                gens = gens[:ckpt_gen]
            except Exception as e:
                print(f"  seed {seed}: checkpoint load failed "
                      f"({type(e).__name__}: {e}); rebuilding from gen 1")
                resume_from = 0
                prev_cell = None
                gens = []

    if resume_from == 0:
        print(f"[{time.strftime('%H:%M:%S')}] seed {seed} gen 1: building")
        doc = build_document(
            initial_state=initial_state,
            configs=configs,
            unique_names=unique_names,
            dry_mass_inc_dict=dry_mass_inc,
            core=core,
            seed=seed,
        )
        _strip_emitter(doc)
        composite = Composite(doc, core=core)

        gen1 = run_generation(composite, 1, max_duration)
        snaps = gen1["snapshots"]
        print(f"  seed {seed} gen 1: {gen1['wall_time']:.0f}s wall, "
              f"sim {gen1['duration']:.0f}s, "
              f"plasmids {snaps[0]['n_full_plasmids']}→{snaps[-1]['n_full_plasmids']}, "
              f"DnaG {snaps[0]['dnaG']}→{snaps[-1]['dnaG']}, "
              f"divided={gen1['divided']}")
        prev_cell = gen1["cell_data_after"]
        gens.append({k: v for k, v in gen1.items() if k != "cell_data_after"})
        if ckpt_dir is not None and prev_cell is not None:
            _save_checkpoint(ckpt_dir, seed, 1, prev_cell)
        if not gen1["divided"]:
            stopped = "gen 1 did not divide"
        resume_from = 1

    for gen_idx in range(resume_from + 1, n_gens + 1):
        if stopped:
            break
        if prev_cell is None or "bulk" not in prev_cell:
            stopped = f"no prior cell state at gen {gen_idx}"
            break
        print(f"[{time.strftime('%H:%M:%S')}] seed {seed} gen {gen_idx}: "
              f"dividing, building daughter")
        try:
            d1_state, _d2 = divide_cell(prev_cell)
            doc = build_document(
                d1_state, configs, unique_names,
                dry_mass_inc_dict=dry_mass_inc, seed=seed * 1000 + gen_idx,
            )
            _strip_emitter(doc)
            composite = Composite(doc, core=core)
            gen = run_generation(composite, gen_idx, max_duration)
        except BaseException as e:
            import traceback
            print(f"  seed {seed} gen {gen_idx}: "
                  f"{type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            stopped = f"gen {gen_idx} setup failed: {type(e).__name__}"
            break
        snaps = gen["snapshots"]
        print(f"  seed {seed} gen {gen_idx}: {gen['wall_time']:.0f}s wall, "
              f"sim {gen['duration']:.0f}s, "
              f"plasmids {snaps[0]['n_full_plasmids']}→{snaps[-1]['n_full_plasmids']}, "
              f"DnaG {snaps[0]['dnaG']}→{snaps[-1]['dnaG']}, "
              f"divided={gen['divided']}")
        prev_cell = gen["cell_data_after"]
        gens.append({k: v for k, v in gen.items() if k != "cell_data_after"})
        if ckpt_dir is not None and prev_cell is not None:
            _save_checkpoint(ckpt_dir, seed, gen_idx, prev_cell)
        if not gen["divided"]:
            stopped = f"gen {gen_idx} did not divide"
            break

    return {
        "seed": seed,
        "generations": gens,
        "wall_time": time.time() - t_wall0,
        "stopped_reason": stopped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--cache-dir", type=str,
        default="out/cache_plasmid_mechanistic",
        help="Default uses the mechanistic cache so both chromosome and "
             "plasmid replication gate on the replisome-subunit pool.")
    parser.add_argument("--max-duration", type=int,
        default=MAX_GEN_DURATION_DEFAULT,
        help="per-generation safety cap in simulated seconds")
    parser.add_argument("--out-json", type=str, default=OUT_JSON_DEFAULT)
    parser.add_argument("--checkpoint-dir", type=str,
        default=CHECKPOINT_DIR_DEFAULT,
        help="Directory for per-(seed, gen) dill sidecars. Set to empty "
             "string to disable checkpointing.")
    parser.add_argument("--resume", action="store_true",
        help="If --out-json and --checkpoint-dir exist, extend each "
             "seed's lineage from its highest-gen checkpoint. Completed "
             "generations are preserved verbatim from the existing JSON.")
    args = parser.parse_args()

    if not os.path.isdir(args.cache_dir):
        raise SystemExit(f"cache dir not found: {args.cache_dir}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Preload prior generations per seed if --resume.
    prior_by_seed: dict[int, list] = {}
    if args.resume and os.path.exists(args.out_json):
        try:
            with open(args.out_json) as f:
                prior = json.load(f)
            for s in prior.get("seeds", []):
                prior_by_seed[int(s["seed"])] = list(s.get("generations", []))
            print(f"[{time.strftime('%H:%M:%S')}] resume: loaded "
                  f"{len(prior_by_seed)} prior seeds from {args.out_json}")
        except Exception as e:
            print(f"  resume load failed: {type(e).__name__}: {e}")

    print(f"[{time.strftime('%H:%M:%S')}] building core")
    core = _build_core()

    # Rebind pint quantities so cache unpickling doesn't crash on first load.
    try:
        with open(os.path.join(args.cache_dir, "sim_data_cache.dill"),
                  "rb") as f:
            cache_probe = dill.load(f)
        from v2ecoli.library.unit_bridge import rebind_cache_quantities
        rebind_cache_quantities(cache_probe)
    except ImportError:
        pass
    except Exception as e:
        print(f"  unit rebind warning: {type(e).__name__}: {e}")

    results: list[dict] = []
    t_pipeline = time.time()
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        prior = prior_by_seed.get(seed)
        if prior and len(prior) >= args.generations:
            print(f"[{time.strftime('%H:%M:%S')}] seed {seed}: already has "
                  f"{len(prior)}/{args.generations} gens; skipping")
            results.append({
                "seed": seed, "generations": prior,
                "wall_time": 0.0, "stopped_reason": "already complete",
            })
        else:
            r = run_seed_lineage(seed, args.generations, args.cache_dir,
                                 args.max_duration, core,
                                 ckpt_dir=ckpt_dir, prior_gens=prior)
            results.append(r)
        with open(args.out_json, "w") as f:
            json.dump({
                "cache_dir": args.cache_dir,
                "n_generations_requested": args.generations,
                "max_gen_duration": args.max_duration,
                "snapshot_interval": SNAPSHOT_INTERVAL,
                "use_rna_control": True,
                "pipeline_wall_time": time.time() - t_pipeline,
                "seeds": results,
            }, f, indent=1)
        r = results[-1]
        print(f"  seed {seed}: {len(r['generations'])}/{args.generations} "
              f"gens, {r['wall_time']:.0f}s wall, "
              f"stopped_reason={r['stopped_reason']}")

    print(f"[{time.strftime('%H:%M:%S')}] wrote {args.out_json} "
          f"({len(results)} seeds, {time.time() - t_pipeline:.0f}s wall)")


if __name__ == "__main__":
    main()
