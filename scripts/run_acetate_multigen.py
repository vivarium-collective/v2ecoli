"""Run a chromosome-only acetate lineage for N generations to test whether
the early-gen FBA over-supply / fast-growth pattern equilibrates by gen 3-5.

Tracks per-tick:
- cycle time, dry/cell mass
- ppGpp concentration, fraction tRNA charged, elongation rate
- division trigger

No plasmid replication, no overrides. Uses cache_acetate_canonical (the
apr24-extracted PI fixture against which acetate sims work cleanly).

    .venv/bin/python scripts/run_acetate_multigen.py --generations 5
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time

import dill
import numpy as np

faulthandler.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli import build_core, build_document
from v2ecoli.composites.baseline import baseline as baseline_doc
from v2ecoli.library.division import divide_cell
from process_bigraph import Composite


CACHE_DIR_DEFAULT = "out/cache_acetate_canonical"
OUT_JSON = "out/acetate_multigen/acetate_multigen_timeseries.json"
DILL_DIR = "out/acetate_multigen/gen_dills"
SNAPSHOT_INTERVAL = 60
MAX_GEN_DURATION_DEFAULT = 12000  # 200 min cap — long enough for τ≈136


def _strip_emitter(doc: dict) -> dict:
    agent = (doc.get("state", {}).get("agents", {}).get("0", {}))
    if "emitter" in agent:
        del agent["emitter"]
    return doc


def _make_composite(cache_dir: str, core, seed: int = 0):
    doc = build_document("baseline", core=core, seed=seed, cache_dir=cache_dir)
    _strip_emitter(doc)
    return Composite(doc, core=core)


def _mean(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(np.mean(x)) if len(x) else 0.0
    return float(x or 0.0)


def capture(t: float, cell: dict) -> dict:
    mass = cell.get("listeners", {}).get("mass", {}) or {}
    gl = cell.get("listeners", {}).get("growth_limits", {}) or {}
    rd = cell.get("listeners", {}).get("ribosome_data", {}) or {}
    unique = cell.get("unique", {}) or {}

    fc = unique.get("full_chromosome")
    n_chrom = 0
    if fc is not None and hasattr(fc, "dtype") and "_entryState" in fc.dtype.names:
        n_chrom = int(fc["_entryState"].sum())
    rep = unique.get("active_replisome")
    n_forks = 0
    if rep is not None and hasattr(rep, "dtype") and "_entryState" in rep.dtype.names:
        n_forks = int(rep["_entryState"].sum())

    return {
        "time": float(t),
        "dry_mass": float(mass.get("dry_mass", 0.0) or 0.0),
        "cell_mass": float(mass.get("cell_mass", 0.0) or 0.0),
        "protein_mass": float(mass.get("protein_mass", 0.0) or 0.0),
        "rna_mass": float(mass.get("rRna_mass", 0.0) or 0.0)
                    + float(mass.get("tRna_mass", 0.0) or 0.0)
                    + float(mass.get("mRna_mass", 0.0) or 0.0),
        "ppgpp_conc": _mean(gl.get("ppgpp_conc")),
        "fraction_trna_charged": _mean(gl.get("fraction_trna_charged")),
        "rela_conc": _mean(gl.get("rela_conc")),
        "spot_conc": _mean(gl.get("spot_conc")),
        "ribosome_conc": _mean(gl.get("ribosome_conc")),
        "effective_elongation_rate": _mean(rd.get("effective_elongation_rate")),
        "n_full_chromosomes": n_chrom,
        "n_active_replisomes": n_forks,
    }


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
            if "divide" in err.lower() or "_add" in err or "_remove" in err:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--max-duration", type=int,
                        default=MAX_GEN_DURATION_DEFAULT)
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"cache dir not found: {cache_dir}")

    with open(os.path.join(cache_dir, "sim_data_cache.dill"), "rb") as f:
        cache = dill.load(f)
    try:
        from v2ecoli.library.unit_bridge import rebind_cache_quantities
        rebind_cache_quantities(cache)
    except ImportError:
        pass
    configs = cache.get("configs", {})
    unique_names = cache.get("unique_names", [])
    dry_mass_inc = cache.get("dry_mass_inc_dict", {})

    print(f"[{time.strftime('%H:%M:%S')}] Gen 1: building from {cache_dir}")
    core = build_core()
    composite = _make_composite(cache_dir, core)
    gens: list[dict] = []
    t_pipeline = time.time()

    def _flush(n_req: int):
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({
                "cache_dir": cache_dir,
                "n_generations_requested": n_req,
                "n_generations_completed": len(gens),
                "pipeline_wall_time": time.time() - t_pipeline,
                "snapshot_interval": SNAPSHOT_INTERVAL,
                "generations": gens,
            }, f, indent=1)

    def _dump_cell(gen_idx: int, cell_data: dict | None):
        if cell_data is None:
            return
        os.makedirs(DILL_DIR, exist_ok=True)
        path = os.path.join(DILL_DIR, f"gen{gen_idx}.dill")
        with open(path, "wb") as f:
            dill.dump(cell_data, f)
        print(f"    wrote {path}")

    def _summary(gen_idx: int, gen: dict) -> str:
        snaps = gen["snapshots"]
        s0, s1 = snaps[0], snaps[-1]
        return (
            f"  gen {gen_idx}: {gen['wall_time']:.0f}s wall, "
            f"sim {gen['duration']:.0f}s ({gen['duration']/60:.1f} min), "
            f"dry_mass {s0['dry_mass']:.0f}→{s1['dry_mass']:.0f} fg, "
            f"ppGpp {s0['ppgpp_conc']:.1f}→{s1['ppgpp_conc']:.1f} uM, "
            f"charged {s0['fraction_trna_charged']:.3f}→{s1['fraction_trna_charged']:.3f}, "
            f"elong {s0['effective_elongation_rate']:.1f}→{s1['effective_elongation_rate']:.1f}, "
            f"divided={gen['divided']}"
        )

    gen1 = run_generation(composite, 1, args.max_duration)
    print(_summary(1, gen1))
    gens.append({k: v for k, v in gen1.items() if k != "cell_data_after"})
    prev_cell = gen1["cell_data_after"]
    _flush(args.generations)
    _dump_cell(1, prev_cell)
    if not gen1["divided"]:
        print("  gen 1 did not divide — stopping lineage.")

    for gen_idx in range(2, args.generations + 1):
        if not gens[-1]["divided"]:
            break
        if prev_cell is None or "bulk" not in prev_cell:
            print(f"  gen {gen_idx}: no prior cell state — stopping lineage")
            break
        print(f"[{time.strftime('%H:%M:%S')}] Gen {gen_idx}: dividing, "
              f"keeping daughter 1")
        try:
            d1_state, _d2_state = divide_cell(prev_cell)
            t_build = time.time()
            daughter_bundle = {
                "initial_state": d1_state,
                "configs": configs,
                "unique_names": unique_names,
                "dry_mass_inc_dict": dry_mass_inc,
            }
            doc = baseline_doc(core=core, seed=gen_idx, cache_dir=cache_dir,
                               bundle=daughter_bundle)
            _strip_emitter(doc)
            composite = Composite(doc, core=core)
            print(f"    built daughter composite in {time.time() - t_build:.1f}s")
            gen = run_generation(composite, gen_idx, args.max_duration)
        except BaseException as e:
            import traceback
            print(f"  gen {gen_idx}: lineage setup/run failed — "
                  f"{type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            break
        print(_summary(gen_idx, gen))
        gens.append({k: v for k, v in gen.items() if k != "cell_data_after"})
        prev_cell = gen["cell_data_after"]
        _flush(args.generations)
        _dump_cell(gen_idx, prev_cell)
        if not gen["divided"]:
            print(f"  gen {gen_idx} did not divide — stopping lineage.")
            break

    pipeline_wall = time.time() - t_pipeline
    _flush(args.generations)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {OUT_JSON} "
          f"({len(gens)} gens, {pipeline_wall:.0f}s wall, "
          f"{pipeline_wall/60:.1f} min)")


if __name__ == "__main__":
    main()
