"""Run a plasmid-enabled lineage for N generations, tracking DnaG and
replisome subunit pools at every snapshot interval.  Writes
``out/plasmid/multigen_timeseries.json`` for consumption by
``scripts/plasmid_report.py`` (renders the "Chromosome dynamics across
generations" section alongside the single-generation plasmid analysis).

Mirrors the lineage logic of ``reports/multigeneration_report.py`` — start
from the plasmid-enabled initial state, run to division, keep daughter 1,
rebuild a fresh ``Composite`` from that state, repeat — but captures bulk
counts for the mechanistic-gate subunits at every chunk rather than just
mass totals, so the gen-over-gen DnaG depletion trajectory becomes
plottable in the plasmid report.

    .venv/bin/python scripts/run_plasmid_multigen.py --generations 10
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time

import dill

# Catch segfaults / SIGABRT / SIGBUS from C extensions (numpy, scipy, glpk)
# and print a Python-level traceback before the process dies. Without this,
# a C-level crash silently exits the process and we see neither a traceback
# nor the GLPK warnings — which is exactly the Gen 5 silent-exit signature.
faulthandler.enable()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli.composite import _build_core, _build_from_cache
from v2ecoli.library.division import divide_cell
from v2ecoli.generate import build_document
from process_bigraph import Composite


def _strip_emitter(doc: dict) -> dict:
    """Delete the RAMEmitter from the document before Composite() so the
    emitter's per-tick history never accumulates. For multigen we already
    capture() the needed fields per chunk into JSON, and the emitter's
    ~7 MB/tick footprint otherwise drives the process to ~35 GB at the
    10800s cap → macOS jetsam SIGKILLs it silently mid-Gen-5. The §8 plots
    in plasmid_report.py read only the JSON, never composite.gather_results,
    so stripping the emitter is safe here."""
    agent = (doc.get("state", {}).get("agents", {}).get("0", {}))
    if "emitter" in agent:
        del agent["emitter"]
    return doc


def _make_composite_no_emitter(cache_dir: str, core, seed: int = 0):
    """Drop-in replacement for make_composite(cache_dir=...) that strips
    the emitter from the document before construction."""
    doc = _build_from_cache(cache_dir, core, seed=seed)
    _strip_emitter(doc)
    return Composite(doc, core=core)

from scripts.run_plasmid_experiment import (
    DNAG_INVESTIGATION_IDS,
    _bulk_count,
    _count_unique,
)


CACHE_DIR_DEFAULT = "out/cache_plasmid"
OUT_JSON = "out/plasmid/multigen_timeseries.json"
DILL_DIR = "out/plasmid/gen_dills"   # sidecar per-gen cell_data_after dills,
                                     # so debug_plasmid_gen5.py can resume
                                     # any gen's setup without replaying the
                                     # whole lineage.
SNAPSHOT_INTERVAL = 1            # 1s matches the single-gen plasmid loop
                                 # in workflow_report.py — needed to catch
                                 # RNA-II / initiation events that fire on a
                                 # ~360s interval but set state in one tick.
MAX_GEN_DURATION_DEFAULT = 5400   # per-generation safety cap in sim seconds.
                                  # 1.5h — long enough to confirm a
                                  # lineage-collapse generation (DnaG below
                                  # the mechanistic-gate minimum → no second
                                  # chromosome → no division) as a
                                  # non-dividing gen that runs to this cap,
                                  # short enough that we don't burn wall
                                  # time on a cell we already know is
                                  # stalled. Dropped from vEcoli's 3h cap
                                  # because Gen 5+ stall trajectories don't
                                  # change past ~30 min sim.


def capture(t: float, cell: dict) -> dict:
    """Snapshot all fields needed for the chromosome-dynamics plots.

    Mass listeners + unique-molecule counts + the seven DnaG-investigation
    bulk counts + the BP1993 ODE state (10 plasmid-DNA / RNA / Rom
    species + replication accumulator) so the report can overlay
    copy-number-control dynamics on the DnaG-depletion trajectory and
    check the plots are self-consistent across generations.
    """
    mass = cell.get("listeners", {}).get("mass", {}) or {}
    unique = cell.get("unique", {}) or {}
    rna_control = cell.get("process_state", {}).get(
        "plasmid_rna_control", {}) or {}

    snap = {
        "time": float(t),
        "dry_mass": float(mass.get("dry_mass", 0.0) or 0.0),
        "cell_mass": float(mass.get("cell_mass", 0.0) or 0.0),
        "protein_mass": float(mass.get("protein_mass", 0.0) or 0.0),
        "dna_mass": float(mass.get("dna_mass", 0.0) or 0.0),
        "rRna_mass": float(mass.get("rRna_mass", 0.0) or 0.0),
        "tRna_mass": float(mass.get("tRna_mass", 0.0) or 0.0),
        "mRna_mass": float(mass.get("mRna_mass", 0.0) or 0.0),
        "n_full_chromosomes": _count_unique(unique, "full_chromosome"),
        "n_active_replisomes": _count_unique(unique, "active_replisome"),
        "n_full_plasmids": _count_unique(unique, "full_plasmid"),
        "n_oriV": _count_unique(unique, "oriV"),
        # BP1993 species
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
        "n_plasmid_active_replisomes": _count_unique(
            unique, "plasmid_active_replisome"
        ),
    }
    for key, mid in DNAG_INVESTIGATION_IDS.items():
        c = _bulk_count(cell, mid)
        snap[key] = int(c) if c is not None else 0
    return snap


def run_generation(composite: Composite, gen_idx: int,
                   max_duration: float) -> dict:
    """Advance the composite in fixed chunks until division or the cap.

    Returns a dict ``{index, duration, wall_time, divided, snapshots,
    cell_data_after}``. On the realize-Array "post-add" failure (currently
    blocks true in-place division), we detect that the mother agent is gone
    and still treat the generation as divided — the last pre-division
    snapshot has already been captured.
    """
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
            if (
                "divide" in err.lower()
                or "_add" in err
                or "_remove" in err
            ):
                divided = True
                break
            if composite.state.get("agents", {}).get("0") is None:
                divided = True
                break
            print(
                f"    gen {gen_idx} warning at t={total_run:.0f}: "
                f"{type(e).__name__}: {err[:160]}"
            )
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
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
    parser.add_argument(
        "--max-duration", type=int, default=MAX_GEN_DURATION_DEFAULT,
        help="per-generation safety cap in simulated seconds",
    )
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
    core = _build_core()
    composite = _make_composite_no_emitter(cache_dir, core)
    gens: list[dict] = []
    t_pipeline = time.time()

    def _flush(n_req: int):
        """Persist whatever gens have completed so a later crash can't lose
        the lineage. Called after every gen."""
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({
                "cache_dir": cache_dir,
                "n_generations_requested": n_req,
                "n_generations_completed": len(gens),
                "pipeline_wall_time": time.time() - t_pipeline,
                "snapshot_interval": SNAPSHOT_INTERVAL,
                "tracked_molecules": DNAG_INVESTIGATION_IDS,
                "generations": gens,
            }, f, indent=1)

    def _dump_cell(gen_idx: int, cell_data: dict | None):
        """Persist a gen's cell_data_after to out/plasmid/gen_dills/genN.dill.
        Enables debug_plasmid_gen5.py (and any future standalone-gen harness)
        to resume setup without replaying the whole lineage."""
        if cell_data is None:
            return
        os.makedirs(DILL_DIR, exist_ok=True)
        path = os.path.join(DILL_DIR, f"gen{gen_idx}.dill")
        with open(path, "wb") as f:
            dill.dump(cell_data, f)
        print(f"    wrote {path}")

    gen1 = run_generation(composite, 1, args.max_duration)
    snaps = gen1["snapshots"]
    print(
        f"  gen 1: {gen1['wall_time']:.0f}s wall, sim {gen1['duration']:.0f}s, "
        f"dry_mass {snaps[0]['dry_mass']:.0f}→{snaps[-1]['dry_mass']:.0f}, "
        f"DnaG {snaps[0]['dnaG']}→{snaps[-1]['dnaG']}, divided={gen1['divided']}"
    )
    gens.append({k: v for k, v in gen1.items() if k != "cell_data_after"})
    prev_cell = gen1["cell_data_after"]
    _flush(args.generations)
    _dump_cell(1, prev_cell)
    if not gen1["divided"]:
        print("  gen 1 did not divide — stopping lineage. Forced "
              "division on a non-dividing cell produces an artifact "
              "trajectory (mass/plasmids accumulate without partition), "
              "not a biologically meaningful daughter.")

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
            doc = build_document(
                d1_state, configs, unique_names,
                dry_mass_inc_dict=dry_mass_inc, seed=gen_idx,
            )
            _strip_emitter(doc)
            composite = Composite(doc, core=core)
            print(f"    built daughter composite in {time.time() - t_build:.1f}s")

            gen = run_generation(composite, gen_idx, args.max_duration)
        except BaseException as e:
            # BaseException, not Exception — catches SystemExit and
            # KeyboardInterrupt. A silent gen exit would otherwise slip past
            # an ``except Exception``.
            import traceback
            print(f"  gen {gen_idx}: lineage setup/run failed — "
                  f"{type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            break
        snaps = gen["snapshots"]
        print(
            f"  gen {gen_idx}: {gen['wall_time']:.0f}s wall, "
            f"sim {gen['duration']:.0f}s, "
            f"dry_mass {snaps[0]['dry_mass']:.0f}→{snaps[-1]['dry_mass']:.0f}, "
            f"DnaG {snaps[0]['dnaG']}→{snaps[-1]['dnaG']}, "
            f"divided={gen['divided']}"
        )
        gens.append({k: v for k, v in gen.items() if k != "cell_data_after"})
        prev_cell = gen["cell_data_after"]
        _flush(args.generations)
        _dump_cell(gen_idx, prev_cell)
        if not gen["divided"]:
            print(f"  gen {gen_idx} did not divide — stopping lineage "
                  "(see gen 1 note).")
            break

    pipeline_wall = time.time() - t_pipeline
    _flush(args.generations)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {OUT_JSON} "
          f"({len(gens)} gens, {pipeline_wall:.0f}s wall)")


if __name__ == "__main__":
    main()
