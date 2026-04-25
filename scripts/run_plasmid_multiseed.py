"""Run a single-cell, single-generation plasmid simulation over N seeds
with copy-number control disabled (``use_rna_control=False``).

Each seed gets an independent RNG stream but the same cache and initial
state. Without RNA I/RNA II feedback, every idle plasmid fires each
timestep (see ``plasmid_replication.py`` line 460 — uncontrolled branch),
so copy number grows until host resources limit it. The ceiling is the
first-limiting-resource signature we want to characterize.

    uv run python scripts/run_plasmid_multiseed.py \\
        --seeds 10 --duration 2400 \\
        --cache-dir out/cache_plasmid_mechanistic

Writes ``out/plasmid/multiseed_timeseries.json`` consumed by
``scripts/plot_plasmid_multiseed.py``.
"""
from __future__ import annotations

import argparse
import binascii
import copy
import json
import os
import sys
import time

import dill
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli.composite import _build_core, _load_cache_bundle
from v2ecoli.generate import build_document
from process_bigraph import Composite

from scripts.run_plasmid_experiment import (
    DNAG_INVESTIGATION_IDS, _bulk_count, _count_unique,
)


OUT_JSON_DEFAULT = "out/plasmid/multiseed_timeseries.json"
SNAPSHOT_INTERVAL = 1  # seconds; full resolution for uncontrolled dynamics


def _strip_emitter(doc: dict) -> dict:
    agent = doc.get("state", {}).get("agents", {}).get("0", {})
    if "emitter" in agent:
        del agent["emitter"]
    return doc


# Replisome subunits whose per-process allocations we want to track.
# trimers need 6 each per oriC; monomers need 2 each per oriC (see
# chromosome_replication.py:244-245).
REPLISOME_SUBUNIT_IDS = {
    "pol_core":   "CPLX0-2361[c]",     # trimer
    "beta_clamp": "CPLX0-3761[c]",     # trimer
    "dnaB":       "CPLX0-3621[c]",     # monomer
    "dnaG":       "EG10239-MONOMER[c]",  # monomer (primase, bottleneck)
    "holA":       "EG11412-MONOMER[c]",  # monomer
    "EG11500":    "EG11500-MONOMER[c]",  # monomer (4th in replisome_monomer_subunits)
}

_bulk_id_to_idx_cache: dict = {}        # for bulk-array reads
_alloc_molecule_idx_cache: dict = {}    # for allocate-store reads

# Global cache of the allocator's moleculeNames list (captured in run_seed).
# The allocate store is indexed by position in this list, NOT bulk-array
# position (see allocator.py:183 — partitioned_counts is shape
# (n_molecules_monitored, n_processes) indexed by allocator.moleculeNames).
_allocator_molecule_names: list = []


def _build_bulk_idx(cell: dict) -> dict:
    """Resolve bulk-array indices for the replisome subunits once.
    ``cell['bulk']`` is a numpy structured array with an ``id`` field."""
    bulk = cell.get("bulk")
    if bulk is None:
        return {}
    try:
        ids = list(bulk["id"])
    except (ValueError, KeyError, TypeError):
        return {}
    out = {}
    for name, mid in REPLISOME_SUBUNIT_IDS.items():
        try:
            out[name] = ids.index(mid)
        except ValueError:
            out[name] = None
    return out


def _build_alloc_molecule_idx() -> dict:
    """Resolve indices into the allocator's moleculeNames list. The
    allocate store is shape n_molecules_monitored (often a strict subset
    of bulk), so bulk-array indices are the wrong indexing."""
    if not _allocator_molecule_names:
        return {}
    out = {}
    for name, mid in REPLISOME_SUBUNIT_IDS.items():
        try:
            out[name] = _allocator_molecule_names.index(mid)
        except ValueError:
            out[name] = None
    return out


def _allocated(cell: dict, proc: str, subunit_idx: dict, key: str):
    """Return the count allocated to ``proc`` for subunit ``key``. The
    allocate store is a per-process bulk-length numpy array written by
    allocator.update each tick (see allocator.py:181-185)."""
    idx = subunit_idx.get(key)
    if idx is None:
        return 0
    alloc = cell.get("allocate", {}).get(proc, {}).get("bulk")
    if alloc is None:
        return 0
    try:
        return int(alloc[idx])
    except (IndexError, TypeError, KeyError):
        return 0


def _requested(cell: dict, proc: str, subunit_idx: dict, key: str):
    """Sum the request count for subunit ``key`` from ``proc``'s request
    list. request.bulk is a list of (idx, count) tuples (see
    allocator.py:117)."""
    idx = subunit_idx.get(key)
    if idx is None:
        return 0
    reqs = cell.get("request", {}).get(proc, {}).get("bulk")
    if not reqs:
        return 0
    total = 0
    try:
        for r_idx, r_val in reqs:
            if r_idx == idx:
                val = int(r_val) if np.isscalar(r_val) else int(np.sum(r_val))
                total += val
    except (TypeError, ValueError):
        return 0
    return total


def snapshot(t: float, cell: dict) -> dict:
    unique = cell.get("unique", {}) or {}
    mass = cell.get("listeners", {}).get("mass", {}) or {}
    rna_control = cell.get("process_state", {}).get(
        "plasmid_rna_control", {}) or {}
    repl_data = cell.get("listeners", {}).get(
        "replication_data", {}) or {}
    snap = {
        "time": float(t),
        "n_full_plasmids": _count_unique(unique, "full_plasmid"),
        "n_oriV": _count_unique(unique, "oriV"),
        # Unique-store key is singular "oriC" (topology maps
        # process-internal "oriCs" → ("unique", "oriC"); see
        # generate.py:677, 690).
        "n_oriC": _count_unique(unique, "oriC"),
        "n_plasmid_active_replisomes": _count_unique(
            unique, "plasmid_active_replisome"),
        "n_active_replisomes": _count_unique(unique, "active_replisome"),
        "n_full_chromosomes": _count_unique(unique, "full_chromosome"),
        "cell_mass": float(mass.get("cell_mass", 0.0) or 0.0),
        "dry_mass": float(mass.get("dry_mass", 0.0) or 0.0),
        "dna_mass": float(mass.get("dna_mass", 0.0) or 0.0),
        "repl_accum": float(rna_control.get("repl_accum", 0.0) or 0.0),
        "n_rna_initiations": int(rna_control.get("n_rna_initiations", 0) or 0),
        "critical_initiation_mass": float(
            repl_data.get("critical_initiation_mass", 0.0) or 0.0),
        "critical_mass_per_oriC": float(
            repl_data.get("critical_mass_per_oriC", 0.0) or 0.0),
    }
    for key, mid in DNAG_INVESTIGATION_IDS.items():
        c = _bulk_count(cell, mid)
        snap[key] = int(c) if c is not None else 0
    # Also record bulk count for every replisome subunit so we can see
    # which subunit hits floor during uncontrolled runaway.
    for key, mid in REPLISOME_SUBUNIT_IDS.items():
        c = _bulk_count(cell, mid)
        snap[f"bulk_{key}"] = int(c) if c is not None else 0

    # Per-process allocations for the replisome subunits. Resolve both
    # index tables once and cache on the first snapshot.
    # - bulk-array index: for bulk-count reads
    # - allocator-moleculeNames index: for allocate/request reads
    global _bulk_id_to_idx_cache, _alloc_molecule_idx_cache
    if not _bulk_id_to_idx_cache:
        _bulk_id_to_idx_cache = _build_bulk_idx(cell)
    if not _alloc_molecule_idx_cache:
        _alloc_molecule_idx_cache = _build_alloc_molecule_idx()
    alloc_idx = _alloc_molecule_idx_cache
    for proc_short, proc_name in [
        ("chrom", "ecoli-chromosome-replication"),
        ("plasmid", "ecoli-plasmid-replication"),
    ]:
        for key in REPLISOME_SUBUNIT_IDS:
            snap[f"alloc_{proc_short}_{key}"] = _allocated(
                cell, proc_name, alloc_idx, key)
            snap[f"req_{proc_short}_{key}"] = _requested(
                cell, proc_name, alloc_idx, key)
    return snap


def run_seed(cache_dir: str, seed: int, duration: float | None, core) -> dict:
    """Build a composite for this seed, disable RNA control, and run
    until the cell divides (no time cap). ``duration`` is an optional
    safety ceiling in simulated seconds; pass None for unbounded.
    Returns {seed, duration_ran, wall_time, divided, snapshots}."""
    initial_state, cache = _load_cache_bundle(cache_dir)
    configs = copy.deepcopy(cache["configs"])
    # Flip BP1993 OFF. The uncontrolled branch (plasmid_replication.py
    # line 460) then fires one initiation per idle plasmid domain per
    # step, giving us runaway replication up to the host-resource cap.
    if "ecoli-plasmid-replication" in configs:
        configs["ecoli-plasmid-replication"]["use_rna_control"] = False
        # Note: previously this script set custom_priorities to give
        # chromosome replication priority 5 (vs plasmid's default 1) so
        # chromosome would win under contention.  That hack is no longer
        # needed — both processes are now PartitionedProcess instances in
        # allocator_2 and the proportional-fairness math handles the
        # competition directly.  Removing the priority override lets the
        # allocator-driven chromosome-arrest phenotype emerge naturally
        # under uncontrolled plasmid load.

    # Capture allocator's molecule_names so we can correctly index the
    # allocate/request stores, which are shape n_monitored_molecules
    # (NOT bulk-array length).
    global _allocator_molecule_names, _alloc_molecule_idx_cache, _bulk_id_to_idx_cache
    _allocator_molecule_names = list(
        configs.get("allocator", {}).get("molecule_names", []))
    _alloc_molecule_idx_cache = {}
    _bulk_id_to_idx_cache = {}

    # Per-process seeds are baked into the cache at build time, so
    # build_document(seed=seed) alone only rerandomizes the allocator RNG.
    # Override each process seed with the same CRC-XOR-name pattern
    # sim_data uses (see sim_data.py:_seedFromName) but keyed off the
    # per-seed loop value, so every process gets a distinct RNG stream
    # per seed. Without this, all loop seeds produce identical trajectories.
    for proc_name, proc_cfg in configs.items():
        if isinstance(proc_cfg, dict) and "seed" in proc_cfg:
            proc_cfg["seed"] = binascii.crc32(
                proc_name.encode("utf-8"), seed) & 0xFFFFFFFF
    doc = build_document(
        initial_state=initial_state,
        configs=configs,
        unique_names=cache["unique_names"],
        dry_mass_inc_dict=cache.get("dry_mass_inc_dict", {}),
        core=core,
        seed=seed,
    )
    _strip_emitter(doc)
    composite = Composite(doc, core=core)

    cell = composite.state["agents"]["0"]
    snaps = [snapshot(0.0, cell)]

    t_wall0 = time.time()
    total_run = 0.0
    divided = False
    while duration is None or total_run < duration:
        chunk = (SNAPSHOT_INTERVAL if duration is None
                 else min(SNAPSHOT_INTERVAL, duration - total_run))
        try:
            composite.run(chunk)
        except Exception as e:
            err = str(e)
            agent_gone = composite.state.get("agents", {}).get("0") is None
            if (agent_gone or "divide" in err.lower() or "_add" in err
                    or "_remove" in err or "_inputs" in err):
                divided = True
                break
            # Non-division runtime error: advance clock and keep going
            # so a transient glitch doesn't lose the whole seed.
            print(f"    seed {seed} warning at t={total_run:.0f}: "
                  f"{type(e).__name__}: {err[:160]}")
            total_run += chunk
            continue
        total_run += chunk
        cur_cell = composite.state.get("agents", {}).get("0")
        if cur_cell is None:
            divided = True
            break
        snaps.append(snapshot(total_run, cur_cell))

    return {
        "seed": seed,
        "duration_ran": total_run,
        "wall_time": time.time() - t_wall0,
        "divided": divided,
        "snapshots": snaps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10,
        help="Number of seeds to run (default 10).")
    parser.add_argument("--start-seed", type=int, default=0,
        help="First seed (default 0). Seeds = start..start+N-1.")
    parser.add_argument("--duration", type=int, default=None,
        help="Optional safety cap in simulated seconds. Default None — "
             "each seed runs until it divides with no time cap.")
    parser.add_argument("--cache-dir", type=str,
        default="out/cache_plasmid_mechanistic",
        help="Plasmid-enabled cache. Default uses the mechanistic-"
             "replisome cache so the DnaG/replisome-subunit consumption "
             "pathway is live.")
    parser.add_argument("--out-json", type=str, default=OUT_JSON_DEFAULT)
    args = parser.parse_args()

    if not os.path.isdir(args.cache_dir):
        raise SystemExit(f"cache dir not found: {args.cache_dir}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # Build core once and reuse; each seed still gets a fresh composite.
    print(f"[{time.strftime('%H:%M:%S')}] building core")
    core = _build_core()

    # Rebind pint quantities so cache unpickling doesn't crash on the
    # first load (matches the pattern in run_plasmid_multigen.py).
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
        print(f"[{time.strftime('%H:%M:%S')}] seed {seed}: running")
        r = run_seed(args.cache_dir, seed, args.duration, core)
        snaps = r["snapshots"]
        first_pc = snaps[0]["n_full_plasmids"]
        last_pc = snaps[-1]["n_full_plasmids"]
        max_pc = max(s["n_full_plasmids"] for s in snaps)
        print(f"  seed {seed}: {r['wall_time']:.0f}s wall, "
              f"sim {r['duration_ran']:.0f}s, plasmids {first_pc}→{last_pc} "
              f"(max {max_pc}), divided={r['divided']}")
        results.append(r)
        # Flush after each seed so a crash doesn't lose progress.
        with open(args.out_json, "w") as f:
            json.dump({
                "cache_dir": args.cache_dir,
                "duration_requested": args.duration,
                "snapshot_interval": SNAPSHOT_INTERVAL,
                "use_rna_control": False,
                "pipeline_wall_time": time.time() - t_pipeline,
                "seeds": results,
            }, f, indent=1)

    print(f"[{time.strftime('%H:%M:%S')}] wrote {args.out_json} "
          f"({len(results)} seeds, {time.time() - t_pipeline:.0f}s wall)")


if __name__ == "__main__":
    main()
