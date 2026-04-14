"""Run a short plasmid-enabled simulation and record plasmid observables.

Loads the plasmid-enabled cache (built with ``save_cache(..., has_plasmid=True)``),
runs for ``DURATION`` seconds, and writes a JSON timeseries that the report
script consumes.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli.composite import make_composite

CACHE_DIR = "out/cache_plasmid"
OUT_JSON = "out/plasmid/timeseries.json"
DURATION = int(os.environ.get("PLASMID_DURATION", 60))
INTERVAL = int(os.environ.get("PLASMID_INTERVAL", 2))


def _count_unique(unique_dict, name):
    arr = unique_dict.get(name)
    if arr is None or not hasattr(arr, "dtype") or "_entryState" not in arr.dtype.names:
        return 0
    return int(arr["_entryState"].sum())


def _bulk_count(cell, name):
    bulk = cell.get("bulk")
    if bulk is None or not hasattr(bulk, "dtype"):
        return None
    try:
        idx = np.where(bulk["id"] == name)[0]
        if idx.size == 0:
            return None
        return int(bulk["count"][idx[0]])
    except Exception:
        return None


def snapshot(t, cell, replisome_trimer_ids, replisome_monomer_ids, dntp_ids):
    unique = cell.get("unique", {})
    mass = cell.get("listeners", {}).get("mass", {})
    rna_control = cell.get("process_state", {}).get("plasmid_rna_control", {})

    snap = {
        "time": float(t),
        "n_full_plasmids": _count_unique(unique, "full_plasmid"),
        "n_oriV": _count_unique(unique, "oriV"),
        "n_plasmid_domains": _count_unique(unique, "plasmid_domain"),
        "n_plasmid_active_replisomes": _count_unique(unique, "plasmid_active_replisome"),
        "n_active_replisomes": _count_unique(unique, "active_replisome"),
        "n_full_chromosomes": _count_unique(unique, "full_chromosome"),
        "cell_mass": float(mass.get("cell_mass", 0.0) or 0.0),
        "dna_mass": float(mass.get("dna_mass", 0.0) or 0.0),
        "rna_I": float(rna_control.get("rna_I", 0.0) or 0.0),
        "rna_II": float(rna_control.get("rna_II", 0.0) or 0.0),
        "hybrid": float(rna_control.get("hybrid", 0.0) or 0.0),
        "time_since_rna_II": float(rna_control.get("time_since_rna_II", 0.0) or 0.0),
        "PL_fractional": float(rna_control.get("PL_fractional", 0.0) or 0.0),
        "n_rna_initiations": int(rna_control.get("n_rna_initiations", 0) or 0),
    }
    # Replisome subunit bulk counts
    trimer_total = 0
    for mid in replisome_trimer_ids:
        c = _bulk_count(cell, mid)
        if c is not None:
            trimer_total += c
    snap["replisome_trimer_min"] = trimer_total  # sum-over-subunits surrogate
    monomer_total = 0
    for mid in replisome_monomer_ids:
        c = _bulk_count(cell, mid)
        if c is not None:
            monomer_total += c
    snap["replisome_monomer_min"] = monomer_total
    return snap


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Loading composite from {CACHE_DIR}")
    t0 = time.time()
    composite = make_composite(cache_dir=CACHE_DIR, seed=0)
    print(f"  loaded in {time.time() - t0:.1f}s")

    cell = composite.state["agents"]["0"]

    # Resolve replisome subunit / dNTP IDs from the plasmid process config
    # for readability in the report.
    import dill
    with open(os.path.join(CACHE_DIR, "sim_data_cache.dill"), "rb") as f:
        cache = dill.load(f)
    cfg = cache["configs"].get("ecoli-plasmid-replication", {})
    trimer_ids = cfg.get("replisome_trimers_subunits", [])
    monomer_ids = cfg.get("replisome_monomers_subunits", [])
    dntp_ids = cfg.get("dntps", [])

    snapshots = [snapshot(0, cell, trimer_ids, monomer_ids, dntp_ids)]
    print(f"  t=0: plasmids={snapshots[0]['n_full_plasmids']}, "
          f"oriV={snapshots[0]['n_oriV']}, "
          f"plasmid_replisomes={snapshots[0]['n_plasmid_active_replisomes']}")

    t_sim = 0
    t_run = time.time()
    while t_sim < DURATION:
        chunk = min(INTERVAL, DURATION - t_sim)
        try:
            composite.run(chunk)
        except Exception as e:
            print(f"  run error at t={t_sim}: {type(e).__name__}: {e}")
            break
        t_sim += chunk
        cell = composite.state.get("agents", {}).get("0")
        if cell is None:
            print(f"  agent disappeared at t={t_sim}")
            break
        snap = snapshot(t_sim, cell, trimer_ids, monomer_ids, dntp_ids)
        snapshots.append(snap)
        print(f"  t={t_sim}: plasmids={snap['n_full_plasmids']}, "
              f"p_replisomes={snap['n_plasmid_active_replisomes']}, "
              f"rna_I={snap['rna_I']:.2f}, rna_II={snap['rna_II']:.3f}, "
              f"n_inits={snap['n_rna_initiations']}")

    wall = time.time() - t_run
    print(f"[{time.strftime('%H:%M:%S')}] sim wall time: {wall:.1f}s "
          f"for {t_sim}s sim time ({wall/max(t_sim,1):.2f}x realtime)")

    result = {
        "duration": DURATION,
        "interval": INTERVAL,
        "wall_time": wall,
        "trimer_subunit_ids": trimer_ids,
        "monomer_subunit_ids": monomer_ids,
        "dntp_ids": dntp_ids,
        "snapshots": snapshots,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {OUT_JSON} ({len(snapshots)} snapshots)")


if __name__ == "__main__":
    main()
