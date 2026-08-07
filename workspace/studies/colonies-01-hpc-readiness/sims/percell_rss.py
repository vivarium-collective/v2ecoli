"""Clean per-cell RSS re-measurement on current main (colonies hardening).

Purpose
-------
The investigation's HPC "cells-per-node" budget rests on the per-cell RSS
footprint. Three numbers were in tension on disk:
  - colonies-01 F-03: ~450 MB/cell within a process -> 384 cells/node;
  - colonies-02 charts: "~1+ GB/cell, each cell loads its own sim_data";
  - investigation executive: "~1000 cells/node, RSS re-derived from the re-run".

The mechanism (v2ecoli/core.py::_load_cache_bundle_cached, @lru_cache by
cache_dir; ecoli_baseline.py deep-copies only initial_state) implies sim_data is
shared BY REFERENCE within a process, so the incremental per-cell cost is the
deep-copied initial_state + mutable process state, NOT a fresh ~1 GB sim_data.
Across separate OS processes (Ray actors) each has its own lru_cache and pays a
full sim_data baseline.

This script re-measures, on CURRENT main, the WITHIN-PROCESS per-cell RSS by
growing N=1 -> 2 -> 4 via forced division (sim_data shared), decomposing RSS
into gc-visible numpy bytes vs native. If numpy stays ~flat as cells are added,
sim_data sharing is confirmed and the per-cell increment is the true additive
footprint. Bounded: short plateaus (few ticks) so the macOS per-tick arena
churn (~1.6 MB/tick) contributes << the ~450 MB/cell step-up.

Run:
  PYTHONPATH=~/code/v2e-hcolonies/.env-shadow:~/code/v2e-hcolonies \
    ~/code/v2ecoli/.venv/bin/python \
    workspace/studies/colonies-01-hpc-readiness/sims/percell_rss.py
"""
import csv
import gc
import os

import numpy as np
import psutil


def rss_mb():
    return psutil.Process().memory_info().rss / 1048576.0


def numpy_mb():
    return sum(o.nbytes for o in gc.get_objects()
               if isinstance(o, np.ndarray)) / 1048576.0


rows = []


def sample(label, n, note=""):
    gc.collect()
    r, m = rss_mb(), numpy_mb()
    row = {"label": label, "n_cells": n, "rss_mb": round(r, 1),
           "numpy_mb": round(m, 1), "native_mb": round(r - m, 1), "note": note}
    rows.append(row)
    print(f"[{label:>10}] n={n} rss={r:7.0f} numpy={m:6.0f} "
          f"native={r - m:7.0f}  {note}", flush=True)
    return row


# --- baseline: bare process before any v2ecoli import -----------------------
sample("import0", 0, "after psutil/numpy import, before v2ecoli")

from v2ecoli.colony import make_colony  # noqa: E402

sample("import1", 0, "after v2ecoli import (imports loaded, no sim_data)")

# --- build N=1 colony (loads + lru-caches sim_data once) ---------------------
c = make_colony(n_cells=1, cache_dir="out/cache", seed=0,
                jitter_per_second=1e-4, init_mass=200.0, emit_cells=False)
sample("build_n1", len(c.state["cells"]), "colony built, sim_data loaded once")

WARM = 20  # short plateau: WARM * ~1.6 MB/tick arena churn << 450 MB/cell step


def warm_and_sample(label, note=""):
    for _ in range(WARM):
        c.run(1.0)
    return sample(label, len(c.state["cells"]), note)


warm_and_sample("plateau_n1", "steady N=1")

# grow N=1 -> 2 (force-divide the one cell) -----------------------------------
for cid in list(c.state["cells"]):
    c.state["cells"][cid]["ecoli"]["instance"]._composite.state["agents"]["0"]["divide"] = True
c.run(1.0)
sample("divided_2", len(c.state["cells"]), "immediately post first division")
warm_and_sample("plateau_n2", "steady N=2")

# grow N=2 -> 4 (force-divide both cells) -------------------------------------
for cid in list(c.state["cells"]):
    c.state["cells"][cid]["ecoli"]["instance"]._composite.state["agents"]["0"]["divide"] = True
c.run(1.0)
sample("divided_4", len(c.state["cells"]), "immediately post second division")
warm_and_sample("plateau_n4", "steady N=4")

# --- report incremental per-cell RSS -----------------------------------------
by = {r["label"]: r for r in rows}
n1 = by["plateau_n1"]
n2 = by["plateau_n2"]
n4 = by["plateau_n4"]
base = by["import1"]["rss_mb"]
print("\n=== per-cell RSS (within one process, sim_data shared) ===", flush=True)
print(f"fixed baseline (imports, pre-sim_data): {base:.0f} MB", flush=True)
print(f"baseline+sim_data+1cell (plateau_n1):   {n1['rss_mb']:.0f} MB", flush=True)
inc_1to2 = n2["rss_mb"] - n1["rss_mb"]
inc_2to4 = (n4["rss_mb"] - n2["rss_mb"]) / 2.0
print(f"incremental per-cell RSS N1->N2:        {inc_1to2:.0f} MB/cell", flush=True)
print(f"incremental per-cell RSS N2->N4:        {inc_2to4:.0f} MB/cell", flush=True)
print(f"numpy delta N1->N4 (sim_data sharing):  "
      f"{n4['numpy_mb'] - n1['numpy_mb']:.0f} MB (flat => shared)", flush=True)

out_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
os.makedirs(out_dir, exist_ok=True)
out = os.path.abspath(os.path.join(out_dir, "percell_rss.csv"))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWROTE {out} ({len(rows)} rows)", flush=True)
print("DONE", flush=True)
