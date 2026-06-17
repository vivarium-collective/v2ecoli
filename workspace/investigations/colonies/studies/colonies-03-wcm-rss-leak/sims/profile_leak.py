"""Leak-localization profiler for the inner EcoliWCM (colonies-03).

colonies-02 proved a single EcoliWCM grows RSS ~7.7 MB/sim-s (emitter- and
transport-independent) until OOM. This isolates ONE EcoliWCM (no colony, no
pymunk) and snapshots, every --snap-every ticks:

  * process RSS (psutil)
  * tracemalloc top allocation sites (cumulative + delta vs previous snap)
  * gc object-type histogram (top growers)
  * len(chromosome_history) on the bridge  (bridge.py:243 appends every tick)
  * a coarse size of the inner composite state

Writes profiling/leak_profile.csv (per-snapshot RSS + the headline counters)
and profiling/top_sites.txt (the tracemalloc top-N at the last snapshot and
the largest growth deltas). Those are the committable evidence for the
leak-localized behavior test.

Usage:
    .venv/bin/python studies/colonies-03-wcm-rss-leak/sims/profile_leak.py \
        --ticks 2500 --snap-every 200
"""
from __future__ import annotations

import argparse
import csv
import gc
import linecache  # noqa: F401  (tracemalloc uses it under the hood)
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def _rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def _gc_top_types(n=12):
    c = Counter(type(o).__name__ for o in gc.get_objects())
    return c.most_common(n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=2500)
    ap.add_argument("--snap-every", type=int, default=200)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = STUDY_DIR / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a single EcoliWCM directly via the bridge (no colony wrapper).
    from v2ecoli.bridge import EcoliWCM
    from process_bigraph import allocate_core

    print("Building single EcoliWCM…")
    t0 = time.perf_counter()
    proc = EcoliWCM({"cache_dir": args.cache_dir, "seed": args.seed},
                    core=allocate_core())
    # Drive it directly: the bridge lazily builds its inner composite on the
    # first update(). state mirrors the colony's cell ports.
    state = {"local": {}, "agent_id": "0", "location": (15.0, 15.0), "angle": 0.0}
    proc.update(state, 1.0)  # warmup builds the inner composite
    print(f"  built + warmup in {time.perf_counter() - t0:.1f}s")

    tracemalloc.start(25)
    base_snap = tracemalloc.take_snapshot()
    prev_snap = base_snap

    rows = []
    top_sites_log = []

    for tick in range(args.ticks):
        proc.update(state, 1.0)

        if tick % args.snap_every == 0 or tick == args.ticks - 1:
            rss = _rss_mb()
            snap = tracemalloc.take_snapshot()
            cum = snap.compare_to(base_snap, "lineno")
            delta = snap.compare_to(prev_snap, "lineno")
            prev_snap = snap

            hist = len(getattr(proc, "chromosome_history", []) or [])
            try:
                inner_keys = len(proc._composite.state or {})
            except Exception:
                inner_keys = -1
            top_type = _gc_top_types(6)

            rows.append({
                "tick": tick, "rss_mb": round(rss, 1),
                "chromosome_history_len": hist,
                "inner_state_keys": inner_keys,
                "gc_total_objects": sum(c for _, c in _gc_top_types(10_000)),
                "top_alloc_site": str(cum[0]) if cum else "",
                "top_alloc_kb": round(cum[0].size / 1024, 1) if cum else 0.0,
            })

            top_sites_log.append(f"\n===== tick {tick}  RSS={rss:.0f}MB  "
                                 f"chrom_hist={hist}  gc_top={top_type} =====")
            top_sites_log.append("-- cumulative top 12 (vs base) --")
            for s in cum[:12]:
                top_sites_log.append(f"  {s}")
            top_sites_log.append("-- growth delta top 8 (vs prev snap) --")
            for s in delta[:8]:
                top_sites_log.append(f"  {s}")

            print(f"  tick {tick:4d}: RSS={rss:6.0f}MB  chrom_hist={hist:4d}  "
                  f"top={cum[0].size/1024:8.0f}KB {cum[0] if cum else ''}")

    csv_path = out_dir / "leak_profile.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    txt_path = out_dir / "top_sites.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(top_sites_log))

    rss0, rss1 = rows[0]["rss_mb"], rows[-1]["rss_mb"]
    span = rows[-1]["tick"] - rows[0]["tick"] or 1
    print(f"\nWrote {csv_path} and {txt_path}")
    print(f"RSS {rss0:.0f} -> {rss1:.0f} MB over {span} ticks "
          f"(~{(rss1 - rss0) / span:.2f} MB/sim-s). chromosome_history "
          f"{rows[0]['chromosome_history_len']} -> "
          f"{rows[-1]['chromosome_history_len']}.")
    print("Inspect top_sites.txt: the site(s) whose cumulative size tracks "
          "the RSS climb are the leak. >=~80% attribution clears leak-localized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
