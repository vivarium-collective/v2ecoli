"""Probe natural division: build the bare v2ecoli baseline composite (NOT
inside the colony bridge), tick it forward, and print the chromosome /
divide state. The point is to isolate whether natural division fires at
all in the inner WCM, separately from the colony+bridge integration.

Run from the worktree root:

    .venv/bin/python studies/colonies-01-hpc-readiness/sims/probe_division.py [--ticks N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))


def snapshot(comp) -> dict:
    cell = comp.state.get("agents", {}).get("0", comp.state)
    listeners = cell.get("listeners", {}) or {}
    mass = listeners.get("mass", {}) or {}
    unique = cell.get("unique", {}) or {}
    fc = unique.get("full_chromosome")
    rep = unique.get("active_replisome")

    n_chrom_entries = 0
    n_chrom_active = 0
    div_times = []
    triggered = []
    if fc is not None and hasattr(fc, "dtype"):
        n_chrom_entries = len(fc)
        if "_entryState" in fc.dtype.names:
            n_chrom_active = int(fc["_entryState"].sum())
        if "division_time" in fc.dtype.names:
            div_times = fc["division_time"].tolist()
        if "has_triggered_division" in fc.dtype.names:
            triggered = fc["has_triggered_division"].tolist()

    n_forks = 0
    if rep is not None and hasattr(rep, "dtype") and "_entryState" in rep.dtype.names:
        n_forks = int(rep["_entryState"].sum())

    return {
        "global_time": float(comp.state.get("global_time", 0)),
        "divide_flag": bool(cell.get("divide", False)),
        "dry_mass": float(mass.get("dry_mass", 0.0)),
        "dna_mass": float(mass.get("dna_mass", 0.0)),
        "n_chrom_active": n_chrom_active,
        "n_chrom_entries": n_chrom_entries,
        "n_forks": n_forks,
        "division_time": div_times,
        "has_triggered_division": triggered,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=3500)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--cache-dir", default="out/cache")
    args = parser.parse_args()

    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    from process_bigraph import Composite
    import v2ecoli.types  # noqa: F401  -- registers dispatch

    core = build_core()
    document = baseline(core=core, seed=0, cache_dir=args.cache_dir)
    comp = Composite(document, core=core)

    print(f"tick     t_sim  divide  dry_mass  dna_mass  chrom  forks  div_time[]  has_triggered[]")
    print(snapshot_line(0, snapshot(comp)))

    t_wall = time.perf_counter()
    for tick in range(1, args.ticks + 1):
        comp.run(1.0)
        if tick % args.every == 0 or tick == args.ticks:
            snap = snapshot(comp)
            print(snapshot_line(tick, snap))
            if snap["divide_flag"]:
                print(f"!! DIVIDE FLAG SET at tick {tick}, stopping")
                break

    print(f"wall: {time.perf_counter() - t_wall:.1f}s")
    return 0


def snapshot_line(tick: int, s: dict) -> str:
    div_t = ",".join(f"{x:.0f}" for x in s["division_time"]) if s["division_time"] else "[]"
    triggered = ",".join("1" if x else "0" for x in s["has_triggered_division"]) if s["has_triggered_division"] else "[]"
    return (
        f"{tick:5d}  {s['global_time']:6.0f}  {str(s['divide_flag']):5s}  "
        f"{s['dry_mass']:7.1f}  {s['dna_mass']:7.2f}  "
        f"{s['n_chrom_active']}/{s['n_chrom_entries']}  "
        f"{s['n_forks']:5d}  {div_t:15s}  {triggered}"
    )


if __name__ == "__main__":
    sys.exit(main())
