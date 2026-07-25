"""Profile one tick of the consensus elongation composite.

Times:
- Composite build (one-shot startup cost — ParCa load, partitioner setup)
- One full tick wall clock
- cProfile of the tick — top-30 by cumulative time + targeted kinetic-class breakdown
- A/B: same tick with consensus flags OFF for comparison (shows what the
  consensus additions actually cost vs the legacy kinetic baseline)

Usage::

    python scripts/profile_consensus_tick.py [--consensus-only]

Output: human-readable summary to stdout. Each tick is multi-minute, so
the full run (build + 2 ticks for A/B) takes 10-30 min depending on
hardware.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


CACHE = "out/cache"


def build_composite(include_aa_supply: bool, ppgpp_regulation: bool):
    from process_bigraph import Composite
    from v2ecoli.composites.kinetic_charging_baseline import (
        kinetic_charging_baseline,
    )
    from v2ecoli.core import build_core

    core = build_core()
    doc = kinetic_charging_baseline(
        core=core,
        seed=0,
        cache_dir=CACHE,
        config_overrides={
            "ecoli-polypeptide-elongation.include_aa_supply": include_aa_supply,
            "ecoli-polypeptide-elongation.ppgpp_regulation": ppgpp_regulation,
        },
    )
    return Composite(doc, core=core)


def profile_one_tick(label: str, include_aa_supply: bool, ppgpp_regulation: bool) -> dict:
    print(f"\n{'='*70}")
    print(f"PROFILE: {label}")
    print(f"  include_aa_supply={include_aa_supply}")
    print(f"  ppgpp_regulation={ppgpp_regulation}")
    print(f"{'='*70}")

    # --- Composite build ---
    t0 = time.perf_counter()
    composite = build_composite(include_aa_supply, ppgpp_regulation)
    build_time = time.perf_counter() - t0
    print(f"\n[build] composite built in {build_time:.2f}s")

    # --- Tick wall clock ---
    print("[tick] running 1 tick under cProfile (this is the slow part)...")
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    composite.run(interval=1.0)
    profiler.disable()
    tick_time = time.perf_counter() - t0
    print(f"[tick] 1 tick completed in {tick_time:.2f}s ({tick_time / 60:.2f} min)")

    # --- cProfile summary ---
    print(f"\n[cprofile] top 30 by cumulative time:")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # --- Targeted kinetic-class breakdown ---
    print(f"[cprofile] kinetic_charging-specific calls (cumulative):")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats("kinetic_charging|polypeptide_elongation|kinetics.py|metabolism.py")
    print(s.getvalue()[:6000])  # cap to keep output manageable

    # --- solve_ivp specifically ---
    print(f"[cprofile] solve_ivp + ode_model + supply hot paths:")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats("solve_ivp|ode_model|supply_function|amino_acid|ppgpp_metabolite")
    print(s.getvalue()[:6000])

    return {
        "label": label,
        "build_time_s": build_time,
        "tick_time_s": tick_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consensus-only", action="store_true",
        help="Skip the flags-off A/B baseline (saves ~half the runtime)",
    )
    args = parser.parse_args()

    summary = []

    # Consensus mode (flags ON) — what the sweep actually runs
    summary.append(profile_one_tick(
        "consensus (flags ON)",
        include_aa_supply=True, ppgpp_regulation=True,
    ))

    if not args.consensus_only:
        # Bare kinetic baseline (flags OFF) — what existed pre-consensus
        summary.append(profile_one_tick(
            "legacy kinetic (flags OFF)",
            include_aa_supply=False, ppgpp_regulation=False,
        ))

    # Final summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for s in summary:
        print(f"  {s['label']!r}:")
        print(f"    build_time = {s['build_time_s']:.2f}s")
        print(f"    tick_time  = {s['tick_time_s']:.2f}s "
              f"({s['tick_time_s'] / 60:.2f} min)")
    if len(summary) == 2:
        ratio = summary[0]["tick_time_s"] / summary[1]["tick_time_s"]
        print(f"\n  consensus / legacy tick ratio = {ratio:.2f}x")


if __name__ == "__main__":
    main()
