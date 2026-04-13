"""
v_max sweep for glucose Michaelis-Menten uptake.

Runs a short baseline sim at each v_max and reports the observed
carbon-replete doubling time, to find the value that matches the K-12
MG1655 target of 44 min.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def _clear_env_flags():
    for k in ("V2ECOLI_DARK_MATTER",):
        os.environ.pop(k, None)


def sweep(vmax_values, duration, env_volume_L, km):
    # Import after arg-parse so matplotlib setup in the report module
    # doesn't fire if this file is imported for testing.
    from nutrient_growth_report import (
        K12_TARGET_DOUBLING_MIN,
        run_single_cell,
        estimate_doubling_time_min,
        _patch_env_volume,
    )
    from v2ecoli.composite import make_composite

    os.environ["V2ECOLI_NUTRIENT_GROWTH"] = "1"
    os.environ["V2ECOLI_DARK_MATTER"] = "0"

    # The MM parameters are applied via env vars read inside
    # metabolic_kinetics.py. Probe whether that's the wiring:
    # fall back to monkey-patching the config if needed.
    rows = []
    for vmax in vmax_values:
        os.environ["V2ECOLI_MM_VMAX"] = f"{vmax}"
        os.environ["V2ECOLI_MM_KM"] = f"{km}"
        print(f"\n=== v_max = {vmax:.1f} mmol/gDCW/h ===")
        t0 = time.time()
        data = run_single_cell(duration, snapshot_interval=10,
                               env_volume_L=env_volume_L,
                               label=f"vmax={vmax}")
        dt = estimate_doubling_time_min(data["snapshots"])
        rows.append((vmax, dt, time.time() - t0))
        dt_str = f"{dt:.2f}" if dt is not None else "—"
        print(f"  vmax={vmax:.1f} → doubling {dt_str} min "
              f"(target {K12_TARGET_DOUBLING_MIN})")

    print("\n" + "=" * 50)
    print(f"{'v_max (mmol/gDCW/h)':>22} | {'doubling (min)':>14} | {'Δ (min)':>8} | wall")
    print("-" * 60)
    for vmax, dt, wall in rows:
        if dt is None:
            print(f"{vmax:>22.1f} | {'—':>14} | {'—':>8} | {wall:.0f}s")
        else:
            delta = dt - K12_TARGET_DOUBLING_MIN
            print(f"{vmax:>22.1f} | {dt:>14.2f} | {delta:>+8.2f} | {wall:.0f}s")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmax-values", type=float, nargs="+",
                        default=[15.0, 17.0, 18.5, 20.0, 23.0, 26.0])
    parser.add_argument("--duration", type=int, default=1500,
                        help="sim seconds per sweep point (default 1500)")
    parser.add_argument("--env-volume-L", type=float, default=1e-13,
                        help="env volume per cell (default 1e-13 = 100 fL "
                             "to delay depletion so fit window is long)")
    parser.add_argument("--km", type=float, default=0.01)
    args = parser.parse_args()

    _clear_env_flags()
    sweep(args.vmax_values, args.duration, args.env_volume_L, args.km)


if __name__ == "__main__":
    main()
