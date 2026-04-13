"""
Sweep starting [GLC]ₑₓₜ for baseline (DM OFF) vs enforced (DM ON).

For each starting glucose concentration, runs both modes and reports
the observed doubling time. The 0 mM column is the key test: a
properly constrained model should not grow without carbon, while the
DM-OFF model will keep growing on the LP's homeostatic slacks.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from nutrient_growth_report import (
    K12_TARGET_DOUBLING_MIN,
    DEFAULT_ENV_VOLUME_L,
    _patch_env_volume,
    _extract_snapshot,
)


def _patch_initial_glc(composite, glc_mM: float):
    """Override boundary.external.GLC for every cell agent in the state."""
    agents = composite.state.get("agents") or {}
    n_patched = 0
    for _aid, agent in agents.items():
        if not isinstance(agent, dict):
            continue
        boundary = agent.setdefault("boundary", {})
        external = boundary.setdefault("external", {})
        external["GLC"] = float(glc_mM)
        n_patched += 1
    return n_patched


def fit_doubling_full(snaps):
    """Fit log(dry_mass) vs t over the entire run. Returns minutes or None."""
    rows = [s for s in snaps if s.get("dry_mass", 0) > 0]
    if len(rows) < 4:
        return None
    t = np.array([r["time"] for r in rows], dtype=float)
    m = np.array([r["dry_mass"] for r in rows], dtype=float)
    slope, _ = np.polyfit(t, np.log(m), 1)
    if slope <= 0:
        return None
    return float(np.log(2) / slope / 60.0)


def fit_doubling_window(snaps, t_start_s: float = 0.0, t_end_s: float = 300.0):
    """Fit only over a narrow early window so the rate reflects the
    initial-glucose condition before depletion changes the regime."""
    rows = [s for s in snaps
            if s.get("dry_mass", 0) > 0
            and t_start_s <= s["time"] <= t_end_s]
    if len(rows) < 4:
        return None
    t = np.array([r["time"] for r in rows], dtype=float)
    m = np.array([r["dry_mass"] for r in rows], dtype=float)
    slope, _ = np.polyfit(t, np.log(m), 1)
    if slope <= 0:
        return None
    return float(np.log(2) / slope / 60.0)


def run_one(glc_mM: float, dark_matter: bool, duration: int,
            snapshot_interval: int, env_volume_L: float):
    from v2ecoli.composite import make_composite

    os.environ["V2ECOLI_NUTRIENT_GROWTH"] = "1"
    os.environ["V2ECOLI_DARK_MATTER"] = "1" if dark_matter else "0"

    composite = make_composite(cache_dir="out/cache", seed=0)
    _patch_env_volume(composite, env_volume_L)
    n = _patch_initial_glc(composite, glc_mM)

    label = f"[glc={glc_mM:5.2f} {'DM-ON ' if dark_matter else 'DM-OFF'}]"
    snaps = [_extract_snapshot(composite.state, 0.0)]
    total = 0.0
    t0 = time.time()
    crashed = None
    while total < duration:
        chunk = min(snapshot_interval, duration - total)
        try:
            composite.run(chunk)
        except Exception as e:
            crashed = total + chunk
            print(f"  {label} crash at t={crashed:.0f}s: {type(e).__name__}")
            break
        total += chunk
        snaps.append(_extract_snapshot(composite.state, total))
    wall = time.time() - t0

    final_dry = snaps[-1]["dry_mass"]
    initial_dry = snaps[0]["dry_mass"]

    # Two doubling-time estimates:
    # - "early" window: 0–300s (5 min) so we measure the initial-glucose
    #   regime even when depletion is fast.
    # - "full" window: whole sim, useful when growth is slow/null.
    dt_early = fit_doubling_window(snaps, 0, 300)
    dt_full = fit_doubling_full(snaps)

    # Final glucose to verify whether the cell actually depleted it.
    glc_final = snaps[-1].get("glc_ext_mM")

    return {
        "glc_initial_mM": glc_mM,
        "dark_matter": dark_matter,
        "wall_s": wall,
        "snaps": len(snaps),
        "crashed_at": crashed,
        "initial_dry_fg": initial_dry,
        "final_dry_fg": final_dry,
        "mass_ratio": (final_dry / initial_dry) if initial_dry > 0 else 0,
        "dt_early_min": dt_early,
        "dt_full_min": dt_full,
        "glc_final_mM": glc_final,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glc-values", type=float, nargs="+",
                   default=[22.0, 5.0, 1.0, 0.1, 0.0])
    p.add_argument("--duration", type=int, default=900,
                   help="sim seconds per condition (default 900)")
    p.add_argument("--snapshot", type=int, default=10)
    p.add_argument("--env-volume-L", type=float,
                   default=DEFAULT_ENV_VOLUME_L,
                   help="env volume per cell (default 10 fL)")
    args = p.parse_args()

    rows = []
    for glc in args.glc_values:
        for dm in (False, True):
            print(f"\n=== glc={glc} mM, DM {'ON' if dm else 'OFF'} ===")
            r = run_one(glc, dm, args.duration, args.snapshot,
                         args.env_volume_L)
            rows.append(r)
            dt_e = f"{r['dt_early_min']:.1f}" if r['dt_early_min'] else "—"
            dt_f = f"{r['dt_full_min']:.1f}" if r['dt_full_min'] else "—"
            print(f"  → mass {r['initial_dry_fg']:.0f} → {r['final_dry_fg']:.0f} fg "
                  f"(ratio {r['mass_ratio']:.3f}); "
                  f"dt early {dt_e} min, full {dt_f} min")

    print("\n" + "=" * 84)
    print("SUMMARY: doubling time vs initial [GLC]")
    print("Target K-12 MG1655 in glucose minimal media: "
          f"{K12_TARGET_DOUBLING_MIN:.0f} min")
    print()
    print(f"{'[GLC]₀ (mM)':>11} | {'mode':>7} | {'dt early (min)':>15} "
          f"| {'dt full (min)':>14} | {'mass ratio':>10} | "
          f"{'final [GLC] mM':>15}")
    print("-" * 84)
    for r in rows:
        mode = "DM ON " if r["dark_matter"] else "DM OFF"
        dt_e = f"{r['dt_early_min']:.2f}" if r['dt_early_min'] else "—"
        dt_f = f"{r['dt_full_min']:.2f}" if r['dt_full_min'] else "—"
        gf = (f"{r['glc_final_mM']:.4f}"
              if r['glc_final_mM'] is not None else "—")
        print(f"{r['glc_initial_mM']:>11.2f} | {mode:>7} | {dt_e:>15} | "
              f"{dt_f:>14} | {r['mass_ratio']:>10.3f} | {gf:>15}")

    # The key finding callout
    print()
    print("INTERPRETATION:")
    zero_rows = [r for r in rows if r["glc_initial_mM"] == 0]
    if len(zero_rows) == 2:
        off = next(r for r in zero_rows if not r["dark_matter"])
        on = next(r for r in zero_rows if r["dark_matter"])
        print(f"  At [GLC]₀ = 0:")
        print(f"    DM OFF mass ratio = {off['mass_ratio']:.3f} "
              f"({'GREW' if off['mass_ratio'] > 1.05 else 'flat'})")
        print(f"    DM ON  mass ratio = {on['mass_ratio']:.3f} "
              f"({'GREW' if on['mass_ratio'] > 1.05 else 'flat'})")
        if off['mass_ratio'] > on['mass_ratio'] + 0.05:
            print("  → DM ON correctly prevents growth without carbon;")
            print("    DM OFF grows via the LP's slack pseudofluxes.")


if __name__ == "__main__":
    main()
