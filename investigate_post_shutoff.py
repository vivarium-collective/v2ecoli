"""
Investigate what sustains the cell after glucose runs out (DM-OFF).

Runs a baseline nutrient-growth sim, partitions snapshots into
pre-shutoff ([GLC] > 1 mM) and post-shutoff ([GLC] < 0.01 mM) phases,
and reports the carbon-bearing imports and secretions weighted by
per-molecule carbon counts. The output answers:

  - What carbon sources is the cell drawing from after glucose is gone?
  - How does that compare to the carbon needed to support observed
    biomass gain in the post-shutoff window?
  - Where is the unaccounted carbon coming from?
"""

from __future__ import annotations

import os
import sys

import numpy as np

from nutrient_growth_report import (
    DEFAULT_ENV_VOLUME_L,
    run_single_cell,
    _glc_shutoff_min,
)
from v2ecoli.library.carbon_counts import (
    CARBON_COUNTS, MOLECULAR_WEIGHTS, carbon_of, mw_of,
)


# Carbon mass fraction of dry mass: 48% (standard E. coli)
C_MASS_FRACTION = 0.48
# Approximate molar mass of biomass C (12 g/mol)
C_MW = 12.0


def split_phases(snaps):
    pre = [s for s in snaps if (s.get("glc_ext_mM") or 0) > 1.0]
    post = [s for s in snaps if (s.get("glc_ext_mM") or 0) <= 0.01]
    return pre, post


def per_molecule_mean(snaps, sign):
    """Mean count/step for each molecule in the snapshots, optionally
    filtered by sign (1 = secretion only, -1 = import only, 0 = both)."""
    mols: set[str] = set()
    for s in snaps:
        mols.update(s.get("exchange_counts", {}) or {})
    out = {}
    for m in mols:
        vals = [s.get("exchange_counts", {}).get(m, 0) for s in snaps]
        if sign > 0:
            vals = [v for v in vals if v > 0]
        elif sign < 0:
            vals = [v for v in vals if v < 0]
        if not vals:
            continue
        out[m] = float(np.mean(vals))
    return out


def carbon_table(per_mol_mean, dt_per_step_s, dry_mass_avg_g, label):
    """Convert count/step per molecule into mmol/gDCW/h and mmol C/h.

    counts/step ÷ N_A ÷ dry_mass_g ÷ dt_h × 1000 = mmol/gDCW/h
    For carbon flux: × carbon_count_per_molecule
    """
    N_A = 6.022e23
    dt_h = dt_per_step_s / 3600
    rows = []
    for m, c_per_step in per_mol_mean.items():
        c_count = carbon_of(m) if "[" in m else carbon_of(m.split("[")[0])
        # try both with and without compartment tag
        if c_count == 0:
            stem = m.split("[")[0] if "[" in m else m
            c_count = CARBON_COUNTS.get(stem, 0)
        flux_mmol_gdcw_h = (c_per_step / N_A / dry_mass_avg_g / dt_h) * 1000
        c_flux_mmol_h = abs(flux_mmol_gdcw_h * c_count) * dry_mass_avg_g
        rows.append((m, c_per_step, flux_mmol_gdcw_h, c_count,
                     c_flux_mmol_h))
    rows.sort(key=lambda r: -r[4])  # by carbon flux desc
    return rows


def print_table(rows, label, n=15):
    print(f"\n=== {label} (top {n} by |C flux|) ===")
    print(f"{'Molecule':<30} {'count/step':>12} {'mmol/gDCW/h':>14} "
          f"{'C/mol':>6} {'mmol C/h':>10}")
    print("-" * 80)
    for m, cps, flux, c, cf in rows[:n]:
        print(f"{m:<30} {cps:>+12.0f} {flux:>+14.4f} "
              f"{c:>6} {cf:>10.4f}")


def investigate(duration_s: int, env_volume_L: float):
    os.environ["V2ECOLI_NUTRIENT_GROWTH"] = "1"
    os.environ["V2ECOLI_DARK_MATTER"] = "0"
    print(f"Running baseline {duration_s}s with env={env_volume_L:g} L "
          f"(no DM)")
    data = run_single_cell(duration_s, snapshot_interval=10,
                           env_volume_L=env_volume_L, label="baseline")
    snaps = data["snapshots"]

    t_shut = _glc_shutoff_min(snaps)
    if t_shut is None:
        print("ERROR: glucose did not deplete in this run; "
              "use a smaller env volume.")
        return
    print(f"\nGlucose depleted at t={t_shut:.1f} min "
          f"(of {duration_s/60:.0f} min total)")

    pre, post = split_phases(snaps)
    print(f"Pre-shutoff: {len(pre)} snapshots, "
          f"post-shutoff: {len(post)} snapshots")

    # Average dry mass during post-shutoff (for unit conversion)
    dry_pre_g  = float(np.mean([s["dry_mass"] for s in pre])) * 1e-15
    dry_post_g = float(np.mean([s["dry_mass"] for s in post])) * 1e-15
    dt_per_step = 10.0  # snapshot interval

    # Imports (negative exchange) — flip sign for printing as positive uptake
    imports_pre  = per_molecule_mean(pre,  sign=-1)
    imports_post = per_molecule_mean(post, sign=-1)
    secret_post  = per_molecule_mean(post, sign=+1)

    rows_imp_pre  = carbon_table(imports_pre,  dt_per_step, dry_pre_g,
                                  "imports pre")
    rows_imp_post = carbon_table(imports_post, dt_per_step, dry_post_g,
                                  "imports post")
    rows_sec_post = carbon_table(secret_post,  dt_per_step, dry_post_g,
                                  "secret post")

    print_table(rows_imp_pre,  "PRE-shutoff imports",  n=10)
    print_table(rows_imp_post, "POST-shutoff imports", n=15)
    print_table(rows_sec_post, "POST-shutoff secretions", n=15)

    # Carbon balance during post-shutoff
    c_in_post  = sum(r[4] for r in rows_imp_post)
    c_out_post = sum(r[4] for r in rows_sec_post)
    c_net_post = c_in_post - c_out_post

    # Biomass C gain rate: Δ dry_mass × C_fraction / C_MW
    if len(post) >= 2:
        dm_first = post[0]["dry_mass"] * 1e-15  # g
        dm_last  = post[-1]["dry_mass"] * 1e-15
        dt_post_h = (post[-1]["time"] - post[0]["time"]) / 3600
        biomass_c_gain_mmol_h = ((dm_last - dm_first) * C_MASS_FRACTION
                                  / C_MW * 1000) / dt_post_h
    else:
        biomass_c_gain_mmol_h = 0

    print(f"\n=== POST-SHUTOFF CARBON BALANCE ===")
    print(f"  Imports (C in):       {c_in_post:>10.4f} mmol C / h")
    print(f"  Secretions (C out):   {c_out_post:>10.4f} mmol C / h")
    print(f"  Net (in − out):       {c_net_post:>+10.4f} mmol C / h")
    print(f"  Biomass C gain:       {biomass_c_gain_mmol_h:>+10.4f} mmol C / h")
    deficit = biomass_c_gain_mmol_h - c_net_post
    print(f"  Carbon deficit:       {deficit:>+10.4f} mmol C / h")
    if deficit > 0:
        print(f"\n  → {deficit:.4f} mmol C/h appears in biomass without a")
        print(f"    matching boundary import. This is the slack-pseudoflux")
        print(f"    contribution — mass conjured by the LP's homeostatic")
        print(f"    objective.")

    # Mass balance summary as well
    print(f"\n=== POST-SHUTOFF MASS BALANCE ===")
    if len(post) >= 2:
        dry_gain_fg = (post[-1]["dry_mass"] - post[0]["dry_mass"])
        dt_post_min = (post[-1]["time"] - post[0]["time"]) / 60
        print(f"  Window: t = {post[0]['time']/60:.1f} – "
              f"{post[-1]['time']/60:.1f} min ({dt_post_min:.1f} min)")
        print(f"  Dry mass: {post[0]['dry_mass']:.1f} → "
              f"{post[-1]['dry_mass']:.1f} fg (Δ = {dry_gain_fg:+.1f} fg)")
        print(f"  Effective post-shutoff growth rate: "
              f"{dry_gain_fg/dt_post_min:.2f} fg/min")


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=int, default=2520)
    p.add_argument("--env-volume-L", type=float, default=DEFAULT_ENV_VOLUME_L)
    args = p.parse_args()
    investigate(args.duration, args.env_volume_L)


if __name__ == "__main__":
    main()
