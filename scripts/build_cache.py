"""Build out/cache/ from the shipped ParCa fixture.

Hydrates models/parca/parca_state.pkl.gz into a SimulationDataEcoli
in memory and emits the simulation-input bundle (initial_state.json,
sim_data_cache.dill, metadata.json, .cache_version) directly — no
intermediate ``simData.cPickle`` round-trip. No ParCa re-run.

Use this whenever:
  - You pulled a branch that changed sim_data.py, the pint boundary,
    or models/parca/parca_state.pkl.gz.
  - Tests or reports abort with StaleCacheError.
  - You removed out/cache/ and want to recreate it.

Usage:
    python scripts/build_cache.py                 # default: out/cache
    python scripts/build_cache.py --cache out/my  # custom destination
    python scripts/build_cache.py --fixture path/to/parca_state.pkl.gz

For a full ParCa re-run (several hours) see docs/generate_full_parca.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)


DEFAULT_FIXTURE = "models/parca/parca_state.pkl.gz"
DEFAULT_CACHE_DIR = "out/cache"


def build_cache(fixture: str, cache_dir: str,
                condition: str | None = None,
                critical_mass_scale: float = 1.0,
                c_period_minutes: float | None = None,
                d_period_minutes: float | None = None,
                dnaa_txn_scale: float = 1.0,
                dnaa_constitutive: bool = False,
                dnaa_stable: bool = False,
                dnaa_translation_efficiency: float | None = None) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading fixture {fixture} ...")
    state = load_parca_state(fixture)
    print(f"    loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Hydrating sim_data ...")
    sim_data = hydrate_sim_data_from_state(state)
    print(f"    hydrated in {time.time()-t1:.1f}s")

    if condition is not None and condition not in sim_data.conditions:
        raise SystemExit(
            f"--condition {condition!r} not in sim_data.conditions "
            f"({sorted(sim_data.conditions.keys())})"
        )
    if condition is not None:
        print(f"    overriding condition: {sim_data.condition!r} → {condition!r}")
    if critical_mass_scale != 1.0:
        print(f"    scaling critical initiation mass M* by {critical_mass_scale:g}")
    if c_period_minutes is not None:
        print(f"    overriding C period: → {c_period_minutes:g} min")
    if d_period_minutes is not None:
        print(f"    overriding D period: → {d_period_minutes:g} min")
    if dnaa_txn_scale != 1.0:
        print(f"    scaling dnaA basal_prob by {dnaa_txn_scale:g}")
    if dnaa_constitutive:
        print(f"    zeroing dnaA delta_prob_matrix rows (constitutive)")
    if dnaa_stable:
        print(f"    zeroing DnaA monomer degradation rate (fully stable)")
    if dnaa_translation_efficiency is not None:
        print(f"    overriding DnaA translation efficiency → {dnaa_translation_efficiency:g}")

    t2 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Building bundle at {cache_dir} ...")
    save_sim_input(sim_data, cache_dir, condition=condition,
                   critical_mass_scale=critical_mass_scale,
                   c_period_minutes=c_period_minutes,
                   d_period_minutes=d_period_minutes,
                   dnaa_txn_scale=dnaa_txn_scale,
                   dnaa_constitutive=dnaa_constitutive,
                   dnaa_stable=dnaa_stable,
                   dnaa_translation_efficiency=dnaa_translation_efficiency)

    version = write_cache_version(cache_dir, repo_root=repo_root)
    print(f"    bundle built in {time.time()-t2:.1f}s")
    print(f"    inputs_hash: {version.inputs_hash[:16]}...")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    print("Bundle contents:")
    for f in sorted(os.listdir(cache_dir)):
        p = os.path.join(cache_dir, f)
        print(f"  {f}: {os.path.getsize(p)/1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help=f"ParCa fixture pickle (default: {DEFAULT_FIXTURE})")
    parser.add_argument("--cache", default=DEFAULT_CACHE_DIR, dest="cache_dir",
                        help=f"output bundle dir (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--condition", default=None,
                        help="Override sim_data.condition before saving the "
                             "bundle (e.g. 'acetate'). Default: leave as-is "
                             "('basal' for a stock ParCa rerun).")
    parser.add_argument("--critical-mass-scale", type=float, default=1.0,
                        help="Multiplier applied to the critical initiation "
                             "mass M* in the chromosome_replication config. "
                             "Use values >1 to delay the per-oriC mass "
                             "criterion (helps prevent mid-cycle re-init "
                             "when C/D are overridden). Default 1.0 = no-op.")
    parser.add_argument("--c-period-minutes", type=float, default=None,
                        help="Direct C-period override (minutes). Solves "
                             "for basal_elongation_rate = replichore_length "
                             "/ (C × 60). Default: leave at ParCa value.")
    parser.add_argument("--d-period-minutes", type=float, default=None,
                        help="Direct D-period override (minutes). Replaces "
                             "the D_period entry in chromosome_replication "
                             "config. Default: leave at ParCa value.")
    parser.add_argument("--dnaa-txn-scale", type=float, default=1.0,
                        help="Multiplier on basal_prob for dnaA-containing "
                             "TUs. e.g. 15 ≈ PDF's 1.5/min/gene rate from "
                             "our observed ~0.1/min baseline. Default 1.0.")
    parser.add_argument("--dnaa-constitutive", action="store_true",
                        help="Zero out dnaA's row in delta_prob_matrix so "
                             "its transcription is invariant to TF state "
                             "(PDF 'constitutive expression' semantics).")
    parser.add_argument("--dnaa-stable", action="store_true",
                        help="Zero DnaA monomer degradation rate (PDF row "
                             "8: 'DnaA degradation rate = 0, fully stable').")
    parser.add_argument("--dnaa-translation-efficiency", type=float,
                        default=None,
                        help="Direct post-ParCa override of DnaA's "
                             "translation efficiency (PDF row 7: 1.0 "
                             "protein/mRNA, Hansen & Atlung 2018). "
                             "Default: leave at ParCa / overrides.py value.")
    args = parser.parse_args()
    build_cache(args.fixture, args.cache_dir, condition=args.condition,
                critical_mass_scale=args.critical_mass_scale,
                c_period_minutes=args.c_period_minutes,
                d_period_minutes=args.d_period_minutes,
                dnaa_txn_scale=args.dnaa_txn_scale,
                dnaa_constitutive=args.dnaa_constitutive,
                dnaa_stable=args.dnaa_stable,
                dnaa_translation_efficiency=args.dnaa_translation_efficiency)


if __name__ == "__main__":
    main()
