"""Make DnaA-ADP binding NON-equilibrium (Haochen 2026-05-31, dnaa-2 Step 3).

v2ecoli's ecoli-equilibrium step integrates every reaction to steady state each
tick, so it cannot represent a kinetically-trapped DnaA-ADP: it instantly drains
hydrolysis-produced DnaA-ADP back to apo -> DnaA-ATP, pinning the DnaA-ATP
fraction at ~0.997. Haochen: the DnaA-ADP unbinding is slow (k_r ~1e-7/s) and
NOT at equilibrium.

This patches a cache so the equilibrium step LEAVES DnaA-ADP alone: it zeroes the
forward AND reverse rates of MONOMER0-4565_RXN (apo + ADP <-> DnaA-ADP) in
configs['ecoli-equilibrium'].fluxesAndMoleculesToSS._data. With that reaction
inert in the equilibrium, DnaA-ADP is formed ONLY by intrinsic hydrolysis
(DnaaIntrinsicHydrolysis) and drained ONLY by the slow kinetic release
(DnaaAdpRelease) — i.e. the `dnaa_nucleotide` feature handles the full cycle
kinetically. The DnaA-ATP equilibrium (MONOMER0-160_RXN) is untouched (fast,
correct).

Usage::

    python scripts/patch_dnaa_adp_nonequilibrium.py \
        --in-cache out/cache-succinate-mechA-1.7e-3 \
        --out-cache out/cache-succinate-mechA-1.7e-3-adpkinetic
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
import v2ecoli  # noqa: F401  (let dill resolve v2ecoli.* pickled classes)

import dill

RXN_ID = "MONOMER0-4565_RXN"  # apo-DnaA + ADP <-> DnaA-ADP


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-cache", default="out/cache-succinate-mechA-1.7e-3")
    p.add_argument("--out-cache", required=True)
    args = p.parse_args()

    src, dst = Path(args.in_cache), Path(args.out_cache)
    if not src.is_dir():
        print(f"ERROR: source cache {src} not found", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"removing existing {dst}")
        shutil.rmtree(dst)
    print(f"cloning {src} -> {dst}")
    shutil.copytree(src, dst)

    dill_path = dst / "sim_data_cache.dill"
    t0 = time.time()
    cache = dill.load(open(dill_path, "rb"))
    eq = cache["configs"]["ecoli-equilibrium"]
    rxn_ids = list(eq["reaction_ids"])
    if RXN_ID not in rxn_ids:
        print(f"ERROR: {RXN_ID} not in ecoli-equilibrium.reaction_ids",
              file=sys.stderr)
        return 1
    i = rxn_ids.index(RXN_ID)
    data = eq["fluxesAndMoleculesToSS"]["_data"]
    before = (data["rates_fwd"][i], data["rates_rev"][i])
    data["rates_fwd"][i] = 0.0
    data["rates_rev"][i] = 0.0
    print(f"  {RXN_ID} (idx {i}) rates fwd/rev: {before} -> (0.0, 0.0)")

    with open(dill_path, "wb") as f:
        dill.dump(cache, f)
    print(f"  saved patched cache in {time.time()-t0:.1f}s")

    manifest = dst / "dnaa_adp_nonequilibrium.json"
    import json
    manifest.write_text(json.dumps({
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_cache": str(src),
        "change": f"zeroed equilibrium rates for {RXN_ID} (DnaA-ADP binding) "
                  "so DnaA-ADP is handled kinetically (hydrolysis source + "
                  "slow DnaaAdpRelease sink), not as fast equilibrium",
        "reaction_index": i,
        "rates_before": list(before),
    }, indent=2))
    print(f"  wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
