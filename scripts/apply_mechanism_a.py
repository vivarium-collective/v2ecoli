"""Apply Mechanism A — runtime overwrite of dnaA's per-promoter init_prob.

PDF feedback's working solution for dnaa-1: set
sim_data.genetic_perturbations["TU00259[c]"] = 2e-3 at runtime, leaving
baseline TE / mRNA t½ / DnaA protein t½ unchanged.

Because v2ecoli's composite generators load the cache bundle directly
(no LoadSimData re-run on composite build — see
v2ecoli/composites/baseline.py:405-409), we apply the perturbation by
patching the bundle's saved config in-place:
``configs['ecoli-transcript-initiation']['perturbations']`` is the dict
the transcript-initiation process iterates each tick to override
basal_prob entries.

Usage::

    python scripts/apply_mechanism_a.py \
        --in-cache out/cache-succinate \
        --out-cache out/cache-succinate-mechA-2e-3 \
        --tu TU00259[c] --value 2e-3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Make v2ecoli importable so dill can resolve pickled classes from the cache.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
# Force v2ecoli import so dill's find_class can resolve v2ecoli.* references
# embedded in the bundle (e.g. v2ecoli.types.quantity types).
import v2ecoli  # noqa: F401

import dill


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-cache", default="out/cache-succinate",
                   help="Source cache directory (default: out/cache-succinate)")
    p.add_argument("--out-cache", required=True,
                   help="Destination cache directory")
    p.add_argument("--tu", default="TU00259[c]",
                   help="Transcription unit id (default: TU00259[c] = dnaA operon)")
    p.add_argument("--value", type=float, default=2e-3,
                   help="Per-promoter init_prob override value (default: 2e-3)")
    args = p.parse_args()

    src = Path(args.in_cache)
    dst = Path(args.out_cache)
    if not src.is_dir():
        print(f"ERROR: source cache {src} not found", file=sys.stderr)
        return 1

    # Clone the cache directory; only sim_data_cache.dill needs patching.
    if dst.exists():
        print(f"removing existing {dst} for fresh patch")
        shutil.rmtree(dst)
    print(f"cloning {src} -> {dst}")
    shutil.copytree(src, dst)

    dill_path = dst / "sim_data_cache.dill"
    t0 = time.time()
    print(f"loading {dill_path}")
    with open(dill_path, "rb") as f:
        cache = dill.load(f)

    tic = cache["configs"]["ecoli-transcript-initiation"]
    perturbations = tic.get("perturbations") or {}
    before = perturbations.get(args.tu, "(absent)")
    perturbations[args.tu] = float(args.value)
    tic["perturbations"] = perturbations

    print(f"  perturbations[{args.tu!r}]: {before} -> {args.value}")

    with open(dill_path, "wb") as f:
        dill.dump(cache, f)
    print(f"  saved patched cache in {time.time()-t0:.1f}s")

    # Write a tiny manifest describing the patch (for provenance).
    manifest = {
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_cache": str(src),
        "mechanism": "A",
        "description": "runtime overwrite of per-promoter init_prob via "
                       "configs.ecoli-transcript-initiation.perturbations",
        "perturbations": {args.tu: float(args.value)},
        "unchanged": [
            "dnaA TE (baseline 0.35)",
            "dnaA mRNA half-life (baseline 1.9 min)",
            "DnaA protein half-life (baseline 280 min)",
            "ParCa C-period (baseline 40 min for succinate)",
            "ParCa D-period (baseline 20 min for succinate)",
        ],
    }
    manifest_path = dst / "mechanism_a.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
