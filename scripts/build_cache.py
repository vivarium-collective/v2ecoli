"""Build out/cache/ from the shipped ParCa fixture.

Fast path: hydrates models/parca/parca_state.pkl.gz (committed to the repo)
into a simData pickle, then runs save_cache to produce initial_state.json,
sim_data_cache.dill, and cache_version.json.  No ParCa re-run.

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

import dill

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2ecoli.composite import save_cache
from v2ecoli.library.cache_version import compute_cache_version, write_cache_version
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)


DEFAULT_FIXTURE = "models/parca/parca_state.pkl.gz"
DEFAULT_CACHE_DIR = "out/cache"
DEFAULT_WORKDIR = "out/workflow"


def build_cache(fixture: str, cache_dir: str, workdir: str) -> None:
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

    os.makedirs(workdir, exist_ok=True)
    sd_path = os.path.join(workdir, "simData.cPickle")
    t2 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Pickling sim_data -> {sd_path} ...")
    with open(sd_path, "wb") as f:
        dill.dump(sim_data, f)
    print(f"    pickled in {time.time()-t2:.1f}s "
          f"({os.path.getsize(sd_path)/1e6:.1f} MB)")

    t3 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Building cache at {cache_dir} ...")
    save_cache(sd_path, cache_dir)

    version = write_cache_version(cache_dir, repo_root=repo_root)
    print(f"    cache built in {time.time()-t3:.1f}s")
    print(f"    inputs_hash: {version.inputs_hash[:16]}...")

    print(f"\nTotal: {time.time()-t0:.1f}s")
    print("Cache contents:")
    for f in sorted(os.listdir(cache_dir)):
        p = os.path.join(cache_dir, f)
        print(f"  {f}: {os.path.getsize(p)/1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help=f"ParCa fixture pickle (default: {DEFAULT_FIXTURE})")
    parser.add_argument("--cache", default=DEFAULT_CACHE_DIR, dest="cache_dir",
                        help=f"output cache dir (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR,
                        help=f"intermediate simData.cPickle dir "
                             f"(default: {DEFAULT_WORKDIR})")
    args = parser.parse_args()
    build_cache(args.fixture, args.cache_dir, args.workdir)


if __name__ == "__main__":
    main()
