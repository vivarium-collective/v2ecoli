"""Single-gen with_aa cell wrapped in parquet_emitter() — post-merge sanity check.

Validates that:
  - the merged main parquet emitter wiring works end-to-end
  - the D-period division trigger fires (with_aa should divide ~21 min)
  - the merged cache_version + LoadSimData(condition=with_aa) path is healthy

Usage:
    python scripts/run_with_aa_parquet_singlegen.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli import build_composite
from v2ecoli.core import build_core
from v2ecoli.composites._helpers import parquet_emitter
from v2ecoli.library.parquet_emitter import ParquetEmitter


CACHE_DIR = "out/cache_with_aa_apr24"
PARQUET_OUT = "out/with_aa_postmerge_parquet"
MAX_DURATION = 3600  # 60 min — plenty of headroom for the ~21 min cycle
SNAPSHOT_INTERVAL = 60


def main() -> None:
    print(f"cache:       {CACHE_DIR}")
    print(f"parquet out: {PARQUET_OUT}")
    print(f"max sim:     {MAX_DURATION}s ({MAX_DURATION/60:.0f} min)")
    print("-" * 60)
    os.makedirs(PARQUET_OUT, exist_ok=True)

    core = build_core()
    t_wall0 = time.time()

    with parquet_emitter(
        out_dir=PARQUET_OUT,
        experiment_id="with_aa_postmerge",
        lineage_seed=0,
        agent_id="0",
        generation=1,
    ):
        t_build = time.time()
        comp = build_composite("baseline", cache_dir=CACHE_DIR)
        cell0 = comp.state["agents"]["0"]
        dry0 = float(cell0["listeners"]["mass"].get("dry_mass", 0))
        print(f"  composite built in {time.time()-t_build:.1f}s — "
              f"initial dry_mass={dry0:.1f} fg")

        total_run = 0.0
        divided = False
        while total_run < MAX_DURATION:
            chunk = min(SNAPSHOT_INTERVAL, MAX_DURATION - total_run)
            try:
                comp.run(chunk)
            except Exception as e:
                err = str(e)
                if "divide" in err.lower() or "_add" in err or "_remove" in err:
                    total_run += chunk
                    divided = True
                    break
                if comp.state.get("agents", {}).get("0") is None:
                    divided = True
                    break
                raise
            total_run += chunk
            cur = comp.state.get("agents", {}).get("0")
            if cur is None:
                divided = True
                break
            dry = float(cur["listeners"]["mass"].get("dry_mass", 0))
            print(f"  t={total_run/60:5.1f} min  dry_mass={dry:.1f} fg")
            if divided:
                break

        # Flush the parquet trailing batch before the ctx mgr clears the override.
        n_closed = ParquetEmitter.flush_all_in_composite(comp, success=divided)
        print(f"  flushed {n_closed} ParquetEmitter instance(s)")

    wall = time.time() - t_wall0
    print("-" * 60)
    print(f"final: sim={total_run/60:.1f} min, wall={wall:.0f}s, divided={divided}")


if __name__ == "__main__":
    main()
