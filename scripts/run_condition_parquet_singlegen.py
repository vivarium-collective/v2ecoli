"""Single-gen baseline parquet sim for one nutrient condition — post-merge.

Generalised from run_with_aa_parquet_singlegen.py. Each condition writes
its parquet history under out/<condition>_postmerge_parquet/.

Usage:
    python scripts/run_condition_parquet_singlegen.py --condition with_aa --max-min 60
"""
from __future__ import annotations

import argparse
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


SNAPSHOT_INTERVAL = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    help="ParCa condition name (basal | with_aa | acetate | "
                         "succinate | no_oxygen)")
    ap.add_argument("--cache-dir", default=None,
                    help="Override cache dir (default: out/cache_<condition>_apr24)")
    ap.add_argument("--out-dir", default=None,
                    help="Override parquet out dir (default: "
                         "out/<condition>_postmerge_parquet)")
    ap.add_argument("--max-min", type=float, default=60.0,
                    help="Max sim duration in minutes (default 60)")
    args = ap.parse_args()

    cache_dir = args.cache_dir or f"out/cache_{args.condition}_apr24"
    out_dir = args.out_dir or f"out/{args.condition}_postmerge_parquet"
    max_duration = int(args.max_min * 60)

    print(f"condition:   {args.condition}")
    print(f"cache:       {cache_dir}")
    print(f"parquet out: {out_dir}")
    print(f"max sim:     {max_duration}s ({max_duration/60:.0f} min)")
    print("-" * 60)
    os.makedirs(out_dir, exist_ok=True)

    core = build_core()
    t_wall0 = time.time()

    with parquet_emitter(
        out_dir=out_dir,
        experiment_id=f"{args.condition}_postmerge",
        lineage_seed=0,
        agent_id="0",
        generation=1,
    ):
        t_build = time.time()
        comp = build_composite("baseline", cache_dir=cache_dir)
        cell0 = comp.state["agents"]["0"]
        dry0 = float(cell0["listeners"]["mass"].get("dry_mass", 0))
        print(f"  composite built in {time.time()-t_build:.1f}s — "
              f"initial dry_mass={dry0:.1f} fg")

        total_run = 0.0
        divided = False
        while total_run < max_duration:
            chunk = min(SNAPSHOT_INTERVAL, max_duration - total_run)
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
            if int(total_run) % 300 == 0 or total_run >= max_duration - 60:
                print(f"  t={total_run/60:5.1f} min  dry_mass={dry:.1f} fg")
            if divided:
                break

        n_closed = ParquetEmitter.flush_all_in_composite(comp, success=divided)
        print(f"  flushed {n_closed} ParquetEmitter instance(s)")

    wall = time.time() - t_wall0
    print("-" * 60)
    print(f"final: sim={total_run/60:.1f} min, wall={wall:.0f}s, divided={divided}")


if __name__ == "__main__":
    main()
