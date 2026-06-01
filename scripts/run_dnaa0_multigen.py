"""dnaa-0 multi-generation succinate runner — proper multi-gen emitter.

The earlier scripts/run_dnaa0_smoke.py used the standalone
parquet_emitter() context manager. Under single_daughters=True, when the
parent agent splits, that emitter loses its observer relationship to the
daughter — all subsequent generations are invisible to the parquet hive
(verified empirically: 6-gen run produced only generation=1/agent_id=1
partition). This runner uses v2ecoli.library.parquet_run.run_multigen_parquet,
which rotates the emitter per generation (close + re-create with new
partition keys), so every gen lands as its own parquet partition that
diagonal_relaxed concat picks up at render time.

Usage::

    python scripts/run_dnaa0_multigen.py --duration 29520 --max-generations 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from v2ecoli import build_composite  # noqa: E402
from v2ecoli.library.parquet_run import run_multigen_parquet  # noqa: E402


STUDY_SLUG = "dnaa-0-parameter-foundation"
INVESTIGATION_SLUG = "dnaa-replication"

# Per-agent listener paths to capture. The runner threads these through
# the per-generation emitter so every cycle's data lands in its
# generation=<N>/agent_id=<id>/ partition.
EMIT_PATHS = [
    "agents/0/listeners/replication_data/number_of_oric",
    "agents/0/listeners/mass/cell_mass",
    "agents/0/listeners/mass/dry_mass",
    "agents/0/listeners/monomer_counts",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=int, default=29520,
                   help="Sim seconds (default 29520 ≈ 6 generations at τ=82 min)")
    p.add_argument("--max-generations", type=int, default=6)
    p.add_argument("--chunk", type=int, default=60,
                   help="Sim seconds per composite.run() chunk (default 60)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="out/cache-succinate")
    p.add_argument("--sim-name", default=None)
    p.add_argument("--out", default=None,
                   help="JSON summary path (default: studies/<slug>/sims/<sim_name>.json)")
    args = p.parse_args()

    sim_name = args.sim_name or f"dnaa0-multigen-seed{args.seed}"
    out_path = args.out or f"studies/{STUDY_SLUG}/sims/{sim_name}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t_load = time.time()
    composite = build_composite("baseline", cache_dir=args.cache_dir, seed=args.seed)
    load_time = time.time() - t_load
    print(f"composite built in {load_time:.1f}s")

    t_run = time.time()
    result = run_multigen_parquet(
        composite,
        experiment_id=sim_name,
        out_dir=f"studies/{STUDY_SLUG}/parquet-runs",
        emit_paths=EMIT_PATHS,
        max_steps=args.duration,
        max_generations=args.max_generations,
        chunk=args.chunk,
        initial_agent_id="0",
        initial_lineage_seed=args.seed,
        single_daughters=True,
        batch_size=400,
        threaded=False,  # avoid the partial-batch race we hit earlier
        study_slug=STUDY_SLUG,
        investigation_slug=INVESTIGATION_SLUG,
    )
    wall = time.time() - t_run

    summary = {
        "study": STUDY_SLUG,
        "investigation": INVESTIGATION_SLUG,
        "sim_name": sim_name,
        "cache_dir": args.cache_dir,
        "seed": args.seed,
        "max_generations": args.max_generations,
        "chunk": args.chunk,
        "duration_requested": args.duration,
        "load_time": load_time,
        "wall_time": wall,
        "run_result": result,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote {out_path}")
    print(f"  sim_name: {sim_name}")
    print(f"  steps completed: {result.get('steps')}")
    print(f"  generations captured: {result.get('generations')}")
    print(f"  out_dir: {result.get('out_dir')}")
    print(f"  wall: {wall:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
