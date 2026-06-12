#!/usr/bin/env python3
"""Fair-lineage v2ecoli multiseed/multigen driver for the perf comparison.

Mirrors scripts/run_phase0_multigen.py but runs a LINEAR lineage
(single_daughters=True) to match vEcoli's single_daughters=True convention —
so both engines simulate the same number of cells (n_seeds × generations),
not 2^gen. Writes the summary.json shape perf_compare.py consumes.

  python scripts/_perf_v2_driver.py --n-seeds 2 --max-generations 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from v2ecoli import build_composite                       # noqa: E402
from v2ecoli.library.sqlite_run import run_multigen_sqlite  # noqa: E402
from v2ecoli.composites._helpers import set_null_emitter_override  # noqa: E402

# The baseline composite declares an internal local:ParquetEmitter that
# captures the FULL WCM state every step (falling back to RAMEmitter, which
# keeps tree_copy(state) forever → tens of GB over a multigen run). This runner
# drives its OWN external SQLiteEmitter (filtered to 6 leaf paths), so the
# internal full-state emitter is pure dead weight. Minimise it to global_time
# only — THE fix for the 57 GB peak RSS. (Applies process-wide; set once.)
set_null_emitter_override(True)

CACHE_DIR = "out/cache"
OUT_ROOT = Path(".pbg/runs/perf-v2")
EMIT_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.volume",
    "listeners.mass.growth",
    "listeners.mass.instantaneous_growth_rate",
]


def run_one(seed, max_steps, max_generations, chunk):
    out_dir = OUT_ROOT / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_file = out_dir / "run.db"
    if db_file.exists():
        db_file.unlink()
    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    try:
        result = run_multigen_sqlite(
            composite, run_id=f"perf-v2-seed{seed:02d}", db_file=str(db_file),
            emit_paths=EMIT_PATHS, max_steps=max_steps,
            max_generations=max_generations, chunk=chunk,
            initial_agent_id="0", single_daughters=True,   # ← linear lineage (fair)
        )
    except Exception as e:
        print(f"  seed={seed:02d} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return {"seed": seed, "error": str(e), "type": type(e).__name__,
                "wall_seconds": round(time.time() - t0, 2)}
    wall = time.time() - t0
    s = {"seed": seed, "max_steps": max_steps, "max_generations": max_generations,
         "actual_steps": result.get("steps"),
         "actual_generations_seen": result.get("generations", []),
         "wall_seconds": round(wall, 2)}
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"  seed={seed:02d}: wall={wall:6.1f}s steps={result.get('steps')} "
          f"gens={result.get('generations')}", flush=True)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--max-generations", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=8000,
                   help="hard tick cap; generous so max-generations is the binding limit")
    p.add_argument("--chunk", type=int, default=60)
    a = p.parse_args()
    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"v2ecoli perf driver: {a.n_seeds} seeds × ≤{a.max_generations} gens "
          f"(single_daughters, max_steps={a.max_steps})", flush=True)
    t0 = time.time()
    per_seed = [run_one(s, a.max_steps, a.max_generations, a.chunk) for s in range(a.n_seeds)]
    total = time.time() - t0
    ok = [r for r in per_seed if "error" not in r]
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "n_seeds_requested": a.n_seeds, "n_seeds_successful": len(ok),
        "max_generations": a.max_generations, "max_steps": a.max_steps,
        "single_daughters": True,
        "total_wall_seconds": round(total, 2), "per_seed": per_seed,
    }, indent=2))
    print(f"\nDone: {len(ok)}/{a.n_seeds} seeds, total {total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
