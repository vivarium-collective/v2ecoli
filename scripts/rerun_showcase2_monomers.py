"""Re-run a showcase-2 multiseed ensemble that emits listeners.monomer_counts,
for regenerating protein_counts_validation after the monomer-counts
accumulate-listener fix (counts_deriver: monomer_counts now uses
overwrite[...] = SET, not the Array-default ACCUMULATE).

Writes the parquet hive under <out-dir>/showcase2_baseline/history so
render_showcase2 / the protein_counts_validation analysis can consume it.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = "out/cache"
EXPERIMENT_ID = "showcase2_baseline"

EMIT_PATHS = [
    "listeners.monomer_counts",
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
]


def run_one(seed: int, out_dir: str, max_steps: int, max_generations: int,
            chunk: int) -> dict:
    from v2ecoli import build_composite
    from v2ecoli.library.parquet_run import run_multigen_parquet
    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=seed)
    result = run_multigen_parquet(
        composite,
        experiment_id=EXPERIMENT_ID,
        out_dir=out_dir,
        emit_paths=EMIT_PATHS,
        max_steps=max_steps,
        max_generations=max_generations,
        chunk=chunk,
        initial_agent_id="0",
        initial_lineage_seed=seed,
    )
    wall = time.time() - t0
    print(f"  seed={seed:02d}: wall={wall:6.1f}s steps={result.get('steps')} "
          f"generations={result.get('generations')}", flush=True)
    return {"seed": seed, "parquet_steps": result.get("steps"),
            "parquet_generations": result.get("generations", []),
            "wall_seconds": round(wall, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=6500)
    p.add_argument("--max-generations", type=int, default=2)
    p.add_argument("--chunk", type=int, default=60)
    p.add_argument("--parallel", choices=["ray", "off"], default="ray")
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    out_dir = str(Path(args.out_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    from v2ecoli.library.parallel_seeds import run_seeds_parallel
    print(f"showcase-2 monomer re-run: N={args.n_seeds} seeds x "
          f"<={args.max_generations} gens x {args.max_steps} steps", flush=True)
    run = run_seeds_parallel(
        range(args.n_seeds), run_one,
        mode=(None if args.parallel == "off" else "ray"),
        run_kwargs=dict(out_dir=out_dir, max_steps=args.max_steps,
                        max_generations=args.max_generations, chunk=args.chunk),
        ray_env={"PYTHONPATH": str(REPO_ROOT)},
    )
    summary = {
        "n_seeds_requested": args.n_seeds, "max_steps": args.max_steps,
        "max_generations": args.max_generations,
        "total_wall_seconds": run.wall_s, "parallel_mode": run.mode,
        "experiment_id": EXPERIMENT_ID, "out_dir": out_dir,
        "per_seed": run.results,
    }
    Path(out_dir, "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: total wall {run.wall_s/60:.1f} min ({run.mode})", flush=True)


if __name__ == "__main__":
    main()
