"""Relaunch the showcase-2 baseline CHROMO ensemble with child_domains emitted.

Mirrors the original showcase2-baseline-chromo run (2 seeds x <=3 gens x 6500
steps, Ray-parallel parquet, V2ECOLI_EMIT_UNIQUE=1) but on a branch where the
unique emit also persists chromosome_domain__child_domains (the parent->child
domain tree, flattened to list<int>). With that column the chromosome_snapshots
renderer places daughter-strand RNAPs ON the replication bubbles, not just the
chromosome rim.

Run from the worktree root with V2ECOLI_EMIT_UNIQUE=1:
    V2ECOLI_EMIT_UNIQUE=1 .venv/bin/python scripts/relaunch_chromo_bubbles.py \
        --out-dir /tmp/v2e-chromo-bubbles-run/.pbg/runs/showcase2-baseline-chromo
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
    "listeners.mass.cell_mass",
    "listeners.mass.dna_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.dry_mass_fold_change",
    "listeners.mass.growth",
    "listeners.mass.instantaneous_growth_rate",
    "listeners.mass.mRna_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.rRna_mass",
    "listeners.mass.rna_mass",
    "listeners.mass.smallMolecule_mass",
    "listeners.mass.tRna_mass",
    "listeners.mass.volume",
    "listeners.mass.water_mass",
    "listeners.growth_limits.fraction_trna_charged",
    "listeners.growth_limits.ppgpp_conc",
    "listeners.replication_data.fork_coordinates",
    "listeners.replication_data.number_of_oric",
    "listeners.ribosome_data.actual_elongations",
    "listeners.ribosome_data.did_terminate",
    "listeners.ribosome_data.effective_elongation_rate",
]


def run_one(seed: int, out_dir: str, max_steps: int, max_generations: int,
            chunk: int) -> dict:
    # V2ECOLI_EMIT_UNIQUE is read at import of parquet_run; ensure it's set in
    # THIS (worker) process before importing.
    os.environ.setdefault("V2ECOLI_EMIT_UNIQUE", "1")
    from v2ecoli import build_composite
    from v2ecoli.library.parquet_run import run_multigen_parquet

    t0 = time.time()
    composite = build_composite("ecoli_baseline", cache_dir=CACHE_DIR, seed=seed)
    try:
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
    except Exception as e:
        print(f"  seed={seed:02d} FAILED: {type(e).__name__}: {str(e)[:120]}")
        return {"seed": seed, "error": str(e), "type": type(e).__name__,
                "wall_seconds": round(time.time() - t0, 2)}
    wall = time.time() - t0
    print(f"  seed={seed:02d}: wall={wall:6.1f}s steps={result.get('steps')} "
          f"generations={result.get('generations')}")
    return {"seed": seed, "parquet_steps": result.get("steps"),
            "parquet_generations": result.get("generations", []),
            "wall_seconds": round(wall, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True,
                   help="run dir; parquet hive goes under <out-dir>/<experiment_id>/history")
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=6500)
    p.add_argument("--max-generations", type=int, default=3)
    p.add_argument("--chunk", type=int, default=60)
    p.add_argument("--parallel", choices=["ray", "off"], default="ray")
    args = p.parse_args()

    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    out_dir = str(Path(args.out_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    from v2ecoli.library.parallel_seeds import run_seeds_parallel

    print(f"showcase-2 baseline CHROMO ensemble (child_domains): N={args.n_seeds} "
          f"seeds x <={args.max_generations} gens x {args.max_steps} steps "
          f"(chunk={args.chunk}, parallel={args.parallel})", flush=True)

    run = run_seeds_parallel(
        range(args.n_seeds), run_one,
        mode=(None if args.parallel == "off" else "ray"),
        run_kwargs=dict(out_dir=out_dir, max_steps=args.max_steps,
                        max_generations=args.max_generations, chunk=args.chunk),
        ray_env={"V2ECOLI_EMIT_UNIQUE": "1",
                 "PYTHONPATH": str(REPO_ROOT)},
    )
    if run.mode == "ray":
        print(f"  [ray] {run.n_seeds} seeds, {run.n_threads_per_worker} thread(s)/worker",
              flush=True)

    summary = {
        "n_seeds_requested": args.n_seeds,
        "max_steps": args.max_steps,
        "max_generations": args.max_generations,
        "total_wall_seconds": run.wall_s,
        "parallel_mode": run.mode,
        "experiment_id": EXPERIMENT_ID,
        "out_dir": out_dir,
        "per_seed": run.results,
    }
    Path(out_dir, "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: total wall {run.wall_s/60:.1f} min ({run.mode})", flush=True)
    for r in run.results:
        print(f"  seed_{r['seed']:02d}: parquet_gens={r.get('parquet_generations')} "
              f"wall={r.get('wall_seconds')}s", flush=True)


if __name__ == "__main__":
    main()
